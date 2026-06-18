"""Unit tests for agentkavach.sender — OTel exporter and TracerProvider setup."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agentkavach.sender import (
    _DEFAULT_RETRY_AFTER,
    _INGEST_TIMEOUT_SECONDS,
    _MAX_EXPORT_BATCH_SIZE,
    _MAX_QUEUE_SIZE,
    _MAX_RETRIES,
    _MAX_RETRY_AFTER,
    _SCHEDULE_DELAY_MILLIS,
    DEV_BACKEND_URL,
    PROD_BACKEND_URL,
    AgentKavachExporter,
    create_tracer_provider,
    resolve_backend_url,
)

try:
    from opentelemetry.sdk.trace.export import SpanExportResult
except ImportError:
    pytest.skip("opentelemetry-sdk not installed", allow_module_level=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_span(
    agent: str = "test-agent",
    model: str = "gpt-4o",
    provider: str = "openai",
    tokens_in: int = 100,
    tokens_out: int = 50,
    cost: float = 0.05,
    partial: bool = False,
) -> MagicMock:
    """Create a mock ReadableSpan with gen_ai attributes."""
    span = MagicMock()
    span.attributes = {
        "gen_ai.agent.name": agent,
        "gen_ai.request.model": model,
        "gen_ai.system": provider,
        "gen_ai.usage.input_tokens": tokens_in,
        "gen_ai.usage.output_tokens": tokens_out,
        "gen_ai.usage.cost": cost,
        "agentkavach.partial": partial,
    }
    span.start_time = 1710288000_000_000_000  # 2024-03-13T00:00:00Z in ns
    span.end_time = 1710288000_150_000_000  # +150ms
    span.context = SimpleNamespace(
        trace_id=0x1234567890ABCDEF1234567890ABCDEF,
        span_id=0x1234567890ABCDEF,
    )
    return span


def _mock_response(status_code: int = 202, headers: dict | None = None, text: str = ""):
    """Create a mock httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.headers = headers or {}
    return resp


def _make_exporter(buffer=None) -> tuple[AgentKavachExporter, MagicMock]:
    """Create an exporter with a mocked HTTP client."""
    exporter = AgentKavachExporter(api_key="cg_test", buffer=buffer)
    mock_client = MagicMock()
    exporter._client = mock_client
    return exporter, mock_client


# ---------------------------------------------------------------------------
# AgentKavachExporter — constructor
# ---------------------------------------------------------------------------


class TestAgentKavachExporter:
    def test_empty_api_key_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            AgentKavachExporter(api_key="")

    def test_endpoint_default(self, monkeypatch):
        # Legacy cg_ keys resolve to the prod Render backend.
        monkeypatch.delenv("AGENTKAVACH_BACKEND_URL", raising=False)
        exporter = AgentKavachExporter(api_key="cg_test_key")
        assert exporter._endpoint == f"{PROD_BACKEND_URL}/v1/ingest"
        exporter.shutdown()

    def test_endpoint_trailing_slash_handled(self):
        exporter = AgentKavachExporter(api_key="cg_test", endpoint="https://example.com/")
        assert exporter._endpoint == "https://example.com/v1/ingest"
        exporter.shutdown()

    def test_authorization_header(self):
        exporter = AgentKavachExporter(api_key="cg_secret_123")
        assert exporter._headers["Authorization"] == "Bearer cg_secret_123"
        exporter.shutdown()

    def test_api_key_not_in_url(self):
        exporter = AgentKavachExporter(api_key="cg_secret_123")
        assert "cg_secret_123" not in exporter._endpoint
        exporter.shutdown()

    def test_buffer_optional(self):
        exporter = AgentKavachExporter(api_key="cg_test")
        assert exporter._buffer is None
        exporter.shutdown()

    def test_buffer_wired(self):
        buf = MagicMock()
        exporter = AgentKavachExporter(api_key="cg_test", buffer=buf)
        assert exporter._buffer is buf
        exporter.shutdown()

    def test_retry_stats_initial(self):
        exporter = AgentKavachExporter(api_key="cg_test")
        assert exporter.retry_stats == {"retries": 0, "buffered": 0, "replayed": 0}
        exporter.shutdown()


class TestSpanToEvent:
    def test_basic_conversion(self):
        span = _make_span()
        event = AgentKavachExporter._span_to_event(span)
        assert event["agent_name"] == "test-agent"
        assert event["model"] == "gpt-4o"
        assert event["provider"] == "openai"
        assert event["input_tokens"] == 100
        assert event["output_tokens"] == 50
        assert event["cost"] == 0.05
        assert event["duration_ms"] == 150
        assert event["timestamp"] == "2024-03-13T00:00:00+00:00"

    def test_missing_attributes(self):
        # Phase 35: timestamp falls back to now() instead of None so
        # the server's IngestEvent (required str field) doesn't 422
        # the whole batch when an upstream instrumentation bug drops
        # start_time.
        from datetime import datetime as _dt

        span = MagicMock()
        span.attributes = {}
        span.start_time = None
        span.end_time = None
        span.context = None
        event = AgentKavachExporter._span_to_event(span)
        assert event["agent_name"] == "unknown"
        assert event["model"] == "unknown"
        assert event["duration_ms"] == 0
        # timestamp must be a non-empty ISO 8601 string close to now
        assert isinstance(event["timestamp"], str)
        parsed = _dt.fromisoformat(event["timestamp"])
        assert (_dt.now(parsed.tzinfo) - parsed).total_seconds() < 5


# ---------------------------------------------------------------------------
# Phase 35: idempotency_key + duration rounding + timestamp fallback
# ---------------------------------------------------------------------------


class TestSpanToEventPhase35:
    def test_idempotency_key_propagated_from_span(self):
        # The client wrapper sets agentkavach.idempotency_key per span.
        # _span_to_event must include it so the server's dedup path
        # (existing_idem check) actually fires on retry.
        span = _make_span()
        span.attributes["agentkavach.idempotency_key"] = "deterministic-uuid-xyz"
        event = AgentKavachExporter._span_to_event(span)
        assert event["idempotency_key"] == "deterministic-uuid-xyz"

    def test_idempotency_key_omitted_when_missing(self):
        # Older SDKs / non-client-wrapped spans won't set the attr.
        # Keep the field absent rather than emitting null so server
        # validation passes either way.
        span = _make_span()
        event = AgentKavachExporter._span_to_event(span)
        assert "idempotency_key" not in event

    def test_duration_rounded_not_truncated(self):
        # Pre-Phase-35 used integer division (//) which truncates.
        # A 1.7 ms span became 1 ms; with round() it's 2 ms.
        span = MagicMock()
        span.attributes = {}
        span.start_time = 1_000_000_000  # 1 s
        span.end_time = 1_001_700_000  # 1.7 ms later
        span.context = None
        event = AgentKavachExporter._span_to_event(span)
        assert event["duration_ms"] == 2  # was 1 under //

    def test_timestamp_fallback_when_start_time_zero(self):
        # 0 start_time is treated as falsy → fallback path. Confirm
        # we get a valid ISO string anyway.
        from datetime import datetime as _dt

        span = MagicMock()
        span.attributes = {}
        span.start_time = 0
        span.end_time = 0
        span.context = None
        event = AgentKavachExporter._span_to_event(span)
        assert isinstance(event["timestamp"], str)
        _dt.fromisoformat(event["timestamp"])  # must parse

    def test_event_is_json_serializable(self):
        span = _make_span()
        event = AgentKavachExporter._span_to_event(span)
        serialized = json.dumps(event)
        assert isinstance(serialized, str)

    def test_prompt_included_when_present(self):
        span = _make_span()
        span.attributes["gen_ai.prompt"] = "What is Python?"
        event = AgentKavachExporter._span_to_event(span)
        assert event["prompt"] == "What is Python?"

    def test_prompt_excluded_when_absent(self):
        span = _make_span()
        event = AgentKavachExporter._span_to_event(span)
        assert "prompt" not in event


# ---------------------------------------------------------------------------
# Export — success path
# ---------------------------------------------------------------------------


class TestExport:
    def test_empty_spans_returns_success(self):
        exporter = AgentKavachExporter(api_key="cg_test")
        result = exporter.export([])
        assert result == SpanExportResult.SUCCESS
        exporter.shutdown()

    def test_successful_export(self):
        exporter, mock_client = _make_exporter()
        mock_client.post.return_value = _mock_response(202)

        result = exporter.export([_make_span()])

        assert result == SpanExportResult.SUCCESS
        mock_client.post.assert_called_once()

        # Verify gzip-compressed body.
        call_kwargs = mock_client.post.call_args
        import gzip as _gzip
        import json as _json

        content = call_kwargs.kwargs.get("content") or call_kwargs[1].get("content")
        payload = _json.loads(_gzip.decompress(content))
        assert "events" in payload
        assert len(payload["events"]) == 1
        assert payload["events"][0]["model"] == "gpt-4o"

    def test_server_error_returns_failure(self):
        exporter, mock_client = _make_exporter()
        mock_client.post.return_value = _mock_response(500, text="Internal Server Error")

        result = exporter.export([_make_span()])
        assert result == SpanExportResult.FAILURE

    def test_timeout_returns_failure_after_retries(self):
        import httpx

        exporter, mock_client = _make_exporter()
        mock_client.post.side_effect = httpx.TimeoutException("timed out")

        with patch("agentkavach.sender.time.sleep"):
            result = exporter.export([_make_span()])
        assert result == SpanExportResult.FAILURE
        # Initial attempt + _MAX_RETRIES retries
        assert mock_client.post.call_count == _MAX_RETRIES + 1

    def test_network_error_returns_failure(self):
        import httpx

        exporter, mock_client = _make_exporter()
        mock_client.post.side_effect = httpx.ConnectError("connection refused")

        result = exporter.export([_make_span()])
        assert result == SpanExportResult.FAILURE


# ---------------------------------------------------------------------------
# Retry on 429
# ---------------------------------------------------------------------------


class TestRetryOn429:
    def test_429_retries_then_succeeds(self):
        """429 on first attempt, 202 on second — should succeed."""
        exporter, mock_client = _make_exporter()
        mock_client.post.side_effect = [
            _mock_response(429, headers={"retry-after": "1"}),
            _mock_response(202),
        ]

        with patch("agentkavach.sender.time.sleep") as mock_sleep:
            result = exporter.export([_make_span()])

        assert result == SpanExportResult.SUCCESS
        assert mock_client.post.call_count == 2
        mock_sleep.assert_called_once_with(1.0)
        assert exporter.retry_stats["retries"] == 1

    def test_429_exhausts_retries(self):
        """429 on all attempts — should fail after _MAX_RETRIES."""
        exporter, mock_client = _make_exporter()
        mock_client.post.return_value = _mock_response(429, headers={"retry-after": "1"})

        with patch("agentkavach.sender.time.sleep"):
            result = exporter.export([_make_span()])

        assert result == SpanExportResult.FAILURE
        # 1 initial + 3 retries = 4 total
        assert mock_client.post.call_count == _MAX_RETRIES + 1
        assert exporter.retry_stats["retries"] == _MAX_RETRIES

    def test_429_honours_retry_after_header(self):
        """Should sleep for the duration specified in Retry-After."""
        exporter, mock_client = _make_exporter()
        mock_client.post.side_effect = [
            _mock_response(429, headers={"retry-after": "3"}),
            _mock_response(202),
        ]

        with patch("agentkavach.sender.time.sleep") as mock_sleep:
            exporter.export([_make_span()])

        mock_sleep.assert_called_once_with(3.0)

    def test_429_missing_retry_after_uses_default(self):
        """No Retry-After header → uses _DEFAULT_RETRY_AFTER."""
        exporter, mock_client = _make_exporter()
        mock_client.post.side_effect = [
            _mock_response(429, headers={}),
            _mock_response(202),
        ]

        with patch("agentkavach.sender.time.sleep") as mock_sleep:
            exporter.export([_make_span()])

        mock_sleep.assert_called_once_with(_DEFAULT_RETRY_AFTER)

    def test_429_retry_after_capped_at_max(self):
        """Absurdly large Retry-After is capped at _MAX_RETRY_AFTER."""
        exporter, mock_client = _make_exporter()
        mock_client.post.side_effect = [
            _mock_response(429, headers={"retry-after": "999"}),
            _mock_response(202),
        ]

        with patch("agentkavach.sender.time.sleep") as mock_sleep:
            exporter.export([_make_span()])

        mock_sleep.assert_called_once_with(_MAX_RETRY_AFTER)

    def test_429_retry_after_floor(self):
        """Retry-After of 0 or negative is floored to _DEFAULT_RETRY_AFTER."""
        exporter, mock_client = _make_exporter()
        mock_client.post.side_effect = [
            _mock_response(429, headers={"retry-after": "0"}),
            _mock_response(202),
        ]

        with patch("agentkavach.sender.time.sleep") as mock_sleep:
            exporter.export([_make_span()])

        mock_sleep.assert_called_once_with(_DEFAULT_RETRY_AFTER)

    def test_429_retry_after_non_numeric(self):
        """Non-numeric Retry-After falls back to default."""
        exporter, mock_client = _make_exporter()
        mock_client.post.side_effect = [
            _mock_response(429, headers={"retry-after": "abc"}),
            _mock_response(202),
        ]

        with patch("agentkavach.sender.time.sleep") as mock_sleep:
            exporter.export([_make_span()])

        mock_sleep.assert_called_once_with(_DEFAULT_RETRY_AFTER)

    def test_timeout_retries_with_backoff(self):
        """Timeouts retry with exponential backoff."""
        import httpx

        exporter, mock_client = _make_exporter()
        mock_client.post.side_effect = [
            httpx.TimeoutException("t/o"),
            httpx.TimeoutException("t/o"),
            _mock_response(202),
        ]

        with patch("agentkavach.sender.time.sleep") as mock_sleep:
            result = exporter.export([_make_span()])

        assert result == SpanExportResult.SUCCESS
        # Backoff: 1.0 * 2^0 = 1.0, 1.0 * 2^1 = 2.0
        assert mock_sleep.call_args_list[0][0][0] == _DEFAULT_RETRY_AFTER * 1
        assert mock_sleep.call_args_list[1][0][0] == _DEFAULT_RETRY_AFTER * 2


# ---------------------------------------------------------------------------
# Buffer integration — events survive 429 exhaustion
# ---------------------------------------------------------------------------


class TestBufferOnFailure:
    def test_429_exhaustion_buffers_events(self):
        """When retries exhausted on 429, events are written to buffer."""
        buf = MagicMock()
        buf.count.return_value = 0
        exporter, mock_client = _make_exporter(buffer=buf)
        mock_client.post.return_value = _mock_response(429, headers={"retry-after": "1"})

        with patch("agentkavach.sender.time.sleep"):
            exporter.export([_make_span(), _make_span()])

        # 2 events should be written to buffer individually
        assert buf.write.call_count == 2
        assert exporter.retry_stats["buffered"] == 2

    def test_timeout_exhaustion_buffers_events(self):
        """When all timeouts exhaust retries, events are buffered."""
        import httpx

        buf = MagicMock()
        buf.count.return_value = 0
        exporter, mock_client = _make_exporter(buffer=buf)
        mock_client.post.side_effect = httpx.TimeoutException("t/o")

        with patch("agentkavach.sender.time.sleep"):
            exporter.export([_make_span()])

        assert buf.write.call_count == 1
        assert exporter.retry_stats["buffered"] == 1

    def test_network_error_buffers_events(self):
        """Network errors (not timeout) buffer immediately — no retry."""
        import httpx

        buf = MagicMock()
        buf.count.return_value = 0
        exporter, mock_client = _make_exporter(buffer=buf)
        mock_client.post.side_effect = httpx.ConnectError("refused")

        exporter.export([_make_span()])

        assert buf.write.call_count == 1
        # Network errors don't retry, so only 1 POST attempt
        assert mock_client.post.call_count == 1

    def test_no_buffer_no_crash_on_failure(self):
        """Without a buffer, 429 exhaustion still returns FAILURE (no crash)."""
        exporter, mock_client = _make_exporter(buffer=None)
        mock_client.post.return_value = _mock_response(429, headers={"retry-after": "1"})

        with patch("agentkavach.sender.time.sleep"):
            result = exporter.export([_make_span()])

        assert result == SpanExportResult.FAILURE

    def test_server_500_does_not_buffer(self):
        """500 errors are not retried and not buffered — only 429 and timeouts."""
        buf = MagicMock()
        buf.count.return_value = 0
        exporter, mock_client = _make_exporter(buffer=buf)
        mock_client.post.return_value = _mock_response(500, text="Server Error")

        exporter.export([_make_span()])

        buf.write.assert_not_called()


# ---------------------------------------------------------------------------
# Buffer replay on success
# ---------------------------------------------------------------------------


class TestBufferReplay:
    def test_success_replays_buffered_events(self):
        """After successful export, buffered events are replayed."""
        buf = MagicMock()
        buf.count.return_value = 2
        buf.read_all.return_value = [
            {"agent": "a", "cost": 0.01},
            {"agent": "b", "cost": 0.02},
        ]

        exporter, mock_client = _make_exporter(buffer=buf)
        # First call: main export succeeds. Second call: replay succeeds.
        mock_client.post.return_value = _mock_response(202)

        exporter.export([_make_span()])

        # Main export + replay chunk
        assert mock_client.post.call_count == 2
        buf.purge.assert_called_once_with(2)
        assert exporter.retry_stats["replayed"] == 2

    def test_no_replay_when_buffer_empty(self):
        """No replay attempt when buffer has 0 events."""
        buf = MagicMock()
        buf.count.return_value = 0

        exporter, mock_client = _make_exporter(buffer=buf)
        mock_client.post.return_value = _mock_response(202)

        exporter.export([_make_span()])

        # Only 1 call — the main export. No replay.
        assert mock_client.post.call_count == 1
        buf.read_all.assert_not_called()

    def test_replay_stops_on_429(self):
        """Buffer replay stops if it hits 429 — leaves remaining in buffer."""
        buf = MagicMock()
        buf.count.return_value = 3
        buf.read_all.return_value = [
            {"agent": "a"},
            {"agent": "b"},
            {"agent": "c"},
        ]

        exporter, mock_client = _make_exporter(buffer=buf)
        # Main export succeeds, replay gets 429
        mock_client.post.side_effect = [
            _mock_response(202),
            _mock_response(429),
        ]

        exporter.export([_make_span()])

        # No purge — 0 events replayed (the entire chunk hit 429)
        buf.purge.assert_not_called()
        assert exporter.retry_stats["replayed"] == 0

    def test_replay_chunks_large_buffer(self):
        """Buffer with > _MAX_EXPORT_BATCH_SIZE events is sent in chunks."""
        chunk_size = _MAX_EXPORT_BATCH_SIZE
        events = [{"agent": f"a-{i}"} for i in range(chunk_size + 10)]
        buf = MagicMock()
        buf.count.return_value = len(events)
        buf.read_all.return_value = events

        exporter, mock_client = _make_exporter(buffer=buf)
        mock_client.post.return_value = _mock_response(202)

        exporter.export([_make_span()])

        # 1 main export + 2 replay chunks (512 + 10)
        assert mock_client.post.call_count == 3
        buf.purge.assert_called_once_with(chunk_size + 10)

    def test_replay_not_attempted_on_export_failure(self):
        """Buffer replay only runs after successful export, not on failure."""
        buf = MagicMock()
        buf.count.return_value = 5

        exporter, mock_client = _make_exporter(buffer=buf)
        mock_client.post.return_value = _mock_response(500, text="error")

        exporter.export([_make_span()])

        buf.read_all.assert_not_called()

    def test_replay_partial_success(self):
        """First replay chunk succeeds, second fails — only purge first chunk."""
        chunk_size = _MAX_EXPORT_BATCH_SIZE
        events = [{"agent": f"a-{i}"} for i in range(chunk_size + 10)]
        buf = MagicMock()
        buf.count.return_value = len(events)
        buf.read_all.return_value = events

        exporter, mock_client = _make_exporter(buffer=buf)
        mock_client.post.side_effect = [
            _mock_response(202),  # main export
            _mock_response(202),  # first replay chunk (512)
            _mock_response(500),  # second replay chunk (10) — fails
        ]

        exporter.export([_make_span()])

        # Only first chunk replayed
        buf.purge.assert_called_once_with(chunk_size)
        assert exporter.retry_stats["replayed"] == chunk_size


# ---------------------------------------------------------------------------
# _parse_retry_after
# ---------------------------------------------------------------------------


class TestParseRetryAfter:
    def test_valid_integer(self):
        resp = _mock_response(429, headers={"retry-after": "5"})
        assert AgentKavachExporter._parse_retry_after(resp) == 5.0

    def test_valid_float(self):
        resp = _mock_response(429, headers={"retry-after": "1.5"})
        assert AgentKavachExporter._parse_retry_after(resp) == 1.5

    def test_zero_floored(self):
        resp = _mock_response(429, headers={"retry-after": "0"})
        assert AgentKavachExporter._parse_retry_after(resp) == _DEFAULT_RETRY_AFTER

    def test_negative_floored(self):
        resp = _mock_response(429, headers={"retry-after": "-5"})
        assert AgentKavachExporter._parse_retry_after(resp) == _DEFAULT_RETRY_AFTER

    def test_huge_value_capped(self):
        resp = _mock_response(429, headers={"retry-after": "300"})
        assert AgentKavachExporter._parse_retry_after(resp) == _MAX_RETRY_AFTER

    def test_missing_header(self):
        resp = _mock_response(429, headers={})
        assert AgentKavachExporter._parse_retry_after(resp) == _DEFAULT_RETRY_AFTER

    def test_non_numeric(self):
        resp = _mock_response(429, headers={"retry-after": "tomorrow"})
        assert AgentKavachExporter._parse_retry_after(resp) == _DEFAULT_RETRY_AFTER


# ---------------------------------------------------------------------------
# TracerProvider factory
# ---------------------------------------------------------------------------


class TestCreateTracerProvider:
    def test_creates_provider(self):
        provider = create_tracer_provider(api_key="cg_test_key", agent_name="test-agent")
        assert provider is not None
        # Resource should carry our service name.
        resource_attrs = dict(provider.resource.attributes)
        assert resource_attrs["service.name"] == "agentkavach"
        assert resource_attrs["agentkavach.agent.name"] == "test-agent"
        provider.shutdown()

    def test_empty_api_key_raises(self):
        with pytest.raises(ValueError, match="API key required"):
            create_tracer_provider(api_key="")

    def test_api_key_not_read_from_env(self, monkeypatch):
        # The key is never read from the environment — an env var alone must
        # not satisfy the requirement; the call still raises.
        monkeypatch.setenv("AGENTKAVACH_API_KEY", "cg_from_env")
        with pytest.raises(ValueError, match="API key required"):
            create_tracer_provider(api_key="")

    def test_custom_endpoint(self):
        provider = create_tracer_provider(
            api_key="cg_test",
            endpoint="https://custom.example.com",
        )
        provider.shutdown()

    def test_env_endpoint_fallback(self, monkeypatch):
        monkeypatch.setenv("AGENTKAVACH_BACKEND_URL", "https://env.example.com")
        provider = create_tracer_provider(api_key="cg_test")
        provider.shutdown()

    def test_buffer_passed_to_exporter(self):
        buf = MagicMock()
        provider = create_tracer_provider(api_key="cg_test", buffer=buf)
        # The exporter inside the processor should have the buffer.
        # We can't easily inspect it, but at least it doesn't crash.
        provider.shutdown()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_queue_size(self):
        assert _MAX_QUEUE_SIZE == 10_000

    def test_batch_size(self):
        assert _MAX_EXPORT_BATCH_SIZE == 512

    def test_flush_interval(self):
        assert _SCHEDULE_DELAY_MILLIS == 5_000

    def test_timeout(self):
        assert _INGEST_TIMEOUT_SECONDS == 10

    def test_max_retries(self):
        assert _MAX_RETRIES == 3

    def test_default_retry_after(self):
        assert _DEFAULT_RETRY_AFTER == 1.0

    def test_max_retry_after(self):
        assert _MAX_RETRY_AFTER == 10.0


# ---------------------------------------------------------------------------
# Backend URL resolver (prefix-based routing)
# ---------------------------------------------------------------------------


class TestResolveBackendUrl:
    def test_ak_prod_key_routes_to_prod(self, monkeypatch):
        monkeypatch.delenv("AGENTKAVACH_BACKEND_URL", raising=False)
        assert resolve_backend_url("ak_prod_abc123") == PROD_BACKEND_URL

    def test_ak_dev_key_routes_to_dev(self, monkeypatch):
        monkeypatch.delenv("AGENTKAVACH_BACKEND_URL", raising=False)
        assert resolve_backend_url("ak_dev_abc123") == DEV_BACKEND_URL

    def test_legacy_cg_key_routes_to_prod(self, monkeypatch):
        # Existing pre-rebrand keys default to prod (where they were minted).
        monkeypatch.delenv("AGENTKAVACH_BACKEND_URL", raising=False)
        assert resolve_backend_url("cg_abc123") == PROD_BACKEND_URL

    def test_unknown_prefix_falls_back_to_prod(self, monkeypatch):
        monkeypatch.delenv("AGENTKAVACH_BACKEND_URL", raising=False)
        assert resolve_backend_url("garbage_xyz") == PROD_BACKEND_URL

    def test_env_var_overrides_prefix(self, monkeypatch):
        monkeypatch.setenv("AGENTKAVACH_BACKEND_URL", "http://localhost:8000")
        # Even a prod-prefixed key respects the explicit override.
        assert resolve_backend_url("ak_prod_abc123") == "http://localhost:8000"
        assert resolve_backend_url("ak_dev_abc123") == "http://localhost:8000"

    def test_prod_url_matches_render_service(self):
        assert PROD_BACKEND_URL == "https://agentcostguard-backend.onrender.com"

    def test_dev_url_matches_render_service(self):
        assert DEV_BACKEND_URL == "https://agentcostguard.onrender.com"


class TestExporterEndpointFromPrefix:
    """End-to-end: the exporter wires resolve_backend_url() correctly."""

    def test_ak_prod_key_targets_prod_ingest(self, monkeypatch):
        monkeypatch.delenv("AGENTKAVACH_BACKEND_URL", raising=False)
        exporter = AgentKavachExporter(api_key="ak_prod_abc123")
        assert exporter._endpoint == f"{PROD_BACKEND_URL}/v1/ingest"

    def test_ak_dev_key_targets_dev_ingest(self, monkeypatch):
        monkeypatch.delenv("AGENTKAVACH_BACKEND_URL", raising=False)
        exporter = AgentKavachExporter(api_key="ak_dev_abc123")
        assert exporter._endpoint == f"{DEV_BACKEND_URL}/v1/ingest"

    def test_explicit_endpoint_wins(self, monkeypatch):
        monkeypatch.setenv("AGENTKAVACH_BACKEND_URL", "http://from-env:8000")
        exporter = AgentKavachExporter(
            api_key="ak_prod_abc123",
            endpoint="http://from-arg:9000",
        )
        assert exporter._endpoint == "http://from-arg:9000/v1/ingest"


# ---------------------------------------------------------------------------
# Auth-failure handling (Phase 31): drop on 401, never buffer
# ---------------------------------------------------------------------------


class TestAuthFailureDropsEvents:
    """Phase 31 regression: 401/403 must drop events instead of buffering.

    Buffering an auth failure is pointless — every replay gets the same
    response and the buffer grows forever. The SDK should drop, log
    loudly at ERROR, and bump auth_failures in retry_stats.
    """

    def _exporter_with_response(self, status_code: int):
        # Buffer(path=None) auto-detects disk → would surface stale events
        # from local agent runs. Use a tracking mock instead so each test
        # gets a clean buffer that records writes deterministically.
        class _CountingBuffer:
            def __init__(self):
                self._events = []

            def write(self, event):
                self._events.append(event)

            def count(self):
                return len(self._events)

            def read_all(self):
                return list(self._events)

            def purge(self, n):
                self._events = self._events[n:]

        buf = _CountingBuffer()
        exporter = AgentKavachExporter(api_key="ak_prod_test", buffer=buf)
        resp = MagicMock()
        resp.status_code = status_code
        resp.text = "unauthorized"
        resp.headers = {}
        exporter._client = MagicMock()
        exporter._client.post = MagicMock(return_value=resp)
        return exporter, buf

    def test_401_response_does_not_buffer(self):
        exporter, buf = self._exporter_with_response(401)
        result = exporter._send_with_retry([{"foo": "bar"}])
        assert result == SpanExportResult.FAILURE
        assert buf.count() == 0  # events DROPPED, not buffered
        assert exporter.retry_stats.get("auth_failures") == 1
        assert exporter.retry_stats.get("dropped_on_auth") == 1

    def test_403_response_does_not_buffer(self):
        exporter, buf = self._exporter_with_response(403)
        result = exporter._send_with_retry([{"foo": "bar"}, {"foo": "baz"}])
        assert result == SpanExportResult.FAILURE
        assert buf.count() == 0
        assert exporter.retry_stats.get("auth_failures") == 1
        assert exporter.retry_stats.get("dropped_on_auth") == 2

    def test_401_does_not_retry(self):
        # Unlike 429, auth failures must not be retried.
        exporter, _ = self._exporter_with_response(401)
        exporter._send_with_retry([{"x": 1}])
        assert exporter._client.post.call_count == 1

    def test_429_still_buffers(self):
        # Sanity: 429 path is unchanged — still buffers after retries.
        exporter, buf = self._exporter_with_response(429)
        exporter._send_with_retry([{"x": 1}])
        # 429 retries _MAX_RETRIES + 1 times then buffers
        assert buf.count() >= 1


# ---------------------------------------------------------------------------
# RuntimeError catch when httpx client is already closed
# ---------------------------------------------------------------------------


class TestRuntimeErrorOnClosedClient:
    """Phase 31 regression: post-shutdown flushes must not crash.

    The SIGTERM handler runs _flush_and_shutdown, which closes the
    httpx client. If the BatchSpanProcessor schedules one more flush
    after that, _client.post() raises RuntimeError. Before this fix the
    error escaped _send_with_retry → uncaught traceback + lost events.
    """

    def test_closed_client_buffers_and_returns_failure(self):
        class _CountingBuffer:
            def __init__(self):
                self._events = []

            def write(self, e):
                self._events.append(e)

            def count(self):
                return len(self._events)

            def read_all(self):
                return list(self._events)

            def purge(self, n):
                self._events = self._events[n:]

        buf = _CountingBuffer()
        exporter = AgentKavachExporter(api_key="ak_prod_test", buffer=buf)
        exporter._client = MagicMock()
        exporter._client.post = MagicMock(
            side_effect=RuntimeError("Cannot send a request, as the client has been closed.")
        )

        result = exporter._send_with_retry([{"x": 1}, {"y": 2}])
        # Must not raise; must buffer for next live exporter.
        assert result == SpanExportResult.FAILURE
        assert buf.count() == 2


# ---------------------------------------------------------------------------
# SIGTERM handler actually terminates the process
# ---------------------------------------------------------------------------


class TestSigtermHandlerExits:
    """Phase 31 regression: the SIGTERM handler used to flush + return
    when the original disposition was SIG_DFL, leaving the process
    alive with a closed httpx client. It must now restore the default
    handler and re-raise so the process actually exits.
    """

    def test_sigterm_runs_handler_in_subprocess(self, tmp_path):
        # Real-process test — verifies the handler doesn't leave the
        # subprocess alive after SIGTERM (the original bug).
        import os
        import signal
        import subprocess
        import sys
        import textwrap
        import time

        script = tmp_path / "sig_test.py"
        script.write_text(
            textwrap.dedent(
                """
                import os, signal, sys, time
                from agentkavach.sender import create_tracer_provider
                provider = create_tracer_provider(api_key="ak_prod_test")
                # Mark ready, then idle until SIGTERM kills us.
                print("READY", flush=True)
                while True:
                    time.sleep(0.1)
                """
            ).strip()
        )

        proc = subprocess.Popen(
            [sys.executable, str(script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            # Wait for the child to fully install its signal handler.
            ready_line = proc.stdout.readline()
            assert "READY" in ready_line

            proc.send_signal(signal.SIGTERM)

            # The fixed handler must let the OS-default terminate-on-SIGTERM
            # action fire after the flush — process should exit within 3s.
            deadline = time.time() + 3.0
            while proc.poll() is None and time.time() < deadline:
                time.sleep(0.05)
            assert proc.poll() is not None, "Process did not exit after SIGTERM"
            # SIGTERM-terminated process has negative return code on POSIX.
            assert proc.returncode in (-signal.SIGTERM, signal.SIGTERM, 0, 1, 143)
        finally:
            if proc.poll() is None:
                os.kill(proc.pid, signal.SIGKILL)
                proc.wait(timeout=2)
