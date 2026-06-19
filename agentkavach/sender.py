"""AgentKavachExporter — OTel SpanExporter that pushes batches to /v1/ingest.

Uses ``BatchSpanProcessor`` for background threading and queue management.
The exporter itself handles retry on HTTP 429 (rate-limited) responses
with exponential backoff, honouring the ``Retry-After`` header.  Failed
batches are written to the disk buffer so they can be replayed on the
next successful export — no events are silently lost.

The exporter converts ``gen_ai.*`` span attributes into compact JSON
events and POSTs them to the AgentKavach backend.

Endpoint resolution (in order of precedence):
    1. ``endpoint=`` argument to ``create_tracer_provider`` / ``AgentKavach``
    2. ``AGENTKAVACH_BACKEND_URL`` environment variable
    3. The API-key prefix:
         - ``ak_prod_…`` → prod backend
         - ``ak_dev_…``  → dev backend
         - ``cg_…``      → prod backend (legacy keys)
         - anything else → prod backend (safe default)

Usage (internal — wired up by ``AgentKavach.__init__``):

    from agentkavach.sender import create_tracer_provider

    provider = create_tracer_provider(
        api_key="ak_prod_xxx",
        agent_name="my-agent",
    )
    tracer = provider.get_tracer("agentkavach")
"""

from __future__ import annotations

import atexit
import gzip
import json
import logging
import os
import signal
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import httpx
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

logger = logging.getLogger(__name__)

# OTel BatchSpanProcessor tuning (from design doc).
_MAX_QUEUE_SIZE = 10_000
_MAX_EXPORT_BATCH_SIZE = 512
_SCHEDULE_DELAY_MILLIS = 5_000

# HTTP timeout for the ingest endpoint.
_INGEST_TIMEOUT_SECONDS = 10

# Retry config for 429 responses.
_MAX_RETRIES = 3
_DEFAULT_RETRY_AFTER = 1.0  # seconds, used when Retry-After header is missing
_MAX_RETRY_AFTER = 10.0  # cap on Retry-After to prevent absurd waits

# Hardcoded backend URLs for env-prefixed keys. The SDK ships with these
# baked in so that an ``ak_prod_*`` key "just works" without any env var.
# If you move off Render or change domains, bump these and release a new
# SDK version.
PROD_BACKEND_URL = "https://agentcostguard-backend.onrender.com"
DEV_BACKEND_URL = "https://agentcostguard.onrender.com"


def resolve_backend_url(api_key: str) -> str:
    """Return the backend URL the SDK should send to for *api_key*.

    Resolution order:
        1. ``AGENTKAVACH_BACKEND_URL`` env var (explicit override — wins)
        2. Key prefix:
             - ``ak_prod_…`` → :data:`PROD_BACKEND_URL`
             - ``ak_dev_…``  → :data:`DEV_BACKEND_URL`
             - ``cg_…``      → :data:`PROD_BACKEND_URL` (legacy)
             - anything else → :data:`PROD_BACKEND_URL` (safe default)
    """
    override = os.environ.get("AGENTKAVACH_BACKEND_URL")
    if override:
        return override
    if api_key.startswith("ak_dev_"):
        return DEV_BACKEND_URL
    return PROD_BACKEND_URL


class AgentKavachExporter(SpanExporter):
    """Custom OTel span exporter that POSTs cost events to ``/v1/ingest``.

    Each span is converted to a compact JSON event dict.  The exporter
    sends the batch as a JSON array in a single HTTP POST.

    On HTTP 429 the exporter retries up to ``_MAX_RETRIES`` times,
    honouring the ``Retry-After`` header with exponential backoff.
    If retries are exhausted, the failed events are written to a
    :class:`Buffer` for replay on the next successful export.

    API key is sent via the ``Authorization`` header — never in the
    request body or URL.
    """

    def __init__(
        self,
        api_key: str,
        endpoint: Optional[str] = None,
        *,
        compress: bool = True,
        buffer: Optional[Any] = None,
        on_backend_reject: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        if not api_key:
            raise ValueError("AgentKavach API key must not be empty")

        resolved = endpoint or resolve_backend_url(api_key)
        self._endpoint = resolved.rstrip("/") + "/v1/ingest"
        self._compress = compress
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "agentkavach-sdk/0.1.0",
        }
        self._client = httpx.Client(timeout=_INGEST_TIMEOUT_SECONDS)
        self._buffer = buffer  # Optional Buffer instance for retry persistence
        self._retry_stats = {"retries": 0, "buffered": 0, "replayed": 0}

        # Phase 59 (B3): callback the exporter invokes when the backend
        # returns 429 with a structured ``reason`` that means "stop
        # sending — retrying won't help" (``tier_agent_limit`` because
        # the org is over its tier's max-agent cap, ``daily_limit``
        # because the org has burned its daily event quota,
        # ``org_budget_exceeded`` because the org cost budget tripped).
        # The AgentKavach client passes a callback that flips
        # ``_backend_paused`` so subsequent ``pre_flight`` calls reject
        # before the LLM is even called — pre-fix the SDK silently
        # swallowed the 429 and the user kept burning provider spend on
        # events that would just be dropped server-side.
        self._on_backend_reject = on_backend_reject

        # De-dupe warning emission per (reason, agent_name) tuple —
        # a noisy agent making 100 calls/min should not produce 100
        # identical warnings.
        self._warned_rejects: set[tuple[str, str]] = set()

    @property
    def retry_stats(self) -> Dict[str, int]:
        """Return retry/buffer statistics for observability."""
        return dict(self._retry_stats)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Convert spans to events and POST to the ingest endpoint.

        Retry logic:
        1. On 429: sleep for ``Retry-After`` seconds (or 1s default), retry
           up to ``_MAX_RETRIES`` times with exponential backoff.
        2. On retry exhaustion: write failed events to disk buffer.
        3. On success: drain any buffered events from prior failures.

        When compress=True, the body is gzip-compressed for 5-10x
        bandwidth reduction on JSON payloads.
        """
        if not spans:
            return SpanExportResult.SUCCESS

        events = [self._span_to_event(span) for span in spans]

        result = self._send_with_retry(events)

        if result == SpanExportResult.SUCCESS:
            # Drain buffered events from prior failures.
            self._replay_buffer()

        return result

    def _send_with_retry(self, events: List[Dict[str, Any]]) -> SpanExportResult:
        """POST events with retry on 429. Buffer on exhaustion."""
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = self._post(events)

                if resp.status_code in (200, 202):
                    return SpanExportResult.SUCCESS

                if resp.status_code == 429:
                    # Phase 59 (B3): inspect the body. A structured
                    # ``reason`` field of ``tier_agent_limit`` /
                    # ``daily_limit`` / ``org_budget_exceeded`` means
                    # "stop sending — retrying won't fix it." Surface
                    # to the user via logger.warning + fire the
                    # on_backend_reject callback, and DO NOT buffer —
                    # buffered events would just produce another 429
                    # on the next successful drain and grow unbounded.
                    body = self._safe_parse_json(resp)
                    reason = body.get("reason") if isinstance(body, dict) else None
                    permanent_reasons = (
                        "tier_agent_limit",
                        "daily_limit",
                        "org_budget_exceeded",
                    )
                    if reason in permanent_reasons:
                        self._handle_permanent_reject(reason, body, events)
                        return SpanExportResult.FAILURE

                    if attempt < _MAX_RETRIES:
                        wait = self._parse_retry_after(resp)
                        self._retry_stats["retries"] += 1
                        logger.info(
                            "Ingest rate-limited (429), retry %d/%d in %.1fs",
                            attempt + 1,
                            _MAX_RETRIES,
                            wait,
                        )
                        time.sleep(wait)
                        continue

                    # Retries exhausted — buffer for later replay.
                    self._buffer_events(events)
                    logger.warning(
                        "Ingest 429 after %d retries, %d events buffered for replay",
                        _MAX_RETRIES,
                        len(events),
                    )
                    return SpanExportResult.FAILURE

                # Auth failures (401, 403): the key is invalid or revoked. Buffering
                # is pointless — every retry will get the same response — and it
                # would silently grow the on-disk buffer forever. Drop the events
                # and bump a counter so operators can see this in `retry_stats`.
                if resp.status_code in (401, 403):
                    self._retry_stats["auth_failures"] = (
                        self._retry_stats.get("auth_failures", 0) + 1
                    )
                    self._retry_stats["dropped_on_auth"] = self._retry_stats.get(
                        "dropped_on_auth", 0
                    ) + len(events)
                    logger.error(
                        "Ingest auth failed (%d) — %d events dropped (key invalid or revoked). "
                        "Check that your AGENTKAVACH_API_KEY is active.",
                        resp.status_code,
                        len(events),
                    )
                    return SpanExportResult.FAILURE

                logger.warning("Ingest returned %d: %s", resp.status_code, resp.text[:200])
                return SpanExportResult.FAILURE

            except httpx.TimeoutException:
                if attempt < _MAX_RETRIES:
                    self._retry_stats["retries"] += 1
                    wait = _DEFAULT_RETRY_AFTER * (2**attempt)
                    logger.info(
                        "Ingest timed out, retry %d/%d in %.1fs", attempt + 1, _MAX_RETRIES, wait
                    )
                    time.sleep(wait)
                    continue
                logger.warning("Ingest timed out after %d retries", _MAX_RETRIES)
                self._buffer_events(events)
                return SpanExportResult.FAILURE
            except httpx.HTTPError as exc:
                logger.warning("Ingest request failed: %s", exc)
                self._buffer_events(events)
                return SpanExportResult.FAILURE
            except RuntimeError as exc:
                # Most common case: the underlying httpx.Client was already
                # closed (e.g. shutdown ran on SIGTERM but the BatchSpanProcessor
                # scheduled one more flush before the process actually exited).
                # Without this, the RuntimeError escapes _send_with_retry, OTel
                # logs an uncaught traceback, and the events are lost without
                # ever hitting the buffer.
                logger.warning("Ingest skipped — exporter client closed: %s", exc)
                self._buffer_events(events)
                return SpanExportResult.FAILURE

        # Should not reach here, but safety net.
        self._buffer_events(events)
        return SpanExportResult.FAILURE

    def _post(self, events: List[Dict[str, Any]]) -> httpx.Response:
        """POST events to the ingest endpoint."""
        payload = {"events": events}
        if self._compress:
            body = gzip.compress(json.dumps(payload).encode())
            headers = {**self._headers, "Content-Encoding": "gzip"}
            return self._client.post(self._endpoint, content=body, headers=headers)
        return self._client.post(self._endpoint, json=payload, headers=self._headers)

    @staticmethod
    def _parse_retry_after(resp: httpx.Response) -> float:
        """Extract wait time from Retry-After header, with backoff floor."""
        raw = resp.headers.get("retry-after", "")
        try:
            wait = float(raw)
            return min(max(wait, _DEFAULT_RETRY_AFTER), _MAX_RETRY_AFTER)
        except (ValueError, TypeError):
            return _DEFAULT_RETRY_AFTER

    @staticmethod
    def _safe_parse_json(resp: httpx.Response) -> Dict[str, Any]:
        """Parse a response body as JSON, returning ``{}`` on any error.

        Used by the 429 reason-detection path so that a malformed or
        empty body falls back to the generic retry-and-buffer flow
        rather than raising.
        """
        try:
            data = resp.json()
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _handle_permanent_reject(
        self,
        reason: str,
        body: Dict[str, Any],
        events: List[Dict[str, Any]],
    ) -> None:
        """Phase 59 (B3): record + warn + notify on a permanent 429.

        ``reason`` is the server's machine-readable code
        (``tier_agent_limit`` / ``daily_limit`` / ``org_budget_exceeded``).
        We bump a stat, emit one warning per (reason, agent) tuple, and
        fire ``on_backend_reject`` so the AgentKavach client can flip
        ``_backend_paused`` and stop wasting LLM spend.
        """
        self._retry_stats[f"rejected_{reason}"] = self._retry_stats.get(
            f"rejected_{reason}", 0
        ) + len(events)

        # The server may include ``rejected_agents`` (list) on
        # tier_agent_limit; otherwise fall back to the agents in
        # the batch.
        rejected_agents = body.get("rejected_agents") or []
        if not rejected_agents:
            rejected_agents = sorted({str(e.get("agent_name", "unknown")) for e in events})

        for agent_name in rejected_agents:
            key = (reason, str(agent_name))
            if key in self._warned_rejects:
                continue
            self._warned_rejects.add(key)
            if reason == "tier_agent_limit":
                logger.warning(
                    "AgentKavach: agent %r rejected by backend (tier limit). "
                    "Events for this agent will not be recorded until the "
                    "agent is reactivated or a slot is freed.",
                    agent_name,
                )
            elif reason == "daily_limit":
                logger.warning(
                    "AgentKavach: daily event limit exhausted for org tier. "
                    "Events will be rejected until UTC midnight reset."
                )
            elif reason == "org_budget_exceeded":
                logger.warning(
                    "AgentKavach: org cost budget exceeded — events will be "
                    "rejected until the budget period rolls over."
                )

        # Fire the callback once per batch (the client's handler is
        # idempotent — flipping ``_backend_paused`` twice is a no-op).
        if self._on_backend_reject is not None:
            try:
                self._on_backend_reject(reason, body)
            except Exception:
                logger.debug("on_backend_reject callback raised", exc_info=True)

    def _buffer_events(self, events: List[Dict[str, Any]]) -> None:
        """Write failed events to disk buffer for later replay."""
        if self._buffer is None:
            return
        for event in events:
            self._buffer.write(event)
        self._retry_stats["buffered"] += len(events)

    def _replay_buffer(self) -> None:
        """Drain buffered events from prior failures on successful export."""
        if self._buffer is None:
            return
        pending = self._buffer.count()
        if pending == 0:
            return

        events = self._buffer.read_all()
        if not events:
            return

        # Send in chunks of _MAX_EXPORT_BATCH_SIZE to avoid oversized payloads.
        chunk_size = _MAX_EXPORT_BATCH_SIZE
        total_replayed = 0
        for i in range(0, len(events), chunk_size):
            chunk = events[i : i + chunk_size]
            try:
                resp = self._post(chunk)
                if resp.status_code in (200, 202):
                    total_replayed += len(chunk)
                elif resp.status_code == 429:
                    # Still rate-limited — stop replaying, leave rest in buffer.
                    logger.info("Buffer replay hit 429 after %d events, pausing", total_replayed)
                    break
                else:
                    logger.warning("Buffer replay got %d, pausing", resp.status_code)
                    break
            except (httpx.TimeoutException, httpx.HTTPError) as exc:
                logger.warning("Buffer replay failed: %s, pausing", exc)
                break

        if total_replayed > 0:
            self._buffer.purge(total_replayed)
            self._retry_stats["replayed"] += total_replayed
            logger.info(
                "Replayed %d buffered events (%d remaining)",
                total_replayed,
                pending - total_replayed,
            )

    def shutdown(self) -> None:
        """Release HTTP resources."""
        self._client.close()

    @staticmethod
    def _span_to_event(span: ReadableSpan) -> Dict[str, Any]:
        """Extract ``gen_ai.*`` attributes from a span into a flat dict.

        Field names match the backend ``IngestEvent`` schema:
        agent_name, provider, model, input_tokens, output_tokens,
        cost, duration_ms, timestamp (ISO 8601), run_id, prompt,
        idempotency_key.

        Phase 35 fixes:
        - **idempotency_key** is now always emitted (sourced from a per-call
          UUID set by the client wrapper). Server's dedup path requires
          this to avoid double-counting on at-least-once redelivery.
        - **timestamp** falls back to ``datetime.now(UTC)`` if the span
          never set ``start_time``. Previously this could be ``None``
          and the server's ``IngestEvent`` (str required) would 422
          the whole batch silently.
        - **duration_ms** uses ``round()`` instead of integer truncation
          for sub-ms accuracy on the dashboard.
        """
        from datetime import datetime, timezone

        attrs = span.attributes or {}

        # Compute duration from span start/end (nanoseconds → milliseconds).
        # Phase 123: prefer the SDK-measured provider latency
        # (``agentkavach.duration_ms``, the time around the real provider
        # call) over the span wall-clock, which also includes our span
        # bookkeeping. Fall back to the span timing when the attribute is
        # absent (older SDK / direct span callers).
        duration_ms = 0
        if span.start_time and span.end_time:
            duration_ms = max(0, round((span.end_time - span.start_time) / 1_000_000))
        measured = attrs.get("agentkavach.duration_ms")
        if measured is not None:
            try:
                duration_ms = max(0, int(measured))
            except (TypeError, ValueError):
                pass

        # Convert nanosecond epoch to ISO 8601 timestamp; fall back to
        # current UTC if the span lost its start_time (defensive — OTel
        # always sets it, but a broken instrumentation upstream
        # shouldn't drop the whole event).
        if span.start_time:
            ts = datetime.fromtimestamp(
                span.start_time / 1_000_000_000, tz=timezone.utc
            ).isoformat()
        else:
            ts = datetime.now(timezone.utc).isoformat()

        idempotency_key = attrs.get("agentkavach.idempotency_key")
        run_id = attrs.get("agentkavach.run_id")
        prompt = attrs.get("gen_ai.prompt")
        # Phase 123: per-call outcome. Only emit when explicitly "error"
        # — a missing/"success" status is the implicit default server-side
        # (NULL counts as success), so omitting it keeps payloads compact
        # and backward-compatible with older backends that ignore the field.
        call_status = attrs.get("agentkavach.status")

        event: Dict[str, Any] = {
            "agent_name": attrs.get("gen_ai.agent.name", "unknown"),
            "model": attrs.get("gen_ai.request.model", "unknown"),
            "provider": attrs.get("gen_ai.system", "unknown"),
            "input_tokens": attrs.get("gen_ai.usage.input_tokens", 0),
            "output_tokens": attrs.get("gen_ai.usage.output_tokens", 0),
            "cost": attrs.get("gen_ai.usage.cost", 0.0),
            "duration_ms": duration_ms,
            "timestamp": ts,
        }
        if idempotency_key:
            event["idempotency_key"] = idempotency_key
        if run_id:
            event["run_id"] = run_id
        if prompt:
            event["prompt"] = prompt
        if call_status:
            event["status"] = call_status
        return event


def create_tracer_provider(
    api_key: str,
    agent_name: str = "default",
    endpoint: Optional[str] = None,
    buffer: Optional[Any] = None,
    on_backend_reject: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> TracerProvider:
    """Build a fully-configured ``TracerProvider`` for AgentKavach telemetry.

    The provider uses a ``BatchSpanProcessor`` with the AgentKavach
    exporter, tuned for 10K queue depth, 512-event batches, and
    5-second flush intervals.

    The API key must be passed explicitly via *api_key*; it is never read
    from the environment.

    Endpoint resolution: *endpoint* argument > ``AGENTKAVACH_BACKEND_URL``
    env var (an infrastructure override, not a credential) > inferred from
    the API-key prefix (see :func:`resolve_backend_url`).

    If *buffer* is provided, the exporter uses it to persist events
    that fail to send (e.g. rate-limited) and replays them on the
    next successful export.
    """
    if not api_key:
        raise ValueError("AgentKavach API key required — pass api_key=")

    resolved_key = api_key
    resolved_endpoint = endpoint or resolve_backend_url(resolved_key)

    resource = Resource.create(
        {
            "service.name": "agentkavach",
            "agentkavach.agent.name": agent_name,
        }
    )

    exporter = AgentKavachExporter(
        api_key=resolved_key,
        endpoint=resolved_endpoint,
        buffer=buffer,
        on_backend_reject=on_backend_reject,
    )

    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(
            exporter,
            max_queue_size=_MAX_QUEUE_SIZE,
            max_export_batch_size=_MAX_EXPORT_BATCH_SIZE,
            schedule_delay_millis=_SCHEDULE_DELAY_MILLIS,
        )
    )

    # Register graceful shutdown to flush pending spans on exit
    _register_shutdown(provider)

    return provider


def _register_shutdown(provider: TracerProvider) -> None:
    """Register atexit and signal handlers to flush the OTel queue on exit.

    Ensures pending spans are exported before the process terminates.
    SIGKILL cannot be caught — some event loss is unavoidable in that case.
    """

    def _flush_and_shutdown() -> None:
        try:
            provider.shutdown()
        except Exception:
            logger.debug("TracerProvider shutdown error (may already be shut down)")

    atexit.register(_flush_and_shutdown)

    # Install signal handlers for SIGTERM (graceful stop).
    # Only install on main thread to avoid RuntimeError.
    try:
        original_sigterm = signal.getsignal(signal.SIGTERM)

        def _sigterm_handler(signum: int, frame: Any) -> None:
            _flush_and_shutdown()
            # Restore + invoke the appropriate downstream behavior so the
            # process actually exits. The previous implementation flushed
            # and *returned* when original_sigterm was SIG_DFL, leaving
            # the process running with a closed httpx client — every
            # subsequent OTel flush then raised RuntimeError. Reset the
            # handler to its default and re-raise the signal so the
            # default terminate-on-SIGTERM behavior fires.
            if callable(original_sigterm) and original_sigterm not in (
                signal.SIG_DFL,
                signal.SIG_IGN,
            ):
                original_sigterm(signum, frame)
                return
            # Default disposition (or SIG_IGN): restore and re-raise so the
            # OS-default action terminates the process. SIG_IGN preserves
            # the previous user intent ("ignore SIGTERM").
            signal.signal(signal.SIGTERM, original_sigterm)
            os.kill(os.getpid(), signal.SIGTERM)

        signal.signal(signal.SIGTERM, _sigterm_handler)
    except (ValueError, OSError):
        # Not on main thread or signal not available — skip
        pass
