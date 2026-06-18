"""Phase 123 SDK tests: per-call status + latency capture.

This is the piece the prior (rejected) PR was missing entirely — the SDK
must actually *emit* a failure event when the provider call raises, and a
success event carrying the measured latency when it returns. These tests
exercise the real ``agentkavach.client.AgentKavach._safe_call`` path with
a stubbed provider so no network or API keys are needed.

Covered:
  1. SUCCESS (non-streaming): a status="success" event with a real
     duration_ms is written to the buffer.
  2. PROVIDER FAILURE: a status="error" event (cost/tokens 0) is written
     AND the original provider exception is re-raised unchanged.
  3. Recording itself failing does not swallow the provider exception
     (fail-open).
  4. A pre-flight BudgetExceededError propagates and emits NO failure
     event.
  5. sender._span_to_event maps the ``agentkavach.status`` /
     ``agentkavach.duration_ms`` span attributes onto the ingest payload.

Each guard gets its own on-disk buffer (``buffer_path``) so events never
leak between tests — the SDK's default buffer is a single shared tmp file,
which would otherwise pollute these (and other) tests' assertions.
"""

from __future__ import annotations

import uuid

import pytest

from agentkavach import AgentKavach, Budget
from agentkavach.exceptions import BudgetExceededError


class _FakeUsage:
    """Minimal OpenAI-style usage object."""

    def __init__(self, prompt_tokens: int = 10, completion_tokens: int = 5) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class _FakeResponse:
    """Minimal OpenAI-style chat completion response."""

    def __init__(self, model: str = "gpt-4o") -> None:
        self.model = model
        self.usage = _FakeUsage()
        self.choices = []


def _make_guard(tmp_path, **overrides):
    """Build a guard with telemetry disabled (no tracer → buffer path).

    Each guard gets a unique on-disk buffer so events never leak between
    tests (the SDK's default buffer is a single shared tmp file).
    """
    buffer_path = str(tmp_path / f"buf_{uuid.uuid4().hex}.jsonl")
    kwargs = dict(
        provider="openai",
        llm_key="sk-test",
        api_key="ak_local_testkey",
        agent_name="latency-bot",
        budget=Budget.daily(100.0),
        buffer_path=buffer_path,
    )
    kwargs.update(overrides)
    guard = AgentKavach(**kwargs)
    # Force the disk-buffer (no-tracer) transport so we can inspect events
    # without standing up an OTel exporter.
    guard._tracer = None
    return guard


def _written_events(guard) -> list:
    """Return everything the SDK wrote to its buffer this test."""
    return guard._buffer.read_all()


def test_success_event_has_status_success_and_duration(monkeypatch, tmp_path):
    guard = _make_guard(tmp_path)

    # Stub the provider call to return quickly with a fake response.
    monkeypatch.setattr(guard, "_call_provider", lambda kwargs: _FakeResponse())
    # Token counting shouldn't hit the network.
    monkeypatch.setattr(guard, "_count_input_tokens", lambda model, messages: 10)

    resp = guard.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    assert isinstance(resp, _FakeResponse)

    events = _written_events(guard)
    assert len(events) == 1
    evt = events[0]
    assert evt["status"] == "success"
    assert evt["agent_name"] == "latency-bot"
    # duration_ms is the measured provider latency — a non-negative int.
    assert isinstance(evt["duration_ms"], int)
    assert evt["duration_ms"] >= 0


def test_provider_failure_emits_error_event_and_reraises(monkeypatch, tmp_path):
    guard = _make_guard(tmp_path)
    monkeypatch.setattr(guard, "_count_input_tokens", lambda model, messages: 10)

    class _Boom(RuntimeError):
        pass

    def _raise(_kwargs):
        raise _Boom("provider exploded")

    monkeypatch.setattr(guard, "_call_provider", _raise)

    with pytest.raises(_Boom, match="provider exploded"):
        guard.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

    events = _written_events(guard)
    assert len(events) == 1
    evt = events[0]
    assert evt["status"] == "error"
    assert evt["cost"] == 0.0
    assert evt["input_tokens"] == 0
    assert evt["output_tokens"] == 0
    assert isinstance(evt["duration_ms"], int)
    assert evt["agent_name"] == "latency-bot"


def test_failure_recording_error_does_not_swallow_provider_exception(monkeypatch, tmp_path):
    guard = _make_guard(tmp_path)
    monkeypatch.setattr(guard, "_count_input_tokens", lambda model, messages: 10)

    class _Boom(RuntimeError):
        pass

    monkeypatch.setattr(guard, "_call_provider", lambda _k: (_ for _ in ()).throw(_Boom("boom")))

    # Make the failure-recording path itself blow up — the original
    # provider exception must still surface (fail-open).
    def _bad_record(*_a, **_kw):
        raise ValueError("recording broke")

    monkeypatch.setattr(guard, "_record_failure", _bad_record)

    with pytest.raises(_Boom, match="boom"):
        guard.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])


def test_preflight_budget_error_emits_no_failure_event(monkeypatch, tmp_path):
    guard = _make_guard(tmp_path)
    monkeypatch.setattr(guard, "_count_input_tokens", lambda model, messages: 10)

    called = {"provider": False}

    def _provider(_kwargs):
        called["provider"] = True
        return _FakeResponse()

    monkeypatch.setattr(guard, "_call_provider", _provider)

    # Force pre_flight to raise a budget error deterministically — this
    # short-circuits before the provider call.
    def _pre_flight(*_a, **_kw):
        raise BudgetExceededError("over budget")

    monkeypatch.setattr(guard._engine, "pre_flight", _pre_flight)

    with pytest.raises(BudgetExceededError):
        guard.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

    # The provider was never called, and NO failure event was emitted —
    # a budget block is not a provider failure.
    assert called["provider"] is False
    assert _written_events(guard) == []


def test_span_to_event_maps_status_and_duration():
    """sender._span_to_event maps the SDK's status / duration span
    attributes onto the ingest payload, so events.status is populated on
    the OTel transport path (the real prod path)."""
    from agentkavach.sender import AgentKavachExporter

    class _Span:
        def __init__(self, attrs):
            self.attributes = attrs
            self.start_time = 1_000_000_000  # 1s in ns
            self.end_time = 1_500_000_000  # 1.5s in ns

    span = _Span(
        {
            "gen_ai.agent.name": "latency-bot",
            "gen_ai.request.model": "gpt-4o",
            "gen_ai.system": "openai",
            "gen_ai.usage.input_tokens": 0,
            "gen_ai.usage.output_tokens": 0,
            "gen_ai.usage.cost": 0.0,
            "agentkavach.status": "error",
            "agentkavach.duration_ms": 1234,
            "agentkavach.idempotency_key": "abc",
        }
    )
    event = AgentKavachExporter._span_to_event(span)
    assert event["status"] == "error"
    # The SDK-measured duration wins over the span wall-clock (which would
    # have computed 500ms from start/end above).
    assert event["duration_ms"] == 1234


def test_span_to_event_omits_status_when_absent():
    """No status attr → no status key in the payload (older-SDK / success
    default; NULL counts as success server-side)."""
    from agentkavach.sender import AgentKavachExporter

    class _Span:
        attributes = {
            "gen_ai.agent.name": "a",
            "gen_ai.request.model": "gpt-4o",
            "gen_ai.system": "openai",
        }
        start_time = 1_000_000_000
        end_time = 1_200_000_000

    event = AgentKavachExporter._span_to_event(_Span())
    assert "status" not in event
    # Falls back to span wall-clock duration (200ms) when no measured attr.
    assert event["duration_ms"] == 200
