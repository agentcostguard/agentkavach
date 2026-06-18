"""Phase 59 SDK blocker regressions.

Covers the three production blockers Phase 58's real-SDK end-to-end run
exposed:

* **B1** — Google non-streaming calls crashed because the new
  ``google-genai`` SDK uses Pydantic v2 response objects and the SDK's
  best-effort ``resp.model = name`` mutation raised ``ValueError``
  (not Attribute/TypeError, which was the only thing the try/except
  caught). Fix: side channel (``self._last_model_requested``) carries
  the model name to ``_post_flight`` without mutating the response.

* **B3** — Tier-cap-rejected and daily-limit 429s were swallowed
  silently. Users had no idea events were being dropped and the SDK
  kept paying the LLM for calls whose telemetry would never land. Fix:
  the OTel exporter detects the structured ``reason`` field, emits a
  ``logger.warning`` (deduped per agent), and fires a callback that
  flips ``self._backend_paused`` so the NEXT ``pre_flight`` rejects
  before the LLM is invoked.

B2 (double-write) is covered in detail by
``tests/test_sdk_no_double_write.py`` — that test exercises the full
SDK -> OTel -> backend round-trip. These tests here focus on the
smaller unit-level invariants of B1 and B3.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agentkavach.budget import Budget
from agentkavach.client import AgentKavach
from agentkavach.exceptions import BudgetExceededError
from agentkavach.sender import AgentKavachExporter


def _guard(**overrides) -> AgentKavach:
    """Build a AgentKavach with sensible test defaults (no real network)."""
    defaults: Dict[str, Any] = dict(
        api_key="ak_test",  # skip OTel construction -> no daemon thread
        llm_key="sk-test-not-real",
        budget=Budget.daily(limit=10.0),
        agent_name="phase59-test",
    )
    defaults.update(overrides)
    return AgentKavach(**defaults)


def _fake_response(status_code: int, body: Dict[str, Any]) -> httpx.Response:
    """Build an ``httpx.Response`` with a JSON body for exporter tests."""
    request = httpx.Request("POST", "https://test-backend.local/v1/ingest")
    return httpx.Response(status_code=status_code, json=body, request=request)


class TestB1GoogleNonStreamingNoCrash:
    """The fix replaces ``resp.model = name`` (which raises on Pydantic v2)
    with a side channel set in ``_call_provider``. Confirm the SDK no
    longer raises and the side channel carries the model forward.
    """

    def test_google_create_does_not_raise_value_error(self):
        """A canned google-genai-shaped response that REFUSES model
        assignment must not crash ``guard.create``.

        Pre-fix: ``resp.model = model_name`` raised ``ValueError`` and
        escaped the try/except (which only caught Attribute/TypeError),
        crashing every non-streaming Gemini call.

        Post-fix: the SDK never mutates ``resp.model`` — it stashes the
        requested name on ``self._last_model_requested`` instead. The
        Pydantic v2 ValueError can't happen.
        """

        class _FrozenResp:
            def __init__(self):
                self.usage_metadata = SimpleNamespace(
                    prompt_token_count=10, candidates_token_count=20
                )

            def __setattr__(self, key, value):
                if key == "model":
                    raise ValueError(
                        f'"AgentKavach simulated google-genai pydantic v2": {key} not in schema'
                    )
                object.__setattr__(self, key, value)

        resp = _FrozenResp()

        fake_client = MagicMock()
        fake_client.models.generate_content.return_value = resp
        fake_client.models.count_tokens.return_value = SimpleNamespace(total_tokens=10)

        guard = _guard(provider="google")
        with patch.object(AgentKavach, "_get_google_client", return_value=fake_client):
            returned = guard.create(model="gemini-2.5-flash", contents="hello")

        assert returned is resp
        assert guard._last_model_requested == "gemini-2.5-flash"

    def test_side_channel_overrides_unknown_in_post_flight(self, tmp_path):
        """When the parser fails to extract a model name,
        ``_post_flight`` must fall back to ``_last_model_requested``.

        This is the half of B1 that turns the crash-fix into a
        completeness-fix: even when the parser returns
        ``model="unknown"`` because the SDK response shape changed,
        pricing still resolves correctly because the side channel
        feeds the real model name in.
        """
        guard = _guard(buffer_path=str(tmp_path / "buf.jsonl"))
        guard._provider = "google"
        guard._last_model_requested = "gemini-2.5-flash"

        response = MagicMock()
        response.usage_metadata = MagicMock(prompt_token_count=1000, candidates_token_count=1000)
        response.model = None
        response.choices = []

        guard._post_flight(response, requested_model="")

        events = guard._buffer.read_all()
        assert len(events) == 1
        assert events[0]["model"] == "gemini-2.5-flash", (
            "Side channel should win when both response and requested_model are empty/unknown."
        )
        assert events[0]["cost"] > 0, (
            "Pricing lookup should succeed via the side-channel model name."
        )


class TestB3ExporterPermanent429:
    """The AgentKavachExporter must surface permanent-reject 429s
    (tier_agent_limit / daily_limit / org_budget_exceeded) to the user
    via logger.warning + fire the on_backend_reject callback. It must
    NOT buffer or retry those — retries would just hit the same wall.
    """

    def _build_exporter(self, on_reject=None) -> AgentKavachExporter:
        return AgentKavachExporter(
            api_key="ak_dev_test_phase59_blocker",
            endpoint="https://test-backend.local",
            on_backend_reject=on_reject,
        )

    def test_tier_agent_limit_warns_once_per_agent(self, caplog):
        """Tier-cap rejection should log ONE warning per (reason, agent)
        even if the same batch hits 429 repeatedly.

        Pre-fix: the SDK swallowed the 429 entirely — no log line, no
        callback. The user had no way to discover their agent's
        telemetry was being dropped.
        """
        exp = self._build_exporter()
        events = [{"agent_name": "ghost-agent", "cost": 0.01}]
        body = {
            "reason": "tier_agent_limit",
            "rejected_agents": ["ghost-agent"],
            "tier": "free",
            "max_agents": 3,
        }

        with caplog.at_level(logging.WARNING, logger="agentkavach.sender"):
            exp._handle_permanent_reject("tier_agent_limit", body, events)
            exp._handle_permanent_reject("tier_agent_limit", body, events)

        msgs = [r.getMessage() for r in caplog.records]
        tier_warnings = [m for m in msgs if "tier limit" in m]
        assert len(tier_warnings) == 1, (
            f"Expected exactly one tier-limit warning, got: {tier_warnings}"
        )
        assert "ghost-agent" in tier_warnings[0]

    def test_daily_limit_logs_one_warning(self, caplog):
        """Daily-event-limit rejection logs ONE warning."""
        exp = self._build_exporter()
        events = [{"agent_name": "any", "cost": 0.01}]
        body = {"reason": "daily_limit", "remaining": 0}

        with caplog.at_level(logging.WARNING, logger="agentkavach.sender"):
            exp._handle_permanent_reject("daily_limit", body, events)

        msgs = [r.getMessage() for r in caplog.records]
        daily_warnings = [m for m in msgs if "daily event limit" in m]
        assert len(daily_warnings) == 1

    def test_callback_fires_with_reason_and_body(self):
        """``on_backend_reject`` callback receives the reason + body."""
        captured: List[Dict[str, Any]] = []

        def _cb(reason: str, body: Dict[str, Any]) -> None:
            captured.append({"reason": reason, "body": body})

        exp = self._build_exporter(on_reject=_cb)
        body = {"reason": "org_budget_exceeded", "detail": "Org budget exceeded."}
        exp._handle_permanent_reject("org_budget_exceeded", body, [])

        assert len(captured) == 1
        assert captured[0]["reason"] == "org_budget_exceeded"
        assert captured[0]["body"] is body

    def test_stats_record_per_reason_count(self):
        """``retry_stats`` should record event counts per permanent reason."""
        exp = self._build_exporter()
        exp._handle_permanent_reject(
            "tier_agent_limit",
            {"rejected_agents": ["a", "b"]},
            [{"agent_name": "a"}, {"agent_name": "a"}, {"agent_name": "b"}],
        )
        exp._handle_permanent_reject("daily_limit", {}, [{"agent_name": "a"}])

        stats = exp.retry_stats
        assert stats["rejected_tier_agent_limit"] == 3
        assert stats["rejected_daily_limit"] == 1


class TestB3ClientPaused:
    """The AgentKavach client's pre-flight gate must reject calls when
    ``_backend_paused`` is set.
    """

    def test_handle_backend_reject_flips_paused(self):
        guard = _guard()
        assert guard._backend_paused is False
        guard._handle_backend_reject("tier_agent_limit", {"reason": "tier"})
        assert guard._backend_paused is True
        assert guard._backend_paused_reason == "tier_agent_limit"
        # Idempotent.
        guard._handle_backend_reject("daily_limit", {})
        assert guard._backend_paused_reason == "tier_agent_limit"

    def test_subsequent_create_rejected_with_budget_exceeded(self):
        """Once paused, the next ``guard.create`` raises BEFORE the
        provider is called — the whole point is to avoid wasted LLM
        spend.
        """
        guard = _guard(provider="openai")
        guard._handle_backend_reject("tier_agent_limit", {})

        fake_client = MagicMock()
        with patch.object(AgentKavach, "_get_openai_client", return_value=fake_client):
            with pytest.raises(BudgetExceededError) as excinfo:
                guard.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "hi"}],
                )

        assert "tier_agent_limit" in str(excinfo.value)
        fake_client.chat.completions.create.assert_not_called()

    def test_daily_limit_path(self):
        guard = _guard(provider="openai")
        guard._handle_backend_reject("daily_limit", {"reason": "daily"})
        assert guard._backend_paused is True
        assert guard._backend_paused_reason == "daily_limit"

        fake_client = MagicMock()
        with patch.object(AgentKavach, "_get_openai_client", return_value=fake_client):
            with pytest.raises(BudgetExceededError) as excinfo:
                guard.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "hi"}],
                )

        assert "daily_limit" in str(excinfo.value)
        fake_client.chat.completions.create.assert_not_called()


class TestB3PermanentReject429RoutingInSendLoop:
    """The exporter's ``_send_with_retry`` loop must short-circuit on a
    permanent-reason 429 — not buffer, not retry.
    """

    def _make_exporter_with_seq(self, status_codes: List[int], bodies: List[Dict[str, Any]]):
        exp = AgentKavachExporter(
            api_key="ak_dev_test_phase59_seq",
            endpoint="https://test-backend.local",
        )
        responses = [_fake_response(s, b) for s, b in zip(status_codes, bodies)]
        seq = iter(responses)

        def _post(events):
            return next(seq)

        exp._post = _post  # type: ignore[assignment]
        return exp

    def test_tier_agent_limit_short_circuits_no_buffer(self, caplog):
        """A 429 with reason=tier_agent_limit must:
        * NOT retry
        * NOT buffer the events
        * Emit a warning
        * Bump rejected_tier_agent_limit stat
        """
        buffer_writes: List[Any] = []
        fake_buffer = MagicMock()
        fake_buffer.write = lambda evt: buffer_writes.append(evt)

        exp = self._make_exporter_with_seq(
            [429],
            [
                {
                    "reason": "tier_agent_limit",
                    "rejected_agents": ["a1"],
                    "tier": "free",
                }
            ],
        )
        exp._buffer = fake_buffer

        events = [{"agent_name": "a1", "cost": 0.01}]
        with caplog.at_level(logging.WARNING, logger="agentkavach.sender"):
            result = exp._send_with_retry(events)

        from opentelemetry.sdk.trace.export import SpanExportResult

        assert result == SpanExportResult.FAILURE
        assert buffer_writes == [], (
            "Events from a permanent 429 must not be buffered — "
            "they would just produce another 429 forever."
        )
        assert exp.retry_stats.get("rejected_tier_agent_limit", 0) == 1
