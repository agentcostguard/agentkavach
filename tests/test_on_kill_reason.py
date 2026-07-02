"""Kill-reason differentiation for ``on_kill`` + ``IngestRejectedError`` (2.1.0).

Before 2.1.0 the ``on_kill`` teardown was a zero-argument signal, so a
customer callback could not tell a budget kill from a backend stop (tier
agent-limit, org daily-limit, dashboard Kill) — teardown logs claimed
"budget exhausted" for agents whose budget was barely touched. Now:

* A callback that declares a parameter receives WHY it fired: engine kills
  pass ``"cost"`` / ``"tokens"`` / ``"duration"`` / ``"calls"`` / ``"loop"``;
  a backend stop passes the rejection code (``"tier_agent_limit"``, ...);
  ``fail_on_error`` paths pass ``"internal_error"``.
* Zero-argument callbacks (the pre-2.1 API) keep being called with no
  arguments — no signature change required to upgrade.
* The backend-stop path raises ``IngestRejectedError`` (a
  ``BudgetExceededError`` subclass, so existing handlers keep catching it)
  carrying the rejection code on ``.reason``.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from agentkavach.alerts import AlertRule
from agentkavach.budget import Budget
from agentkavach.client import AgentKavach
from agentkavach.engine import SpendEngine, ThresholdEvent, _on_kill_reason_mode, invoke_on_kill
from agentkavach.exceptions import (
    BudgetExceededError,
    CallLimitError,
    IngestRejectedError,
    LoopDetectedError,
)


def _guard(**overrides) -> AgentKavach:
    """Build an AgentKavach with sensible test defaults (no real network)."""
    defaults: Dict[str, Any] = dict(
        api_key="ak_test",
        llm_key="sk-test-not-real",
        budget=Budget.daily(limit=10.0),
        agent_name="kill-reason-test",
    )
    defaults.update(overrides)
    return AgentKavach(**defaults)


# ---------------------------------------------------------------------------
# _on_kill_reason_mode — signature introspection
# ---------------------------------------------------------------------------


class TestReasonMode:
    def test_zero_arg_lambda_is_none(self):
        assert _on_kill_reason_mode(lambda: None) == "none"

    def test_zero_arg_def_is_none(self):
        def cb():
            pass

        assert _on_kill_reason_mode(cb) == "none"

    def test_positional_param(self):
        def cb(reason):
            pass

        assert _on_kill_reason_mode(cb) == "positional"

    def test_positional_param_any_name(self):
        # The parameter NAME doesn't matter for positional params.
        def cb(why):
            pass

        assert _on_kill_reason_mode(cb) == "positional"

    def test_positional_with_default(self):
        def cb(reason="unset"):
            pass

        assert _on_kill_reason_mode(cb) == "positional"

    def test_var_positional(self):
        def cb(*args):
            pass

        assert _on_kill_reason_mode(cb) == "positional"

    def test_keyword_only_reason(self):
        def cb(*, reason=None):
            pass

        assert _on_kill_reason_mode(cb) == "keyword"

    def test_keyword_only_other_name_is_none(self):
        # A keyword-only param NOT named "reason" can't receive the reason.
        def cb(*, verbose=False):
            pass

        assert _on_kill_reason_mode(cb) == "none"

    def test_var_keyword_only_is_none(self):
        # **kwargs alone is not an opt-in — we can't guess the expected name.
        def cb(**kwargs):
            pass

        assert _on_kill_reason_mode(cb) == "none"

    def test_bound_method(self):
        class Teardown:
            def on_kill(self, reason):
                pass

        assert _on_kill_reason_mode(Teardown().on_kill) == "positional"

    def test_fully_bound_partial_is_none(self):
        def cb(reason):
            pass

        assert _on_kill_reason_mode(functools.partial(cb, "pre-bound")) == "none"

    def test_unintrospectable_callable_is_none(self):
        # Some C builtins have no retrievable signature — must not raise,
        # must fall back to the zero-arg call.
        try:
            import inspect

            inspect.signature(min)
            pytest.skip("min is introspectable on this runtime")
        except (TypeError, ValueError):
            pass
        assert _on_kill_reason_mode(min) == "none"


# ---------------------------------------------------------------------------
# invoke_on_kill — delivery
# ---------------------------------------------------------------------------


class TestInvokeOnKill:
    def test_positional_receives_reason(self):
        got = []
        invoke_on_kill(lambda reason: got.append(reason), "cost")
        assert got == ["cost"]

    def test_keyword_only_receives_reason(self):
        got = []

        def cb(*, reason=None):
            got.append(reason)

        invoke_on_kill(cb, "tier_agent_limit")
        assert got == ["tier_agent_limit"]

    def test_zero_arg_called_without_args(self):
        got = []
        invoke_on_kill(lambda: got.append("fired"), "cost")
        assert got == ["fired"]

    def test_none_reason_passed_through(self):
        # Legacy kill paths may not record a reason — the callback still fires.
        got = []
        invoke_on_kill(lambda reason: got.append(reason), None)
        assert got == [None]

    def test_raising_callback_is_swallowed(self):
        def bad(reason):
            raise RuntimeError("boom")

        # Must not raise — a failing teardown never masks the kill.
        invoke_on_kill(bad, "cost")

    def test_raising_zero_arg_callback_is_swallowed(self):
        def bad():
            raise RuntimeError("boom")

        invoke_on_kill(bad, "cost")


# ---------------------------------------------------------------------------
# SpendEngine — each kill path delivers its reason
# ---------------------------------------------------------------------------


class TestEngineKillReasons:
    def test_cost_kill_passes_cost(self):
        reasons = []
        engine = SpendEngine(
            budget=Budget.daily(limit=0.01),
            agent_name="test",
            thresholds=(1.0,),
            on_kill=lambda reason: reasons.append(reason),
        )
        engine.post_flight("gpt-4o", input_tokens=0, output_tokens=10_000)
        engine.check_thresholds()
        assert reasons == ["cost"]

    def test_tokens_kill_passes_tokens(self):
        reasons = []
        engine = SpendEngine(
            budget=Budget.daily(limit=100.0),  # cost never crosses
            agent_name="test",
            thresholds=(1.0,),
            max_tokens_per_run=100,
            on_kill=lambda reason: reasons.append(reason),
        )
        engine.post_flight("gpt-4o", input_tokens=500, output_tokens=500)
        engine.check_thresholds()
        assert reasons == ["tokens"]

    def test_duration_kill_passes_duration(self):
        reasons = []
        engine = SpendEngine(
            budget=Budget.daily(limit=100.0),
            agent_name="test",
            thresholds=(1.0,),
            max_runtime_seconds=1.0,
            on_kill=lambda reason: reasons.append(reason),
        )
        # Backdate the first-call clock past the runtime cap.
        engine._first_call_time = time.monotonic() - 60.0
        engine.check_thresholds()
        assert reasons == ["duration"]

    def test_loop_kill_passes_loop(self):
        reasons = []
        engine = SpendEngine(
            budget=Budget.daily(limit=100.0),
            agent_name="test",
            detect_loops=True,
            loop_threshold=3,
            on_kill=lambda reason: reasons.append(reason),
        )
        msgs = [{"role": "user", "content": "hi"}]
        with pytest.raises(LoopDetectedError):
            for _ in range(8):
                engine.pre_flight("gpt-4o", msgs, input_tokens=10)
                engine.post_flight("gpt-4o", input_tokens=10, output_tokens=10)
                engine.check_thresholds()
        assert reasons == ["loop"]

    def test_calls_kill_passes_calls(self):
        reasons = []
        engine = SpendEngine(
            budget=Budget.daily(limit=100.0),
            agent_name="test",
            thresholds=(1.0,),
            max_calls_per_run=2,
            on_kill=lambda reason: reasons.append(reason),
        )
        msgs = [{"role": "user", "content": "hi"}]
        for _ in range(2):
            engine.pre_flight("gpt-4o", msgs, input_tokens=10)
            engine.post_flight("gpt-4o", input_tokens=10, output_tokens=10)
            engine.check_thresholds()
        with pytest.raises(CallLimitError):
            engine.pre_flight("gpt-4o", msgs, input_tokens=10)
        # The calls cap raises in pre-flight; the teardown fires on the next
        # threshold sweep (same latch semantics as every other kill path).
        engine.check_thresholds()
        assert reasons == ["calls"]

    def test_zero_arg_callback_still_works_on_cost_kill(self):
        # Pre-2.1 zero-arg teardowns must keep firing unchanged.
        fired = []
        engine = SpendEngine(
            budget=Budget.daily(limit=0.01),
            agent_name="test",
            thresholds=(1.0,),
            on_kill=lambda: fired.append(True),
        )
        engine.post_flight("gpt-4o", input_tokens=0, output_tokens=10_000)
        engine.check_thresholds()
        assert fired == [True]

    def test_reason_callback_fires_at_most_once(self):
        reasons = []
        engine = SpendEngine(
            budget=Budget.daily(limit=0.01),
            agent_name="test",
            thresholds=(0.5, 0.8, 1.0),
            on_kill=lambda reason: reasons.append(reason),
        )
        engine.post_flight("gpt-4o", input_tokens=0, output_tokens=10_000)
        for _ in range(5):
            engine.check_thresholds()
        assert reasons == ["cost"], f"on_kill fired {len(reasons)}× — must fire once"

    def test_reason_callback_rearms_after_reset(self):
        reasons = []
        engine = SpendEngine(
            budget=Budget.daily(limit=0.01),
            agent_name="test",
            thresholds=(1.0,),
            on_kill=lambda reason: reasons.append(reason),
        )
        engine.post_flight("gpt-4o", input_tokens=0, output_tokens=10_000)
        engine.check_thresholds()
        engine.reset()
        engine.post_flight("gpt-4o", input_tokens=0, output_tokens=10_000)
        engine.check_thresholds()
        assert reasons == ["cost", "cost"]

    def test_raising_reason_callback_does_not_break_thresholds(self):
        def bad(reason):
            raise RuntimeError("boom")

        engine = SpendEngine(
            budget=Budget.daily(limit=0.01),
            agent_name="test",
            thresholds=(1.0,),
            on_kill=bad,
        )
        engine.post_flight("gpt-4o", input_tokens=0, output_tokens=10_000)
        events = engine.check_thresholds()
        assert len(events) == 1


# ---------------------------------------------------------------------------
# IngestRejectedError — the backend-stop exception
# ---------------------------------------------------------------------------


class TestIngestRejectedError:
    def test_is_budget_exceeded_subclass(self):
        # Existing ``except BudgetExceededError`` handlers must keep catching.
        assert issubclass(IngestRejectedError, BudgetExceededError)

    def test_reason_attribute(self):
        exc = IngestRejectedError("rejected", reason="tier_agent_limit")
        assert exc.reason == "tier_agent_limit"

    def test_budget_attrs_are_none(self):
        # No budget was exceeded — spent/limit/period stay None (documented).
        exc = IngestRejectedError("rejected", reason="daily_limit")
        assert exc.spent is None
        assert exc.limit is None
        assert exc.period is None

    def test_exported_from_package_root(self):
        import agentkavach

        assert agentkavach.IngestRejectedError is IngestRejectedError
        assert "IngestRejectedError" in agentkavach.__all__


# ---------------------------------------------------------------------------
# Client backend-stop path — reason reaches both the exception and on_kill
# ---------------------------------------------------------------------------


class TestBackendStopReason:
    def test_raises_ingest_rejected_with_reason(self):
        guard = _guard(provider="openai")
        guard._handle_backend_reject("tier_agent_limit", {})
        guard._call_provider = MagicMock()

        with pytest.raises(IngestRejectedError) as excinfo:
            guard.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

        assert excinfo.value.reason == "tier_agent_limit"
        assert "tier_agent_limit" in str(excinfo.value)
        guard._call_provider.assert_not_called()

    def test_still_catchable_as_budget_exceeded(self):
        guard = _guard(provider="openai")
        guard._handle_backend_reject("daily_limit", {})
        guard._call_provider = MagicMock()

        with pytest.raises(BudgetExceededError) as excinfo:
            guard.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

        assert excinfo.value.reason == "daily_limit"

    def test_on_kill_receives_backend_reason(self):
        reasons = []
        guard = _guard(provider="openai", on_kill=lambda reason: reasons.append(reason))
        guard._handle_backend_reject("tier_agent_limit", {})
        guard._call_provider = MagicMock()

        with pytest.raises(IngestRejectedError):
            guard.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

        assert reasons == ["tier_agent_limit"]

    def test_zero_arg_on_kill_still_fires_on_backend_stop(self):
        fired = []
        guard = _guard(provider="openai", on_kill=lambda: fired.append(True))
        guard._handle_backend_reject("org_budget_exceeded", {})
        guard._call_provider = MagicMock()

        with pytest.raises(IngestRejectedError):
            guard.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

        assert fired == [True]

    def test_backend_kill_fires_at_most_once(self):
        reasons = []
        guard = _guard(provider="openai", on_kill=lambda reason: reasons.append(reason))
        guard._handle_backend_reject("tier_agent_limit", {})
        guard._call_provider = MagicMock()

        for _ in range(3):
            with pytest.raises(IngestRejectedError):
                guard.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

        assert reasons == ["tier_agent_limit"]

    def test_missing_reason_defaults_backend_rejected(self):
        reasons = []
        guard = _guard(provider="openai", on_kill=lambda reason: reasons.append(reason))
        # Simulate a 429 that carried no reason code.
        guard._backend_paused = True
        guard._backend_paused_reason = None
        guard._call_provider = MagicMock()

        with pytest.raises(IngestRejectedError) as excinfo:
            guard.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

        assert excinfo.value.reason == "backend_rejected"
        assert reasons == ["backend_rejected"]


# ---------------------------------------------------------------------------
# Alert-rule "kill" channel — reason is the rule's budget dimension
# ---------------------------------------------------------------------------


class TestKillChannelReason:
    def _event(self, budget_type: str) -> ThresholdEvent:
        return ThresholdEvent(
            threshold=1.0,
            spent=1.0,
            limit=1.0,
            budget_key="cost:daily:x",
            agent_name="kill-reason-test",
            budget_type=budget_type,
        )

    def test_cost_rule_passes_cost(self):
        reasons = []
        guard = _guard(
            on_kill=lambda reason: reasons.append(reason),
            alerts=[AlertRule(threshold=1.0, channels=("kill",), budget_type="cost")],
        )
        guard._dispatcher.dispatch(self._event("cost"))
        assert reasons == ["cost"]

    def test_tokens_rule_normalized_to_tokens(self):
        # The dispatcher speaks "tokens_total"; the engine reason vocabulary
        # (and therefore on_kill) says "tokens".
        reasons = []
        guard = _guard(
            on_kill=lambda reason: reasons.append(reason),
            alerts=[AlertRule(threshold=1.0, channels=("kill",), budget_type="tokens_total")],
        )
        guard._dispatcher.dispatch(self._event("tokens_total"))
        assert reasons == ["tokens"]

    def test_zero_arg_on_kill_still_works_via_kill_channel(self):
        fired = []
        guard = _guard(
            on_kill=lambda: fired.append(True),
            alerts=[AlertRule(threshold=1.0, channels=("kill",), budget_type="cost")],
        )
        guard._dispatcher.dispatch(self._event("cost"))
        assert fired == [True]


# ---------------------------------------------------------------------------
# fail_on_error paths — reason is "internal_error"
# ---------------------------------------------------------------------------


class TestFailOnErrorReason:
    def test_pre_flight_internal_error_passes_internal_error(self):
        reasons = []
        guard = _guard(
            provider="openai",
            on_kill=lambda reason: reasons.append(reason),
            fail_on_error=True,
        )
        guard._engine.pre_flight = MagicMock(side_effect=RuntimeError("engine error"))
        guard._call_provider = MagicMock()

        with pytest.raises(RuntimeError, match="engine error"):
            guard._safe_call({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})

        assert reasons == ["internal_error"]

    def test_post_flight_internal_error_passes_internal_error(self):
        reasons = []
        guard = _guard(
            provider="openai",
            on_kill=lambda reason: reasons.append(reason),
            fail_on_error=True,
        )
        guard._engine.pre_flight = MagicMock()
        guard._call_provider = MagicMock(
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content="hi"))],
                usage=MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="gpt-4o",
            )
        )
        guard._post_flight = MagicMock(side_effect=RuntimeError("post error"))

        with pytest.raises(RuntimeError, match="post error"):
            guard._safe_call({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})

        assert reasons == ["internal_error"]
