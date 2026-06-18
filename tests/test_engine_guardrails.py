"""Tests for SpendEngine guardrails — token, call count, runtime limits."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agentkavach.budget import Budget
from agentkavach.engine import SpendEngine
from agentkavach.exceptions import (
    BudgetExceededError,
    CallLimitError,
    LoopDetectedError,
    RuntimeLimitError,
    TokenLimitError,
)

MESSAGES = [{"role": "user", "content": "Hello"}]


@pytest.fixture()
def engine():
    """Engine with a generous budget and no guardrails."""
    return SpendEngine(budget=Budget.daily(1000), agent_name="test-bot")


# ---------------------------------------------------------------------------
# Token limit
# ---------------------------------------------------------------------------


class TestTokenLimit:
    def test_no_limit_by_default(self, engine: SpendEngine):
        assert engine.max_tokens_per_run is None
        engine.pre_flight("gpt-4o", MESSAGES)
        engine.post_flight("gpt-4o", 5000, 3000)
        # No error — no limit set.

    def test_raises_when_tokens_exceeded(self):
        engine = SpendEngine(
            budget=Budget.daily(1000),
            agent_name="test-bot",
            max_tokens_per_run=1000,
        )
        engine.pre_flight("gpt-4o", MESSAGES)
        # First call: 600 tokens — under limit.
        engine.post_flight("gpt-4o", 400, 200)
        assert engine._total_tokens == 600

        engine.pre_flight("gpt-4o", MESSAGES)
        # Second call: 600 more = 1200 total → over 1000 limit. Phase 40
        # unified the kill path: post_flight no longer raises immediately;
        # check_thresholds owns the kill via the 100% token threshold and
        # the NEXT pre_flight rejects. Phase 102 routes the rejection
        # through TokenLimitError (matching the public docs) instead of
        # the generic BudgetExceededError.
        engine.post_flight("gpt-4o", 400, 200)
        engine.check_thresholds()
        assert engine._total_tokens == 1200
        # _killed is set by check_thresholds at the 100% token rule, so
        # pre_flight's first short-circuit (the "killed" branch) fires
        # — and now replays the token-limit exception specifically.
        with pytest.raises(TokenLimitError):
            engine.pre_flight("gpt-4o", MESSAGES)

    def test_kills_engine_after_token_limit(self):
        engine = SpendEngine(
            budget=Budget.daily(1000),
            agent_name="test-bot",
            max_tokens_per_run=500,
        )
        engine.pre_flight("gpt-4o", MESSAGES)
        engine.post_flight("gpt-4o", 400, 200)
        # 100% threshold dispatches the kill via check_thresholds (Phase 40).
        engine.check_thresholds()
        assert engine._killed is True
        # Subsequent calls are blocked via TokenLimitError (Phase 102 —
        # used to be BudgetExceededError). The exception carries the
        # documented `spent` and `limit` attributes.
        with pytest.raises(TokenLimitError) as exc_info:
            engine.pre_flight("gpt-4o", MESSAGES)
        assert exc_info.value.spent == 600
        assert exc_info.value.limit == 500

    def test_exact_limit_is_not_exceeded(self):
        engine = SpendEngine(
            budget=Budget.daily(1000),
            agent_name="test-bot",
            max_tokens_per_run=1000,
        )
        engine.pre_flight("gpt-4o", MESSAGES)
        # Exactly at limit — should not raise (> not >=).
        engine.post_flight("gpt-4o", 500, 500)
        assert engine._total_tokens == 1000

    def test_reset_clears_token_count(self):
        engine = SpendEngine(
            budget=Budget.daily(1000),
            agent_name="test-bot",
            max_tokens_per_run=1000,
        )
        engine.pre_flight("gpt-4o", MESSAGES)
        engine.post_flight("gpt-4o", 400, 200)
        engine.reset()
        assert engine._total_tokens == 0
        assert engine._call_count == 0


# ---------------------------------------------------------------------------
# Call count limit
# ---------------------------------------------------------------------------


class TestCallCountLimit:
    def test_no_limit_by_default(self, engine: SpendEngine):
        assert engine.max_calls_per_run is None
        for _ in range(10):
            engine.pre_flight("gpt-4o", MESSAGES)
            engine.post_flight("gpt-4o", 100, 50)

    def test_raises_when_calls_exceeded(self):
        engine = SpendEngine(
            budget=Budget.daily(1000),
            agent_name="test-bot",
            max_calls_per_run=3,
        )
        for _ in range(3):
            engine.pre_flight("gpt-4o", MESSAGES)
            engine.post_flight("gpt-4o", 100, 50)

        # 4th call → pre_flight raises.
        with pytest.raises(CallLimitError, match="exceeded call limit"):
            engine.pre_flight("gpt-4o", MESSAGES)

    def test_kills_engine_after_call_limit(self):
        engine = SpendEngine(
            budget=Budget.daily(1000),
            agent_name="test-bot",
            max_calls_per_run=1,
        )
        engine.pre_flight("gpt-4o", MESSAGES)
        engine.post_flight("gpt-4o", 100, 50)

        with pytest.raises(CallLimitError):
            engine.pre_flight("gpt-4o", MESSAGES)

        assert engine._killed is True

    def test_reset_clears_call_count(self):
        engine = SpendEngine(
            budget=Budget.daily(1000),
            agent_name="test-bot",
            max_calls_per_run=2,
        )
        engine.pre_flight("gpt-4o", MESSAGES)
        engine.post_flight("gpt-4o", 100, 50)
        engine.pre_flight("gpt-4o", MESSAGES)
        engine.post_flight("gpt-4o", 100, 50)

        engine.reset()
        # After reset, should be able to make calls again.
        engine.pre_flight("gpt-4o", MESSAGES)
        engine.post_flight("gpt-4o", 100, 50)


# ---------------------------------------------------------------------------
# Runtime limit
# ---------------------------------------------------------------------------


class TestRuntimeLimit:
    def test_no_limit_by_default(self, engine: SpendEngine):
        assert engine.max_runtime_seconds is None
        engine.pre_flight("gpt-4o", MESSAGES)
        engine.post_flight("gpt-4o", 100, 50)

    def test_raises_when_runtime_exceeded(self):
        engine = SpendEngine(
            budget=Budget.daily(1000),
            agent_name="test-bot",
            max_runtime_seconds=5.0,
        )
        # First call sets _first_call_time.
        engine.pre_flight("gpt-4o", MESSAGES)
        engine.post_flight("gpt-4o", 100, 50)

        # Phase 102 restored RuntimeLimitError (the audit'd contract):
        # pre_flight rejects with RuntimeLimitError once elapsed wall
        # clock crosses the limit. Phase 40 wrongly degraded this to
        # BudgetExceededError, silently breaking documented except clauses.
        original_time = engine._first_call_time
        with patch("agentkavach.engine.time") as mock_time:
            mock_time.monotonic.return_value = original_time + 10.0
            with pytest.raises(RuntimeLimitError, match="duration"):
                engine.pre_flight("gpt-4o", MESSAGES)

    def test_within_limit_passes(self):
        engine = SpendEngine(
            budget=Budget.daily(1000),
            agent_name="test-bot",
            max_runtime_seconds=60.0,
        )
        # First call.
        engine.pre_flight("gpt-4o", MESSAGES)
        engine.post_flight("gpt-4o", 100, 50)
        # Second call (immediately after) — well within 60s.
        engine.pre_flight("gpt-4o", MESSAGES)
        engine.post_flight("gpt-4o", 100, 50)

    def test_kills_engine_after_runtime_limit(self):
        engine = SpendEngine(
            budget=Budget.daily(1000),
            agent_name="test-bot",
            max_runtime_seconds=1.0,
        )
        engine.pre_flight("gpt-4o", MESSAGES)
        engine.post_flight("gpt-4o", 100, 50)

        original_time = engine._first_call_time
        with patch("agentkavach.engine.time") as mock_time:
            mock_time.monotonic.return_value = original_time + 5.0
            # Threshold-check fires the kill via the 100% duration rule.
            engine.check_thresholds()
            # Phase 102: rejection is RuntimeLimitError (matches docs).
            with pytest.raises(RuntimeLimitError):
                engine.pre_flight("gpt-4o", MESSAGES)

        assert engine._killed is True

    def test_reset_clears_runtime(self):
        engine = SpendEngine(
            budget=Budget.daily(1000),
            agent_name="test-bot",
            max_runtime_seconds=1.0,
        )
        engine.pre_flight("gpt-4o", MESSAGES)
        engine.post_flight("gpt-4o", 100, 50)
        assert engine._first_call_time is not None

        engine.reset()
        assert engine._first_call_time is None


# ---------------------------------------------------------------------------
# Combined guardrails
# ---------------------------------------------------------------------------


class TestCombinedGuardrails:
    def test_call_limit_checked_before_budget(self):
        """Call limit fires in pre_flight, before budget check."""
        engine = SpendEngine(
            budget=Budget.daily(1000),
            agent_name="test-bot",
            max_calls_per_run=1,
        )
        engine.pre_flight("gpt-4o", MESSAGES)
        engine.post_flight("gpt-4o", 100, 50)

        # Call limit should fire before budget check.
        with pytest.raises(CallLimitError):
            engine.pre_flight("gpt-4o", MESSAGES)

    def test_runtime_checked_before_call_limit(self):
        """Phase 40: call-count is the only immediate-raise guardrail
        remaining. Runtime + tokens flow through the unified threshold
        path → BudgetExceededError on the next pre_flight."""
        engine = SpendEngine(
            budget=Budget.daily(1000),
            agent_name="test-bot",
            max_calls_per_run=1,
            max_runtime_seconds=1.0,
        )
        engine.pre_flight("gpt-4o", MESSAGES)
        engine.post_flight("gpt-4o", 100, 50)

        original_time = engine._first_call_time
        with patch("agentkavach.engine.time") as mock_time:
            mock_time.monotonic.return_value = original_time + 5.0
            # Both limits exceeded. Call cap fires first (still raised
            # eagerly because "+1 over" has no meaningful threshold).
            with pytest.raises(CallLimitError):
                engine.pre_flight("gpt-4o", MESSAGES)

    def test_multiple_guardrails_token_fires_in_post(self):
        """Token limit no longer raises in post_flight (Phase 40). The
        unified kill path fires via check_thresholds at 100% tokens, and
        the next pre_flight rejects with TokenLimitError (Phase 102 —
        previously degraded to BudgetExceededError)."""
        engine = SpendEngine(
            budget=Budget.daily(1000),
            agent_name="test-bot",
            max_calls_per_run=10,
            max_tokens_per_run=500,
        )
        engine.pre_flight("gpt-4o", MESSAGES)
        engine.post_flight("gpt-4o", 400, 200)
        engine.check_thresholds()
        assert engine._killed is True
        with pytest.raises(TokenLimitError):
            engine.pre_flight("gpt-4o", MESSAGES)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestGuardrailProperties:
    def test_total_tokens_tracked(self):
        engine = SpendEngine(
            budget=Budget.daily(1000),
            agent_name="test-bot",
            max_tokens_per_run=100000,
        )
        engine.pre_flight("gpt-4o", MESSAGES)
        engine.post_flight("gpt-4o", 100, 50)
        assert engine._total_tokens == 150

        engine.pre_flight("gpt-4o", MESSAGES)
        engine.post_flight("gpt-4o", 200, 100)
        assert engine._total_tokens == 450

    def test_call_count_tracked(self):
        engine = SpendEngine(
            budget=Budget.daily(1000),
            agent_name="test-bot",
            max_calls_per_run=100,
        )
        for i in range(5):
            engine.pre_flight("gpt-4o", MESSAGES)
            engine.post_flight("gpt-4o", 100, 50)
        assert engine._call_count == 5

    def test_first_call_time_set_on_first_pre_flight(self):
        engine = SpendEngine(
            budget=Budget.daily(1000),
            agent_name="test-bot",
            max_runtime_seconds=60.0,
        )
        assert engine._first_call_time is None
        engine.pre_flight("gpt-4o", MESSAGES)
        assert engine._first_call_time is not None
        first = engine._first_call_time

        engine.post_flight("gpt-4o", 100, 50)
        engine.pre_flight("gpt-4o", MESSAGES)
        # Should not change on subsequent calls.
        assert engine._first_call_time == first


# ---------------------------------------------------------------------------
# Phase 102 — Public exception contract
# ---------------------------------------------------------------------------
#
# The audit in plan/docs-audit-2026-05-28.md flagged that the engine
# raises plain BudgetExceededError for token/runtime overruns and that
# none of the documented `e.<attr>` references actually exist. These
# tests pin every dimension to its documented exception subclass and
# documented attribute name. If they break, the public docs are wrong
# too — see `dashboard/app/public/docs/guardrails/page.tsx` and
# `dashboard/app/public/docs/budgets/page.tsx`.


class TestPublicExceptionContract:
    def test_token_limit_raises_TokenLimitError_not_BudgetExceededError(self):
        engine = SpendEngine(
            budget=Budget.daily(1000),
            agent_name="test-bot",
            max_tokens_per_run=10,
        )
        engine.pre_flight("gpt-4o", MESSAGES)
        engine.post_flight("gpt-4o", 20, 30)  # 50 tokens — over 10
        engine.check_thresholds()
        with pytest.raises(TokenLimitError) as exc_info:
            engine.pre_flight("gpt-4o", MESSAGES)
        # TokenLimitError is a GuardrailError but NOT a BudgetExceededError —
        # customers catch each subclass independently.
        assert not isinstance(exc_info.value, BudgetExceededError)
        assert exc_info.value.spent == 50
        assert exc_info.value.limit == 10

    def test_runtime_limit_raises_RuntimeLimitError_with_elapsed_attr(self):
        engine = SpendEngine(
            budget=Budget.daily(1000),
            agent_name="test-bot",
            max_runtime_seconds=2.0,
        )
        engine.pre_flight("gpt-4o", MESSAGES)
        engine.post_flight("gpt-4o", 100, 50)
        original_time = engine._first_call_time
        with patch("agentkavach.engine.time") as mock_time:
            mock_time.monotonic.return_value = original_time + 7.5
            with pytest.raises(RuntimeLimitError) as exc_info:
                engine.pre_flight("gpt-4o", MESSAGES)
        assert not isinstance(exc_info.value, BudgetExceededError)
        # elapsed is the wall-clock seconds since the first call.
        assert exc_info.value.elapsed is not None
        assert 7.0 <= exc_info.value.elapsed <= 8.0
        assert exc_info.value.limit == 2.0

    def test_call_limit_raises_CallLimitError_with_call_count_attr(self):
        engine = SpendEngine(
            budget=Budget.daily(1000),
            agent_name="test-bot",
            max_calls_per_run=2,
        )
        engine.pre_flight("gpt-4o", MESSAGES)
        engine.post_flight("gpt-4o", 100, 50)
        engine.pre_flight("gpt-4o", MESSAGES)
        engine.post_flight("gpt-4o", 100, 50)
        with pytest.raises(CallLimitError) as exc_info:
            engine.pre_flight("gpt-4o", MESSAGES)
        assert exc_info.value.call_count == 2
        assert exc_info.value.limit == 2

    def test_loop_detected_raises_LoopDetectedError_with_pattern_and_count_attrs(self):
        engine = SpendEngine(
            budget=Budget.daily(1000),
            agent_name="test-bot",
            detect_loops=True,
            loop_threshold=3,
        )
        with pytest.raises(LoopDetectedError) as exc_info:
            # Three consecutive repetitions of the same (model, tool)
            # pair within a window long enough for the detector.
            for _ in range(6):
                engine.record_call_pattern("gpt-4o", "search")
        assert exc_info.value.pattern is not None
        assert exc_info.value.count == 3

    def test_budget_overrun_still_raises_BudgetExceededError(self):
        """Regression: don't accidentally re-classify the actual cost path."""
        engine = SpendEngine(
            budget=Budget.daily(0.01),
            agent_name="test-bot",
        )
        engine.pre_flight("gpt-4o", MESSAGES)
        # gpt-4o input is $0.0025/1k, output $0.01/1k → 1M tokens dwarfs the cap.
        engine.post_flight("gpt-4o", 1_000_000, 1_000_000)
        engine.check_thresholds()
        with pytest.raises(BudgetExceededError) as exc_info:
            engine.pre_flight("gpt-4o", MESSAGES)
        # BudgetExceededError is its own class — NOT a GuardrailError.
        from agentkavach.exceptions import GuardrailError as _G

        assert not isinstance(exc_info.value, _G)
        assert exc_info.value.spent is not None
        assert exc_info.value.spent > 0.01
        assert exc_info.value.limit == 0.01
        assert exc_info.value.period == "daily"

    def test_kill_reason_persists_so_subsequent_calls_raise_same_type(self):
        """The first-100% dimension wins — every subsequent pre_flight
        replays the same exception subclass so customers' specific
        ``except TokenLimitError:`` blocks keep catching."""
        engine = SpendEngine(
            budget=Budget.daily(1000),
            agent_name="test-bot",
            max_tokens_per_run=10,
        )
        engine.pre_flight("gpt-4o", MESSAGES)
        engine.post_flight("gpt-4o", 20, 30)
        engine.check_thresholds()
        # Three independent attempts — all must raise the same type.
        for _ in range(3):
            with pytest.raises(TokenLimitError):
                engine.pre_flight("gpt-4o", MESSAGES)
