"""Budget enforcement unit tests for agentkavach.engine."""

from __future__ import annotations

import threading
from unittest.mock import patch

import pytest

from agentkavach.budget import Budget
from agentkavach.engine import SpendEngine, ThresholdEvent, _fast_token_count
from agentkavach.exceptions import BudgetExceededError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def engine() -> SpendEngine:
    """A fresh engine with a $10 daily budget."""
    return SpendEngine(budget=Budget.daily(limit=10.0), agent_name="test-agent")


@pytest.fixture()
def tight_engine() -> SpendEngine:
    """Engine with a very small budget for testing limits."""
    return SpendEngine(budget=Budget.daily(limit=0.01), agent_name="tight-agent")


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------


class TestPreFlight:
    def test_allows_call_under_budget(self, engine: SpendEngine):
        messages = [{"role": "user", "content": "hello"}]
        cost = engine.pre_flight("gpt-4o", messages)
        assert cost >= 0

    def test_rejects_call_over_budget(self, tight_engine: SpendEngine):
        # Force the counter near the limit first.
        tight_engine.post_flight("gpt-4o", input_tokens=0, output_tokens=1000)
        messages = [{"role": "user", "content": "x" * 5000}]
        with pytest.raises(BudgetExceededError):
            tight_engine.pre_flight("gpt-4o", messages)

    def test_unknown_model_passes(self, engine: SpendEngine):
        messages = [{"role": "user", "content": "test"}]
        cost = engine.pre_flight("totally-unknown-model", messages)
        assert cost == 0.0

    def test_killed_agent_rejects_immediately(self, engine: SpendEngine):
        engine._killed = True
        with pytest.raises(BudgetExceededError, match="killed"):
            engine.pre_flight("gpt-4o", [{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# Post-flight
# ---------------------------------------------------------------------------


class TestPostFlight:
    def test_records_spend(self, engine: SpendEngine):
        cost = engine.post_flight("gpt-4o", input_tokens=1000, output_tokens=500)
        assert cost > 0
        assert engine.spent == cost

    def test_accumulates_spend(self, engine: SpendEngine):
        c1 = engine.post_flight("gpt-4o", input_tokens=100, output_tokens=50)
        c2 = engine.post_flight("gpt-4o", input_tokens=200, output_tokens=100)
        assert engine.spent == pytest.approx(c1 + c2)

    def test_unknown_model_records_zero(self, engine: SpendEngine):
        cost = engine.post_flight("mystery-model-v9", input_tokens=100, output_tokens=50)
        assert cost == 0.0
        assert engine.spent == 0.0


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


class TestThresholds:
    def test_no_events_under_threshold(self, engine: SpendEngine):
        engine.post_flight("gpt-4o", input_tokens=10, output_tokens=5)
        events = engine.check_thresholds()
        assert events == []

    def test_70_percent_fires(self):
        engine = SpendEngine(
            budget=Budget.daily(limit=1.0),
            agent_name="test",
            thresholds=(0.70,),
        )
        # Push to 80% of $1.00 = $0.80.
        # gpt-4o: output=$0.010/1k → need 80 tokens for $0.80
        engine.post_flight("gpt-4o", input_tokens=0, output_tokens=80_000)
        events = engine.check_thresholds()
        assert len(events) == 1
        assert events[0].threshold == 0.70

    def test_threshold_fires_only_once(self):
        engine = SpendEngine(
            budget=Budget.daily(limit=1.0),
            agent_name="test",
            thresholds=(0.70,),
        )
        engine.post_flight("gpt-4o", input_tokens=0, output_tokens=80_000)
        engine.check_thresholds()
        # Second check should return no new events.
        events = engine.check_thresholds()
        assert events == []

    def test_multiple_thresholds_fire_in_order(self):
        engine = SpendEngine(
            budget=Budget.daily(limit=0.10),
            agent_name="test",
            thresholds=(0.50, 0.80, 1.0),
        )
        # Push spend way past all thresholds.
        engine.post_flight("gpt-4o", input_tokens=0, output_tokens=50_000)
        events = engine.check_thresholds()
        thresholds_fired = [e.threshold for e in events]
        assert thresholds_fired == [0.50, 0.80, 1.0]

    def test_kill_callback_fires_at_100(self):
        killed = []
        engine = SpendEngine(
            budget=Budget.daily(limit=0.01),
            agent_name="test",
            thresholds=(1.0,),
            on_kill=lambda: killed.append(True),
        )
        engine.post_flight("gpt-4o", input_tokens=0, output_tokens=10_000)
        engine.check_thresholds()
        assert killed == [True]
        assert engine._killed is True

    def test_kill_callback_fires_at_most_once(self):
        # Regression: check_thresholds() re-fired on_kill on EVERY post-kill
        # call (the guard was only `if self._killed`), so a teardown callback
        # could page / shut down / notify repeatedly. It must fire exactly once.
        killed = []
        engine = SpendEngine(
            budget=Budget.daily(limit=0.01),
            agent_name="test",
            thresholds=(0.5, 0.8, 1.0),
            on_kill=lambda: killed.append(True),
        )
        engine.post_flight("gpt-4o", input_tokens=0, output_tokens=10_000)
        for _ in range(5):
            engine.check_thresholds()
        assert killed == [True], f"on_kill fired {len(killed)}× — must fire once"

    def test_loop_kill_fires_on_kill_once(self):
        # A loop kill terminates the agent like any other guardrail, so on_kill
        # must fire — it raises in record_call_pattern before check_thresholds,
        # so it was previously the one kill path that never ran the teardown.
        from agentkavach.exceptions import LoopDetectedError

        killed = []
        engine = SpendEngine(
            budget=Budget.daily(limit=100.0),
            agent_name="test",
            detect_loops=True,
            loop_threshold=3,
            on_kill=lambda: killed.append(True),
        )
        msgs = [{"role": "user", "content": "hi"}]
        raised = False
        for _ in range(8):
            try:
                engine.pre_flight("gpt-4o", msgs, input_tokens=10)
                engine.post_flight("gpt-4o", input_tokens=10, output_tokens=10)
                engine.check_thresholds()
            except LoopDetectedError:
                raised = True
                break
        assert raised, "loop should have been detected"
        assert killed == [True], f"loop kill must fire on_kill exactly once, got {len(killed)}"

    def test_kill_callback_rearms_after_reset(self):
        killed = []
        engine = SpendEngine(
            budget=Budget.daily(limit=0.01),
            agent_name="test",
            thresholds=(1.0,),
            on_kill=lambda: killed.append(True),
        )
        engine.post_flight("gpt-4o", input_tokens=0, output_tokens=10_000)
        engine.check_thresholds()
        engine.reset()
        engine.post_flight("gpt-4o", input_tokens=0, output_tokens=10_000)
        engine.check_thresholds()
        assert killed == [True, True]

    def test_kill_callback_exception_is_logged(self):
        def bad_callback():
            raise RuntimeError("boom")

        engine = SpendEngine(
            budget=Budget.daily(limit=0.01),
            agent_name="test",
            thresholds=(1.0,),
            on_kill=bad_callback,
        )
        engine.post_flight("gpt-4o", input_tokens=0, output_tokens=10_000)
        # Should not raise — exception is caught and logged.
        events = engine.check_thresholds()
        assert len(events) == 1


# ---------------------------------------------------------------------------
# ThresholdEvent
# ---------------------------------------------------------------------------


class TestThresholdEvent:
    def test_fields(self):
        event = ThresholdEvent(
            threshold=0.70,
            spent=7.0,
            limit=10.0,
            budget_key="daily:2026-03-13",
            agent_name="agent-a",
        )
        assert event.threshold == 0.70
        assert event.spent == 7.0
        assert event.agent_name == "agent-a"
        # Phase 40 default — backward-compatible with cost-only callers.
        assert event.budget_type == "cost"


class TestMultiDimensionThresholds:
    """Phase 40: check_thresholds evaluates cost / tokens / duration
    independently. A token budget without a cost budget still fires alerts
    at 50/80/100% of token usage; same for duration. Previously alerts
    were cost-percent only, so token/duration agents silently never paged."""

    def test_tokens_threshold_fires_independently_of_cost(self):
        engine = SpendEngine(
            budget=Budget.daily(10.0),  # plenty of cost headroom
            agent_name="tokens-bot",
            max_tokens_per_run=1000,
            thresholds=(0.5, 0.8, 1.0),
        )
        engine.pre_flight("gpt-4o", [{"role": "user", "content": "x"}])
        # 500 tokens — exactly 50% of tokens cap.
        engine.post_flight("gpt-4o", input_tokens=300, output_tokens=200)
        events = engine.check_thresholds()
        token_events = [e for e in events if e.budget_type == "tokens_total"]
        assert any(e.threshold == 0.5 for e in token_events), (
            "50% tokens threshold must fire even when cost is at 0%"
        )
        assert all(e.budget_key == "tokens_total:per_run" for e in token_events)

    def test_duration_threshold_fires_independently_of_cost(self):
        engine = SpendEngine(
            budget=Budget.daily(10.0),
            agent_name="duration-bot",
            max_runtime_seconds=10.0,
            thresholds=(0.5, 0.8, 1.0),
        )
        engine.pre_flight("gpt-4o", [{"role": "user", "content": "x"}])
        # Pin the first-call clock to a small, exactly-representable float so
        # ``(start + 8.0) - start`` is exactly 8.0. Using the real
        # ``time.monotonic()`` value (a large float ~1e6) loses low-order
        # precision: ``(big + 8.0) - big`` can come back as 7.9999…, making
        # ``elapsed / 10.0`` fall just under 0.8 and the 80% threshold flake
        # under xdist.
        engine._first_call_time = 1000.0
        original_time = engine._first_call_time
        with patch("agentkavach.engine.time") as mock_time:
            # Simulate 8s elapsed — exactly 80% of the 10s duration cap.
            mock_time.monotonic.return_value = original_time + 8.0
            engine.post_flight("gpt-4o", 50, 50)
            events = engine.check_thresholds()
        duration_events = [e for e in events if e.budget_type == "duration"]
        # 50 % and 80 % should both fire on first check after crossing them.
        thresholds_fired = {e.threshold for e in duration_events}
        assert 0.5 in thresholds_fired
        assert 0.8 in thresholds_fired

    def test_dimensions_track_fired_state_independently(self):
        # Token threshold firing must not silence cost thresholds, and vice
        # versa. They live in separate _fired buckets.
        engine = SpendEngine(
            budget=Budget.daily(0.001),
            agent_name="multi-bot",
            max_tokens_per_run=1000,
            thresholds=(1.0,),
        )
        engine.pre_flight("gpt-4o", [{"role": "user", "content": "x"}])
        # Push BOTH dimensions over the limit in a single call.
        engine.post_flight("gpt-4o", input_tokens=600, output_tokens=600)
        events = engine.check_thresholds()
        btypes = {e.budget_type for e in events}
        assert "cost" in btypes
        assert "tokens_total" in btypes


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestEngineProperties:
    def test_remaining(self, engine: SpendEngine):
        assert engine.remaining == 10.0
        engine.post_flight("gpt-4o", input_tokens=1000, output_tokens=500)
        assert engine.remaining < 10.0
        assert engine.remaining == pytest.approx(10.0 - engine.spent)

    def test_utilization(self, engine: SpendEngine):
        assert engine.utilization == 0.0
        engine.post_flight("gpt-4o", input_tokens=1000, output_tokens=500)
        assert 0 < engine.utilization < 1.0

    def test_remaining_never_negative(self, tight_engine: SpendEngine):
        tight_engine.post_flight("gpt-4o", input_tokens=0, output_tokens=100_000)
        assert tight_engine.remaining == 0.0


# ---------------------------------------------------------------------------
# Partial recording
# ---------------------------------------------------------------------------


class TestRecordPartial:
    def test_records_output_only(self, engine: SpendEngine):
        cost = engine.record_partial("gpt-4o", output_tokens=500)
        assert cost > 0
        assert engine.spent == cost

    def test_unknown_model_returns_zero(self, engine: SpendEngine):
        assert engine.record_partial("mystery-model", output_tokens=500) == 0.0


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_clears_spend(self, engine: SpendEngine):
        engine.post_flight("gpt-4o", input_tokens=100, output_tokens=50)
        engine.reset()
        assert engine.spent == 0.0
        assert engine._killed is False


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_post_flights(self):
        engine = SpendEngine(budget=Budget.daily(limit=1000.0), agent_name="thread-test")
        errors: list[Exception] = []

        def do_post_flights():
            try:
                for _ in range(100):
                    engine.post_flight("gpt-4o", input_tokens=10, output_tokens=5)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=do_post_flights) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert engine.spent > 0


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


class TestPreFlightWithInputTokens:
    """Tests for pre_flight with exact input_tokens from native counting."""

    def test_uses_exact_input_tokens(self, engine: SpendEngine):
        messages = [{"role": "user", "content": "hello"}]
        cost = engine.pre_flight("gpt-4o", messages, input_tokens=100)
        assert cost > 0

    def test_input_cost_uses_input_pricing(self):
        # gpt-4o: input=$0.0025/1k, output=$0.010/1k
        # 1000 input tokens: input_cost=$0.0025, estimated_output=$0.010
        # total = $0.0125
        engine = SpendEngine(budget=Budget.daily(limit=10.0), agent_name="test")
        cost = engine.pre_flight("gpt-4o", [], input_tokens=1000)
        assert cost == pytest.approx(0.0025 + 0.010, rel=1e-4)

    def test_pre_flight_allows_until_actual_spend_exceeds(self):
        # Phase 36 design change: pre-flight no longer rejects based on the
        # *estimated* next-call cost. As long as recorded spend hasn't
        # crossed the limit, the call is allowed — the user keeps the work
        # they paid for. Only the NEXT pre-flight (after post-flight has
        # pushed spend past the limit) rejects.
        engine = SpendEngine(budget=Budget.daily(limit=0.001), agent_name="test")
        # First call must be allowed even when estimated cost dwarfs the limit.
        engine.pre_flight("gpt-4o", [], input_tokens=10000)
        # Simulate the call landing and pushing recorded spend over the limit.
        engine.post_flight("gpt-4o", input_tokens=10000, output_tokens=10000)
        # Now the NEXT pre-flight is the one that should reject.
        with pytest.raises(BudgetExceededError, match="Exceeded budget"):
            engine.pre_flight("gpt-4o", [], input_tokens=100)

    def test_falls_back_to_heuristic_when_none(self, engine: SpendEngine):
        messages = [{"role": "user", "content": "hello world"}]
        cost_no_tokens = engine.pre_flight("gpt-4o", messages, input_tokens=None)
        assert cost_no_tokens >= 0

    def test_triggers_thresholds_on_budget_exceeded(self):
        killed = []

        def on_kill():
            killed.append(True)

        engine = SpendEngine(
            budget=Budget.daily(limit=0.001),
            agent_name="test",
            thresholds=(1.0,),
            on_kill=on_kill,
        )
        # Spend up to 100% first.
        engine.post_flight("gpt-4o", input_tokens=0, output_tokens=1000)
        events = engine.check_thresholds()
        assert len(events) == 1
        assert killed == [True]


class TestTokenCount:
    def test_simple_message(self):
        count = _fast_token_count([{"role": "user", "content": "Hello world"}])
        assert count > 0

    def test_empty_messages(self):
        assert _fast_token_count([]) == 0

    def test_multipart_content(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }
        ]
        count = _fast_token_count(messages)
        assert count > 0

    def test_missing_content_key(self):
        # Should not crash on messages without "content".
        count = _fast_token_count([{"role": "system"}])
        assert count >= 0


# ---------------------------------------------------------------------------
# Shared budget
# ---------------------------------------------------------------------------


class TestSharedBudget:
    def test_two_engines_share_counter(self):
        shared = Budget.org_budget(limit=1.0)
        engine_a = SpendEngine(budget=shared, agent_name="agent-a")
        engine_b = SpendEngine(budget=shared, agent_name="agent-b")

        # Share the same _spend dict to simulate shared memory.
        engine_b._spend = engine_a._spend
        engine_b._fired = engine_a._fired

        engine_a.post_flight("gpt-4o", input_tokens=0, output_tokens=50_000)
        engine_b.post_flight("gpt-4o", input_tokens=0, output_tokens=50_000)

        # Both should see the combined spend.
        assert engine_a.spent == engine_b.spent
        assert engine_a.spent > 0


# ---------------------------------------------------------------------------
# Org budget (dual-budget enforcement)
# ---------------------------------------------------------------------------


class TestOrgBudget:
    def test_post_flight_records_both_budgets(self):
        """post_flight should record spend against both primary and org budget keys."""
        primary = Budget.daily(limit=10.0)
        org = Budget.org_budget(limit=50.0)
        engine = SpendEngine(budget=primary, agent_name="test", org_budget=org)

        cost = engine.post_flight("gpt-4o", input_tokens=1000, output_tokens=500)
        assert cost > 0
        # Primary budget key should have the cost
        assert engine._spend.get(primary.key, 0.0) == cost
        # Org budget key should also have the cost
        assert engine._spend.get(org.key, 0.0) == cost

    def test_pre_flight_rejects_when_org_budget_exceeded(self):
        """pre_flight should reject when org budget is exhausted, even if primary has room."""
        primary = Budget.daily(limit=100.0)  # plenty of room
        org = Budget.org_budget(limit=0.01)  # very small org budget
        engine = SpendEngine(budget=primary, agent_name="test", org_budget=org)

        # Burn through the org budget
        engine.post_flight("gpt-4o", input_tokens=0, output_tokens=10_000)

        # Primary has room but org budget is exhausted
        with pytest.raises(BudgetExceededError, match="org budget"):
            engine.pre_flight("gpt-4o", [{"role": "user", "content": "x" * 5000}])

    def test_pre_flight_rejects_when_primary_budget_exceeded(self):
        """pre_flight should reject when primary budget is exhausted, even if org has room."""
        primary = Budget.daily(limit=0.01)  # very small primary
        org = Budget.org_budget(limit=100.0)  # plenty of org room
        engine = SpendEngine(budget=primary, agent_name="test", org_budget=org)

        # Burn through the primary budget
        engine.post_flight("gpt-4o", input_tokens=0, output_tokens=10_000)

        with pytest.raises(BudgetExceededError):
            engine.pre_flight("gpt-4o", [{"role": "user", "content": "x" * 5000}])

    def test_pre_flight_allows_when_both_budgets_have_room(self):
        """pre_flight should allow calls when both budgets have remaining capacity."""
        primary = Budget.daily(limit=100.0)
        org = Budget.org_budget(limit=200.0)
        engine = SpendEngine(budget=primary, agent_name="test", org_budget=org)

        cost = engine.pre_flight("gpt-4o", [{"role": "user", "content": "hello"}])
        assert cost >= 0

    def test_no_org_budget_works_normally(self):
        """Engine without org_budget should work exactly as before."""
        engine = SpendEngine(budget=Budget.daily(limit=10.0), agent_name="test")
        assert engine.org_budget is None
        cost = engine.post_flight("gpt-4o", input_tokens=1000, output_tokens=500)
        assert cost > 0
        assert engine.spent == cost

    def test_org_budget_accumulates_across_calls(self):
        """Org budget spend should accumulate across multiple post_flight calls."""
        primary = Budget.daily(limit=100.0)
        org = Budget.org_budget(limit=50.0)
        engine = SpendEngine(budget=primary, agent_name="test", org_budget=org)

        c1 = engine.post_flight("gpt-4o", input_tokens=100, output_tokens=50)
        c2 = engine.post_flight("gpt-4o", input_tokens=200, output_tokens=100)

        assert engine._spend.get(org.key, 0.0) == pytest.approx(c1 + c2)
