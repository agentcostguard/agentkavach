"""Tests for runaway loop detection in SpendEngine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agentkavach import AgentKavach, Budget
from agentkavach.engine import SpendEngine
from agentkavach.exceptions import LoopDetectedError

MESSAGES = [{"role": "user", "content": "Hello"}]


@pytest.fixture()
def engine():
    """Engine with loop detection enabled."""
    return SpendEngine(
        budget=Budget.daily(1000),
        agent_name="test-bot",
        detect_loops=True,
        loop_threshold=3,
    )


# ---------------------------------------------------------------------------
# Basic detection
# ---------------------------------------------------------------------------


class TestLoopDetection:
    def test_no_detection_when_disabled(self):
        engine = SpendEngine(
            budget=Budget.daily(1000),
            agent_name="test-bot",
            detect_loops=False,
        )
        # Repeat same pattern many times — should not raise.
        for _ in range(20):
            engine.record_call_pattern("gpt-4o", "search")
            engine.record_call_pattern("gpt-4o", "respond")

    def test_detects_length_2_loop(self, engine: SpendEngine):
        """Pattern of length 2 repeated 3 times triggers detection."""
        engine.record_call_pattern("gpt-4o", "search")
        engine.record_call_pattern("gpt-4o", "respond")
        engine.record_call_pattern("gpt-4o", "search")
        engine.record_call_pattern("gpt-4o", "respond")
        # 3rd repetition triggers.
        engine.record_call_pattern("gpt-4o", "search")
        with pytest.raises(LoopDetectedError, match="repeating pattern"):
            engine.record_call_pattern("gpt-4o", "respond")

    def test_detects_length_3_loop(self, engine: SpendEngine):
        """Pattern of length 3 repeated 3 times."""
        pattern = [("gpt-4o", "search"), ("gpt-4o", "parse"), ("gpt-4o", "respond")]
        for _ in range(2):
            for model, tool in pattern:
                engine.record_call_pattern(model, tool)
        # 3rd repetition.
        engine.record_call_pattern("gpt-4o", "search")
        engine.record_call_pattern("gpt-4o", "parse")
        with pytest.raises(LoopDetectedError):
            engine.record_call_pattern("gpt-4o", "respond")

    def test_no_false_positive_varied_calls(self, engine: SpendEngine):
        """Diverse call patterns should not trigger detection."""
        tools = ["search", "parse", "respond", "save", "log", "check"]
        for i in range(15):
            engine.record_call_pattern("gpt-4o", tools[i % len(tools)])

    def test_no_false_positive_insufficient_repeats(self, engine: SpendEngine):
        """Pattern repeated only twice (below threshold=3) doesn't trigger."""
        engine.record_call_pattern("gpt-4o", "search")
        engine.record_call_pattern("gpt-4o", "respond")
        engine.record_call_pattern("gpt-4o", "search")
        engine.record_call_pattern("gpt-4o", "respond")
        # Only 2 repetitions — should not raise.

    def test_kills_engine_on_loop(self, engine: SpendEngine):
        """Engine is killed after loop detection."""
        for _ in range(2):
            engine.record_call_pattern("gpt-4o", "a")
            engine.record_call_pattern("gpt-4o", "b")
        engine.record_call_pattern("gpt-4o", "a")
        with pytest.raises(LoopDetectedError):
            engine.record_call_pattern("gpt-4o", "b")
        assert engine._killed is True


# ---------------------------------------------------------------------------
# Custom threshold
# ---------------------------------------------------------------------------


class TestLoopThreshold:
    def test_threshold_2(self):
        """Lower threshold catches loops sooner."""
        engine = SpendEngine(
            budget=Budget.daily(1000),
            agent_name="test-bot",
            detect_loops=True,
            loop_threshold=2,
        )
        engine.record_call_pattern("gpt-4o", "a")
        engine.record_call_pattern("gpt-4o", "b")
        # 2nd repetition triggers.
        engine.record_call_pattern("gpt-4o", "a")
        with pytest.raises(LoopDetectedError):
            engine.record_call_pattern("gpt-4o", "b")

    def test_threshold_5(self):
        """Higher threshold allows more repetitions."""
        engine = SpendEngine(
            budget=Budget.daily(1000),
            agent_name="test-bot",
            detect_loops=True,
            loop_threshold=5,
        )
        # 4 repetitions — under threshold.
        for _ in range(4):
            engine.record_call_pattern("gpt-4o", "a")
            engine.record_call_pattern("gpt-4o", "b")
        # 5th triggers.
        engine.record_call_pattern("gpt-4o", "a")
        with pytest.raises(LoopDetectedError):
            engine.record_call_pattern("gpt-4o", "b")


# ---------------------------------------------------------------------------
# Tool name None (model-only loops)
# ---------------------------------------------------------------------------


class TestModelOnlyLoops:
    def test_model_only_pattern(self, engine: SpendEngine):
        """Loops detected even with tool_name=None."""
        for _ in range(2):
            engine.record_call_pattern("gpt-4o", None)
            engine.record_call_pattern("claude-3-5-sonnet-20241022", None)
        engine.record_call_pattern("gpt-4o", None)
        with pytest.raises(LoopDetectedError):
            engine.record_call_pattern("claude-3-5-sonnet-20241022", None)


# ---------------------------------------------------------------------------
# Integration with post_flight
# ---------------------------------------------------------------------------


class TestLoopInPostFlight:
    def test_post_flight_triggers_loop_detection(self):
        engine = SpendEngine(
            budget=Budget.daily(1000),
            agent_name="test-bot",
            detect_loops=True,
            loop_threshold=3,
        )
        for _ in range(2):
            engine.pre_flight("gpt-4o", MESSAGES)
            engine.post_flight("gpt-4o", 100, 50, tool_name="search")
            engine.pre_flight("gpt-4o", MESSAGES)
            engine.post_flight("gpt-4o", 100, 50, tool_name="respond")
        engine.pre_flight("gpt-4o", MESSAGES)
        engine.post_flight("gpt-4o", 100, 50, tool_name="search")
        engine.pre_flight("gpt-4o", MESSAGES)
        with pytest.raises(LoopDetectedError):
            engine.post_flight("gpt-4o", 100, 50, tool_name="respond")


# ---------------------------------------------------------------------------
# Reset clears loop state
# ---------------------------------------------------------------------------


class TestLoopReset:
    def test_reset_clears_recent_calls(self, engine: SpendEngine):
        for _ in range(2):
            engine.record_call_pattern("gpt-4o", "a")
            engine.record_call_pattern("gpt-4o", "b")

        engine.reset()
        assert len(engine._recent_calls) == 0

        # Should be able to make the same pattern again without triggering.
        for _ in range(2):
            engine.record_call_pattern("gpt-4o", "a")
            engine.record_call_pattern("gpt-4o", "b")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestLoopEdgeCases:
    def test_single_repeated_call_detected(self, engine: SpendEngine):
        """Same call repeated is detected as a loop (pattern of identical pairs)."""
        with pytest.raises(LoopDetectedError):
            for _ in range(20):
                engine.record_call_pattern("gpt-4o", "search")

    def test_deque_maxlen_respected(self, engine: SpendEngine):
        """Deque stays at maxlen=20."""
        for i in range(30):
            engine.record_call_pattern(f"model-{i}", f"tool-{i}")
        assert len(engine._recent_calls) == 20

    def test_length_5_pattern(self, engine: SpendEngine):
        """Longest checked pattern length (5) works."""
        pattern = [
            ("gpt-4o", "a"),
            ("gpt-4o", "b"),
            ("gpt-4o", "c"),
            ("gpt-4o", "d"),
            ("gpt-4o", "e"),
        ]
        for _ in range(2):
            for model, tool in pattern:
                engine.record_call_pattern(model, tool)
        # 3rd repetition of length-5 pattern.
        for i in range(4):
            engine.record_call_pattern(pattern[i][0], pattern[i][1])
        with pytest.raises(LoopDetectedError):
            engine.record_call_pattern("gpt-4o", "e")


# ---------------------------------------------------------------------------
# Client integration
# ---------------------------------------------------------------------------


class TestClientLoopDetection:
    def test_constructor_passes_detect_loops(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            g = AgentKavach(
                provider="openai",
                api_key="ak_test",
                llm_key="sk-test",
                budget=Budget.daily(1000),
                detect_loops=True,
                loop_threshold=5,
            )
        assert g.engine.detect_loops is True
        assert g.engine.loop_threshold == 5

    def test_extract_tool_name_openai(self):
        """Extracts tool name from OpenAI-style response."""
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.tool_calls = [MagicMock()]
        resp.choices[0].message.tool_calls[0].function.name = "search"
        assert AgentKavach._extract_tool_name(resp) == "search"

    def test_extract_tool_name_no_tools(self):
        """Returns None when no tool calls."""
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.tool_calls = None
        resp.choices[0].message.content = "just text"
        # content is not a list, so Anthropic check skips
        assert AgentKavach._extract_tool_name(resp) is None

    def test_extract_tool_name_anthropic(self):
        """Extracts tool name from Anthropic-style response."""
        block = MagicMock()
        block.type = "tool_use"
        block.name = "search_api"
        resp = MagicMock()
        resp.choices = []  # No OpenAI choices
        resp.content = [block]
        assert AgentKavach._extract_tool_name(resp) == "search_api"
