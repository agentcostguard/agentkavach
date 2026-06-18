"""Integration tests for AgentKavach client guardrails."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agentkavach import AgentKavach, Budget
from agentkavach.exceptions import (
    CallLimitError,
    GuardrailError,
    RuntimeLimitError,
    TokenLimitError,
)


def _mock_openai_response(input_tokens: int = 100, output_tokens: int = 50):
    """Create a mock OpenAI response."""
    resp = MagicMock()
    resp.usage.prompt_tokens = input_tokens
    resp.usage.completion_tokens = output_tokens
    resp.usage.total_tokens = input_tokens + output_tokens
    resp.model = "gpt-4o"
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = "test response"
    return resp


@pytest.fixture()
def guard():
    """AgentKavach with guardrails and mocked OpenAI."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
        g = AgentKavach(
            provider="openai",
            api_key="ak_test",
            llm_key="sk-test",
            agent_name="test-bot",
            budget=Budget.daily(1000),
            max_tokens_per_run=500,
            max_calls_per_run=3,
            max_runtime_seconds=60.0,
        )
    return g


# ---------------------------------------------------------------------------
# Constructor params
# ---------------------------------------------------------------------------


class TestGuardrailConstructor:
    def test_params_passed_to_engine(self, guard: AgentKavach):
        assert guard.engine.max_tokens_per_run == 500
        assert guard.engine.max_calls_per_run == 3
        assert guard.engine.max_runtime_seconds == 60.0

    def test_defaults_are_none(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            g = AgentKavach(
                provider="openai",
                api_key="ak_test",
                llm_key="sk-test",
                budget=Budget.daily(100),
            )
        assert g.engine.max_tokens_per_run is None
        assert g.engine.max_calls_per_run is None
        assert g.engine.max_runtime_seconds is None


# ---------------------------------------------------------------------------
# Call limit propagation
# ---------------------------------------------------------------------------


class TestClientCallLimit:
    def test_call_limit_propagates(self, guard: AgentKavach):
        """CallLimitError propagates through _safe_call."""
        mock_resp = _mock_openai_response()

        with patch.object(guard, "_call_provider", return_value=mock_resp):
            with patch.object(guard, "_count_input_tokens", return_value=50):
                # 3 successful calls.
                for _ in range(3):
                    guard.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

                # 4th call raises.
                with pytest.raises(CallLimitError):
                    guard.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

    def test_call_limit_is_guardrail_error(self, guard: AgentKavach):
        """CallLimitError is catchable as GuardrailError."""
        mock_resp = _mock_openai_response()

        with patch.object(guard, "_call_provider", return_value=mock_resp):
            with patch.object(guard, "_count_input_tokens", return_value=50):
                for _ in range(3):
                    guard.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

                with pytest.raises(GuardrailError):
                    guard.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# Token limit propagation
# ---------------------------------------------------------------------------


class TestClientTokenLimit:
    def test_token_limit_propagates(self, guard: AgentKavach):
        """Phase 40 unified the kill path; Phase 102 restored the specific
        exception subclass so the public contract holds: when the token cap
        fires, the NEXT call raises ``TokenLimitError`` (not the generic
        ``BudgetExceededError``). The first call succeeds even when its
        tokens push spend over (matches the Phase 37 cost design)."""
        mock_resp = _mock_openai_response(input_tokens=300, output_tokens=300)

        with patch.object(guard, "_call_provider", return_value=mock_resp):
            with patch.object(guard, "_count_input_tokens", return_value=300):
                # First call: 600 tokens > 500 limit, but lands.
                guard.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
                # Second call rejected once spend has crossed — with the
                # dimension-specific exception the docs advertise.
                with pytest.raises(TokenLimitError):
                    guard.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# Runtime limit propagation
# ---------------------------------------------------------------------------


class TestClientRuntimeLimit:
    def test_runtime_limit_propagates(self, guard: AgentKavach):
        """Phase 40 unified the kill path; Phase 102 restored
        ``RuntimeLimitError`` (the audit'd public contract). After the
        elapsed clock crosses the cap, the next pre_flight rejects with
        ``RuntimeLimitError`` carrying the documented ``elapsed`` attribute.
        """
        mock_resp = _mock_openai_response()

        with patch.object(guard, "_call_provider", return_value=mock_resp):
            with patch.object(guard, "_count_input_tokens", return_value=50):
                # First call to set _first_call_time.
                guard.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

                # Simulate time passing.
                original_time = guard.engine._first_call_time
                with patch("agentkavach.engine.time") as mock_time:
                    mock_time.monotonic.return_value = original_time + 120.0
                    with pytest.raises(RuntimeLimitError) as exc_info:
                        guard.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
                    assert exc_info.value.limit == 60.0
