"""Tests for agentkavach.exceptions — guardrail exception hierarchy."""

from __future__ import annotations

import pytest

from agentkavach.exceptions import (
    BudgetExceededError,
    CallLimitError,
    GuardrailError,
    LoopDetectedError,
    RateLimitedError,
    RuntimeLimitError,
    TokenLimitError,
)

# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------


class TestHierarchy:
    def test_guardrail_error_is_exception(self):
        assert issubclass(GuardrailError, Exception)

    def test_token_limit_is_guardrail(self):
        assert issubclass(TokenLimitError, GuardrailError)

    def test_call_limit_is_guardrail(self):
        assert issubclass(CallLimitError, GuardrailError)

    def test_runtime_limit_is_guardrail(self):
        assert issubclass(RuntimeLimitError, GuardrailError)

    def test_loop_detected_is_guardrail(self):
        assert issubclass(LoopDetectedError, GuardrailError)

    def test_budget_exceeded_is_not_guardrail(self):
        """BudgetExceededError stays independent — no breaking change."""
        assert not issubclass(BudgetExceededError, GuardrailError)

    def test_rate_limited_is_not_guardrail(self):
        assert not issubclass(RateLimitedError, GuardrailError)


# ---------------------------------------------------------------------------
# Catching
# ---------------------------------------------------------------------------


class TestCatching:
    def test_catch_guardrail_catches_token_limit(self):
        with pytest.raises(GuardrailError):
            raise TokenLimitError("too many tokens")

    def test_catch_guardrail_catches_call_limit(self):
        with pytest.raises(GuardrailError):
            raise CallLimitError("too many calls")

    def test_catch_guardrail_catches_runtime_limit(self):
        with pytest.raises(GuardrailError):
            raise RuntimeLimitError("too long")

    def test_catch_guardrail_catches_loop_detected(self):
        with pytest.raises(GuardrailError):
            raise LoopDetectedError("loop found")

    def test_catch_exception_catches_guardrail(self):
        with pytest.raises(Exception):
            raise GuardrailError("generic guardrail")

    def test_catch_guardrail_does_not_catch_budget(self):
        with pytest.raises(BudgetExceededError):
            raise BudgetExceededError("over budget")


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


class TestMessages:
    def test_token_limit_message(self):
        err = TokenLimitError("exceeded 10000 tokens")
        assert str(err) == "exceeded 10000 tokens"

    def test_call_limit_message(self):
        err = CallLimitError("exceeded 50 calls")
        assert str(err) == "exceeded 50 calls"

    def test_runtime_limit_message(self):
        err = RuntimeLimitError("exceeded 300s")
        assert str(err) == "exceeded 300s"

    def test_loop_detected_message(self):
        err = LoopDetectedError("pattern repeated 3 times")
        assert str(err) == "pattern repeated 3 times"

    def test_guardrail_no_message(self):
        err = GuardrailError()
        assert str(err) == ""


# ---------------------------------------------------------------------------
# Public API exports
# ---------------------------------------------------------------------------


class TestExports:
    def test_all_exceptions_importable_from_package(self):
        import agentkavach

        assert hasattr(agentkavach, "GuardrailError")
        assert hasattr(agentkavach, "TokenLimitError")
        assert hasattr(agentkavach, "CallLimitError")
        assert hasattr(agentkavach, "RuntimeLimitError")
        assert hasattr(agentkavach, "LoopDetectedError")
        assert hasattr(agentkavach, "BudgetExceededError")
        assert hasattr(agentkavach, "RateLimitedError")
