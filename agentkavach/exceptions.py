"""AgentKavach exceptions.

All public exception classes accept structured attributes so customer code
can introspect *why* a budget or guardrail fired:

    except TokenLimitError as e:
        log.warning("token cap hit at %d/%d", e.spent, e.limit)

The attributes are documented in
``dashboard/app/public/docs/guardrails/page.tsx`` and
``dashboard/app/public/docs/budgets/page.tsx``.  Adding or renaming an
attribute is a public API change — update both docs and the tests in
``tests/test_engine_guardrails.py`` whenever this file changes.
"""

from __future__ import annotations

from typing import Optional, Sequence


class BudgetExceededError(Exception):
    """Raised when a request would exceed the configured budget.

    Attributes:
        spent: USD already accumulated against the budget when the cap fired.
        limit: USD limit configured on the budget.
        period: Budget period string ("daily", "monthly", "total"); ``None``
            when the exception is raised outside a normal budget (e.g.
            backend-rejected ingest, kill-switch reuse).
    """

    def __init__(
        self,
        message: str = "",
        *,
        spent: Optional[float] = None,
        limit: Optional[float] = None,
        period: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.spent = spent
        self.limit = limit
        self.period = period


class RateLimitedError(Exception):
    """Raised when the AgentKavach API rate limit is hit."""


class GuardrailError(Exception):
    """Base class for all guardrail violations.

    Guardrail errors propagate to the caller (same as BudgetExceededError).
    All other internal errors are caught and logged (fail-open design).
    """


class TokenLimitError(GuardrailError):
    """Raised when ``max_tokens_per_run`` is exceeded.

    Attributes:
        spent: Total tokens (input + output) consumed when the cap fired.
        limit: Token cap configured via ``max_tokens_per_run``.
    """

    def __init__(
        self,
        message: str = "",
        *,
        spent: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.spent = spent
        self.limit = limit


class CallLimitError(GuardrailError):
    """Raised when ``max_calls_per_run`` is exceeded.

    Attributes:
        call_count: Number of LLM calls made when the cap fired.
        limit: Call cap configured via ``max_calls_per_run``.
    """

    def __init__(
        self,
        message: str = "",
        *,
        call_count: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.call_count = call_count
        self.limit = limit


class RuntimeLimitError(GuardrailError):
    """Raised when ``max_runtime_seconds`` is exceeded.

    Attributes:
        elapsed: Wall-clock seconds since the first LLM call.
        limit: Runtime cap (seconds) configured via ``max_runtime_seconds``.
    """

    def __init__(
        self,
        message: str = "",
        *,
        elapsed: Optional[float] = None,
        limit: Optional[float] = None,
    ) -> None:
        super().__init__(message)
        self.elapsed = elapsed
        self.limit = limit


class LoopDetectedError(GuardrailError):
    """Raised when a runaway loop pattern is detected.

    Attributes:
        pattern: The repeating ``(model, tool_name)`` sequence that fired
            the detector. ``None`` if the raise site couldn't supply it.
        count: How many consecutive repetitions of ``pattern`` were seen.
    """

    def __init__(
        self,
        message: str = "",
        *,
        pattern: Optional[Sequence[object]] = None,
        count: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.pattern = pattern
        self.count = count
