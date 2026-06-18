"""In-memory spend tracking, pre/post flight checks.

The engine is the hot-path performance core of AgentKavach.  All budget
checks are sub-millisecond dictionary lookups — no I/O, no network.

Usage (internal — called by ``AgentKavach`` client, not by end users):

    engine = SpendEngine(budget=Budget.daily(50), agent_name="my-agent")
    engine.pre_flight("gpt-4o", messages)   # raises BudgetExceededError
    engine.post_flight(usage_record)         # updates counters
    engine.check_thresholds()                # returns triggered alerts
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

from agentkavach.budget import Budget
from agentkavach.exceptions import (
    BudgetExceededError,
    CallLimitError,
    LoopDetectedError,
    RuntimeLimitError,
    TokenLimitError,
)
from agentkavach.pricing import estimate_cost, get_price

logger = logging.getLogger(__name__)

# Default alert thresholds when none are configured.
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.70, 0.90, 1.0)


@dataclass
class ThresholdEvent:
    """Emitted when usage on a budget dimension crosses a threshold.

    ``budget_type`` identifies which dimension fired:
      - ``"cost"`` (default; spent/limit in USD)
      - ``"tokens_total"`` (spent/limit are integer token counts)
      - ``"duration"`` (spent/limit in seconds, wall-clock since first call)

    Channel handlers + the backend's alert evaluator use this to route to
    the right AlertConfig and format the right message units.
    """

    threshold: float  # e.g. 0.70
    spent: float
    limit: float
    budget_key: str
    agent_name: str
    budget_type: str = "cost"


@dataclass
class SpendEngine:
    """In-memory spend tracker with pre/post-flight budget enforcement.

    Thread-safe: all mutations go through ``_lock``.  Read-only
    accessors (``spent``, ``remaining``) are atomic on CPython due to
    the GIL, but we still lock for correctness on other runtimes.
    """

    budget: Budget
    agent_name: str = "default"
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS
    on_kill: Optional[Callable[[], None]] = None

    # Optional org-level budget — enforced alongside the primary budget.
    # The most restrictive wins: both must have remaining capacity.
    org_budget: Optional[Budget] = None

    # Guardrail limits (per-run).
    max_tokens_per_run: Optional[int] = None
    max_calls_per_run: Optional[int] = None
    max_runtime_seconds: Optional[float] = None

    # Loop detection.
    detect_loops: bool = False
    loop_threshold: int = 3  # consecutive repetitions before kill

    # Internal state — not part of the public API.
    _spend: Dict[str, float] = field(default_factory=dict, repr=False)
    _fired: Dict[str, set[float]] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _killed: bool = field(default=False, repr=False)
    # First dimension to cross 100% — preserved so post-kill pre_flight
    # rejects with the specific exception type the customer documented
    # (TokenLimitError / RuntimeLimitError / BudgetExceededError) rather
    # than always BudgetExceededError. One of:
    # ``"cost"``, ``"tokens"``, ``"duration"``, ``"loop"``, ``"calls"``.
    _kill_reason: Optional[str] = field(default=None, repr=False)
    # Guards the on_kill teardown callback so it fires AT MOST ONCE per engine
    # lifetime. Without this, every subsequent ``check_thresholds`` call after a
    # kill re-invokes on_kill (the guard is only ``if self._killed``), so a
    # customer's teardown — paging, process shutdown, a one-shot notification —
    # would fire repeatedly.
    _on_kill_fired: bool = field(default=False, repr=False)
    _total_tokens: int = field(default=0, repr=False)
    _call_count: int = field(default=0, repr=False)
    _first_call_time: Optional[float] = field(default=None, repr=False)
    _recent_calls: Deque[Tuple[str, Optional[str]]] = field(
        default_factory=lambda: collections.deque(maxlen=20), repr=False
    )

    # -- public interface ---------------------------------------------------

    @property
    def spent(self) -> float:
        """Current spend for the active budget period."""
        with self._lock:
            return self._spend.get(self.budget.key, 0.0)

    @property
    def remaining(self) -> float:
        """Dollars remaining before the budget is exhausted."""
        return max(0.0, self.budget.limit - self.spent)

    @property
    def utilization(self) -> float:
        """Fraction of budget consumed (0.0–1.0+)."""
        if self.budget.limit == 0:
            return 0.0
        return self.spent / self.budget.limit

    def pre_flight(
        self,
        model: str,
        messages: Sequence[dict],
        input_tokens: Optional[int] = None,
    ) -> float:
        """Reject the call only if the budget is *already* exhausted.

        Previous behavior estimated the next call's cost and rejected when
        ``current + estimated > limit``. That stranded budget at the edge —
        e.g. an agent at 50% with a query estimated to push to 60% was
        blocked, even though it had plenty of headroom. Worse, when the
        block did fire the recorded spend never actually crossed the limit,
        so dashboards reported the agent as "Active" indefinitely.

        New behavior: pre-flight is a *post-fact* gate. As long as recorded
        spend hasn't crossed the limit, the call is allowed to proceed. The
        post-flight record may push spend over — that's by design; the
        threshold-check fires the kill on the way back and the *next*
        pre-flight rejects. The user keeps the work they already paid for,
        and the agent terminates with a clean "exceeded" state visible in
        the dashboard.

        Returns the estimated cost in USD (informational only — no longer
        used as a gating signal).
        """
        if self._killed:
            # Replay the original kill reason so customers' specific
            # ``except TokenLimitError`` / ``except RuntimeLimitError``
            # handlers fire on subsequent calls too — not just the first
            # one to cross. Defaults to BudgetExceededError when the
            # reason wasn't recorded (legacy kill paths, on_kill from
            # callbacks).
            self._raise_for_kill_reason()

        # Stamp the first-call clock so duration thresholds can fire
        # against wall-clock from the agent's first activity.
        with self._lock:
            if self._first_call_time is None:
                self._first_call_time = time.monotonic()

        # Call-count cap is the only guardrail still enforced via immediate
        # raise — it has no natural threshold semantics (every call is +1,
        # and exceeding by one is meaningless).
        with self._lock:
            if self.max_calls_per_run is not None and self._call_count >= self.max_calls_per_run:
                self._killed = True
                self._kill_reason = "calls"
                raise CallLimitError(
                    f"Agent {self.agent_name!r} exceeded call limit "
                    f"({self._call_count} >= {self.max_calls_per_run})",
                    call_count=self._call_count,
                    limit=self.max_calls_per_run,
                )

        # Reject if ANY dimension has already crossed its limit. Cost,
        # tokens, and duration each raise the dimension-specific exception
        # the public docs advertise (BudgetExceededError / TokenLimitError /
        # RuntimeLimitError). check_thresholds is what set _killed (handled
        # at the top of this method).
        with self._lock:
            current = self._spend.get(self.budget.key, 0.0)
            if current >= self.budget.limit:
                self._kill_reason = "cost"
                raise BudgetExceededError(
                    f"Exceeded budget (${current:.4f} of ${self.budget.limit:.4f})",
                    spent=current,
                    limit=self.budget.limit,
                    period=self.budget.period.value,
                )
            if (
                self.max_tokens_per_run is not None
                and self._total_tokens >= self.max_tokens_per_run
            ):
                self._kill_reason = "tokens"
                raise TokenLimitError(
                    f"Exceeded tokens budget "
                    f"({self._total_tokens} of {self.max_tokens_per_run} tokens)",
                    spent=self._total_tokens,
                    limit=self.max_tokens_per_run,
                )
            if self.max_runtime_seconds is not None and self._first_call_time is not None:
                elapsed = time.monotonic() - self._first_call_time
                if elapsed >= self.max_runtime_seconds:
                    self._kill_reason = "duration"
                    raise RuntimeLimitError(
                        f"Exceeded duration budget "
                        f"({elapsed:.1f}s of {self.max_runtime_seconds:.1f}s)",
                        elapsed=elapsed,
                        limit=self.max_runtime_seconds,
                    )
            if self.org_budget is not None:
                org_current = self._spend.get(self.org_budget.key, 0.0)
                if org_current >= self.org_budget.limit:
                    self._kill_reason = "cost"
                    raise BudgetExceededError(
                        f"Exceeded org budget (${org_current:.4f} of ${self.org_budget.limit:.4f})",
                        spent=org_current,
                        limit=self.org_budget.limit,
                        period=self.org_budget.period.value,
                    )

        # Estimated cost is still returned for callers that record/forecast,
        # but it no longer gates the call.
        price = get_price(model)
        if price is None:
            logger.warning(
                "Unknown model %r — budget tracking will record $0.00 for this call. "
                "Add pricing via PRICE_TABLE or use a known model.",
                model,
            )
            return 0.0

        token_count = input_tokens if input_tokens is not None else _fast_token_count(messages)
        input_cost = token_count / 1000 * price.input_per_1k
        estimated_output_cost = token_count / 1000 * price.output_per_1k
        return input_cost + estimated_output_cost

    def post_flight(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        tool_name: Optional[str] = None,
    ) -> float:
        """Record actual spend after a successful LLM call.

        Returns the actual cost in USD.
        """
        cost = estimate_cost(model, input_tokens, output_tokens)
        if cost is None:
            logger.warning(
                "Unknown model %r — budget tracking will record $0.00 for this call. "
                "Add pricing via PRICE_TABLE or use a known model.",
                model,
            )
            return 0.0

        with self._lock:
            key = self.budget.key
            self._spend[key] = self._spend.get(key, 0.0) + cost

            # Record against org budget too
            if self.org_budget is not None:
                org_key = self.org_budget.key
                self._spend[org_key] = self._spend.get(org_key, 0.0) + cost

            # Track calls and tokens. The token cap is no longer enforced
            # by raising here — check_thresholds owns the kill path for all
            # three dimensions (cost/tokens/duration) uniformly. The next
            # pre_flight rejects with BudgetExceededError once spend crossed.
            self._call_count += 1
            self._total_tokens += input_tokens + output_tokens

        # Loop detection (outside lock — record_call_pattern acquires its own).
        self.record_call_pattern(model, tool_name)

        return cost

    def check_thresholds(self) -> List[ThresholdEvent]:
        """Evaluate every configured budget dimension and emit threshold events.

        Three dimensions evaluated independently:
          - ``cost`` (always, against ``self.budget.limit``)
          - ``tokens_total`` (when ``max_tokens_per_run`` is set)
          - ``duration`` (when ``max_runtime_seconds`` is set; wall-clock
            since the first call's pre-flight)

        Each (dimension, threshold) fires at most once per budget period
        (cost uses the period key; tokens/duration use per-run keys).
        Any dimension hitting 1.0 sets ``_killed`` so the next pre-flight
        rejects with ``BudgetExceededError`` — unified kill path across
        all three.
        """
        events: List[ThresholdEvent] = []
        any_at_100 = False
        # First-100% reason wins — used by post-kill pre_flight to surface
        # the right exception subclass to the caller.
        kill_reason: Optional[str] = None
        with self._lock:
            # ---- Cost ----
            if self.budget is not None and self.budget.limit > 0:
                key = self.budget.key
                current = self._spend.get(key, 0.0)
                fired = self._fired.setdefault(key, set())
                for threshold in sorted(self.thresholds):
                    if threshold in fired:
                        continue
                    if current / self.budget.limit >= threshold:
                        fired.add(threshold)
                        events.append(
                            ThresholdEvent(
                                threshold=threshold,
                                spent=current,
                                limit=self.budget.limit,
                                budget_key=key,
                                agent_name=self.agent_name,
                                budget_type="cost",
                            )
                        )
                        if threshold == 1.0:
                            any_at_100 = True
                            if kill_reason is None:
                                kill_reason = "cost"

            # ---- Tokens ----
            if self.max_tokens_per_run is not None and self.max_tokens_per_run > 0:
                key = "tokens_total:per_run"
                fired = self._fired.setdefault(key, set())
                current = float(self._total_tokens)
                limit = float(self.max_tokens_per_run)
                for threshold in sorted(self.thresholds):
                    if threshold in fired:
                        continue
                    if current / limit >= threshold:
                        fired.add(threshold)
                        events.append(
                            ThresholdEvent(
                                threshold=threshold,
                                spent=current,
                                limit=limit,
                                budget_key=key,
                                agent_name=self.agent_name,
                                budget_type="tokens_total",
                            )
                        )
                        if threshold == 1.0:
                            any_at_100 = True
                            if kill_reason is None:
                                kill_reason = "tokens"

            # ---- Duration (wall-clock since first call) ----
            if (
                self.max_runtime_seconds is not None
                and self.max_runtime_seconds > 0
                and self._first_call_time is not None
            ):
                key = "duration:per_run"
                fired = self._fired.setdefault(key, set())
                elapsed = time.monotonic() - self._first_call_time
                limit = float(self.max_runtime_seconds)
                for threshold in sorted(self.thresholds):
                    if threshold in fired:
                        continue
                    if elapsed / limit >= threshold:
                        fired.add(threshold)
                        events.append(
                            ThresholdEvent(
                                threshold=threshold,
                                spent=elapsed,
                                limit=limit,
                                budget_key=key,
                                agent_name=self.agent_name,
                                budget_type="duration",
                            )
                        )
                        if threshold == 1.0:
                            any_at_100 = True
                            if kill_reason is None:
                                kill_reason = "duration"

            if any_at_100 and not self._killed:
                self._killed = True
                if self._kill_reason is None and kill_reason is not None:
                    self._kill_reason = kill_reason

        # Execute kill callback outside the lock to avoid deadlocks.
        self._maybe_fire_on_kill()

        return events

    def _maybe_fire_on_kill(self) -> None:
        """Run the on_kill teardown exactly once per kill, for ANY kill reason.

        Must be called outside ``self._lock`` (the customer callback may do
        arbitrary work). The ``_on_kill_fired`` latch makes it idempotent, so
        every kill path — cost / tokens / duration / calls (via check_thresholds)
        AND loop detection (which raises before check_thresholds runs) — can call
        this and the teardown still fires exactly once.
        """
        if self._killed and self.on_kill is not None and not self._on_kill_fired:
            self._on_kill_fired = True
            try:
                self.on_kill()
            except Exception:
                logger.exception("on_kill callback raised an exception")

    def record_partial(
        self,
        model: str,
        output_tokens: int,
    ) -> float:
        """Record spend for a partially-consumed stream.

        Uses only output token pricing since we don't know the full
        prompt cost for a disconnected stream.
        """
        price = get_price(model)
        if price is None:
            return 0.0

        cost = output_tokens / 1000 * price.output_per_1k

        with self._lock:
            key = self.budget.key
            self._spend[key] = self._spend.get(key, 0.0) + cost

        return cost

    def record_call_pattern(
        self,
        model: str,
        tool_name: Optional[str] = None,
    ) -> None:
        """Record a call pattern for loop detection.

        Appends ``(model, tool_name)`` to the recent-calls deque and
        checks for repeating patterns of length 2–5.  If any pattern
        repeats >= ``loop_threshold`` consecutive times, raises
        ``LoopDetectedError``.

        Only active when ``detect_loops`` is True.
        """
        if not self.detect_loops:
            return

        with self._lock:
            self._recent_calls.append((model, tool_name))
            calls = list(self._recent_calls)

        # Check for repeating patterns of length 2–5.
        for pattern_len in range(2, 6):
            if len(calls) < pattern_len * self.loop_threshold:
                continue
            tail = calls[-(pattern_len * self.loop_threshold) :]
            pattern = tail[:pattern_len]
            is_loop = all(
                tail[i * pattern_len : (i + 1) * pattern_len] == pattern
                for i in range(1, self.loop_threshold)
            )
            if is_loop:
                self._killed = True
                self._kill_reason = "loop"
                # A loop kill terminates the agent just like a budget/token/
                # duration kill, so the teardown must fire here too — this path
                # raises before check_thresholds would otherwise run it.
                self._maybe_fire_on_kill()
                raise LoopDetectedError(
                    f"Agent {self.agent_name!r} detected repeating pattern "
                    f"{pattern} repeated {self.loop_threshold} times",
                    pattern=pattern,
                    count=self.loop_threshold,
                )

    def reset(self) -> None:
        """Clear all spend counters and guardrail state (useful for testing)."""
        with self._lock:
            self._spend.clear()
            self._fired.clear()
            self._killed = False
            self._kill_reason = None
            self._on_kill_fired = False
            self._total_tokens = 0
            self._call_count = 0
            self._first_call_time = None
            self._recent_calls.clear()

    # -- internal helpers ---------------------------------------------------

    def _raise_for_kill_reason(self) -> None:
        """Raise the dimension-specific exception matching ``_kill_reason``.

        Called from the top of pre_flight when ``_killed`` is set. Replays
        the exception subclass the customer's first try/except already saw
        — so ``except TokenLimitError:`` keeps catching subsequent calls,
        not just the moment the cap fired. Falls back to
        ``BudgetExceededError`` when the reason wasn't recorded.
        """
        reason = self._kill_reason
        agent = self.agent_name
        if reason == "tokens":
            raise TokenLimitError(
                f"Agent {agent!r} has been killed — token limit exhausted",
                spent=self._total_tokens,
                limit=self.max_tokens_per_run,
            )
        if reason == "duration":
            elapsed = (
                time.monotonic() - self._first_call_time
                if self._first_call_time is not None
                else None
            )
            raise RuntimeLimitError(
                f"Agent {agent!r} has been killed — runtime limit exhausted",
                elapsed=elapsed,
                limit=self.max_runtime_seconds,
            )
        if reason == "calls":
            raise CallLimitError(
                f"Agent {agent!r} has been killed — call limit exhausted",
                call_count=self._call_count,
                limit=self.max_calls_per_run,
            )
        if reason == "loop":
            raise LoopDetectedError(
                f"Agent {agent!r} has been killed — loop pattern detected",
            )
        # "cost" or unknown — fall through to the documented default.
        current = self._spend.get(self.budget.key, 0.0) if self.budget else None
        raise BudgetExceededError(
            f"Agent {agent!r} has been killed — budget exhausted",
            spent=current,
            limit=self.budget.limit if self.budget else None,
            period=self.budget.period.value if self.budget else None,
        )


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

# Lazy-loaded tiktoken encoder (cached after first call).
_encoder = None
_encoder_lock = threading.Lock()


def _get_encoder():  # type: ignore[no-untyped-def]
    """Return a cached tiktoken encoder, falling back to a simple heuristic."""
    global _encoder
    if _encoder is not None:
        return _encoder
    with _encoder_lock:
        if _encoder is not None:
            return _encoder
        try:
            import tiktoken

            _encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            logger.info("tiktoken unavailable — using 4-chars-per-token heuristic")
            _encoder = _FallbackEncoder()
    return _encoder


class _FallbackEncoder:
    """Simple chars/4 estimator when tiktoken is not installed."""

    def encode(self, text: str) -> list[int]:
        return [0] * (len(text) // 4 + 1)


def _fast_token_count(messages: Sequence[dict]) -> int:
    """Estimate total output tokens for a message sequence.

    This is a *rough* upper-bound used for pre-flight checks only.
    Post-flight uses the actual token count from the provider response.
    """
    encoder = _get_encoder()
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(encoder.encode(content))
        elif isinstance(content, list):
            # Multi-part content (text + images).
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += len(encoder.encode(part.get("text", "")))
    # Add per-message overhead (~4 tokens per message for role/delimiters).
    total += len(messages) * 4
    return total
