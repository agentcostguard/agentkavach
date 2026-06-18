"""Budget types: daily(), monthly(), total(), org_budget().

A Budget encapsulates the spend limit, reset period, and optional
shared-budget key.  Budgets are value objects — construct them once and
pass into the engine.

Usage:
    from agentkavach.budget import Budget

    b = Budget.daily(limit=50.0)
    b = Budget.monthly(limit=500.0)
    b = Budget.org_budget(limit=200.0, period="daily")
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Union


class Period(enum.Enum):
    """Reset cadence for a budget."""

    DAILY = "daily"
    MONTHLY = "monthly"
    TOTAL = "total"  # never resets


def _budget_key(period: Period, shared_name: Optional[str] = None) -> str:
    """Build a deterministic spend-counter key for the current period.

    Examples:
        ``"daily:2026-03-13"``
        ``"monthly:2026-03"``
        ``"total"``
        ``"shared:team-daily:daily:2026-03-13"``
    """
    now = datetime.now(timezone.utc)
    if period is Period.DAILY:
        suffix = now.strftime("%Y-%m-%d")
    elif period is Period.MONTHLY:
        suffix = now.strftime("%Y-%m")
    else:
        suffix = ""

    base = f"{period.value}:{suffix}" if suffix else period.value
    if shared_name:
        return f"shared:{shared_name}:{base}"
    return base


@dataclass(frozen=True)
class Budget:
    """Immutable budget specification.

    Prefer the class-method constructors (``daily``, ``monthly``,
    ``org_budget``) over direct instantiation.
    """

    limit: float
    period: Period = Period.DAILY
    shared_name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.limit <= 0:
            raise ValueError(f"Budget limit must be positive, got {self.limit}")

    # -- constructors -------------------------------------------------------

    @classmethod
    def daily(cls, limit: float) -> Budget:
        """Create a budget that resets every midnight UTC."""
        return cls(limit=limit, period=Period.DAILY)

    @classmethod
    def monthly(cls, limit: float) -> Budget:
        """Create a budget that resets on the 1st of each month UTC."""
        return cls(limit=limit, period=Period.MONTHLY)

    @classmethod
    def total(cls, limit: float) -> Budget:
        """Create a budget that never resets (lifetime cap)."""
        return cls(limit=limit, period=Period.TOTAL)

    @classmethod
    def org_budget(cls, limit: float, period: Union[str, Period] = "daily") -> Budget:
        """Create an org-level budget that applies across ALL agents.

        This creates a shared budget with the sentinel name ``__org__``.
        When used with the backend, spend from every agent in the org
        is pooled against this single limit.

        The most restrictive budget wins: if an agent has a $10/day
        budget and the org has a $50/day budget, both are enforced.

        Usage::

            org = Budget.org_budget(limit=50.0, period="daily")
        """
        resolved = Period(period) if isinstance(period, str) else period
        return cls(limit=limit, period=resolved, shared_name="__org__")

    # -- helpers ------------------------------------------------------------

    @property
    def key(self) -> str:
        """Return the current spend-counter key for this budget."""
        return _budget_key(self.period, self.shared_name)

    @property
    def is_shared(self) -> bool:
        return self.shared_name is not None
