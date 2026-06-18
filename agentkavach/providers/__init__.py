"""Provider-specific response/usage parsing.

Each provider module exposes two functions:

    parse_usage(response) -> UsageRecord
    calculate_cost(usage: UsageRecord) -> float

The UsageRecord dataclass is the common currency across all providers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class UsageRecord:
    """Provider-agnostic usage extracted from an LLM response."""

    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: Optional[float] = None  # filled in by calculate_cost
