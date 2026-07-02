"""AgentKavach: Hard budget limits for LLM APIs."""

__version__ = "2.1.0"

from agentkavach.alerts import ChannelConfig, ChannelType
from agentkavach.budget import Budget
from agentkavach.client import AgentKavach
from agentkavach.exceptions import (
    BudgetExceededError,
    CallLimitError,
    GuardrailError,
    IngestRejectedError,
    LoopDetectedError,
    RateLimitedError,
    RuntimeLimitError,
    TokenLimitError,
)
from agentkavach.pricing import register_price

# Backward-compat alias for pre-rebrand users
CostGuard = AgentKavach

__all__ = [
    "Budget",
    "ChannelConfig",
    "ChannelType",
    "AgentKavach",
    "CostGuard",
    "BudgetExceededError",
    "CallLimitError",
    "GuardrailError",
    "IngestRejectedError",
    "LoopDetectedError",
    "RateLimitedError",
    "RuntimeLimitError",
    "TokenLimitError",
    "register_price",
]
