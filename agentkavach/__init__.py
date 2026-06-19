"""AgentKavach: Hard budget limits for LLM APIs."""

__version__ = "1.0.2"

from agentkavach.alerts import ChannelConfig, ChannelType
from agentkavach.budget import Budget
from agentkavach.client import AgentKavach
from agentkavach.exceptions import (
    BudgetExceededError,
    CallLimitError,
    GuardrailError,
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
    "LoopDetectedError",
    "RateLimitedError",
    "RuntimeLimitError",
    "TokenLimitError",
    "register_price",
]
