"""Model price tables for LLM providers.

Prices are per 1K tokens. Updated manually — run scripts/update_prices.py
to refresh from provider documentation.

Usage:
    from agentkavach.pricing import get_price, estimate_cost

    price = get_price("gpt-4o")
    cost = estimate_cost("gpt-4o", input_tokens=500, output_tokens=200)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ModelPrice:
    """Immutable price entry for a single model."""

    input_per_1k: float
    output_per_1k: float


# ---------------------------------------------------------------------------
# Canonical price table
#
# Sources (as of 2026-03):
#   OpenAI:    https://openai.com/pricing
#   Anthropic: https://docs.anthropic.com/en/docs/about-claude/models
#   Google:    https://ai.google.dev/pricing
#
# Prices are USD per 1 000 tokens.
# ---------------------------------------------------------------------------
PRICE_TABLE: Dict[str, ModelPrice] = {
    # OpenAI
    "gpt-4o": ModelPrice(input_per_1k=0.0025, output_per_1k=0.010),
    "gpt-4o-2024-11-20": ModelPrice(input_per_1k=0.0025, output_per_1k=0.010),
    "gpt-4o-mini": ModelPrice(input_per_1k=0.00015, output_per_1k=0.0006),
    "gpt-4o-mini-2024-07-18": ModelPrice(input_per_1k=0.00015, output_per_1k=0.0006),
    "gpt-4-turbo": ModelPrice(input_per_1k=0.01, output_per_1k=0.03),
    "gpt-4": ModelPrice(input_per_1k=0.03, output_per_1k=0.06),
    "gpt-3.5-turbo": ModelPrice(input_per_1k=0.0005, output_per_1k=0.0015),
    "o1": ModelPrice(input_per_1k=0.015, output_per_1k=0.06),
    "o1-mini": ModelPrice(input_per_1k=0.003, output_per_1k=0.012),
    "o3-mini": ModelPrice(input_per_1k=0.0011, output_per_1k=0.0044),
    "o3": ModelPrice(input_per_1k=0.01, output_per_1k=0.04),
    "o4-mini": ModelPrice(input_per_1k=0.0011, output_per_1k=0.0044),
    "gpt-4.1": ModelPrice(input_per_1k=0.002, output_per_1k=0.008),
    "gpt-4.1-mini": ModelPrice(input_per_1k=0.0004, output_per_1k=0.0016),
    "gpt-4.1-nano": ModelPrice(input_per_1k=0.0001, output_per_1k=0.0004),
    "gpt-4.5-preview": ModelPrice(input_per_1k=0.075, output_per_1k=0.15),
    "codex-mini": ModelPrice(input_per_1k=0.0015, output_per_1k=0.006),
    # Anthropic
    "claude-opus-4-0": ModelPrice(input_per_1k=0.015, output_per_1k=0.075),
    "claude-sonnet-4-0": ModelPrice(input_per_1k=0.003, output_per_1k=0.015),
    "claude-3-5-sonnet-20241022": ModelPrice(input_per_1k=0.003, output_per_1k=0.015),
    "claude-3-5-haiku-20241022": ModelPrice(input_per_1k=0.0008, output_per_1k=0.004),
    "claude-3-opus-20240229": ModelPrice(input_per_1k=0.015, output_per_1k=0.075),
    "claude-3-haiku-20240307": ModelPrice(input_per_1k=0.00025, output_per_1k=0.00125),
    "claude-opus-4-6": ModelPrice(input_per_1k=0.015, output_per_1k=0.075),
    "claude-sonnet-4-6": ModelPrice(input_per_1k=0.003, output_per_1k=0.015),
    "claude-haiku-4-5": ModelPrice(input_per_1k=0.0008, output_per_1k=0.004),
    # Google
    "gemini-2.0-flash": ModelPrice(input_per_1k=0.0001, output_per_1k=0.0004),
    "gemini-1.5-pro": ModelPrice(input_per_1k=0.00125, output_per_1k=0.005),
    "gemini-1.5-flash": ModelPrice(input_per_1k=0.000075, output_per_1k=0.0003),
    "gemini-2.5-pro": ModelPrice(input_per_1k=0.00125, output_per_1k=0.01),
    "gemini-2.5-flash": ModelPrice(input_per_1k=0.00015, output_per_1k=0.0035),
    # Mistral
    "mistral-large-latest": ModelPrice(input_per_1k=0.002, output_per_1k=0.006),
    "mistral-large-2411": ModelPrice(input_per_1k=0.002, output_per_1k=0.006),
    "mistral-small-latest": ModelPrice(input_per_1k=0.0001, output_per_1k=0.0003),
    "mistral-small-2503": ModelPrice(input_per_1k=0.0001, output_per_1k=0.0003),
    "codestral-latest": ModelPrice(input_per_1k=0.0003, output_per_1k=0.0009),
    "pixtral-large-latest": ModelPrice(input_per_1k=0.002, output_per_1k=0.006),
    "ministral-8b-latest": ModelPrice(input_per_1k=0.0001, output_per_1k=0.0001),
    "mistral-embed": ModelPrice(input_per_1k=0.0001, output_per_1k=0.0001),
}

# Alias mappings — short names that resolve to the canonical entry above.
_ALIASES: Dict[str, str] = {
    # OpenAI aliases
    "o3-reasoning": "o3",
    "gpt-4.5": "gpt-4.5-preview",
    "codex": "codex-mini",
    # Anthropic aliases
    "claude-opus": "claude-opus-4-6",
    "claude-sonnet": "claude-sonnet-4-6",
    "claude-haiku": "claude-haiku-4-5",
    "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3.5-haiku": "claude-3-5-haiku-20241022",
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-3-haiku": "claude-3-haiku-20240307",
    # Google aliases
    "gemini-pro": "gemini-2.5-pro",
    "gemini-flash": "gemini-2.5-flash",
    # Mistral aliases
    "mistral-large": "mistral-large-2411",
    "mistral-small": "mistral-small-2503",
    "codestral": "codestral-latest",
    "pixtral-large": "pixtral-large-latest",
    "ministral-8b": "ministral-8b-latest",
}


def resolve_model(model: str) -> str:
    """Return the canonical model name, resolving aliases."""
    return _ALIASES.get(model, model)


def _prefix_match(model: str) -> Optional[str]:
    """Try to match a model by progressively stripping trailing segments.

    Handles date-suffixed models like ``claude-sonnet-4-20250514`` by
    stripping ``-20250514`` and matching ``claude-sonnet-4``, which
    may itself be an alias resolving to a canonical entry.

    Also handles provider snapshot suffixes like ``gpt-4o-2024-11-20``
    by stripping date parts (segments that are all digits).
    """
    parts = model.split("-")
    # Try removing trailing segments one at a time
    for i in range(len(parts) - 1, 0, -1):
        candidate = "-".join(parts[:i])
        # Check aliases first, then price table
        resolved = _ALIASES.get(candidate, candidate)
        if resolved in PRICE_TABLE:
            return resolved
    return None


# User-registered custom prices (set via register_price)
_CUSTOM_PRICES: Dict[str, ModelPrice] = {}


def register_price(model: str, input_per_1k: float, output_per_1k: float) -> None:
    """Register custom pricing for a model.

    Use this for new/private models not yet in the built-in table::

        from agentkavach.pricing import register_price
        register_price("my-fine-tuned-model", input_per_1k=0.005, output_per_1k=0.015)
    """
    _CUSTOM_PRICES[model] = ModelPrice(input_per_1k=input_per_1k, output_per_1k=output_per_1k)


def get_price(model: str) -> Optional[ModelPrice]:
    """Look up pricing for *model*.

    Resolution order:
    1. Custom prices (registered via ``register_price``)
    2. Exact match in price table (after alias resolution)
    3. Prefix match — strip date suffixes and retry
       e.g. ``claude-sonnet-4-20250514`` → ``claude-sonnet-4-0``

    Returns ``None`` only if no match is found at any level.
    """
    # 1. Custom prices
    if model in _CUSTOM_PRICES:
        return _CUSTOM_PRICES[model]

    # 2. Exact match (with alias)
    canonical = resolve_model(model)
    price = PRICE_TABLE.get(canonical)
    if price is not None:
        return price

    # 3. Prefix match — strip date suffixes
    prefix_canonical = _prefix_match(model)
    if prefix_canonical is not None:
        return PRICE_TABLE[prefix_canonical]

    return None


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> Optional[float]:
    """Estimate cost in USD for a given token count.

    Returns ``None`` if the model is not in the price table.
    """
    price = get_price(model)
    if price is None:
        return None
    return input_tokens / 1000 * price.input_per_1k + output_tokens / 1000 * price.output_per_1k


def supported_models() -> list[str]:
    """Return all known model identifiers (canonical + aliases)."""
    return sorted(set(PRICE_TABLE.keys()) | set(_ALIASES.keys()))
