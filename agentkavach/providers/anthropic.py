"""Anthropic response/usage parsing.

Handles ``anthropic.types.Message`` responses from the Anthropic Python
SDK.  The usage structure differs from OpenAI — Anthropic uses
``input_tokens`` / ``output_tokens`` directly on the ``usage`` object.
"""

from __future__ import annotations

import logging
from typing import Any

from agentkavach.pricing import estimate_cost
from agentkavach.providers import UsageRecord

logger = logging.getLogger(__name__)

PROVIDER = "anthropic"


def parse_usage(response: Any) -> UsageRecord:
    """Extract token counts from an Anthropic Message response.

    Expects *response* to have ``.model`` and ``.usage`` attributes
    (standard ``anthropic.types.Message``).

    Raises ``ValueError`` if the response lacks usage data.
    """
    model = getattr(response, "model", None)
    if model is None:
        raise ValueError("Response is missing 'model' attribute")

    usage = getattr(response, "usage", None)
    if usage is None:
        raise ValueError("Response is missing 'usage' attribute")

    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0

    return UsageRecord(
        provider=PROVIDER,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def count_tokens(model: str, messages: list[dict], client: Any = None) -> int:
    """Count input tokens using the Anthropic count_tokens API.

    Requires an active Anthropic client. This makes a network call
    (~100-200ms) but returns the exact input token count.

    Falls back to a 4-chars-per-token heuristic if the API call fails
    or no client is provided.
    """
    if client is None:
        return _heuristic_count(messages)

    try:
        result = client.messages.count_tokens(
            model=model,
            messages=messages,
        )
        return result.input_tokens
    except Exception:
        logger.warning("Anthropic count_tokens API failed, using heuristic", exc_info=True)
        return _heuristic_count(messages)


def _heuristic_count(messages: list[dict]) -> int:
    """Simple 4-chars-per-token fallback."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content) // 4 + 1
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += len(part.get("text", "")) // 4 + 1
        total += 4
    return total


def calculate_cost(usage: UsageRecord) -> float:
    """Return estimated USD cost for the given *usage*.

    Returns ``0.0`` if the model is not in the price table (fail-open).
    """
    cost = estimate_cost(usage.model, usage.input_tokens, usage.output_tokens)
    if cost is None:
        logger.warning("Unknown model %r — cost estimated as $0.00", usage.model)
        return 0.0
    return cost
