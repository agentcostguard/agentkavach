"""OpenAI response/usage parsing.

Handles both ChatCompletion and streaming chunk responses from the
``openai`` Python SDK.  Falls back gracefully when usage data is absent
(e.g. streamed responses without ``stream_options.include_usage``).
"""

from __future__ import annotations

import logging
from typing import Any

from agentkavach.pricing import estimate_cost
from agentkavach.providers import UsageRecord

logger = logging.getLogger(__name__)

PROVIDER = "openai"


def parse_usage(response: Any) -> UsageRecord:
    """Extract token counts from an OpenAI ChatCompletion response.

    Expects *response* to have ``.model`` and ``.usage`` attributes
    (standard ``openai.types.chat.ChatCompletion``).

    Raises ``ValueError`` if the response lacks usage data.
    """
    model = getattr(response, "model", None)
    if model is None:
        raise ValueError("Response is missing 'model' attribute")

    usage = getattr(response, "usage", None)
    if usage is None:
        raise ValueError(
            "Response is missing 'usage' — enable stream_options.include_usage for streaming calls"
        )

    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0

    return UsageRecord(
        provider=PROVIDER,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def count_tokens(model: str, messages: list[dict], client: Any = None) -> int:
    """Count input tokens locally using tiktoken.

    This is a local operation with no network call (~0.1ms).
    Falls back to a 4-chars-per-token heuristic if tiktoken is unavailable.
    """
    try:
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        total = 0
        for msg in messages:
            total += 4  # role/delimiter overhead per message
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(encoding.encode(content))
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        total += len(encoding.encode(part.get("text", "")))
            role = msg.get("role", "")
            total += len(encoding.encode(role))
        total += 2  # assistant reply priming
        return total
    except ImportError:
        logger.info("tiktoken unavailable — using 4-chars-per-token heuristic")
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
