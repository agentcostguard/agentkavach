"""Google (Gemini) response/usage parsing.

Handles responses from the ``google-genai`` SDK (the successor to the
now-deprecated ``google-generativeai`` package).  Google nests usage
under ``usage_metadata`` with ``prompt_token_count`` and
``candidates_token_count`` — that shape is unchanged between the old
and new SDK, so the parser code is the same.
"""

from __future__ import annotations

import logging
from typing import Any

from agentkavach.pricing import estimate_cost
from agentkavach.providers import UsageRecord

logger = logging.getLogger(__name__)

PROVIDER = "google"


def parse_usage(response: Any) -> UsageRecord:
    """Extract token counts from a Google GenerateContent response.

    Expects *response* to carry ``model`` (str) and ``usage_metadata``
    with ``prompt_token_count`` / ``candidates_token_count``.

    For the Google SDK the model name often needs to be passed
    separately since the response object may not carry it.  We accept
    it as an attribute or fall back to ``"unknown"``.

    Raises ``ValueError`` if usage metadata is absent.
    """
    model = getattr(response, "model", None) or "unknown"

    usage_metadata = getattr(response, "usage_metadata", None)
    if usage_metadata is None:
        raise ValueError("Response is missing 'usage_metadata' attribute")

    input_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
    output_tokens = getattr(usage_metadata, "candidates_token_count", 0) or 0

    return UsageRecord(
        provider=PROVIDER,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def count_tokens(model: str, messages: list[dict], client: Any = None) -> int:
    """Count input tokens using the Google count_tokens API.

    Requires a configured ``google.genai.Client`` instance. This makes
    a network call (~100-200ms) but returns the exact input token count.

    Falls back to a 4-chars-per-token heuristic if the API call fails
    or no client is provided.
    """
    if client is None:
        return _heuristic_count(messages)

    try:
        # Convert messages to Google's contents format.
        contents = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                contents.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        contents.append(part.get("text", ""))

        # New SDK: client.models.count_tokens(model=..., contents=...)
        result = client.models.count_tokens(model=model, contents=contents)
        return result.total_tokens
    except Exception:
        logger.warning("Google count_tokens API failed, using heuristic", exc_info=True)
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
