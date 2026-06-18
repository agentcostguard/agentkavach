"""Streaming wrapper with GeneratorExit handling.

Wraps an OpenAI-style streaming response to track token usage in
real-time.  Handles client disconnects (``GeneratorExit``) gracefully,
ensuring partial usage is always recorded even when the consumer
abandons the stream.

Usage (internal — called by ``AgentKavach`` client):

    wrapped = StreamWrapper(
        stream=openai_stream,
        model="gpt-4o",
        engine=spend_engine,
        on_complete=callback,
    )
    for chunk in wrapped:
        print(chunk.choices[0].delta.content)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterator, Optional

from agentkavach.exceptions import BudgetExceededError

logger = logging.getLogger(__name__)


class StreamWrapper:
    """Wraps a provider streaming response to track and record usage.

    Transparently yields chunks while counting output tokens.  On
    stream completion (or disconnect) it records the spend via the
    engine.

    Attributes:
        output_tokens: Running count of output tokens seen so far.
        completed: Whether the stream finished normally.
    """

    def __init__(
        self,
        stream: Iterator[Any],
        model: str,
        engine: Any,
        on_complete: Optional[Callable[[str, int, bool], None]] = None,
    ) -> None:
        """
        Args:
            stream: The raw provider streaming iterator.
            model: Model identifier for cost lookups.
            engine: ``SpendEngine`` instance for recording spend.
            on_complete: Optional callback invoked when the stream ends.
                Signature: ``(model, output_tokens, partial) -> None``.
        """
        self._stream = stream
        self._model = model
        self._engine = engine
        self._on_complete = on_complete

        self.output_tokens: int = 0
        self.completed: bool = False
        self._finalized: bool = False

    def __iter__(self) -> StreamWrapper:
        return self

    def __next__(self) -> Any:
        # Check if budget has been exhausted (kill switch). Defer to the
        # engine so the customer's specific guardrail except-handler
        # (TokenLimitError / RuntimeLimitError / etc.) fires here too —
        # not just on the next non-streaming call.
        if getattr(self._engine, "_killed", False):
            self._finalize(partial=True)
            replay = getattr(self._engine, "_raise_for_kill_reason", None)
            if callable(replay):
                replay()
            raise BudgetExceededError(
                f"Stream interrupted — budget exhausted for model {self._model!r}"
            )

        try:
            chunk = next(self._stream)
            self.output_tokens += _count_chunk_tokens(chunk)
            return chunk
        except StopIteration:
            self.completed = True
            self._finalize(partial=False)
            raise
        except GeneratorExit:
            logger.warning(
                "Stream for %s disconnected after ~%d output tokens",
                self._model,
                self.output_tokens,
            )
            self._finalize(partial=True)
            raise

    def __enter__(self) -> StreamWrapper:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Explicitly close the stream, recording any remaining usage."""
        partial = not self.completed
        self._finalize(partial=partial)
        # Close the underlying stream if it supports it.
        close_fn = getattr(self._stream, "close", None)
        if callable(close_fn):
            close_fn()

    def _finalize(self, partial: bool) -> None:
        """Record spend exactly once."""
        if self._finalized:
            return
        self._finalized = True

        if self.output_tokens > 0:
            try:
                self._engine.record_partial(self._model, self.output_tokens)
            except Exception:
                logger.exception("Failed to record partial spend")

        if self._on_complete is not None:
            try:
                self._on_complete(self._model, self.output_tokens, partial)
            except Exception:
                logger.exception("on_complete callback raised an exception")


def _count_chunk_tokens(chunk: Any) -> int:
    """Extract token count from a streaming chunk.

    Each provider emits a different chunk shape. We probe them in
    order, falling through to the next strategy when an attribute is
    missing. All probes are defensive (``getattr(..., None)``) so an
    unknown chunk type yields ``0`` rather than raising.

    Strategies (in order):

    1. ``chunk.usage.completion_tokens`` — OpenAI chat completion
       stream with ``stream_options={"include_usage": True}``.
    2. ``chunk.choices[0].delta.content`` — OpenAI per-chunk text
       (Mistral chunks also expose this shape when wrapped by
       ``chunk.data``, handled by strategy 5).
    3. Anthropic event types:
       - ``chunk.usage.output_tokens`` on ``MessageDeltaEvent`` / the
         final ``message_delta`` event — exact count.
       - ``chunk.delta.text`` on ``RawContentBlockDeltaEvent`` —
         estimate from text length.
    4. Google (Gemini) ``GenerateContentResponse`` chunks:
       - ``chunk.usage_metadata.candidates_token_count`` on the final
         chunk — exact count.
       - ``chunk.text`` on every text chunk — estimate from length.
    5. Mistral ``CompletionEvent`` (``client.chat.stream`` yields
       these): the real chunk is ``chunk.data``; recurse with that.
    """
    # Strategy 1: Direct usage data (OpenAI stream with include_usage=True).
    usage = getattr(chunk, "usage", None)
    if usage is not None:
        completion = getattr(usage, "completion_tokens", None)
        if completion is not None and completion > 0:
            return completion
        # Strategy 3a: Anthropic MessageDeltaEvent carries
        # ``usage.output_tokens`` (not ``completion_tokens``).
        output = getattr(usage, "output_tokens", None)
        if output is not None and output > 0:
            return output

    # Strategy 2: Estimate from delta content (OpenAI per-chunk).
    choices = getattr(chunk, "choices", None)
    if choices:
        for choice in choices:
            delta = getattr(choice, "delta", None)
            if delta is not None:
                content = getattr(delta, "content", None)
                if isinstance(content, str) and content:
                    # Rough: ~4 chars per token.
                    return max(1, len(content) // 4)

    # Strategy 3b: Anthropic RawContentBlockDeltaEvent — top-level
    # ``delta.text``. Guard against OpenAI shape (which also has
    # ``delta`` but always nested under ``choices``, already handled).
    delta = getattr(chunk, "delta", None)
    if delta is not None and not choices:
        text = getattr(delta, "text", None)
        if isinstance(text, str) and text:
            return max(1, len(text) // 4)

    # Strategy 4: Google (Gemini) GenerateContentResponse chunks.
    usage_metadata = getattr(chunk, "usage_metadata", None)
    if usage_metadata is not None:
        cand = getattr(usage_metadata, "candidates_token_count", None)
        if cand is not None and cand > 0:
            return cand
    text = getattr(chunk, "text", None)
    if isinstance(text, str) and text:
        return max(1, len(text) // 4)

    # Strategy 5: Mistral CompletionEvent wraps the actual chunk
    # under ``.data``. Recurse to reuse strategies 1/2.
    data = getattr(chunk, "data", None)
    if data is not None and data is not chunk:
        return _count_chunk_tokens(data)

    return 0
