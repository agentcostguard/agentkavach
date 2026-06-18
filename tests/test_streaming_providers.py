"""Per-provider streaming tests for Phase 44.

Each provider emits a different streaming chunk shape. These tests
fabricate the shape using ``SimpleNamespace`` (no SDK calls) and
assert that ``_count_chunk_tokens`` and ``StreamWrapper`` extract the
right token counts.

Coverage:
- OpenAI: ``chunk.choices[0].delta.content`` and
  ``chunk.usage.completion_tokens``.
- Anthropic: ``chunk.delta.text`` (RawContentBlockDeltaEvent) and
  ``chunk.usage.output_tokens`` (MessageDeltaEvent).
- Google: ``chunk.text`` and ``chunk.usage_metadata.candidates_token_count``.
- Mistral: nested ``chunk.data.choices[0].delta.content`` (CompletionEvent
  wrapping CompletionChunk) and ``chunk.data.usage.completion_tokens``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from agentkavach.stream import StreamWrapper, _count_chunk_tokens

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _engine() -> MagicMock:
    """Mock SpendEngine for StreamWrapper tests."""
    engine = MagicMock()
    engine.record_partial = MagicMock(return_value=0.001)
    engine._killed = False
    return engine


# ---------------------------------------------------------------------------
# OpenAI — both shapes
# ---------------------------------------------------------------------------


class TestOpenAIStream:
    def test_openai_stream_counts_via_delta_content(self):
        """OpenAI per-chunk ``delta.content`` is counted (≈chars/4)."""
        chunk = MagicMock()
        chunk.usage = None
        chunk.choices = [MagicMock(delta=MagicMock(content="hello world"))]
        # Avoid MagicMock auto-creating .delta.text — strip non-OpenAI attrs.
        # MagicMock() returns truthy for any getattr; explicitly None them.
        chunk.delta = None
        chunk.usage_metadata = None
        chunk.text = None
        chunk.data = None
        # "hello world" → 11 chars → 11//4 = 2 tokens.
        assert _count_chunk_tokens(chunk) == 2

    def test_openai_stream_counts_via_include_usage(self):
        """Final OpenAI chunk with ``stream_options={include_usage: True}``."""
        chunk = SimpleNamespace(
            usage=SimpleNamespace(completion_tokens=137),
            choices=[],
        )
        assert _count_chunk_tokens(chunk) == 137

    def test_openai_full_stream_records_spend(self):
        """End-to-end: a sequence of OpenAI deltas updates output_tokens."""
        chunks = [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello, "))],
                usage=None,
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="world!"))],
                usage=None,
            ),
        ]
        engine = _engine()
        wrapper = StreamWrapper(stream=iter(chunks), model="gpt-4o", engine=engine)
        list(wrapper)
        assert wrapper.output_tokens > 0
        engine.record_partial.assert_called_once()


# ---------------------------------------------------------------------------
# Anthropic — RawContentBlockDeltaEvent + MessageDeltaEvent
# ---------------------------------------------------------------------------


class TestAnthropicStream:
    def test_anthropic_stream_counts_via_delta_text(self):
        """``RawContentBlockDeltaEvent`` exposes ``delta.text`` directly."""
        # Anthropic chunk: no ``choices``, top-level ``delta.text``.
        chunk = SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="text_delta", text="hello world"),
        )
        # 11 chars // 4 = 2 tokens.
        assert _count_chunk_tokens(chunk) == 2

    def test_anthropic_stream_counts_via_message_delta_usage(self):
        """``MessageDeltaEvent.usage.output_tokens`` returns exact count."""
        chunk = SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="end_turn"),
            usage=SimpleNamespace(output_tokens=42),
        )
        assert _count_chunk_tokens(chunk) == 42

    def test_anthropic_full_stream(self):
        """Stream of Anthropic events accumulates output_tokens correctly."""
        chunks = [
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="abcd"),
            ),
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="efgh"),
            ),
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="end_turn"),
                usage=SimpleNamespace(output_tokens=50),
            ),
        ]
        engine = _engine()
        wrapper = StreamWrapper(stream=iter(chunks), model="claude-sonnet-4-0", engine=engine)
        list(wrapper)
        # 2 deltas of 4 chars (1 token each) + 50 from MessageDeltaEvent = 52.
        assert wrapper.output_tokens == 52
        engine.record_partial.assert_called_once()


# ---------------------------------------------------------------------------
# Google (Gemini) — text and usage_metadata
# ---------------------------------------------------------------------------


class TestGoogleStream:
    def test_google_stream_counts_via_text(self):
        """Per-chunk Google response exposes ``.text`` (string)."""
        # Build a Google-shaped chunk without OpenAI/Anthropic noise.
        chunk = SimpleNamespace(
            text="hello world four",  # 16 chars → 4 tokens
            usage_metadata=None,
            candidates=[],
        )
        assert _count_chunk_tokens(chunk) == 4

    def test_google_stream_counts_via_usage_metadata(self):
        """Final Google chunk carries ``usage_metadata.candidates_token_count``."""
        chunk = SimpleNamespace(
            text=None,  # final chunk often has no text
            usage_metadata=SimpleNamespace(
                prompt_token_count=10,
                candidates_token_count=33,
                total_token_count=43,
            ),
        )
        assert _count_chunk_tokens(chunk) == 33

    def test_google_full_stream(self):
        """Streaming a Gemini response accumulates tokens."""
        chunks = [
            SimpleNamespace(text="abcd", usage_metadata=None),
            SimpleNamespace(text="efghijkl", usage_metadata=None),  # 8 chars → 2
            SimpleNamespace(
                text=None,
                usage_metadata=SimpleNamespace(
                    prompt_token_count=5,
                    candidates_token_count=99,
                    total_token_count=104,
                ),
            ),
        ]
        engine = _engine()
        wrapper = StreamWrapper(stream=iter(chunks), model="gemini-1.5-pro", engine=engine)
        list(wrapper)
        # 1 + 2 + 99 = 102
        assert wrapper.output_tokens == 102
        engine.record_partial.assert_called_once()


# ---------------------------------------------------------------------------
# Mistral — CompletionEvent wraps the chunk in `.data`
# ---------------------------------------------------------------------------


class TestMistralStream:
    def test_mistral_stream_counts_via_delta_content(self):
        """``CompletionEvent.data.choices[0].delta.content`` is unwrapped."""
        inner = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="hello world"))],
            usage=None,
        )
        chunk = SimpleNamespace(data=inner)
        # 11 chars // 4 = 2 tokens.
        assert _count_chunk_tokens(chunk) == 2

    def test_mistral_stream_counts_via_completion_tokens(self):
        """Mistral may emit a final chunk with ``usage.completion_tokens``."""
        inner = SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(completion_tokens=77, prompt_tokens=10),
        )
        chunk = SimpleNamespace(data=inner)
        assert _count_chunk_tokens(chunk) == 77

    def test_mistral_full_stream(self):
        """Stream of CompletionEvent wrappers is processed end-to-end."""
        chunks = [
            SimpleNamespace(
                data=SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="abcdefgh"))],
                    usage=None,
                )
            ),
            SimpleNamespace(
                data=SimpleNamespace(
                    choices=[],
                    usage=SimpleNamespace(completion_tokens=20, prompt_tokens=5),
                )
            ),
        ]
        engine = _engine()
        wrapper = StreamWrapper(stream=iter(chunks), model="mistral-large-latest", engine=engine)
        list(wrapper)
        # 8 chars // 4 = 2, then 20 = 22 total.
        assert wrapper.output_tokens == 22
        engine.record_partial.assert_called_once()


# ---------------------------------------------------------------------------
# Defensive — unknown chunk shape must yield 0, not raise
# ---------------------------------------------------------------------------


class TestUnknownChunk:
    def test_unknown_shape_returns_zero(self):
        chunk = SimpleNamespace(foo="bar", baz=123)
        assert _count_chunk_tokens(chunk) == 0

    def test_bare_object_returns_zero(self):
        assert _count_chunk_tokens(object()) == 0
