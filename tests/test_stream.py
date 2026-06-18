"""Streaming edge case tests for agentkavach.stream."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from agentkavach.stream import StreamWrapper, _count_chunk_tokens

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(content: str = "Hello", usage: object = None) -> SimpleNamespace:
    """Create a mock streaming chunk."""
    chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(delta=SimpleNamespace(content=content)),
        ],
        usage=usage,
    )
    return chunk


def _make_engine() -> MagicMock:
    """Create a mock SpendEngine."""
    engine = MagicMock()
    engine.record_partial = MagicMock(return_value=0.001)
    engine._killed = False
    return engine


def _chunks(contents: list[str]):
    """Yield mock chunks from a list of content strings."""
    for content in contents:
        yield _make_chunk(content=content)


# ---------------------------------------------------------------------------
# Normal streaming
# ---------------------------------------------------------------------------


class TestStreamNormal:
    def test_yields_all_chunks(self):
        engine = _make_engine()
        stream = StreamWrapper(
            stream=_chunks(["Hello", " world", "!"]),
            model="gpt-4o",
            engine=engine,
        )
        results = list(stream)
        assert len(results) == 3
        assert results[0].choices[0].delta.content == "Hello"

    def test_records_spend_on_completion(self):
        engine = _make_engine()
        stream = StreamWrapper(
            stream=_chunks(["Hello", " world"]),
            model="gpt-4o",
            engine=engine,
        )
        list(stream)  # exhaust
        engine.record_partial.assert_called_once()
        args = engine.record_partial.call_args
        assert args[0][0] == "gpt-4o"
        assert args[0][1] > 0  # output_tokens

    def test_completed_flag(self):
        engine = _make_engine()
        stream = StreamWrapper(
            stream=_chunks(["Hi"]),
            model="gpt-4o",
            engine=engine,
        )
        list(stream)
        assert stream.completed is True

    def test_on_complete_callback(self):
        engine = _make_engine()
        callback_calls = []
        stream = StreamWrapper(
            stream=_chunks(["test"]),
            model="gpt-4o",
            engine=engine,
            on_complete=lambda m, t, p: callback_calls.append((m, t, p)),
        )
        list(stream)
        assert len(callback_calls) == 1
        model, tokens, partial = callback_calls[0]
        assert model == "gpt-4o"
        assert partial is False


# ---------------------------------------------------------------------------
# Disconnect / GeneratorExit
# ---------------------------------------------------------------------------


class TestStreamDisconnect:
    def test_partial_recording_on_close(self):
        engine = _make_engine()
        stream = StreamWrapper(
            stream=_chunks(["Hello", " world", " foo", " bar"]),
            model="gpt-4o",
            engine=engine,
        )
        # Consume only 2 chunks.
        next(stream)
        next(stream)
        stream.close()

        engine.record_partial.assert_called_once()
        assert stream.output_tokens > 0

    def test_close_with_context_manager(self):
        engine = _make_engine()
        with StreamWrapper(
            stream=_chunks(["Hello"]),
            model="gpt-4o",
            engine=engine,
        ) as stream:
            next(stream)
        # __exit__ calls close(), which finalizes.
        engine.record_partial.assert_called_once()

    def test_on_complete_called_with_partial_true(self):
        engine = _make_engine()
        callback_calls = []
        stream = StreamWrapper(
            stream=_chunks(["a", "b", "c"]),
            model="gpt-4o",
            engine=engine,
            on_complete=lambda m, t, p: callback_calls.append((m, t, p)),
        )
        next(stream)
        stream.close()
        assert callback_calls[0][2] is True  # partial=True

    def test_finalize_only_once(self):
        engine = _make_engine()
        stream = StreamWrapper(
            stream=_chunks(["hi"]),
            model="gpt-4o",
            engine=engine,
        )
        next(stream)
        stream.close()
        stream.close()  # should not double-record
        engine.record_partial.assert_called_once()


# ---------------------------------------------------------------------------
# Empty / no-content streams
# ---------------------------------------------------------------------------


class TestStreamEdgeCases:
    def test_empty_stream(self):
        engine = _make_engine()
        stream = StreamWrapper(
            stream=iter([]),
            model="gpt-4o",
            engine=engine,
        )
        results = list(stream)
        assert results == []
        assert stream.completed is True
        # No tokens to record.
        engine.record_partial.assert_not_called()

    def test_chunk_with_no_content(self):
        engine = _make_engine()
        empty_chunk = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=None))],
            usage=None,
        )
        stream = StreamWrapper(
            stream=iter([empty_chunk]),
            model="gpt-4o",
            engine=engine,
        )
        results = list(stream)
        assert len(results) == 1
        assert stream.output_tokens == 0

    def test_chunk_with_no_choices(self):
        engine = _make_engine()
        bare_chunk = SimpleNamespace(choices=[], usage=None)
        stream = StreamWrapper(
            stream=iter([bare_chunk]),
            model="gpt-4o",
            engine=engine,
        )
        list(stream)
        assert stream.output_tokens == 0

    def test_underlying_stream_closed(self):
        engine = _make_engine()
        inner = MagicMock()
        inner.__next__ = MagicMock(side_effect=StopIteration)
        inner.__iter__ = MagicMock(return_value=inner)
        inner.close = MagicMock()

        stream = StreamWrapper(
            stream=inner,
            model="gpt-4o",
            engine=engine,
        )
        list(stream)
        stream.close()
        inner.close.assert_called_once()


# ---------------------------------------------------------------------------
# _count_chunk_tokens
# ---------------------------------------------------------------------------


class TestCountChunkTokens:
    def test_from_usage(self):
        chunk = SimpleNamespace(
            usage=SimpleNamespace(completion_tokens=42),
            choices=[],
        )
        assert _count_chunk_tokens(chunk) == 42

    def test_from_delta_content(self):
        chunk = _make_chunk(content="Hello world test")  # 16 chars → 4 tokens
        assert _count_chunk_tokens(chunk) == 4

    def test_short_content_at_least_one(self):
        chunk = _make_chunk(content="Hi")  # 2 chars → rounds to 1
        assert _count_chunk_tokens(chunk) >= 1

    def test_no_content_returns_zero(self):
        chunk = SimpleNamespace(choices=[], usage=None)
        assert _count_chunk_tokens(chunk) == 0

    def test_usage_takes_priority(self):
        chunk = SimpleNamespace(
            usage=SimpleNamespace(completion_tokens=10),
            choices=[SimpleNamespace(delta=SimpleNamespace(content="x" * 100))],
        )
        # Should use usage, not content estimate.
        assert _count_chunk_tokens(chunk) == 10


# ---------------------------------------------------------------------------
# Callback exception handling
# ---------------------------------------------------------------------------


class TestCallbackErrors:
    def test_engine_error_does_not_propagate(self):
        engine = MagicMock()
        engine._killed = False
        engine.record_partial.side_effect = RuntimeError("engine boom")

        stream = StreamWrapper(
            stream=_chunks(["hi"]),
            model="gpt-4o",
            engine=engine,
        )
        # Should not raise.
        list(stream)

    def test_on_complete_error_does_not_propagate(self):
        engine = _make_engine()

        def bad_callback(m, t, p):
            raise RuntimeError("callback boom")

        stream = StreamWrapper(
            stream=_chunks(["hi"]),
            model="gpt-4o",
            engine=engine,
            on_complete=bad_callback,
        )
        # Should not raise.
        list(stream)
