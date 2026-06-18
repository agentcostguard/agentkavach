"""Unit tests for agentkavach.providers — OpenAI, Anthropic, Google parsers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentkavach.providers import UsageRecord
from agentkavach.providers.anthropic import calculate_cost as anthropic_cost
from agentkavach.providers.anthropic import count_tokens as anthropic_count
from agentkavach.providers.anthropic import parse_usage as anthropic_parse
from agentkavach.providers.google import calculate_cost as google_cost
from agentkavach.providers.google import count_tokens as google_count
from agentkavach.providers.google import parse_usage as google_parse
from agentkavach.providers.mistral import calculate_cost as mistral_cost
from agentkavach.providers.mistral import count_tokens as mistral_count
from agentkavach.providers.mistral import parse_usage as mistral_parse
from agentkavach.providers.openai import calculate_cost as openai_cost
from agentkavach.providers.openai import count_tokens as openai_count
from agentkavach.providers.openai import parse_usage as openai_parse

# ---------------------------------------------------------------------------
# Helpers — lightweight response stubs (no real SDK dependency)
# ---------------------------------------------------------------------------


def _openai_response(
    model: str = "gpt-4o",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    )


def _anthropic_response(
    model: str = "claude-opus-4-0",
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


def _mistral_response(
    model: str = "mistral-large-latest",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    )


def _google_response(
    model: str = "gemini-1.5-pro",
    prompt_token_count: int = 100,
    candidates_token_count: int = 50,
) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt_token_count,
            candidates_token_count=candidates_token_count,
        ),
    )


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


class TestOpenAIParser:
    def test_parse_basic(self):
        usage = openai_parse(_openai_response())
        assert isinstance(usage, UsageRecord)
        assert usage.provider == "openai"
        assert usage.model == "gpt-4o"
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

    def test_parse_different_model(self):
        usage = openai_parse(_openai_response(model="gpt-4o-mini"))
        assert usage.model == "gpt-4o-mini"

    def test_parse_missing_usage_raises(self):
        response = SimpleNamespace(model="gpt-4o", usage=None)
        with pytest.raises(ValueError, match="missing 'usage'"):
            openai_parse(response)

    def test_parse_missing_model_raises(self):
        response = SimpleNamespace(
            model=None, usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1)
        )
        with pytest.raises(ValueError, match="missing 'model'"):
            openai_parse(response)

    def test_calculate_cost_known_model(self):
        usage = openai_parse(_openai_response(prompt_tokens=1000, completion_tokens=1000))
        cost = openai_cost(usage)
        assert cost > 0

    def test_calculate_cost_unknown_model(self):
        usage = UsageRecord(
            provider="openai", model="unknown-v99", input_tokens=100, output_tokens=50
        )
        assert openai_cost(usage) == 0.0


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


class TestAnthropicParser:
    def test_parse_basic(self):
        usage = anthropic_parse(_anthropic_response())
        assert usage.provider == "anthropic"
        assert usage.model == "claude-opus-4-0"
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

    def test_parse_missing_usage_raises(self):
        response = SimpleNamespace(model="claude-opus-4-0", usage=None)
        with pytest.raises(ValueError, match="missing 'usage'"):
            anthropic_parse(response)

    def test_parse_missing_model_raises(self):
        response = SimpleNamespace(
            model=None, usage=SimpleNamespace(input_tokens=1, output_tokens=1)
        )
        with pytest.raises(ValueError, match="missing 'model'"):
            anthropic_parse(response)

    def test_calculate_cost_known_model(self):
        usage = anthropic_parse(_anthropic_response(input_tokens=1000, output_tokens=1000))
        cost = anthropic_cost(usage)
        assert cost > 0

    def test_calculate_cost_unknown_model(self):
        usage = UsageRecord(
            provider="anthropic", model="unknown-v99", input_tokens=100, output_tokens=50
        )
        assert anthropic_cost(usage) == 0.0


# ---------------------------------------------------------------------------
# Google
# ---------------------------------------------------------------------------


class TestGoogleParser:
    def test_parse_basic(self):
        usage = google_parse(_google_response())
        assert usage.provider == "google"
        assert usage.model == "gemini-1.5-pro"
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

    def test_parse_missing_usage_metadata_raises(self):
        response = SimpleNamespace(model="gemini-1.5-pro", usage_metadata=None)
        with pytest.raises(ValueError, match="missing 'usage_metadata'"):
            google_parse(response)

    def test_parse_missing_model_defaults_to_unknown(self):
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=5),
        )
        usage = google_parse(response)
        assert usage.model == "unknown"

    def test_calculate_cost_known_model(self):
        usage = google_parse(_google_response(prompt_token_count=1000, candidates_token_count=1000))
        cost = google_cost(usage)
        assert cost > 0

    def test_calculate_cost_unknown_model(self):
        usage = UsageRecord(
            provider="google", model="unknown-v99", input_tokens=100, output_tokens=50
        )
        assert google_cost(usage) == 0.0


# ---------------------------------------------------------------------------
# Mistral
# ---------------------------------------------------------------------------


class TestMistralParser:
    def test_parse_basic(self):
        usage = mistral_parse(_mistral_response())
        assert isinstance(usage, UsageRecord)
        assert usage.provider == "mistral"
        assert usage.model == "mistral-large-latest"
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

    def test_parse_different_model(self):
        usage = mistral_parse(_mistral_response(model="mistral-small-latest"))
        assert usage.model == "mistral-small-latest"

    def test_parse_missing_usage_raises(self):
        response = SimpleNamespace(model="mistral-large-latest", usage=None)
        with pytest.raises(ValueError, match="missing 'usage'"):
            mistral_parse(response)

    def test_parse_missing_model_raises(self):
        response = SimpleNamespace(
            model=None, usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1)
        )
        with pytest.raises(ValueError, match="missing 'model'"):
            mistral_parse(response)

    def test_calculate_cost_known_model(self):
        usage = mistral_parse(_mistral_response(prompt_tokens=1000, completion_tokens=1000))
        cost = mistral_cost(usage)
        assert cost > 0

    def test_calculate_cost_unknown_model(self):
        usage = UsageRecord(
            provider="mistral", model="unknown-v99", input_tokens=100, output_tokens=50
        )
        assert mistral_cost(usage) == 0.0


# ---------------------------------------------------------------------------
# UsageRecord
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


class TestOpenAICountTokens:
    def test_counts_simple_message(self):
        messages = [{"role": "user", "content": "Hello world"}]
        count = openai_count("gpt-4o", messages)
        assert count > 0

    def test_counts_multipart_content(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }
        ]
        count = openai_count("gpt-4o", messages)
        assert count > 0

    def test_empty_messages(self):
        count = openai_count("gpt-4o", [])
        assert count == 2  # assistant reply priming only

    def test_no_client_needed(self):
        # OpenAI counting is local — no client required.
        count = openai_count("gpt-4o", [{"role": "user", "content": "test"}])
        assert count > 0

    def test_unknown_model_falls_back_to_cl100k(self):
        """When tiktoken doesn't recognize the model, it falls back to cl100k_base."""
        messages = [{"role": "user", "content": "Hello world"}]
        count = openai_count("totally-unknown-model-xyz", messages)
        assert count > 0

    def test_tiktoken_import_error_heuristic_fallback(self, monkeypatch):
        """When tiktoken is unavailable, use 4-chars-per-token heuristic."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "tiktoken":
                raise ImportError("no tiktoken")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        messages = [{"role": "user", "content": "Hello world, this is a test message"}]
        count = openai_count("gpt-4o", messages)
        assert count > 0

    def test_tiktoken_import_error_multipart_content(self, monkeypatch):
        """Heuristic fallback handles multipart content correctly."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "tiktoken":
                raise ImportError("no tiktoken")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }
        ]
        count = openai_count("gpt-4o", messages)
        assert count > 0


class TestAnthropicCountTokens:
    def test_falls_back_to_heuristic_without_client(self):
        messages = [{"role": "user", "content": "Hello world, this is a test"}]
        count = anthropic_count("claude-opus-4-0", messages)
        assert count > 0

    def test_uses_client_when_provided(self):
        mock_client = SimpleNamespace(
            messages=SimpleNamespace(
                count_tokens=lambda model, messages: SimpleNamespace(input_tokens=42),
            )
        )
        count = anthropic_count(
            "claude-opus-4-0", [{"role": "user", "content": "hi"}], client=mock_client
        )
        assert count == 42

    def test_falls_back_on_api_error(self):
        def raise_error(**kwargs):
            raise RuntimeError("API error")

        mock_client = SimpleNamespace(messages=SimpleNamespace(count_tokens=raise_error))
        count = anthropic_count(
            "claude-opus-4-0", [{"role": "user", "content": "test"}], client=mock_client
        )
        assert count > 0  # heuristic fallback


class TestGoogleCountTokens:
    def test_falls_back_to_heuristic_without_client(self):
        messages = [{"role": "user", "content": "Hello world, this is a test"}]
        count = google_count("gemini-1.5-pro", messages)
        assert count > 0

    def test_uses_client_when_provided(self):
        # Phase 45: new ``google-genai`` shape is ``client.models.count_tokens(
        # model=..., contents=...)`` returning a ``CountTokensResponse``
        # with ``.total_tokens``.
        mock_models = SimpleNamespace(
            count_tokens=lambda model, contents: SimpleNamespace(total_tokens=55),
        )
        mock_client = SimpleNamespace(models=mock_models)
        count = google_count(
            "gemini-1.5-pro", [{"role": "user", "content": "hi"}], client=mock_client
        )
        assert count == 55

    def test_falls_back_on_api_error(self):
        def raise_error(model, contents):
            raise RuntimeError("API error")

        mock_client = SimpleNamespace(models=SimpleNamespace(count_tokens=raise_error))
        count = google_count(
            "gemini-1.5-pro", [{"role": "user", "content": "test"}], client=mock_client
        )
        assert count > 0  # heuristic fallback

    def test_client_with_multipart_content(self):
        """count_tokens with a client extracts text from multipart content."""
        captured = {}

        def mock_count_tokens(model, contents):
            captured["model"] = model
            captured["contents"] = contents
            return SimpleNamespace(total_tokens=77)

        mock_client = SimpleNamespace(models=SimpleNamespace(count_tokens=mock_count_tokens))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }
        ]
        count = google_count("gemini-1.5-pro", messages, client=mock_client)
        assert count == 77
        # Only text parts should be passed to the API.
        assert captured["contents"] == ["Describe this"]
        assert captured["model"] == "gemini-1.5-pro"

    def test_heuristic_multipart_content(self):
        """Heuristic count handles multipart content lists."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello world test message here"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }
        ]
        count = google_count("gemini-1.5-pro", messages)
        assert count > 0


class TestMistralCountTokens:
    def test_counts_simple_message(self):
        messages = [{"role": "user", "content": "Hello world"}]
        count = mistral_count("mistral-large-latest", messages)
        assert count > 0

    def test_no_client_needed(self):
        count = mistral_count("mistral-large-latest", [{"role": "user", "content": "test"}])
        assert count > 0

    def test_unknown_model_falls_back_to_cl100k(self):
        messages = [{"role": "user", "content": "Hello world"}]
        count = mistral_count("totally-unknown-model-xyz", messages)
        assert count > 0

    def test_tiktoken_import_error_heuristic_fallback(self, monkeypatch):
        """When tiktoken is unavailable, use 4-chars-per-token heuristic."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "tiktoken":
                raise ImportError("no tiktoken")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        messages = [{"role": "user", "content": "Hello world, this is a test message"}]
        count = mistral_count("mistral-large-latest", messages)
        assert count > 0

    def test_empty_messages(self):
        count = mistral_count("mistral-large-latest", [])
        assert count == 2  # assistant reply priming only


class TestUsageRecord:
    def test_frozen(self):
        record = UsageRecord(provider="openai", model="gpt-4o", input_tokens=100, output_tokens=50)
        with pytest.raises(AttributeError):
            record.input_tokens = 200  # type: ignore[misc]

    def test_default_cost_is_none(self):
        record = UsageRecord(provider="openai", model="gpt-4o", input_tokens=100, output_tokens=50)
        assert record.cost is None
