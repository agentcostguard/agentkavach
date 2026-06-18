"""Tests for multi-provider AgentKavach client support.

Covers Anthropic / Google provider integration at the AgentKavach client level,
multi-provider shared budgets, and provider dispatch routing.
"""

from __future__ import annotations

import textwrap
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agentkavach.budget import Budget
from agentkavach.client import AgentKavach
from agentkavach.exceptions import BudgetExceededError

# ---------------------------------------------------------------------------
# Response stubs
# ---------------------------------------------------------------------------


def _mock_openai_response(
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


def _mock_anthropic_response(
    model: str = "claude-3-5-sonnet-20241022",
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


def _mock_google_response(
    model: str = "gemini-2.0-flash",
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


def _client(provider: str = "openai", **overrides) -> AgentKavach:
    defaults = dict(
        api_key="ak_test",
        provider=provider,
        openai_api_key="sk-test" if provider == "openai" else "",
        anthropic_api_key="sk-ant-test" if provider == "anthropic" else "",
        google_api_key="AIza-test" if provider == "google" else "",
        llm_key="mistral-test" if provider == "mistral" else "",
        budget=Budget.daily(limit=10.0),
        agent_name="test-agent",
    )
    defaults.update(overrides)
    return AgentKavach(**defaults)


# ---------------------------------------------------------------------------
# Provider init
# ---------------------------------------------------------------------------


class TestProviderInit:
    def test_default_provider_is_openai(self):
        cg = AgentKavach(api_key="ak_test", openai_api_key="sk-test")
        assert cg._provider == "openai"

    def test_anthropic_provider(self):
        cg = _client(provider="anthropic")
        assert cg._provider == "anthropic"

    def test_google_provider(self):
        cg = _client(provider="google")
        assert cg._provider == "google"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            _client(provider="cohere")

    def test_case_insensitive_provider(self):
        cg = AgentKavach(
            api_key="ak_test",
            provider="Anthropic",
            anthropic_api_key="sk-ant-test",
            budget=Budget.daily(10.0),
        )
        assert cg._provider == "anthropic"

    def test_anthropic_llm_key_not_read_from_env(self, monkeypatch):
        # A provider key in the environment must NOT satisfy llm_key — the SDK
        # never reads it, so construction without an explicit llm_key raises.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
        with pytest.raises(ValueError, match="llm_key is required"):
            AgentKavach(api_key="ak_test", provider="anthropic")

    def test_google_llm_key_not_read_from_env(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "AIza-from-env")
        with pytest.raises(ValueError, match="llm_key is required"):
            AgentKavach(api_key="ak_test", provider="google")

    def test_has_messages_namespace(self):
        cg = _client(provider="anthropic")
        assert hasattr(cg, "messages")
        assert hasattr(cg.messages, "create")

    def test_has_chat_namespace(self):
        cg = _client(provider="openai")
        assert hasattr(cg, "chat")
        assert hasattr(cg.chat, "completions")

    def test_has_generate_content_method(self):
        cg = _client(provider="google")
        assert callable(getattr(cg, "generate_content", None))


# ---------------------------------------------------------------------------
# OpenAI calls (existing behavior, regression)
# ---------------------------------------------------------------------------


class TestOpenAICalls:
    @patch("agentkavach.client.AgentKavach._get_openai_client")
    def test_chat_completions_create(self, mock_get):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response()
        mock_get.return_value = mock_client

        cg = _client(provider="openai")
        response = cg.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert response.model == "gpt-4o"
        assert cg.spent > 0

    @patch("agentkavach.client.AgentKavach._get_openai_client")
    def test_buffers_openai_event(self, mock_get, tmp_path):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response()
        mock_get.return_value = mock_client

        cg = _client(provider="openai", buffer_path=str(tmp_path / "buf.jsonl"))
        cg.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        )
        events = cg._buffer.read_all()
        assert len(events) == 1
        assert events[0]["provider"] == "openai"


# ---------------------------------------------------------------------------
# Anthropic calls
# ---------------------------------------------------------------------------


class TestAnthropicCalls:
    @patch("agentkavach.client.AgentKavach._get_anthropic_client")
    def test_messages_create(self, mock_get):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_anthropic_response()
        mock_get.return_value = mock_client

        cg = _client(provider="anthropic")
        response = cg.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "hello"}],
        )
        assert response.model == "claude-3-5-sonnet-20241022"
        assert cg.spent > 0

    @patch("agentkavach.client.AgentKavach._get_anthropic_client")
    def test_spend_recorded_correctly(self, mock_get):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_anthropic_response(
            input_tokens=200,
            output_tokens=100,
        )
        mock_get.return_value = mock_client

        cg = _client(provider="anthropic", budget=Budget.daily(100.0))
        cg.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "hello"}],
        )
        assert cg.spent > 0
        assert cg.remaining < 100.0

    @patch("agentkavach.client.AgentKavach._get_anthropic_client")
    def test_budget_exceeded_raises(self, mock_get):
        cg = _client(provider="anthropic", budget=Budget.daily(0.001))
        cg._engine.post_flight("claude-3-5-sonnet-20241022", 0, 100_000)
        cg._engine.check_thresholds()

        with pytest.raises(BudgetExceededError):
            cg.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": "x" * 5000}],
            )

    @patch("agentkavach.client.AgentKavach._get_anthropic_client")
    def test_buffers_anthropic_event(self, mock_get, tmp_path):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_anthropic_response()
        mock_get.return_value = mock_client

        cg = _client(
            provider="anthropic",
            buffer_path=str(tmp_path / "buf.jsonl"),
        )
        cg.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "hi"}],
        )
        events = cg._buffer.read_all()
        assert len(events) == 1
        assert events[0]["provider"] == "anthropic"
        assert events[0]["model"] == "claude-3-5-sonnet-20241022"


# ---------------------------------------------------------------------------
# Google calls
# ---------------------------------------------------------------------------


def _mock_google_client(response):
    """Build a mock ``google.genai.Client`` whose
    ``client.models.generate_content(...)`` returns *response*.

    Phase 45: the new SDK exposes a ``Client`` with a ``models``
    namespace instead of the old ``GenerativeModel`` factory.
    """
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = response
    return mock_client


class TestGoogleCalls:
    @patch("agentkavach.client.AgentKavach._get_google_client")
    def test_generate_content(self, mock_get):
        mock_get.return_value = _mock_google_client(_mock_google_response())

        cg = _client(provider="google")
        response = cg.generate_content(
            model="gemini-2.0-flash",
            contents="Hello!",
        )
        assert response.model == "gemini-2.0-flash"
        assert cg.spent > 0

    @patch("agentkavach.client.AgentKavach._get_google_client")
    def test_spend_recorded(self, mock_get):
        mock_get.return_value = _mock_google_client(
            _mock_google_response(prompt_token_count=500, candidates_token_count=200)
        )

        cg = _client(provider="google", budget=Budget.daily(100.0))
        cg.generate_content(model="gemini-2.0-flash", contents="Hello!")
        assert cg.spent > 0

    @patch("agentkavach.client.AgentKavach._get_google_client")
    def test_budget_exceeded_raises(self, mock_get):
        cg = _client(provider="google", budget=Budget.daily(0.001))
        cg._engine.post_flight("gemini-2.0-flash", 0, 100_000)
        cg._engine.check_thresholds()

        with pytest.raises(BudgetExceededError):
            cg.generate_content(
                model="gemini-2.0-flash",
                contents="x" * 5000,
            )

    @patch("agentkavach.client.AgentKavach._get_google_client")
    def test_buffers_google_event(self, mock_get, tmp_path):
        mock_get.return_value = _mock_google_client(_mock_google_response())

        cg = _client(
            provider="google",
            buffer_path=str(tmp_path / "buf.jsonl"),
        )
        cg.generate_content(model="gemini-2.0-flash", contents="Hi!")
        events = cg._buffer.read_all()
        assert len(events) == 1
        assert events[0]["provider"] == "google"
        assert events[0]["model"] == "gemini-2.0-flash"

    @patch("agentkavach.client.AgentKavach._get_google_client")
    def test_list_contents(self, mock_get):
        mock_get.return_value = _mock_google_client(_mock_google_response())

        cg = _client(provider="google")
        cg.generate_content(
            model="gemini-2.0-flash",
            contents=["Hello!", "How are you?"],
        )
        assert cg.spent > 0


# ---------------------------------------------------------------------------
# Shared budgets across providers
# ---------------------------------------------------------------------------


class TestOrgBudgets:
    def test_org_budget_across_openai_and_anthropic(self):
        shared = Budget.org_budget(
            limit=100.0,
            period="daily",
        )
        cg_oai = _client(
            provider="openai",
            budget=shared,
            agent_name="openai-bot",
        )
        cg_ant = _client(
            provider="anthropic",
            budget=shared,
            agent_name="anthropic-bot",
        )

        # org_budget gets routed via the org_budget engine slot.
        assert cg_oai._engine.org_budget is not None
        assert cg_ant._engine.org_budget is not None
        assert cg_oai._engine.org_budget.shared_name == "__org__"

    def test_org_budget_spend_independent_engines(self):
        """Each AgentKavach has its own engine; org budget coordination
        happens server-side via org_budget."""
        shared = Budget.org_budget(
            limit=50.0,
            period="daily",
        )
        agents = [
            _client(provider="openai", budget=shared, agent_name="a"),
            _client(provider="anthropic", budget=shared, agent_name="b"),
            _client(provider="google", budget=shared, agent_name="c"),
        ]
        # All engines have the same org budget
        org_keys = {a._engine.org_budget.key for a in agents}
        assert len(org_keys) == 1  # all same org budget key

    def test_org_budget_limit(self):
        shared = Budget.org_budget(
            limit=200.0,
            period="monthly",
        )
        assert shared.limit == 200.0
        assert shared.shared_name == "__org__"


# ---------------------------------------------------------------------------
# Message extraction
# ---------------------------------------------------------------------------


class TestMessageExtraction:
    def test_openai_messages(self):
        cg = _client(provider="openai")
        msgs = cg._extract_messages(
            {
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
        assert len(msgs) == 1
        assert msgs[0]["content"] == "hi"

    def test_google_string_contents(self):
        cg = _client(provider="google")
        msgs = cg._extract_messages({"contents": "hello world"})
        assert len(msgs) == 1
        assert msgs[0]["content"] == "hello world"

    def test_google_list_contents(self):
        cg = _client(provider="google")
        msgs = cg._extract_messages({"contents": ["a", "b"]})
        assert len(msgs) == 2

    def test_empty_kwargs(self):
        cg = _client(provider="openai")
        msgs = cg._extract_messages({})
        assert msgs == []


# ---------------------------------------------------------------------------
# Stream complete callback
# ---------------------------------------------------------------------------


class TestStreamComplete:
    def test_stream_callback_uses_correct_provider(self, tmp_path):
        cg = _client(
            provider="anthropic",
            buffer_path=str(tmp_path / "buf.jsonl"),
        )
        cg._on_stream_complete("claude-3-5-sonnet-20241022", 50, False)
        events = cg._buffer.read_all()
        assert events[0]["provider"] == "anthropic"

    def test_stream_callback_google(self, tmp_path):
        cg = _client(
            provider="google",
            buffer_path=str(tmp_path / "buf.jsonl"),
        )
        cg._on_stream_complete("gemini-2.0-flash", 30, True)
        events = cg._buffer.read_all()
        assert events[0]["provider"] == "google"
        assert events[0]["partial"] is True


# ---------------------------------------------------------------------------
# YAML config with provider
# ---------------------------------------------------------------------------


class TestYamlMultiProvider:
    def test_provider_in_yaml(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            textwrap.dedent("""\
            agents:
              claude-bot:
                provider: anthropic
                budget: { type: daily, limit: 50 }
              gemini-bot:
                provider: google
                budget: { type: daily, limit: 30 }
              gpt-bot:
                provider: openai
                budget: { type: daily, limit: 40 }
        """)
        )
        clients = AgentKavach.from_yaml(str(config), api_key="ak_test", llm_key="sk-test")
        assert clients["claude-bot"]._provider == "anthropic"
        assert clients["gemini-bot"]._provider == "google"
        assert clients["gpt-bot"]._provider == "openai"

    def test_default_provider_in_yaml(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            textwrap.dedent("""\
            defaults:
              provider: anthropic
            agents:
              bot-a: {}
              bot-b:
                provider: google
        """)
        )
        clients = AgentKavach.from_yaml(str(config), api_key="ak_test", llm_key="sk-test")
        assert clients["bot-a"]._provider == "anthropic"
        assert clients["bot-b"]._provider == "google"

    def test_org_budget_multi_provider_yaml(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            textwrap.dedent("""\
            org_budget:
              limit: 200
              period: daily
            agents:
              oai-bot:
                provider: openai
                budget: { type: daily, limit: 50 }
              ant-bot:
                provider: anthropic
                budget: { type: daily, limit: 50 }
        """)
        )
        clients = AgentKavach.from_yaml(str(config), api_key="ak_test", llm_key="sk-test")
        assert clients["oai-bot"]._provider == "openai"
        assert clients["ant-bot"]._provider == "anthropic"
        # org_budget propagates as __org__ sentinel on every engine.
        assert clients["oai-bot"]._engine.org_budget is not None
        assert clients["oai-bot"]._engine.org_budget.shared_name == "__org__"
        assert clients["ant-bot"]._engine.org_budget is not None
        assert clients["ant-bot"]._engine.org_budget.shared_name == "__org__"


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------


class TestProviderDispatch:
    @patch("agentkavach.client.AgentKavach._get_openai_client")
    def test_openai_dispatch(self, mock_get):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response()
        mock_get.return_value = mock_client

        cg = _client(provider="openai")
        cg._call_provider({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
        mock_client.chat.completions.create.assert_called_once()

    @patch("agentkavach.client.AgentKavach._get_anthropic_client")
    def test_anthropic_dispatch(self, mock_get):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_anthropic_response()
        mock_get.return_value = mock_client

        cg = _client(provider="anthropic")
        cg._call_provider(
            {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
        mock_client.messages.create.assert_called_once()

    @patch("agentkavach.client.AgentKavach._get_google_client")
    def test_google_dispatch(self, mock_get):
        # Phase 45: new google-genai surface — ``client.models.generate_content(
        # model=..., contents=...)`` instead of ``GenerativeModel(model).generate_content(...)``.
        mock_client = _mock_google_client(_mock_google_response())
        mock_get.return_value = mock_client

        cg = _client(provider="google")
        cg._call_provider({"model": "gemini-2.0-flash", "contents": "hello"})
        mock_client.models.generate_content.assert_called_once()
        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        assert call_kwargs["model"] == "gemini-2.0-flash"
        assert call_kwargs["contents"] == "hello"


# ---------------------------------------------------------------------------
# Threshold alerts across providers
# ---------------------------------------------------------------------------


class TestCrossProviderAlerts:
    @patch("agentkavach.client.AgentKavach._get_anthropic_client")
    def test_threshold_fires_for_anthropic(self, mock_get):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_anthropic_response(
            input_tokens=10000,
            output_tokens=5000,
        )
        mock_get.return_value = mock_client

        kills = []
        cg = _client(
            provider="anthropic",
            budget=Budget.daily(0.10),
            on_kill=lambda: kills.append(True),
        )
        # Exhaust budget.
        cg._engine.post_flight("claude-3-5-sonnet-20241022", 50000, 50000)
        cg._engine.check_thresholds()

        assert cg._engine._killed is True
        assert len(kills) == 1

    @patch("agentkavach.client.AgentKavach._get_google_client")
    def test_threshold_fires_for_google(self, mock_get):
        mock_get.return_value = _mock_google_client(
            _mock_google_response(prompt_token_count=10000, candidates_token_count=5000)
        )

        kills = []
        cg = _client(
            provider="google",
            budget=Budget.daily(0.001),
            on_kill=lambda: kills.append(True),
        )
        cg._engine.post_flight("gemini-2.0-flash", 100000, 100000)
        cg._engine.check_thresholds()

        assert cg._engine._killed is True


# ---------------------------------------------------------------------------
# Mistral init
# ---------------------------------------------------------------------------


def _mock_mistral_response(
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


class TestMistralInit:
    def test_mistral_provider(self):
        cg = _client(provider="mistral")
        assert cg._provider == "mistral"

    def test_mistral_llm_key_not_read_from_env(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "mistral-key-from-env")
        with pytest.raises(ValueError, match="llm_key is required"):
            AgentKavach(api_key="ak_test", provider="mistral")


# ---------------------------------------------------------------------------
# Mistral calls
# ---------------------------------------------------------------------------


class TestMistralCalls:
    @patch("agentkavach.client.AgentKavach._get_mistral_client")
    def test_create(self, mock_get):
        mock_client = MagicMock()
        mock_client.chat.complete.return_value = _mock_mistral_response()
        mock_get.return_value = mock_client

        cg = _client(provider="mistral")
        response = cg.create(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert response.model == "mistral-large-latest"
        assert cg.spent > 0

    @patch("agentkavach.client.AgentKavach._get_mistral_client")
    def test_buffers_mistral_event(self, mock_get, tmp_path):
        mock_client = MagicMock()
        mock_client.chat.complete.return_value = _mock_mistral_response()
        mock_get.return_value = mock_client

        cg = _client(provider="mistral", buffer_path=str(tmp_path / "buf.jsonl"))
        cg.create(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": "hi"}],
        )
        events = cg._buffer.read_all()
        assert len(events) == 1
        assert events[0]["provider"] == "mistral"
        assert events[0]["model"] == "mistral-large-latest"

    @patch("agentkavach.client.AgentKavach._get_mistral_client")
    def test_budget_exceeded_raises(self, mock_get):
        cg = _client(provider="mistral", budget=Budget.daily(0.001))
        cg._engine.post_flight("mistral-large-latest", 0, 100_000)
        cg._engine.check_thresholds()

        with pytest.raises(BudgetExceededError):
            cg.create(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": "x" * 5000}],
            )


# ---------------------------------------------------------------------------
# Mistral dispatch
# ---------------------------------------------------------------------------


class TestMistralDispatch:
    @patch("agentkavach.client.AgentKavach._get_mistral_client")
    def test_mistral_dispatch(self, mock_get):
        mock_client = MagicMock()
        mock_client.chat.complete.return_value = _mock_mistral_response()
        mock_get.return_value = mock_client

        cg = _client(provider="mistral")
        cg._call_provider(
            {"model": "mistral-large-latest", "messages": [{"role": "user", "content": "hi"}]}
        )
        mock_client.chat.complete.assert_called_once()
