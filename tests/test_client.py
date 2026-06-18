"""Unit tests for agentkavach.client: AgentKavach wrapper and YAML config."""

from __future__ import annotations

import textwrap
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agentkavach.budget import Budget
from agentkavach.client import (
    AgentKavach,
    _build_channel_configs_from_yaml,
    _parse_alerts,
    _parse_budget,
    _parse_channel_defs,
    _parse_guardrails,
    _resolve_channel_creds,
)
from agentkavach.exceptions import BudgetExceededError

# ---------------------------------------------------------------------------
# Helpers
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


def _client(**overrides) -> AgentKavach:
    """Create a AgentKavach with sensible test defaults (no real API keys)."""
    defaults = dict(
        api_key="ak_test",  # skip OTel setup
        llm_key="sk-test-not-real",
        budget=Budget.daily(limit=10.0),
        agent_name="test-agent",
    )
    defaults.update(overrides)
    return AgentKavach(**defaults)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestAgentKavachInit:
    def test_default_budget(self):
        cg = AgentKavach(api_key="ak_test", llm_key="sk-test")
        assert cg._budget.limit == 100.0  # default

    def test_custom_budget(self):
        cg = _client(budget=Budget.daily(limit=25.0))
        assert cg._budget.limit == 25.0

    def test_agent_name(self):
        cg = _client(agent_name="my-bot")
        assert cg._agent_name == "my-bot"

    def test_missing_api_key_raises(self):
        with pytest.raises(ValueError, match="api_key is required"):
            AgentKavach(llm_key="sk-test")

    def test_missing_llm_key_raises(self):
        with pytest.raises(ValueError, match="llm_key is required"):
            AgentKavach(api_key="ak_test")

    def test_llm_key_not_read_from_env(self, monkeypatch):
        # An env var alone must not satisfy llm_key — the SDK never reads it.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        with pytest.raises(ValueError, match="llm_key is required"):
            AgentKavach(api_key="ak_test")

    def test_chat_namespace(self):
        cg = _client()
        assert hasattr(cg, "chat")
        assert hasattr(cg.chat, "completions")
        assert hasattr(cg.chat.completions, "create")

    def test_create_method(self):
        cg = _client()
        assert callable(cg.create)

    def test_llm_key_precedence(self):
        cg = AgentKavach(api_key="ak_test", llm_key="sk-primary", openai_api_key="sk-legacy")
        assert cg._llm_key == "sk-primary"


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestAgentKavachProperties:
    def test_spent(self):
        cg = _client()
        assert cg.spent == 0.0

    def test_remaining(self):
        cg = _client(budget=Budget.daily(limit=50.0))
        assert cg.remaining == 50.0

    def test_engine_exposed(self):
        cg = _client()
        assert cg.engine is cg._engine


# ---------------------------------------------------------------------------
# _safe_call
# ---------------------------------------------------------------------------


class TestSafeCall:
    @patch("agentkavach.client.AgentKavach._get_openai_client")
    def test_normal_call(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response()
        mock_get_client.return_value = mock_client

        cg = _client()
        response = cg.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert response.model == "gpt-4o"
        assert cg.spent > 0

    @patch("agentkavach.client.AgentKavach._get_openai_client")
    def test_budget_exceeded_raises(self, mock_get_client):
        cg = _client(budget=Budget.daily(limit=0.001))
        cg._engine.post_flight("gpt-4o", input_tokens=0, output_tokens=100_000)
        cg._engine.check_thresholds()

        with pytest.raises(BudgetExceededError):
            cg.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "x" * 5000}],
            )

    @patch("agentkavach.client.AgentKavach._get_openai_client")
    def test_streaming_returns_wrapper(self, mock_get_client):
        mock_client = MagicMock()
        mock_stream = iter([_mock_openai_response()])
        mock_client.chat.completions.create.return_value = mock_stream
        mock_get_client.return_value = mock_client

        cg = _client()
        result = cg.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        from agentkavach.stream import StreamWrapper

        assert isinstance(result, StreamWrapper)

    @patch("agentkavach.client.AgentKavach._get_openai_client")
    def test_post_flight_buffers_event(self, mock_get_client, tmp_path):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response()
        mock_get_client.return_value = mock_client

        cg = _client(buffer_path=str(tmp_path / "buf.jsonl"))
        cg.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        )
        events = cg._buffer.read_all()
        assert len(events) == 1
        # Phase 35: buffer events use server's IngestEvent field names
        # so buffer-replay POSTs validate. Pre-fix, `agent` / `tokens_in`
        # caused every replay to 422.
        assert events[0]["agent_name"] == "test-agent"
        assert events[0]["model"] == "gpt-4o"
        assert "idempotency_key" in events[0]

    @patch("agentkavach.client.AgentKavach._get_openai_client")
    def test_create_method_works(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response()
        mock_get_client.return_value = mock_client

        cg = _client()
        response = cg.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert response.model == "gpt-4o"

    @patch("agentkavach.client.AgentKavach._get_openai_client")
    def test_engine_runs_even_without_explicit_budget(self, mock_get_client):
        """With keys present, the engine always runs (no passthrough mode)."""
        mock_client = MagicMock()
        mock_response = _mock_openai_response()
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        # No explicit budget → defaults to $100/day, enforcement still active.
        cg = AgentKavach(api_key="ak_test", llm_key="sk-test-not-real", agent_name="test-agent")
        cg._engine.pre_flight = MagicMock()
        cg._engine.post_flight = MagicMock()

        response = cg.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert response.model == "gpt-4o"
        # Engine runs on every call now that passthrough mode is gone.
        cg._engine.pre_flight.assert_called()
        cg._engine.post_flight.assert_called()

    @patch("agentkavach.client.AgentKavach._get_openai_client")
    def test_spend_recorded_with_default_budget(self, mock_get_client):
        """A call records spend even when no explicit budget is set."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response()
        mock_get_client.return_value = mock_client

        cg = AgentKavach(api_key="ak_test", llm_key="sk-test-not-real", agent_name="test-agent")
        cg.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
        assert cg.spent > 0

    @patch("agentkavach.client.AgentKavach._get_openai_client")
    def test_enforcement_with_explicit_budget(self, mock_get_client):
        """Explicit budget → enforcement active, spend recorded."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response()
        mock_get_client.return_value = mock_client

        cg = _client(api_key="ak_test", budget=Budget.daily(limit=10.0))
        cg.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
        assert cg.spent > 0  # Budget enforcement active, spend recorded


# ---------------------------------------------------------------------------
# save_prompts
# ---------------------------------------------------------------------------


class TestSavePrompts:
    def test_save_prompts_default_false(self):
        cg = _client()
        assert cg.save_prompts is False

    def test_save_prompts_true(self):
        cg = _client(save_prompts=True)
        assert cg.save_prompts is True

    @patch("agentkavach.client.AgentKavach._get_openai_client")
    def test_prompt_not_in_buffer_when_disabled(self, mock_get_client, tmp_path):
        # Phase 59 (B2): when an api_key is present the OTel exporter
        # becomes the canonical ingest transport and the disk buffer
        # write in ``_post_flight`` is intentionally skipped to avoid
        # double-counting. This test asserts on the buffer contents, so
        # it must run in no-tracer mode (api_key="ak_test") to keep the buffer
        # as the event sink.
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response()
        mock_get_client.return_value = mock_client

        cg = _client(
            api_key="ak_test",
            save_prompts=False,
            buffer_path=str(tmp_path / "buf.jsonl"),
        )
        cg.create(model="gpt-4o", messages=[{"role": "user", "content": "secret prompt"}])
        events = cg._buffer.read_all()
        assert len(events) == 1
        assert "prompt" not in events[0]

    @patch("agentkavach.client.AgentKavach._get_openai_client")
    def test_prompt_in_buffer_when_enabled(self, mock_get_client, tmp_path):
        # Phase 59 (B2): see test_prompt_not_in_buffer_when_disabled —
        # api_key="ak_test" is required so the buffer remains the event sink.
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response()
        mock_get_client.return_value = mock_client

        cg = _client(
            api_key="ak_test",
            save_prompts=True,
            buffer_path=str(tmp_path / "buf.jsonl"),
        )
        cg.create(model="gpt-4o", messages=[{"role": "user", "content": "test prompt text"}])
        events = cg._buffer.read_all()
        assert len(events) == 1
        assert events[0]["prompt"] == "test prompt text"


# ---------------------------------------------------------------------------
# Phase 53: prompt length cap + save_prompts privacy warning
# ---------------------------------------------------------------------------


class TestPromptPrivacyHardening:
    """Length cap + once-per-process privacy warning."""

    @patch("agentkavach.client.AgentKavach._get_openai_client")
    def test_long_prompt_truncated_in_buffer(self, mock_get_client, tmp_path):
        """Buffered prompt must be capped to 2048 chars with a suffix marker."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response()
        mock_get_client.return_value = mock_client

        # Phase 59 (B2): api_key="ak_test" keeps OTel off so the buffer
        # remains the event sink — see ``_post_flight`` tracer guard.
        long_prompt = "y" * 5000
        cg = _client(
            api_key="ak_test",
            save_prompts=True,
            buffer_path=str(tmp_path / "buf.jsonl"),
        )
        cg.create(model="gpt-4o", messages=[{"role": "user", "content": long_prompt}])

        events = cg._buffer.read_all()
        assert len(events) == 1
        stored = events[0]["prompt"]
        # Total length must equal the 2048-char cap, suffix included.
        assert len(stored) == 2048
        assert stored.endswith("... [truncated]")
        # The head must come from the original prompt (i.e. truncation
        # didn't accidentally replace content with the marker).
        assert stored.startswith("y" * 100)

    @patch("agentkavach.client.AgentKavach._get_openai_client")
    def test_short_prompt_not_modified(self, mock_get_client, tmp_path):
        """Prompts at or below the cap pass through unchanged."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response()
        mock_get_client.return_value = mock_client

        # Phase 59 (B2): api_key="ak_test" keeps OTel off so the buffer
        # remains the event sink.
        cg = _client(
            api_key="ak_test",
            save_prompts=True,
            buffer_path=str(tmp_path / "buf.jsonl"),
        )
        msg = "short and sweet"
        cg.create(model="gpt-4o", messages=[{"role": "user", "content": msg}])

        events = cg._buffer.read_all()
        assert events[0]["prompt"] == msg
        assert "[truncated]" not in events[0]["prompt"]

    def test_save_prompts_true_emits_warning(self, caplog):
        """save_prompts=True must log a privacy warning at WARNING level."""
        import logging

        # Reset the class-level latch so the warning fires for this test.
        AgentKavach._save_prompts_warning_emitted = False

        with caplog.at_level(logging.WARNING, logger="agentkavach.client"):
            _client(save_prompts=True)

        messages = [r.getMessage() for r in caplog.records]
        assert any("save_prompts=True" in m for m in messages), messages
        assert any("30" in m and "days" in m for m in messages), messages
        assert any("privacy" in m.lower() for m in messages), messages

    def test_save_prompts_false_emits_no_warning(self, caplog):
        """Default save_prompts=False must NOT emit the privacy warning."""
        import logging

        AgentKavach._save_prompts_warning_emitted = False

        with caplog.at_level(logging.WARNING, logger="agentkavach.client"):
            _client(save_prompts=False)

        messages = [r.getMessage() for r in caplog.records]
        assert not any("save_prompts=True" in m for m in messages), messages

    def test_save_prompts_warning_only_once_per_process(self, caplog):
        """Constructing many opted-in clients should not spam the operator."""
        import logging

        AgentKavach._save_prompts_warning_emitted = False

        with caplog.at_level(logging.WARNING, logger="agentkavach.client"):
            _client(save_prompts=True)
            _client(save_prompts=True)
            _client(save_prompts=True)

        warnings = [r for r in caplog.records if "save_prompts=True" in r.getMessage()]
        assert len(warnings) == 1, [r.getMessage() for r in warnings]


# ---------------------------------------------------------------------------
# YAML config
# ---------------------------------------------------------------------------


class TestFromYaml:
    def test_single_agent(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            textwrap.dedent("""\
            agents:
              my-bot:
                budget:
                  type: daily
                  limit: 50
        """)
        )
        clients = AgentKavach.from_yaml(str(config), api_key="ak_test", llm_key="sk-test")
        assert "my-bot" in clients
        assert clients["my-bot"]._budget.limit == 50.0

    def test_select_single_agent(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            textwrap.dedent("""\
            agents:
              bot-a:
                budget: { type: daily, limit: 25 }
              bot-b:
                budget: { type: daily, limit: 75 }
        """)
        )
        client = AgentKavach.from_yaml(
            str(config), api_key="ak_test", llm_key="sk-test", agent="bot-a"
        )
        assert isinstance(client, AgentKavach)
        assert client._budget.limit == 25.0

    def test_missing_agent_raises(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text("agents:\n  bot-a:\n    budget: { type: daily, limit: 10 }\n")
        with pytest.raises(KeyError, match="nonexistent"):
            AgentKavach.from_yaml(
                str(config), api_key="ak_test", llm_key="sk-test", agent="nonexistent"
            )

    def test_defaults_applied(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            textwrap.dedent("""\
            defaults:
              budget:
                type: daily
                limit: 100
            agents:
              bot-a: {}
        """)
        )
        clients = AgentKavach.from_yaml(str(config), api_key="ak_test", llm_key="sk-test")
        assert clients["bot-a"]._budget.limit == 100.0

    def test_agent_overrides_defaults(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            textwrap.dedent("""\
            defaults:
              budget:
                type: daily
                limit: 100
            agents:
              bot-a:
                budget:
                  type: daily
                  limit: 25
        """)
        )
        clients = AgentKavach.from_yaml(str(config), api_key="ak_test", llm_key="sk-test")
        assert clients["bot-a"]._budget.limit == 25.0

    def test_budget_default_keyword(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            textwrap.dedent("""\
            defaults:
              budget:
                type: daily
                limit: 200
            agents:
              bot-a:
                budget: default
        """)
        )
        clients = AgentKavach.from_yaml(str(config), api_key="ak_test", llm_key="sk-test")
        assert clients["bot-a"]._budget.limit == 200.0

    def test_alerts_from_yaml(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            textwrap.dedent("""\
            agents:
              bot-a:
                budget: { type: daily, limit: 50 }
                alerts:
                  - { threshold: 0.50, channels: [email] }
                  - { threshold: 1.0, channels: [pagerduty, kill] }
        """)
        )
        clients = AgentKavach.from_yaml(str(config), api_key="ak_test", llm_key="sk-test")
        rules = clients["bot-a"]._dispatcher.rules
        assert len(rules) == 2
        assert rules[0].threshold == 0.50
        assert "kill" in rules[1].channels

    def test_yaml_org_budget_key_parses_correctly(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            textwrap.dedent("""\
            org_budget:
              limit: 200
              period: daily
            agents:
              bot-a:
                budget: { type: daily, limit: 50 }
              bot-b:
                budget: { type: daily, limit: 50 }
        """)
        )
        clients = AgentKavach.from_yaml(str(config), api_key="ak_test", llm_key="sk-test")
        # org_budget propagates as __org__ sentinel on every engine.
        assert clients["bot-a"]._engine.org_budget is not None
        assert clients["bot-a"]._engine.org_budget.shared_name == "__org__"
        assert clients["bot-a"]._engine.org_budget.limit == 200.0
        assert clients["bot-b"]._engine.org_budget is not None
        assert clients["bot-b"]._engine.org_budget.shared_name == "__org__"

    def test_yaml_shared_budgets_key_no_longer_recognized(self, tmp_path):
        """The legacy `shared_budgets:` top-level key is now ignored;
        agents referencing `budget: { shared: ... }` get rejected."""
        config = tmp_path / "config.yaml"
        config.write_text(
            textwrap.dedent("""\
            shared_budgets:
              team-daily:
                limit: 200
                period: daily
            agents:
              bot-a:
                budget: { type: daily, limit: 50 }
        """)
        )
        clients = AgentKavach.from_yaml(str(config), api_key="ak_test", llm_key="sk-test")
        # The shared_budgets block is silently ignored — no org budget.
        assert clients["bot-a"]._engine.org_budget is None
        # And bot-a falls back to its own per-agent budget.
        assert clients["bot-a"]._budget.limit == 50.0

    def test_invalid_yaml_raises(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text("just a string")
        with pytest.raises(ValueError, match="Invalid YAML"):
            AgentKavach.from_yaml(str(config), api_key="ak_test", llm_key="sk-test")

    def test_monthly_budget(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            textwrap.dedent("""\
            agents:
              bot-a:
                budget: { type: monthly, limit: 500 }
        """)
        )
        clients = AgentKavach.from_yaml(str(config), api_key="ak_test", llm_key="sk-test")
        from agentkavach.budget import Period

        assert clients["bot-a"]._budget.period is Period.MONTHLY

    def test_invalid_budget_type_raises(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            textwrap.dedent("""\
            agents:
              bot-a:
                budget: { type: weekly, limit: 50 }
        """)
        )
        with pytest.raises(ValueError, match="Unknown budget type"):
            AgentKavach.from_yaml(str(config), api_key="ak_test", llm_key="sk-test")

    def test_invalid_channel_in_yaml_raises(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            textwrap.dedent("""\
            agents:
              bot-a:
                budget: { type: daily, limit: 50 }
                alerts:
                  - { threshold: 0.50, channels: [phone] }
        """)
        )
        with pytest.raises(ValueError, match="Unknown channel type"):
            AgentKavach.from_yaml(str(config), api_key="ak_test", llm_key="sk-test")

    def test_save_prompts_in_yaml(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            textwrap.dedent("""\
            agents:
              my-bot:
                save_prompts: true
                budget:
                  type: daily
                  limit: 50
        """)
        )
        clients = AgentKavach.from_yaml(str(config), api_key="ak_test", llm_key="sk-test")
        assert clients["my-bot"].save_prompts is True

    def test_save_prompts_default_false_in_yaml(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            textwrap.dedent("""\
            agents:
              my-bot:
                budget: { type: daily, limit: 50 }
        """)
        )
        clients = AgentKavach.from_yaml(str(config), api_key="ak_test", llm_key="sk-test")
        assert clients["my-bot"].save_prompts is False


# ---------------------------------------------------------------------------
# _parse_budget / _parse_alerts helpers
# ---------------------------------------------------------------------------


class TestParseHelpers:
    def test_parse_budget_empty(self):
        budget = _parse_budget({}, "agent")
        assert budget.limit == 100.0

    def test_parse_budget_daily(self):
        budget = _parse_budget({"type": "daily", "limit": 50}, "agent")
        assert budget.limit == 50.0

    def test_parse_budget_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown budget type"):
            _parse_budget({"type": "hourly", "limit": 10}, "agent")

    def test_parse_budget_default_string(self):
        budget = _parse_budget("default", "agent")
        assert budget.limit == 100.0

    def test_parse_alerts_empty(self):
        assert _parse_alerts([]) == []

    def test_parse_alerts_basic(self):
        rules = _parse_alerts([{"threshold": 0.70, "channels": ["email"]}])
        assert len(rules) == 1
        assert rules[0].threshold == 0.70
        assert rules[0].channels == ("email",)

    def test_parse_alerts_invalid_channel_raises(self):
        with pytest.raises(ValueError, match="Unknown channel type"):
            _parse_alerts([{"threshold": 0.50, "channels": ["sms"]}])

    def test_parse_alerts_legacy_at(self):
        rules = _parse_alerts([{"at": 0.70, "channels": ["email"]}])
        assert rules[0].threshold == 0.70

    def test_parse_alerts_validates_against_channel_defs(self):
        defs = {"slack": {"type": "slack", "webhook_url": "https://x"}}
        rules = _parse_alerts(
            [{"threshold": 0.80, "channels": ["slack"]}],
            channel_defs=defs,
        )
        assert rules[0].channels == ("slack",)

    def test_parse_alerts_rejects_undefined_channel(self):
        defs = {"slack": {"type": "slack", "webhook_url": "https://x"}}
        with pytest.raises(ValueError, match="not defined in the channels section"):
            _parse_alerts(
                [{"threshold": 0.70, "channels": ["email"]}],
                channel_defs=defs,
            )

    def test_parse_alerts_kill_always_allowed(self):
        defs = {"slack": {"type": "slack", "webhook_url": "https://x"}}
        rules = _parse_alerts(
            [{"threshold": 1.0, "channels": ["kill"]}],
            channel_defs=defs,
        )
        assert "kill" in rules[0].channels

    def test_parse_alerts_no_channel_defs_skips_validation(self):
        rules = _parse_alerts(
            [{"threshold": 0.70, "channels": ["email"]}],
            channel_defs=None,
        )
        assert rules[0].channels == ("email",)


# ---------------------------------------------------------------------------
# _parse_channel_defs / _resolve_channel_creds helpers
# ---------------------------------------------------------------------------


class TestParseChannelDefs:
    def test_empty(self):
        assert _parse_channel_defs({}) == {}

    def test_valid_channels(self):
        defs = _parse_channel_defs(
            {
                "slack": {"type": "slack", "webhook_url": "https://hooks.slack.com/x"},
                "email": {"type": "email", "to": "team@acme.com"},
            }
        )
        assert "slack" in defs
        assert "email" in defs

    def test_type_inferred_from_name(self):
        defs = _parse_channel_defs(
            {
                "slack": {"webhook_url": "https://hooks.slack.com/x"},
            }
        )
        assert "slack" in defs

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="unknown type"):
            _parse_channel_defs({"sms": {"type": "sms"}})

    def test_non_dict_config_raises(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            _parse_channel_defs({"slack": "https://hooks.slack.com/x"})


class TestResolveChannelCreds:
    def test_empty(self):
        assert _resolve_channel_creds({}) == {}

    def test_slack(self):
        defs = {"slack": {"type": "slack", "webhook_url": "https://hooks.slack.com/x"}}
        result = _resolve_channel_creds(defs)
        assert result == {"slack_webhook_url": "https://hooks.slack.com/x"}

    def test_email(self):
        defs = {"email": {"type": "email", "to": "team@acme.com", "api_key": "re_test"}}
        result = _resolve_channel_creds(defs)
        assert result == {"alert_email": "team@acme.com", "resend_api_key": "re_test"}

    def test_pagerduty(self):
        defs = {"pagerduty": {"type": "pagerduty", "routing_key": "R0xxx"}}
        result = _resolve_channel_creds(defs)
        assert result == {"pagerduty_routing_key": "R0xxx"}

    def test_webhook_with_secret(self):
        defs = {"webhook": {"type": "webhook", "url": "https://x.com/hook", "secret": "s3c"}}
        result = _resolve_channel_creds(defs)
        assert result == {"webhook_url": "https://x.com/hook", "webhook_secret": "s3c"}

    def test_multiple_channels(self):
        defs = {
            "slack": {"type": "slack", "webhook_url": "https://hooks.slack.com/x"},
            "pagerduty": {"type": "pagerduty", "routing_key": "R0xxx"},
        }
        result = _resolve_channel_creds(defs)
        assert result["slack_webhook_url"] == "https://hooks.slack.com/x"
        assert result["pagerduty_routing_key"] == "R0xxx"


# ---------------------------------------------------------------------------
# YAML with channels section
# ---------------------------------------------------------------------------


class TestYamlChannels:
    @patch("agentkavach.channels.slack.httpx.Client")
    def test_yaml_with_channels_section(self, mock_cls, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            textwrap.dedent("""\
            channels:
              slack:
                type: slack
                webhook_url: https://hooks.slack.com/x
                dispatch: sdk

            agents:
              bot-a:
                budget: { type: daily, limit: 50 }
                alerts:
                  - { threshold: 0.80, channels: [slack] }
                  - { threshold: 1.0, channels: [kill] }
        """)
        )
        clients = AgentKavach.from_yaml(str(config), api_key="ak_test", llm_key="sk-test")
        assert "bot-a" in clients
        # dispatch: sdk → the SDK registers a client-side handler for delivery.
        assert "slack" in clients["bot-a"]._dispatcher._channels

    def test_yaml_channels_rejects_undefined_reference(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            textwrap.dedent("""\
            channels:
              slack:
                type: slack
                webhook_url: https://hooks.slack.com/x

            agents:
              bot-a:
                budget: { type: daily, limit: 50 }
                alerts:
                  - { threshold: 0.70, channels: [email] }
        """)
        )
        with pytest.raises(ValueError, match="not defined in the channels section"):
            AgentKavach.from_yaml(str(config), api_key="ak_test", llm_key="sk-test")

    def test_yaml_without_channels_section_works(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            textwrap.dedent("""\
            agents:
              bot-a:
                budget: { type: daily, limit: 50 }
                alerts:
                  - { threshold: 0.70, channels: [email] }
                  - { threshold: 1.0, channels: [kill] }
        """)
        )
        clients = AgentKavach.from_yaml(str(config), api_key="ak_test", llm_key="sk-test")
        assert "bot-a" in clients
        rules = clients["bot-a"]._dispatcher.rules
        assert len(rules) == 2

    @patch("agentkavach.channels.slack.httpx.Client")
    def test_yaml_slack_with_env_var_interpolation(self, mock_cls, tmp_path, monkeypatch):
        monkeypatch.setenv("AGENTKAVACH_SLACK_WEBHOOK_URL", "https://hooks.slack.com/env-yaml")
        config = tmp_path / "config.yaml"
        config.write_text(
            textwrap.dedent("""\
            channels:
              slack:
                type: slack
                webhook_url: https://hooks.slack.com/env-yaml
                dispatch: sdk

            agents:
              bot-slack:
                budget: { type: daily, limit: 100 }
                alerts:
                  - { threshold: 0.80, channels: [slack] }
        """)
        )
        clients = AgentKavach.from_yaml(str(config), api_key="ak_test", llm_key="sk-test")
        assert "slack" in clients["bot-slack"]._dispatcher._channels

    @patch("agentkavach.channels.slack.httpx.Client")
    def test_yaml_backend_slack_not_registered_client_side(self, mock_cls, tmp_path):
        """Default (backend) dispatch slack produces a rule but no client-side handler.

        The backend delivers it, so the SDK must NOT register a local dispatcher
        channel for it (that would double-deliver).
        """
        config = tmp_path / "config.yaml"
        config.write_text(
            textwrap.dedent("""\
            channels:
              slack:
                type: slack
                webhook_url: https://hooks.slack.com/x

            agents:
              bot-a:
                budget: { type: daily, limit: 50 }
                alerts:
                  - { threshold: 0.80, channels: [slack] }
        """)
        )
        clients = AgentKavach.from_yaml(str(config), api_key="ak_test", llm_key="sk-test")
        dispatcher = clients["bot-a"]._dispatcher
        # Backend-mode → no client-side handler...
        assert "slack" not in dispatcher._channels
        # ...but the rule still exists (so it appears in the synced config).
        assert any("slack" in r.channels for r in dispatcher.rules)

    @patch("agentkavach.channels.slack.httpx.Client")
    def test_yaml_slack_at_multiple_thresholds(self, mock_cls, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            textwrap.dedent("""\
            channels:
              slack:
                type: slack
                webhook_url: https://hooks.slack.com/multi

            agents:
              bot-multi:
                budget: { type: daily, limit: 50 }
                alerts:
                  - { threshold: 0.70, channels: [slack] }
                  - { threshold: 0.90, channels: [slack] }
                  - { threshold: 1.0, channels: [slack, kill] }
        """)
        )
        clients = AgentKavach.from_yaml(str(config), api_key="ak_test", llm_key="sk-test")
        rules = clients["bot-multi"]._dispatcher.rules
        assert len(rules) == 3
        assert all("slack" in r.channels for r in rules)

    @patch("agentkavach.channels.webhook.httpx.Client")
    @patch("agentkavach.channels.slack.httpx.Client")
    def test_yaml_internal_endpoints_docs_example(self, mock_slack, mock_wh, tmp_path):
        """End-to-end load of the Internal Endpoints docs YAML: distinct channel
        names, two webhooks of the same type, mixed backend/sdk dispatch.

        Regression guard for the docs example — it must actually load and route
        correctly through from_yaml.
        """
        config = tmp_path / "config.yaml"
        config.write_text(
            textwrap.dedent("""\
            channels:
              public_hook:
                type: webhook
                url: https://hooks.example.com/public
              internal_slack:
                type: slack
                webhook_url: https://mattermost.internal/hooks/abc
                dispatch: sdk
              internal_hook:
                type: webhook
                url: http://10.0.0.5/budget-alerts
                secret: s3cr3t
                dispatch: sdk

            agents:
              research-bot:
                provider: openai
                budget: { type: daily, limit: 50 }
                alerts:
                  - { threshold: 0.50, channels: [internal_slack] }
                  - { threshold: 0.80, channels: [public_hook] }
                  - { threshold: 0.90, channels: [internal_hook] }
        """)
        )
        clients = AgentKavach.from_yaml(str(config), api_key="ak_test", llm_key="sk-test")
        guard = clients["research-bot"]
        dispatcher = guard._dispatcher
        # sdk channels are delivered client-side → registered + engine fires at
        # their configured thresholds.
        assert "slack" in dispatcher._channels
        assert "webhook" in dispatcher._channels
        assert 0.50 in guard._engine.thresholds
        assert 0.90 in guard._engine.thresholds
        # The public (backend) webhook does NOT add a client-side fire threshold.
        # Build the sync payload and confirm dispatch routing per channel.
        payload = guard._build_sync_payload()
        acs = {(a["channel"], a["threshold_pct"]): a for a in payload["alert_configs"]}
        # public webhook → backend, target synced.
        assert acs[("webhook", 0.80)]["dispatch"] == "backend"
        assert acs[("webhook", 0.80)]["target"] == "https://hooks.example.com/public"
        # internal webhook → sdk, target/secret NOT synced (kept local).
        assert acs[("webhook", 0.90)]["dispatch"] == "sdk"
        assert "target" not in acs[("webhook", 0.90)]
        assert "secret" not in acs[("webhook", 0.90)]
        # internal slack → sdk, url not synced.
        assert acs[("slack", 0.50)]["dispatch"] == "sdk"
        assert "target" not in acs[("slack", 0.50)]


# ---------------------------------------------------------------------------
# _build_channel_configs_from_yaml — dispatch-aware ChannelConfig construction
# ---------------------------------------------------------------------------


class TestBuildChannelConfigsFromYaml:
    def test_credentials_and_dispatch_mapped_per_channel(self):
        channel_defs = {
            "slack": {"webhook_url": "https://hooks.slack/x", "dispatch": "sdk"},
            "pagerduty": {"routing_key": "R0xxx"},
            "email": {"to": "ops@acme.com"},
        }
        alerts = [
            {"threshold": 0.5, "channels": ["slack"]},
            {"threshold": 0.8, "channels": ["pagerduty", "email"]},
        ]
        out = _build_channel_configs_from_yaml(alerts, channel_defs)
        by = {(c.channel_type, c.threshold): c for c in out}
        assert by[("slack", 0.5)].webhook_url == "https://hooks.slack/x"
        assert by[("slack", 0.5)].dispatch == "sdk"
        assert by[("pagerduty", 0.8)].routing_key == "R0xxx"
        assert by[("pagerduty", 0.8)].dispatch == "backend"  # default
        assert by[("email", 0.8)].to == "ops@acme.com"

    def test_webhook_secret_and_email_api_key_carried(self):
        channel_defs = {
            "webhook": {
                "url": "http://10.0.0.5/hook",
                "secret": "sign-me",
                "dispatch": "sdk",
            },
            "email": {"to": "ops@acme.com", "api_key": "re_legacy"},
        }
        alerts = [{"threshold": 0.9, "channels": ["webhook", "email"]}]
        out = _build_channel_configs_from_yaml(alerts, channel_defs)
        webhook = next(c for c in out if c.channel_type == "webhook")
        email = next(c for c in out if c.channel_type == "email")
        assert webhook.secret == "sign-me"
        assert webhook.dispatch == "sdk"
        assert email.api_key == "re_legacy"

    def test_action_kill_appends_kill_channel(self):
        # Legacy `action: kill` shorthand → a kill ChannelConfig is appended.
        channel_defs = {"slack": {"webhook_url": "https://hooks.slack/x"}}
        alerts = [{"threshold": 1.0, "channels": ["slack"], "action": "kill"}]
        out = _build_channel_configs_from_yaml(alerts, channel_defs)
        types = {c.channel_type for c in out}
        assert types == {"slack", "kill"}

    def test_budget_type_carried_to_channel_and_kill(self):
        # A YAML alert can target a non-cost dimension; budget_type must reach
        # both the channel ChannelConfig and the kill ChannelConfig.
        channel_defs = {"team_slack": {"type": "slack", "webhook_url": "https://hooks/x"}}
        alerts = [
            {"threshold": 0.8, "channels": ["team_slack"], "budget_type": "tokens_total"},
            {"threshold": 1.0, "channels": ["kill"], "budget_type": "duration"},
        ]
        out = _build_channel_configs_from_yaml(alerts, channel_defs)
        slack = next(c for c in out if c.channel_type == "slack")
        kill = next(c for c in out if c.channel_type == "kill")
        assert slack.budget_type == "tokens_total"
        assert kill.budget_type == "duration"

    def test_budget_type_defaults_cost(self):
        channel_defs = {"team_slack": {"type": "slack", "webhook_url": "https://hooks/x"}}
        out = _build_channel_configs_from_yaml(
            [{"threshold": 0.8, "channels": ["team_slack"]}], channel_defs
        )
        assert out[0].budget_type == "cost"

    def test_template_passed_through(self):
        channel_defs = {"webhook": {"url": "https://hooks/x"}}
        tmpl = {"text": "custom {agent_name}"}
        alerts = [{"threshold": 0.7, "channels": ["webhook"], "template": tmpl}]
        out = _build_channel_configs_from_yaml(alerts, channel_defs)
        assert out[0].template == tmpl

    def test_at_alias_for_threshold(self):
        # `at:` is accepted as an alias for `threshold:`.
        channel_defs = {"slack": {"webhook_url": "https://hooks/x"}}
        alerts = [{"at": 0.6, "channels": ["slack"]}]
        out = _build_channel_configs_from_yaml(alerts, channel_defs)
        assert out[0].threshold == 0.6

    def test_arbitrary_names_resolve_type_and_dispatch(self):
        # Mirrors the Internal Endpoints docs example: a public webhook and an
        # internal webhook declared under DISTINCT names, same `type: webhook`,
        # with independent dispatch modes.
        channel_defs = {
            "public_hook": {"type": "webhook", "url": "https://hooks.example.com/public"},
            "internal_slack": {
                "type": "slack",
                "webhook_url": "https://mattermost.internal/hooks/abc",
                "dispatch": "sdk",
            },
            "internal_hook": {
                "type": "webhook",
                "url": "http://10.0.0.5/budget-alerts",
                "secret": "s3cr3t",
                "dispatch": "sdk",
            },
        }
        alerts = [
            {"threshold": 0.5, "channels": ["internal_slack"]},
            {"threshold": 0.8, "channels": ["public_hook"]},
            {"threshold": 0.9, "channels": ["internal_hook"]},
        ]
        out = _build_channel_configs_from_yaml(alerts, channel_defs)
        by = {c.threshold: c for c in out}
        # public webhook → backend
        assert by[0.8].channel_type == "webhook"
        assert by[0.8].url == "https://hooks.example.com/public"
        assert by[0.8].dispatch == "backend"
        # internal slack → sdk
        assert by[0.5].channel_type == "slack"
        assert by[0.5].dispatch == "sdk"
        # internal webhook → sdk + secret
        assert by[0.9].channel_type == "webhook"
        assert by[0.9].url == "http://10.0.0.5/budget-alerts"
        assert by[0.9].secret == "s3cr3t"
        assert by[0.9].dispatch == "sdk"


# ---------------------------------------------------------------------------
# YAML guardrails (per-run caps via from_yaml)
# ---------------------------------------------------------------------------


class TestYamlGuardrails:
    def test_parse_guardrails_all_keys_typed(self):
        out = _parse_guardrails(
            {
                "max_tokens_per_run": "1000",
                "max_calls_per_run": "5",
                "loop_threshold": "4",
                "max_runtime_seconds": "15",
                "detect_loops": True,
            },
            "bot",
        )
        assert out == {
            "max_tokens_per_run": 1000,
            "max_calls_per_run": 5,
            "loop_threshold": 4,
            "max_runtime_seconds": 15.0,
            "detect_loops": True,
        }
        assert isinstance(out["max_tokens_per_run"], int)
        assert isinstance(out["max_runtime_seconds"], float)

    def test_parse_guardrails_empty(self):
        assert _parse_guardrails({}, "bot") == {}
        assert _parse_guardrails(None, "bot") == {}

    def test_parse_guardrails_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown guardrail key"):
            _parse_guardrails({"max_tokens": 100}, "bot")

    def test_parse_guardrails_non_mapping_raises(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            _parse_guardrails([1, 2], "bot")

    def test_from_yaml_applies_guardrails_to_engine(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            textwrap.dedent("""\
            agents:
              guard-bot:
                budget: { type: daily, limit: 50 }
                guardrails:
                  max_tokens_per_run: 1000
                  max_calls_per_run: 5
                  max_runtime_seconds: 15
                  detect_loops: true
                  loop_threshold: 3
        """)
        )
        guard = AgentKavach.from_yaml(
            str(config), api_key="ak_test", llm_key="sk-test", agent="guard-bot"
        )
        eng = guard._engine
        assert eng.max_tokens_per_run == 1000
        assert eng.max_calls_per_run == 5
        assert eng.max_runtime_seconds == 15.0
        assert eng.detect_loops is True
        assert eng.loop_threshold == 3

    def test_from_yaml_no_guardrails_leaves_defaults(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            textwrap.dedent("""\
            agents:
              plain-bot:
                budget: { type: daily, limit: 50 }
        """)
        )
        guard = AgentKavach.from_yaml(
            str(config), api_key="ak_test", llm_key="sk-test", agent="plain-bot"
        )
        assert guard._engine.max_tokens_per_run is None
        assert guard._engine.max_calls_per_run is None
        assert guard._engine.detect_loops is False


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


class TestShutdown:
    def test_shutdown_without_tracer(self):
        cg = _client()
        cg.shutdown()  # should not raise

    def test_shutdown_with_tracer(self):
        cg = _client()
        mock_provider = MagicMock()
        cg._tracer_provider = mock_provider
        cg.shutdown()
        mock_provider.shutdown.assert_called_once()


# ---------------------------------------------------------------------------
# Phase 33: _post_flight repairs missing usage.model from requested_model
# ---------------------------------------------------------------------------


class TestPostFlightModelRepair:
    """Regression for the 2026-05-25 Gemini-cost-as-$0 bug.

    Some provider SDKs (notably google-genai) return immutable
    response objects where ``resp.model = name`` silently fails. The
    parser then reads ``response.model`` as ``None`` → defaults to
    ``"unknown"`` → pricing lookup misses → cost = $0 → budgets never
    trip. ``_post_flight`` now patches ``usage.model`` from the
    ``requested_model`` kwarg.
    """

    def _engine_post_flight_args(self):
        captured = {}

        def _capture(**kw):
            captured.update(kw)

        return captured, _capture

    def test_unknown_model_repaired_with_requested(self, tmp_path):
        # End-to-end: simulate a Gemini-style immutable response that
        # reports model = "unknown" and confirm the buffered event has
        # the real model + non-zero cost after _post_flight runs.
        cg = _client(buffer_path=str(tmp_path / "buf.jsonl"))
        cg._provider = "google"

        response = MagicMock()
        response.usage_metadata = MagicMock(prompt_token_count=1000, candidates_token_count=1000)
        response.model = "unknown"
        # tool extraction expects an iterable; make it tidy
        response.choices = []

        cg._post_flight(response, requested_model="gemini-2.5-flash")

        events = cg._buffer.read_all()
        assert len(events) == 1
        assert events[0]["model"] == "gemini-2.5-flash"
        assert events[0]["cost"] > 0, "After model repair, pricing lookup should hit and cost > 0."

    def test_known_response_model_replaced_by_requested(self, tmp_path):
        # Phase 86: even when the response carries a real model, prefer
        # the requested name. Pre-Phase-86 the provider echo won, which
        # produced two rows in the dashboard's Model Breakdown for the
        # same logical model (alias vs versioned snapshot — e.g.
        # ``gpt-4o-mini`` vs ``gpt-4o-mini-2024-07-18``). The customer
        # wrote the requested name in their code; that's what they
        # expect to see.
        cg = _client(buffer_path=str(tmp_path / "buf.jsonl"))
        cg._provider = "google"

        response = MagicMock()
        response.usage_metadata = MagicMock(prompt_token_count=1000, candidates_token_count=1000)
        response.model = "gemini-1.5-flash"
        response.choices = []

        cg._post_flight(response, requested_model="gemini-2.5-flash")

        events = cg._buffer.read_all()
        assert len(events) == 1
        # Phase 86: requested wins over provider echo.
        assert events[0]["model"] == "gemini-2.5-flash"
