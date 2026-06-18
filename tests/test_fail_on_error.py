"""Tests for the fail_on_error parameter."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from agentkavach import AgentKavach, Budget
from agentkavach.exceptions import BudgetExceededError


class TestFailOnErrorDefault:
    """Default (fail_on_error=False): fail-open behavior."""

    def test_default_is_false(self):
        """fail_on_error defaults to False."""
        guard = AgentKavach(
            provider="openai",
            api_key="ak_test",
            llm_key="sk-test",
            agent_name="test",
            budget=Budget.daily(50),
        )
        assert guard._fail_on_error is False

    def test_pre_flight_error_swallowed_by_default(self):
        """Internal pre-flight errors are swallowed in fail-open mode."""
        kill_fn = MagicMock()
        guard = AgentKavach(
            provider="openai",
            api_key="ak_test",
            llm_key="sk-test",
            agent_name="test",
            budget=Budget.daily(50),
            on_kill=kill_fn,
        )
        # Force pre_flight to raise a non-budget error
        guard._engine.pre_flight = MagicMock(side_effect=RuntimeError("internal error"))
        # Mock the provider call
        guard._call_provider = MagicMock(
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content="hi"))],
                usage=MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="gpt-4o",
            )
        )
        guard._post_flight = MagicMock()

        # Should NOT raise — fail-open
        result = guard._safe_call(
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert result is not None
        kill_fn.assert_not_called()

    def test_post_flight_error_swallowed_by_default(self):
        """Internal post-flight errors are swallowed in fail-open mode."""
        kill_fn = MagicMock()
        guard = AgentKavach(
            provider="openai",
            api_key="ak_test",
            llm_key="sk-test",
            agent_name="test",
            budget=Budget.daily(50),
            on_kill=kill_fn,
        )
        guard._engine.pre_flight = MagicMock()
        guard._call_provider = MagicMock(
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content="hi"))],
                usage=MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="gpt-4o",
            )
        )
        guard._post_flight = MagicMock(side_effect=RuntimeError("post error"))

        # Should NOT raise — fail-open
        result = guard._safe_call(
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert result is not None
        kill_fn.assert_not_called()


class TestFailOnErrorTrue:
    """fail_on_error=True: fail-closed behavior."""

    def test_pre_flight_error_raises_and_calls_on_kill(self):
        """Pre-flight internal error raises and calls on_kill when fail_on_error=True."""
        kill_fn = MagicMock()
        guard = AgentKavach(
            provider="openai",
            api_key="ak_test",
            llm_key="sk-test",
            agent_name="test",
            budget=Budget.daily(50),
            on_kill=kill_fn,
            fail_on_error=True,
        )
        guard._engine.pre_flight = MagicMock(side_effect=RuntimeError("engine error"))
        guard._call_provider = MagicMock()

        with pytest.raises(RuntimeError, match="engine error"):
            guard._safe_call({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})

        kill_fn.assert_called_once()

    def test_post_flight_error_raises_and_calls_on_kill(self):
        """Post-flight internal error raises and calls on_kill when fail_on_error=True."""
        kill_fn = MagicMock()
        guard = AgentKavach(
            provider="openai",
            api_key="ak_test",
            llm_key="sk-test",
            agent_name="test",
            budget=Budget.daily(50),
            on_kill=kill_fn,
            fail_on_error=True,
        )
        guard._engine.pre_flight = MagicMock()
        guard._call_provider = MagicMock(
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content="hi"))],
                usage=MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="gpt-4o",
            )
        )
        guard._post_flight = MagicMock(side_effect=RuntimeError("post error"))

        with pytest.raises(RuntimeError, match="post error"):
            guard._safe_call({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})

        kill_fn.assert_called_once()

    def test_budget_exceeded_still_raises_regardless(self):
        """BudgetExceededError always propagates, regardless of fail_on_error."""
        guard = AgentKavach(
            provider="openai",
            api_key="ak_test",
            llm_key="sk-test",
            agent_name="test",
            budget=Budget.daily(50),
            fail_on_error=False,
        )
        guard._engine.pre_flight = MagicMock(side_effect=BudgetExceededError("over budget"))
        guard._engine.check_thresholds = MagicMock(return_value=[])

        with pytest.raises(BudgetExceededError):
            guard._safe_call({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})

    def test_fail_on_error_without_on_kill_still_raises(self):
        """fail_on_error=True without on_kill still raises the error."""
        guard = AgentKavach(
            provider="openai",
            api_key="ak_test",
            llm_key="sk-test",
            agent_name="test",
            budget=Budget.daily(50),
            fail_on_error=True,
            # No on_kill
        )
        guard._engine.pre_flight = MagicMock(side_effect=RuntimeError("no kill"))
        guard._call_provider = MagicMock()

        with pytest.raises(RuntimeError, match="no kill"):
            guard._safe_call({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})

    def test_on_kill_exception_does_not_mask_original(self):
        """If on_kill itself raises, the original error still propagates."""
        kill_fn = MagicMock(side_effect=Exception("kill failed"))
        guard = AgentKavach(
            provider="openai",
            api_key="ak_test",
            llm_key="sk-test",
            agent_name="test",
            budget=Budget.daily(50),
            on_kill=kill_fn,
            fail_on_error=True,
        )
        guard._engine.pre_flight = MagicMock(side_effect=RuntimeError("original"))
        guard._call_provider = MagicMock()

        with pytest.raises(RuntimeError, match="original"):
            guard._safe_call({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})

        kill_fn.assert_called_once()


class TestFailOnErrorYaml:
    """Test YAML config loading of fail_on_error."""

    def test_yaml_fail_on_error_true(self, tmp_path):
        """fail_on_error in YAML config is respected."""
        config = tmp_path / "config.yaml"
        config.write_text(
            """
agents:
  test-bot:
    provider: openai
    budget:
      daily: 50
    fail_on_error: true
    save_prompts: true
"""
        )
        guard = AgentKavach.from_yaml(
            str(config), api_key="ak_test", llm_key="sk-test", agent="test-bot"
        )
        assert guard._fail_on_error is True
        assert guard._save_prompts is True

    def test_yaml_fail_on_error_default_false(self, tmp_path):
        """fail_on_error defaults to false when not in YAML."""
        config = tmp_path / "config.yaml"
        config.write_text(
            """
agents:
  test-bot:
    provider: openai
    budget:
      daily: 50
"""
        )
        guard = AgentKavach.from_yaml(
            str(config), api_key="ak_test", llm_key="sk-test", agent="test-bot"
        )
        assert guard._fail_on_error is False


class TestFailOnErrorWarningLog:
    """Test that warning is logged when fail_on_error fires without on_kill."""

    def test_warning_logged_when_no_on_kill_pre_flight(self, caplog):
        """Pre-flight fail_on_error without on_kill logs a warning."""
        guard = AgentKavach(
            provider="openai",
            api_key="ak_test",
            llm_key="sk-test",
            agent_name="warn-bot",
            budget=Budget.daily(50),
            fail_on_error=True,
        )
        guard._engine.pre_flight = MagicMock(side_effect=RuntimeError("boom"))
        guard._call_provider = MagicMock()

        with caplog.at_level(logging.WARNING):
            with pytest.raises(RuntimeError, match="boom"):
                guard._safe_call(
                    {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
                )

        assert any(
            "fail_on_error triggered but no on_kill callback" in r.message
            and "warn-bot" in r.message
            for r in caplog.records
        )

    def test_warning_logged_when_no_on_kill_post_flight(self, caplog):
        """Post-flight fail_on_error without on_kill logs a warning."""
        guard = AgentKavach(
            provider="openai",
            api_key="ak_test",
            llm_key="sk-test",
            agent_name="warn-bot",
            budget=Budget.daily(50),
            fail_on_error=True,
        )
        guard._engine.pre_flight = MagicMock()
        guard._call_provider = MagicMock(return_value=MagicMock())
        guard._post_flight = MagicMock(side_effect=RuntimeError("post boom"))

        with caplog.at_level(logging.WARNING):
            with pytest.raises(RuntimeError, match="post boom"):
                guard._safe_call(
                    {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
                )

        assert any(
            "fail_on_error triggered but no on_kill callback" in r.message
            and "warn-bot" in r.message
            for r in caplog.records
        )

    def test_post_flight_fail_on_error_without_on_kill_still_raises(self):
        """fail_on_error=True without on_kill still raises in post-flight."""
        guard = AgentKavach(
            provider="openai",
            api_key="ak_test",
            llm_key="sk-test",
            agent_name="test",
            budget=Budget.daily(50),
            fail_on_error=True,
        )
        guard._engine.pre_flight = MagicMock()
        guard._call_provider = MagicMock(return_value=MagicMock())
        guard._post_flight = MagicMock(side_effect=RuntimeError("post fail"))

        with pytest.raises(RuntimeError, match="post fail"):
            guard._safe_call({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})

    def test_no_warning_when_on_kill_defined(self, caplog):
        """No warning logged when on_kill IS defined."""
        kill_fn = MagicMock()
        guard = AgentKavach(
            provider="openai",
            api_key="ak_test",
            llm_key="sk-test",
            agent_name="ok-bot",
            budget=Budget.daily(50),
            fail_on_error=True,
            on_kill=kill_fn,
        )
        guard._engine.pre_flight = MagicMock(side_effect=RuntimeError("boom"))
        guard._call_provider = MagicMock()

        with caplog.at_level(logging.WARNING):
            with pytest.raises(RuntimeError, match="boom"):
                guard._safe_call(
                    {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
                )

        assert not any(
            "fail_on_error triggered but no on_kill callback" in r.message for r in caplog.records
        )
