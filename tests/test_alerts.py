"""Unit tests for agentkavach.alerts: alert rules, channels, dispatcher, formatting."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agentkavach.alerts import (
    VALID_CHANNEL_TYPES,
    AlertDispatcher,
    AlertRule,
    ChannelConfig,
    format_alert_message,
)
from agentkavach.engine import ThresholdEvent

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _event(
    threshold: float = 0.70,
    spent: float = 7.0,
    limit: float = 10.0,
    agent: str = "test-agent",
) -> ThresholdEvent:
    return ThresholdEvent(
        threshold=threshold,
        spent=spent,
        limit=limit,
        budget_key="daily:2026-03-13",
        agent_name=agent,
    )


# ---------------------------------------------------------------------------
# ChannelConfig
# ---------------------------------------------------------------------------


class TestChannelConfig:
    def test_valid_email(self):
        ch = ChannelConfig(channel_type="email", threshold=0.50, to="team@acme.com")
        assert ch.channel_type == "email"
        assert ch.to == "team@acme.com"

    def test_valid_slack(self):
        ch = ChannelConfig(
            channel_type="slack", threshold=0.80, webhook_url="https://hooks.slack.com/test"
        )
        assert ch.webhook_url == "https://hooks.slack.com/test"

    def test_valid_pagerduty(self):
        ch = ChannelConfig(channel_type="pagerduty", threshold=0.95, routing_key="R0xxx")
        assert ch.routing_key == "R0xxx"

    def test_valid_webhook(self):
        ch = ChannelConfig(channel_type="webhook", threshold=0.90, url="https://example.com/hook")
        assert ch.url == "https://example.com/hook"

    def test_valid_kill(self):
        ch = ChannelConfig(channel_type="kill", threshold=1.0)
        assert ch.channel_type == "kill"

    def test_invalid_channel_type(self):
        with pytest.raises(ValueError, match="Unknown channel type"):
            ChannelConfig(channel_type="phone", threshold=0.50)

    def test_invalid_threshold(self):
        with pytest.raises(ValueError, match="Threshold"):
            ChannelConfig(channel_type="kill", threshold=0.0)

    def test_slack_requires_webhook_url(self):
        with pytest.raises(ValueError, match="webhook_url"):
            ChannelConfig(channel_type="slack", threshold=0.80)

    def test_email_requires_to(self):
        with pytest.raises(ValueError, match="to"):
            ChannelConfig(channel_type="email", threshold=0.50)

    def test_pagerduty_requires_routing_key(self):
        with pytest.raises(ValueError, match="routing_key"):
            ChannelConfig(channel_type="pagerduty", threshold=0.95)

    def test_webhook_requires_url(self):
        with pytest.raises(ValueError, match="url"):
            ChannelConfig(channel_type="webhook", threshold=0.90)

    def test_frozen(self):
        ch = ChannelConfig(channel_type="kill", threshold=1.0)
        with pytest.raises(AttributeError):
            ch.threshold = 0.5  # type: ignore[misc]


class TestValidChannelTypes:
    def test_known_types(self):
        assert VALID_CHANNEL_TYPES == {"email", "slack", "pagerduty", "webhook", "kill"}


# ---------------------------------------------------------------------------
# AlertRule
# ---------------------------------------------------------------------------


class TestAlertRule:
    def test_valid_rule(self):
        rule = AlertRule(threshold=0.70, channels=("email",))
        assert rule.threshold == 0.70

    def test_kill_channel(self):
        rule = AlertRule(threshold=1.0, channels=("pagerduty", "kill"))
        assert "kill" in rule.channels

    def test_invalid_threshold_zero(self):
        with pytest.raises(ValueError, match="Threshold"):
            AlertRule(threshold=0.0, channels=("email",))

    def test_invalid_threshold_negative(self):
        with pytest.raises(ValueError, match="Threshold"):
            AlertRule(threshold=-0.5, channels=("email",))

    def test_invalid_threshold_above_one(self):
        with pytest.raises(ValueError, match="Threshold"):
            AlertRule(threshold=1.5, channels=("email",))

    def test_invalid_channel_type(self):
        with pytest.raises(ValueError, match="Unknown channel type"):
            AlertRule(threshold=0.70, channels=("phone",))

    def test_frozen(self):
        rule = AlertRule(threshold=0.70)
        with pytest.raises(AttributeError):
            rule.threshold = 0.90  # type: ignore[misc]

    def test_custom_template(self):
        rule = AlertRule(
            threshold=0.70,
            template={"body": "Budget at {pct}%!"},
        )
        assert rule.template is not None


# ---------------------------------------------------------------------------
# DEFAULT_RULES
# ---------------------------------------------------------------------------


class TestDefaultRules:
    def test_three_default_rules(self):
        from agentkavach.alerts import DEFAULT_RULES

        assert len(DEFAULT_RULES) == 3

    def test_default_thresholds(self):
        from agentkavach.alerts import DEFAULT_RULES

        thresholds = [r.threshold for r in DEFAULT_RULES]
        assert thresholds == [0.70, 0.90, 1.0]

    def test_kill_at_100(self):
        from agentkavach.alerts import DEFAULT_RULES

        assert "kill" in DEFAULT_RULES[2].channels


# ---------------------------------------------------------------------------
# AlertDispatcher
# ---------------------------------------------------------------------------


class TestAlertDispatcher:
    def test_dispatch_matching_rule(self):
        dispatcher = AlertDispatcher(
            rules=(AlertRule(threshold=0.70, channels=("email",)),),
        )
        dispatched = dispatcher.dispatch(_event(threshold=0.70))
        assert "email" in dispatched

    def test_no_dispatch_for_unmatched_threshold(self):
        dispatcher = AlertDispatcher(
            rules=(AlertRule(threshold=0.90, channels=("email",)),),
        )
        dispatched = dispatcher.dispatch(_event(threshold=0.70))
        assert dispatched == []

    def test_multiple_channels(self):
        dispatcher = AlertDispatcher(
            rules=(AlertRule(threshold=0.70, channels=("email", "slack")),),
        )
        dispatched = dispatcher.dispatch(_event(threshold=0.70))
        assert set(dispatched) == {"email", "slack"}

    def test_registered_handler_called(self):
        handler = MagicMock()
        dispatcher = AlertDispatcher(
            rules=(AlertRule(threshold=0.70, channels=("slack",)),),
        )
        dispatcher.register_channel("slack", handler)
        dispatcher.dispatch(_event(threshold=0.70))
        handler.assert_called_once()

    def test_handler_receives_event_and_template(self):
        handler = MagicMock()
        tmpl = {"body": "custom"}
        dispatcher = AlertDispatcher(
            rules=(AlertRule(threshold=0.70, channels=("slack",), template=tmpl),),
        )
        dispatcher.register_channel("slack", handler)
        event = _event(threshold=0.70)
        dispatcher.dispatch(event)
        handler.assert_called_once_with(event, tmpl)

    def test_handler_exception_caught(self):
        handler = MagicMock(side_effect=RuntimeError("boom"))
        dispatcher = AlertDispatcher(
            rules=(AlertRule(threshold=0.70, channels=("slack",)),),
        )
        dispatcher.register_channel("slack", handler)
        dispatched = dispatcher.dispatch(_event(threshold=0.70))
        assert "slack" not in dispatched

    def test_cooldown_prevents_repeated_dispatch(self):
        dispatcher = AlertDispatcher(
            rules=(AlertRule(threshold=0.70, channels=("email",)),),
            cooldown_seconds=60,
        )
        dispatcher.dispatch(_event(threshold=0.70))
        dispatched = dispatcher.dispatch(_event(threshold=0.70))
        assert dispatched == []

    def test_rules_for_threshold(self):
        dispatcher = AlertDispatcher()
        rules = dispatcher.rules_for_threshold(0.70)
        assert len(rules) == 1
        assert rules[0].threshold == 0.70


# ---------------------------------------------------------------------------
# format_alert_message
# ---------------------------------------------------------------------------


class TestFormatAlertMessage:
    def test_default_format(self):
        msg = format_alert_message(_event())
        assert "test-agent" in msg
        assert "70%" in msg
        assert "$10.00" in msg

    def test_custom_template(self):
        template = {"body": "Agent {agent_name} at {pct}% — ${spent} of ${budget}"}
        msg = format_alert_message(_event(), template=template)
        assert "Agent test-agent" in msg
        assert "70%" in msg

    def test_remaining_calculation(self):
        msg = format_alert_message(_event(spent=7.0, limit=10.0))
        assert "$3.00 remaining" in msg

    def test_100_percent(self):
        msg = format_alert_message(_event(threshold=1.0, spent=10.0, limit=10.0))
        assert "100%" in msg
        assert "$0.00 remaining" in msg


class TestDispatcherBudgetTypeRouting:
    """Phase 40: AlertDispatcher.dispatch must match BOTH threshold AND
    budget_type. A cost rule must NOT fire on a tokens event, and vice
    versa — otherwise an agent with three identical rule percentages (one
    per dimension) would triple-fire each channel for one usage tick."""

    def test_cost_rule_skips_token_event(self):
        from agentkavach.alerts import AlertDispatcher, AlertRule

        fired: list[tuple] = []
        dispatcher = AlertDispatcher(
            rules=(AlertRule(threshold=0.5, channels=("slack",), budget_type="cost"),)
        )
        dispatcher.register_channel("slack", lambda evt, tmpl=None: fired.append(("slack", evt)))

        tokens_event = ThresholdEvent(
            threshold=0.5,
            spent=500.0,
            limit=1000.0,
            budget_key="tokens_total:per_run",
            agent_name="agent-a",
            budget_type="tokens_total",
        )
        dispatched = dispatcher.dispatch(tokens_event)
        assert dispatched == [], "cost rule must not handle a tokens event"
        assert fired == []

    def test_tokens_rule_fires_only_for_tokens_event(self):
        from agentkavach.alerts import AlertDispatcher, AlertRule

        fired: list[str] = []
        dispatcher = AlertDispatcher(
            rules=(
                AlertRule(threshold=0.5, channels=("slack",), budget_type="cost"),
                AlertRule(threshold=0.5, channels=("slack",), budget_type="tokens_total"),
            )
        )
        dispatcher.register_channel(
            "slack",
            lambda evt, tmpl=None: fired.append(evt.budget_type),
        )

        cost_event = ThresholdEvent(
            threshold=0.5,
            spent=5.0,
            limit=10.0,
            budget_key="daily:2026-05-25",
            agent_name="agent-a",
            budget_type="cost",
        )
        tokens_event = ThresholdEvent(
            threshold=0.5,
            spent=500.0,
            limit=1000.0,
            budget_key="tokens_total:per_run",
            agent_name="agent-a",
            budget_type="tokens_total",
        )
        dispatcher.dispatch(cost_event)
        dispatcher.dispatch(tokens_event)
        assert fired == ["cost", "tokens_total"]
