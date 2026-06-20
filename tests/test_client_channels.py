"""Tests for AgentKavach.channel(), AgentKavach.alert(), and channel auto-registration."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agentkavach.alerts import AlertRule, ChannelConfig
from agentkavach.budget import Budget
from agentkavach.client import AgentKavach

# ---------------------------------------------------------------------------
# AgentKavach.channel() factory
# ---------------------------------------------------------------------------


class TestAgentKavachChannel:
    def test_email_channel(self):
        ch = AgentKavach.channel("email", threshold=0.50, to="team@acme.com")
        assert isinstance(ch, ChannelConfig)
        assert ch.channel_type == "email"
        assert ch.threshold == 0.50
        assert ch.to == "team@acme.com"

    def test_slack_channel(self):
        ch = AgentKavach.channel("slack", threshold=0.80, webhook_url="https://hooks.slack.com/x")
        assert ch.channel_type == "slack"
        assert ch.webhook_url == "https://hooks.slack.com/x"

    def test_pagerduty_channel(self):
        ch = AgentKavach.channel("pagerduty", threshold=0.95, routing_key="R0xxx")
        assert ch.channel_type == "pagerduty"
        assert ch.routing_key == "R0xxx"

    def test_webhook_channel(self):
        ch = AgentKavach.channel("webhook", threshold=0.90, url="https://example.com/hook")
        assert ch.channel_type == "webhook"

    def test_webhook_with_secret(self):
        ch = AgentKavach.channel(
            "webhook", threshold=0.90, url="https://example.com/hook", secret="s3cret"
        )
        assert ch.secret == "s3cret"

    def test_kill_channel(self):
        ch = AgentKavach.channel("kill", threshold=1.0)
        assert ch.channel_type == "kill"

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Unknown channel type"):
            AgentKavach.channel("phone", threshold=0.50)

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="Threshold"):
            AgentKavach.channel("kill", threshold=0.0)

    def test_slack_missing_webhook_raises(self):
        with pytest.raises(ValueError, match="webhook_url"):
            AgentKavach.channel("slack", threshold=0.80)

    def test_email_missing_to_raises(self):
        with pytest.raises(ValueError, match="to"):
            AgentKavach.channel("email", threshold=0.50)

    def test_pagerduty_missing_key_raises(self):
        with pytest.raises(ValueError, match="routing_key"):
            AgentKavach.channel("pagerduty", threshold=0.95)


# ---------------------------------------------------------------------------
# AgentKavach.alert() legacy convenience method
# ---------------------------------------------------------------------------


class TestAgentKavachAlert:
    def test_basic_alert_rule(self):
        rule = AgentKavach.alert(threshold=0.70, channels=["email"])
        assert isinstance(rule, AlertRule)
        assert rule.threshold == 0.70
        assert rule.channels == ("email",)

    def test_kill_channel(self):
        rule = AgentKavach.alert(threshold=1.0, channels=["kill"])
        assert "kill" in rule.channels

    def test_multiple_channels(self):
        rule = AgentKavach.alert(threshold=0.90, channels=["slack", "email", "pagerduty"])
        assert rule.channels == ("slack", "email", "pagerduty")

    def test_custom_template(self):
        tmpl = {"text": "Custom: {agent_name} at {pct}%"}
        rule = AgentKavach.alert(threshold=0.80, channels=["slack"], template=tmpl)
        assert rule.template == tmpl

    def test_default_channels(self):
        rule = AgentKavach.alert(threshold=0.50)
        assert rule.channels == ("email",)

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            AgentKavach.alert(threshold=0.0)

    def test_usage_in_constructor(self):
        cg = AgentKavach(
            api_key="ak_test",
            llm_key="sk-test",
            budget=Budget.daily(50),
            alerts=[
                AgentKavach.alert(threshold=0.50, channels=["email"]),
                AgentKavach.alert(threshold=0.90, channels=["slack", "email"]),
                AgentKavach.alert(threshold=1.0, channels=["kill"]),
            ],
        )
        assert len(cg._dispatcher.rules) == 3


# ---------------------------------------------------------------------------
# Constructor with channels param
# ---------------------------------------------------------------------------


class TestChannelsParam:
    def test_channels_produce_rules_no_client_handler(self):
        # The cloud delivers every channel, so no channel registers a
        # client-side handler — but each still produces a rule and its target
        # is synced for cloud delivery.
        cg = AgentKavach(
            api_key="ak_test",
            llm_key="sk-test",
            budget=Budget.daily(50),
            channels=[
                AgentKavach.channel(
                    "slack",
                    threshold=0.80,
                    webhook_url="https://hooks.slack.com/x",
                ),
                AgentKavach.channel("kill", threshold=1.0),
            ],
        )
        # slack is cloud-delivered (no client-side handler); only kill ever
        # appears in the dispatcher channels.
        assert "slack" not in cg._dispatcher._channels
        assert len(cg._dispatcher.rules) == 2
        payload = cg._build_sync_payload()
        acs = {ac["channel"]: ac for ac in payload["alert_configs"]}
        assert acs["slack"]["target"] == "https://hooks.slack.com/x"
        assert "dispatch" not in acs["slack"]

    def test_backend_channels_not_registered_client_side(self):
        cg = AgentKavach(
            api_key="ak_test",
            llm_key="sk-test",
            budget=Budget.daily(50),
            channels=[
                AgentKavach.channel(
                    "slack", threshold=0.80, webhook_url="https://hooks.slack.com/x"
                ),
            ],
        )
        assert "slack" not in cg._dispatcher._channels
        payload = cg._build_sync_payload()
        acs = {ac["channel"]: ac for ac in payload["alert_configs"]}
        assert acs["slack"]["target"] == "https://hooks.slack.com/x"

    def test_channels_builds_rules_from_thresholds(self):
        cg = AgentKavach(
            api_key="ak_test",
            llm_key="sk-test",
            budget=Budget.daily(50),
            channels=[
                AgentKavach.channel("kill", threshold=1.0),
            ],
        )
        assert len(cg._dispatcher.rules) == 1
        assert cg._dispatcher.rules[0].threshold == 1.0
        assert "kill" in cg._dispatcher.rules[0].channels


# ---------------------------------------------------------------------------
# llm_key param
# ---------------------------------------------------------------------------


class TestLlmKey:
    def test_llm_key_stored(self):
        cg = AgentKavach(api_key="ak_test", llm_key="sk-test-key")
        assert cg._llm_key == "sk-test-key"

    def test_legacy_openai_key_resolves(self):
        cg = AgentKavach(api_key="ak_test", openai_api_key="sk-legacy")
        assert cg._llm_key == "sk-legacy"

    def test_llm_key_takes_precedence(self):
        cg = AgentKavach(api_key="ak_test", llm_key="sk-new", openai_api_key="sk-old")
        assert cg._llm_key == "sk-new"

    def test_llm_key_not_read_from_env(self, monkeypatch):
        # The SDK never reads the provider key from the environment.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        with pytest.raises(ValueError, match="llm_key is required"):
            AgentKavach(api_key="ak_test")

    def test_anthropic_llm_key_not_read_from_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
        with pytest.raises(ValueError, match="llm_key is required"):
            AgentKavach(api_key="ak_test", provider="anthropic")


# ---------------------------------------------------------------------------
# Unified create() method
# ---------------------------------------------------------------------------


class TestCreateMethod:
    def test_create_exists(self):
        cg = AgentKavach(api_key="ak_test", llm_key="sk-test")
        assert callable(cg.create)


# ---------------------------------------------------------------------------
# Channel auto-registration (legacy params)
# ---------------------------------------------------------------------------


class TestChannelAutoRegistration:
    def test_no_channels_by_default(self):
        cg = AgentKavach(api_key="ak_test", llm_key="sk-test")
        assert len(cg._channels) == 0

    def test_slack_captured_from_arg(self):
        # Legacy kwargs no longer register a client-side handler; they capture
        # the target into _channel_configs so the cloud can deliver it.
        cg = AgentKavach(
            api_key="ak_test",
            llm_key="sk-test",
            slack_webhook_url="https://hooks.slack.com/test",
        )
        assert "slack" not in cg._dispatcher._channels
        assert len(cg._channels) == 0
        slack_cfgs = [c for c in cg._channel_configs if c.channel_type == "slack"]
        assert len(slack_cfgs) == 1
        assert slack_cfgs[0].webhook_url == "https://hooks.slack.com/test"
        assert cg._resolve_channel_target("slack") == "https://hooks.slack.com/test"

    def test_slack_not_captured_from_env(self, monkeypatch):
        # Channel credentials are never read from the environment — an env var
        # alone must not capture a channel.
        monkeypatch.setenv("AGENTKAVACH_SLACK_WEBHOOK_URL", "https://hooks.slack.com/env")
        cg = AgentKavach(api_key="ak_test", llm_key="sk-test")
        assert "slack" not in cg._dispatcher._channels
        assert [c for c in cg._channel_configs if c.channel_type == "slack"] == []

    def test_email_captured(self):
        cg = AgentKavach(
            api_key="ak_test",
            llm_key="sk-test",
            alert_email="team@acme.com",
        )
        # No client-side handler; the recipient is captured for cloud delivery.
        assert "email" not in cg._dispatcher._channels
        assert cg._find_email_target() == "team@acme.com"

    def test_email_without_address_not_captured(self):
        cg = AgentKavach(
            api_key="ak_test",
            llm_key="sk-test",
        )
        assert "email" not in cg._dispatcher._channels
        assert cg._find_email_target() is None

    def test_pagerduty_captured(self):
        cg = AgentKavach(
            api_key="ak_test",
            llm_key="sk-test",
            pagerduty_routing_key="R0xxx",
        )
        assert "pagerduty" not in cg._dispatcher._channels
        assert cg._resolve_channel_target("pagerduty") == "R0xxx"

    def test_webhook_captured(self):
        cg = AgentKavach(
            api_key="ak_test",
            llm_key="sk-test",
            webhook_url="https://example.com/hook",
        )
        assert "webhook" not in cg._dispatcher._channels
        assert cg._resolve_channel_target("webhook") == "https://example.com/hook"

    def test_webhook_with_secret(self):
        cg = AgentKavach(
            api_key="ak_test",
            llm_key="sk-test",
            webhook_url="https://example.com/hook",
            webhook_secret="s3cret",
        )
        assert "webhook" not in cg._dispatcher._channels
        assert cg._resolve_channel_target("webhook") == "https://example.com/hook"
        assert cg._resolve_webhook_secret() == "s3cret"

    def test_multiple_channels_captured(self):
        cg = AgentKavach(
            api_key="ak_test",
            llm_key="sk-test",
            slack_webhook_url="https://hooks.slack.com/test",
            pagerduty_routing_key="R0xxx",
        )
        assert "slack" not in cg._dispatcher._channels
        assert "pagerduty" not in cg._dispatcher._channels
        assert len(cg._channels) == 0
        assert cg._resolve_channel_target("slack") == "https://hooks.slack.com/test"
        assert cg._resolve_channel_target("pagerduty") == "R0xxx"


# ---------------------------------------------------------------------------
# Shutdown closes channels
# ---------------------------------------------------------------------------


class TestSlackNotRegisteredWithoutUrl:
    def test_slack_not_registered_when_no_url(self):
        cg = AgentKavach(api_key="ak_test", llm_key="sk-test")
        assert "slack" not in cg._dispatcher._channels

    def test_slack_not_registered_when_no_env(self, monkeypatch):
        monkeypatch.delenv("AGENTKAVACH_SLACK_WEBHOOK_URL", raising=False)
        cg = AgentKavach(api_key="ak_test", llm_key="sk-test")
        assert "slack" not in cg._dispatcher._channels


class TestShutdownChannels:
    def test_shutdown_does_not_raise_with_captured_channels(self):
        # No channel is delivered client-side anymore, so there are no senders
        # to close. shutdown() must still complete cleanly.
        cg = AgentKavach(
            api_key="ak_test",
            llm_key="sk-test",
            slack_webhook_url="https://hooks.slack.com/test",
        )
        assert cg._channels == []
        cg.shutdown()


# ---------------------------------------------------------------------------
# format_alert_message with template system
# ---------------------------------------------------------------------------


class TestFormatAlertMessageIntegration:
    def test_default_message(self):
        from agentkavach.alerts import format_alert_message
        from agentkavach.engine import ThresholdEvent

        event = ThresholdEvent(
            threshold=0.70,
            spent=35.0,
            limit=50.0,
            budget_key="daily:2026-03-13",
            agent_name="my-bot",
        )
        msg = format_alert_message(event)
        assert "my-bot" in msg
        assert "70%" in msg
        assert "$50.00" in msg

    def test_custom_template(self):
        from agentkavach.alerts import format_alert_message
        from agentkavach.engine import ThresholdEvent

        event = ThresholdEvent(
            threshold=0.90,
            spent=45.0,
            limit=50.0,
            budget_key="daily:2026-03-13",
            agent_name="my-bot",
        )
        tmpl = {"body": "Alert: {agent_name} at {pct}% — ${spent}/${budget}"}
        msg = format_alert_message(event, template=tmpl)
        assert "Alert: my-bot at 90%" in msg
        assert "$45.00/$50.00" in msg


class TestBackendDispatchedEmail:
    """Phase 37 regression: when the customer provides only an email recipient
    (no Resend api_key — the backend's Resend key dispatches on their behalf),
    the SDK must still expose the recipient to the sync-config payload so the
    backend's AlertConfig.target gets populated. Previously the EmailChannel
    constructor raised, no handler registered, and `_find_email_target`
    returned None — backend had no idea where to send the email.
    """

    def test_email_target_survives_missing_api_key(self):
        ch = ChannelConfig(channel_type="email", threshold=0.5, to="ops@acme.com")
        # No api_key, no RESEND_API_KEY env var → SDK should NOT raise; it
        # should fall back to backend dispatch and remember the recipient.
        with patch.dict("os.environ", {}, clear=False):
            guard = AgentKavach(
                provider="openai",
                llm_key="sk-test",
                agent_name="bot",
                budget=Budget.daily(1.0),
                channels=[ch],
                api_key="ak_dev_x",
            )
        # No SDK-side handler should have registered (no key to send with).
        email_handlers = [
            c for c in guard._channels if getattr(c, "to_email", None) == "ops@acme.com"
        ]
        assert email_handlers == []
        # But the recipient must be recoverable for sync-config.
        assert guard._find_email_target() == "ops@acme.com"

    def test_sync_payload_includes_email_target_for_backend_dispatch(self):
        ch = ChannelConfig(channel_type="email", threshold=0.5, to="ops@acme.com")
        with patch.dict("os.environ", {}, clear=False):
            guard = AgentKavach(
                provider="openai",
                llm_key="sk-test",
                agent_name="bot",
                budget=Budget.daily(1.0),
                channels=[ch],
                api_key="ak_dev_x",
            )
        payload = guard._build_sync_payload()
        email_acs = [ac for ac in payload.get("alert_configs", []) if ac.get("channel") == "email"]
        assert email_acs, "sync payload must include email alert configs"
        assert all(ac.get("target") == "ops@acme.com" for ac in email_acs)


class TestBackendDeliveredChannels:
    """The cloud delivers every channel. None register a client-side handler;
    each channel's target (and webhook secret) is synced with no dispatch key."""

    def test_webhook_synced_not_registered(self):
        guard = AgentKavach(
            provider="openai",
            llm_key="sk-test",
            agent_name="bot",
            budget=Budget.daily(1.0),
            api_key="ak_dev_x",
            channels=[
                ChannelConfig(
                    channel_type="webhook",
                    threshold=0.05,
                    url="https://hooks.example.com/alerts",
                )
            ],
        )
        # No client-side handler — the cloud delivers it.
        assert "webhook" not in guard._dispatcher._channels
        payload = guard._build_sync_payload()
        acs = {ac["channel"]: ac for ac in payload["alert_configs"]}
        assert acs["webhook"]["target"] == "https://hooks.example.com/alerts"
        assert "dispatch" not in acs["webhook"]

    def test_slack_synced_not_registered(self):
        guard = AgentKavach(
            provider="openai",
            llm_key="sk-test",
            agent_name="bot",
            budget=Budget.daily(1.0),
            api_key="ak_dev_x",
            channels=[
                ChannelConfig(
                    channel_type="slack",
                    threshold=0.05,
                    webhook_url="https://hooks.slack.com/x",
                )
            ],
        )
        assert "slack" not in guard._dispatcher._channels
        payload = guard._build_sync_payload()
        acs = {ac["channel"]: ac for ac in payload["alert_configs"]}
        assert acs["slack"]["target"] == "https://hooks.slack.com/x"
        assert "dispatch" not in acs["slack"]

    def test_pagerduty_synced_not_registered(self):
        guard = AgentKavach(
            provider="openai",
            llm_key="sk-test",
            agent_name="bot",
            budget=Budget.daily(1.0),
            api_key="ak_dev_x",
            channels=[
                ChannelConfig(
                    channel_type="pagerduty",
                    threshold=0.07,
                    routing_key="R0xxx",
                )
            ],
        )
        assert "pagerduty" not in guard._dispatcher._channels
        payload = guard._build_sync_payload()
        acs = {ac["channel"]: ac for ac in payload["alert_configs"]}
        assert acs["pagerduty"]["target"] == "R0xxx"
        assert "dispatch" not in acs["pagerduty"]
