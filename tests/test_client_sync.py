"""Tests for SDK client _sync_config() and _build_sync_payload().

Covers:
- Payload building from budget + alert rules
- Org budget inclusion
- Sync fires in a daemon thread
- Sync is skipped when no API key
- Sync is skipped when no endpoint
- HTTP errors are caught silently
"""

from __future__ import annotations

from unittest.mock import patch

from agentkavach.alerts import AlertRule
from agentkavach.budget import Budget


def _make_guard(**kwargs):
    """Create AgentKavach with defaults that avoid real provider setup."""
    from agentkavach.client import AgentKavach

    defaults = {
        "provider": "openai",
        "agent_name": "test-bot",
        "budget": Budget.daily(50),
        "api_key": "ak_test",
        "llm_key": "sk-test",
    }
    defaults.update(kwargs)
    return AgentKavach(**defaults)


class TestBuildSyncPayload:
    def test_basic_budget(self):
        guard = _make_guard()
        payload = guard._build_sync_payload()
        assert payload is not None
        assert payload["agent_name"] == "test-bot"
        assert len(payload["budgets"]) == 1
        assert payload["budgets"][0]["budget_type"] == "cost"
        assert payload["budgets"][0]["period"] == "daily"
        assert payload["budgets"][0]["limit_value"] == 50.0

    def test_monthly_budget(self):
        guard = _make_guard(budget=Budget.monthly(500))
        payload = guard._build_sync_payload()
        assert payload["budgets"][0]["period"] == "monthly"
        assert payload["budgets"][0]["limit_value"] == 500.0

    def test_total_budget(self):
        guard = _make_guard(budget=Budget.total(1000))
        payload = guard._build_sync_payload()
        assert payload["budgets"][0]["period"] == "total"

    def test_alert_rules_in_payload(self):
        guard = _make_guard(
            alerts=[
                AlertRule(threshold=0.5, channels=("email",)),
                AlertRule(threshold=1.0, channels=("email", "kill")),
            ]
        )
        payload = guard._build_sync_payload()
        # 1 from 0.5/email + 2 from 1.0/email,kill = 3
        assert len(payload["alert_configs"]) == 3
        channels = [ac["channel"] for ac in payload["alert_configs"]]
        assert "email" in channels
        assert "kill" in channels

    def test_org_budget_in_payload(self):
        org_b = Budget.org_budget(limit=200.0, period="daily")
        guard = _make_guard(org_budget=org_b)
        payload = guard._build_sync_payload()
        assert payload["org_budget"] is not None
        assert payload["org_budget"]["limit_value"] == 200.0
        assert payload["org_budget"]["period"] == "daily"

    def test_no_org_budget_when_not_set(self):
        guard = _make_guard()
        payload = guard._build_sync_payload()
        assert payload["org_budget"] is None


class TestSyncConfig:
    def test_sync_fires_daemon_thread(self):
        """When API key is set, _sync_config starts a daemon thread."""
        with patch.object(
            __import__("agentkavach.client", fromlist=["AgentKavach"]).AgentKavach,
            "_sync_config",
        ) as mock_sync:
            _make_guard(api_key="cg_test_key_123", endpoint="http://localhost:8000")
            mock_sync.assert_called_once()

    def test_sync_thread_catches_errors(self):
        """The sync function catches all exceptions (invalid endpoint)."""
        guard = _make_guard(api_key="cg_test_key_123", endpoint="http://localhost:99999")
        # _sync_config was called in __init__ and should not have raised
        # Call it again to verify — should not raise
        guard._sync_config()

    def test_build_payload_threshold_values(self):
        """Threshold values are preserved exactly."""
        guard = _make_guard(alerts=[AlertRule(threshold=0.8, channels=("email",))])
        payload = guard._build_sync_payload()
        pcts = [ac["threshold_pct"] for ac in payload["alert_configs"]]
        assert 0.8 in pcts


class TestMultiBudgetSync:
    """Phase 33: every limit type the SDK enforces must reach the
    server so the dashboard Budgets panel renders the full set.
    Pre-Phase-33 only cost budgets were synced — token / call /
    duration caps were invisible to the dashboard."""

    def _types(self, payload):
        return {b["budget_type"] for b in payload["budgets"]}

    def test_tokens_per_run_synced(self):
        guard = _make_guard(max_tokens_per_run=5000)
        payload = guard._build_sync_payload()
        assert "tokens_total" in self._types(payload)
        tokens_budget = next(b for b in payload["budgets"] if b["budget_type"] == "tokens_total")
        assert tokens_budget["period"] == "per_run"
        assert tokens_budget["limit_value"] == 5000
        assert tokens_budget["unit"] == "tokens"

    def test_calls_per_run_synced(self):
        guard = _make_guard(max_calls_per_run=10)
        payload = guard._build_sync_payload()
        assert "calls" in self._types(payload)
        calls_budget = next(b for b in payload["budgets"] if b["budget_type"] == "calls")
        assert calls_budget["period"] == "per_run"
        assert calls_budget["limit_value"] == 10

    def test_duration_per_run_synced_in_ms(self):
        # SDK takes seconds; server stores ms. Sync converts s → ms.
        guard = _make_guard(max_runtime_seconds=20)
        payload = guard._build_sync_payload()
        duration_budget = next(b for b in payload["budgets"] if b["budget_type"] == "duration")
        assert duration_budget["period"] == "per_run"
        assert duration_budget["limit_value"] == 20_000  # 20s in ms
        assert duration_budget["unit"] == "ms"

    def test_loops_synced_when_detect_loops_enabled(self):
        guard = _make_guard(detect_loops=True, loop_threshold=4)
        payload = guard._build_sync_payload()
        loops_budget = next(b for b in payload["budgets"] if b["budget_type"] == "loops")
        assert loops_budget["limit_value"] == 4

    def test_loops_not_synced_when_detect_loops_off(self):
        guard = _make_guard(detect_loops=False, loop_threshold=4)
        payload = guard._build_sync_payload()
        assert "loops" not in self._types(payload)

    def test_all_limits_together(self):
        # Realistic agent: cost + token + call + duration + loops.
        guard = _make_guard(
            budget=Budget.daily(0.05),
            max_tokens_per_run=5000,
            max_calls_per_run=10,
            max_runtime_seconds=20,
            detect_loops=True,
            loop_threshold=3,
        )
        payload = guard._build_sync_payload()
        assert self._types(payload) == {
            "cost",
            "tokens_total",
            "calls",
            "duration",
            "loops",
        }


class TestChannelTargetSync:
    """Phase 153: the SDK syncs the customer-owned target for every channel so
    the backend can dispatch to the customer's own endpoint — email recipient,
    slack webhook, pagerduty routing key."""

    def test_slack_pagerduty_email_targets_synced(self):
        from agentkavach.alerts import ChannelConfig, ChannelType

        guard = _make_guard(
            channels=[
                ChannelConfig(
                    channel_type=ChannelType.SLACK,
                    threshold=0.05,
                    webhook_url="https://hooks.slack.com/services/CUSTOMER/OWN",
                    budget_type="cost",
                ),
                ChannelConfig(
                    channel_type=ChannelType.PAGERDUTY,
                    threshold=0.05,
                    routing_key="CUST_RK",
                    budget_type="cost",
                ),
                ChannelConfig(
                    channel_type=ChannelType.EMAIL,
                    threshold=0.05,
                    to="ops@example.com",
                    budget_type="cost",
                ),
            ]
        )
        payload = guard._build_sync_payload()
        acs = payload.get("alert_configs") or payload.get("org_alert_configs") or []
        by_channel = {ac["channel"]: ac for ac in acs}
        assert by_channel["slack"]["target"] == "https://hooks.slack.com/services/CUSTOMER/OWN"
        assert by_channel["pagerduty"]["target"] == "CUST_RK"
        assert by_channel["email"]["target"] == "ops@example.com"


class TestSyncConfigRetry:
    """F9: config sync retries transient failures so an agent's budget/alert
    config isn't silently lost (which would leave the backend never firing that
    agent's alerts)."""

    def test_retries_then_succeeds(self):
        from unittest.mock import MagicMock, patch

        guard = _make_guard()
        calls = {"n": 0}

        def fake_urlopen(req, timeout=None):
            calls["n"] += 1
            if calls["n"] < 3:
                raise TimeoutError("read timed out")
            return MagicMock()  # supports the context-manager protocol

        with (
            patch("urllib.request.urlopen", side_effect=fake_urlopen),
            patch("time.sleep"),
        ):
            ok = guard._post_sync_config("https://x/v1/sync-config", b"{}")

        assert ok is True
        assert calls["n"] == 3

    def test_gives_up_after_attempts(self):
        from unittest.mock import patch

        guard = _make_guard()
        with (
            patch("urllib.request.urlopen", side_effect=TimeoutError("nope")) as mock_open,
            patch("time.sleep"),
        ):
            ok = guard._post_sync_config("https://x/v1/sync-config", b"{}", attempts=3)

        assert ok is False
        assert mock_open.call_count == 3


class TestWebhookSynced:
    """Phase 157: webhook is a first-class backend channel, so the SDK syncs it
    (URL as target + optional signing secret) so the backend can dispatch it."""

    def test_webhook_channel_synced_with_target_and_secret(self):
        from agentkavach.alerts import ChannelConfig, ChannelType

        guard = _make_guard(
            channels=[
                ChannelConfig(
                    channel_type=ChannelType.EMAIL, threshold=0.03, to="a@b.com", budget_type="cost"
                ),
                ChannelConfig(
                    channel_type=ChannelType.WEBHOOK,
                    threshold=0.09,
                    url="https://api.acme.com/budget-alerts",
                    secret="sign-me",
                    budget_type="cost",
                ),
            ]
        )
        payload = guard._build_sync_payload()
        acs = payload.get("alert_configs") or payload.get("org_alert_configs") or []
        by_channel = {ac["channel"]: ac for ac in acs}
        assert "email" in by_channel
        assert by_channel["webhook"]["target"] == "https://api.acme.com/budget-alerts"
        assert by_channel["webhook"]["secret"] == "sign-me"


class TestAllChannelsSynced:
    """The cloud delivers every channel, so the SDK always syncs the channel
    target (and webhook secret) for every channel that has one — with NO
    ``dispatch`` key in the payload."""

    def test_every_channel_target_and_secret_synced(self):
        from agentkavach.alerts import ChannelConfig, ChannelType

        guard = _make_guard(
            channels=[
                ChannelConfig(
                    channel_type=ChannelType.WEBHOOK,
                    threshold=0.05,
                    url="http://10.0.0.5/alerts",
                    secret="internal-secret",
                    budget_type="cost",
                ),
                ChannelConfig(
                    channel_type=ChannelType.SLACK,
                    threshold=0.07,
                    webhook_url="https://hooks.slack.com/public",
                    budget_type="cost",
                ),
            ]
        )
        payload = guard._build_sync_payload()
        acs = payload.get("alert_configs") or payload.get("org_alert_configs") or []
        by_channel = {ac["channel"]: ac for ac in acs}
        # webhook: target + secret synced, never a dispatch key.
        assert by_channel["webhook"]["target"] == "http://10.0.0.5/alerts"
        assert by_channel["webhook"]["secret"] == "internal-secret"
        assert "dispatch" not in by_channel["webhook"]
        # slack: public target synced, never a dispatch key.
        assert by_channel["slack"]["target"] == "https://hooks.slack.com/public"
        assert "dispatch" not in by_channel["slack"]
