"""Unit tests for agentkavach.channels.templates — variable rendering and defaults."""

from __future__ import annotations

import pytest

from agentkavach.budget import Period
from agentkavach.channels.templates import (
    EMAIL_DEFAULT_TEMPLATE,
    PAGERDUTY_DEFAULT_TEMPLATE,
    SLACK_DEFAULT_TEMPLATE,
    WEBHOOK_DEFAULT_TEMPLATE,
    _infer_period,
    _next_reset,
    _period_label,
    _SafeFormatter,
    _threshold_to_level,
    _threshold_to_severity,
    build_variables,
    render,
    render_dict,
)
from agentkavach.engine import ThresholdEvent

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _event(
    threshold: float = 0.70,
    spent: float = 35.0,
    limit: float = 50.0,
    agent: str = "research-bot",
) -> ThresholdEvent:
    return ThresholdEvent(
        threshold=threshold,
        spent=spent,
        limit=limit,
        budget_key="daily:2026-03-13",
        agent_name=agent,
    )


# ---------------------------------------------------------------------------
# build_variables
# ---------------------------------------------------------------------------


class TestBuildVariables:
    def test_all_keys_present(self):
        v = build_variables(_event())
        expected_keys = {
            "agent_name",
            "pct",
            "spent",
            "budget",
            "remaining",
            "spent_fmt",
            "budget_fmt",
            "remaining_fmt",
            "period",
            "level",
            "severity",
            "resets_at",
            "dashboard_url",
            "budget_type",
            "unit",
        }
        assert set(v.keys()) == expected_keys

    def test_values(self):
        v = build_variables(_event())
        assert v["agent_name"] == "research-bot"
        assert v["pct"] == "70"
        assert v["spent"] == "35.00"
        assert v["budget"] == "50.00"
        assert v["remaining"] == "15.00"
        assert v["level"] == "WARN"
        assert v["severity"] == "warning"

    def test_remaining_clamps_to_zero(self):
        v = build_variables(_event(spent=60.0, limit=50.0))
        assert v["remaining"] == "0.00"

    def test_dashboard_url(self):
        v = build_variables(_event())
        assert v["dashboard_url"] == "https://agentkavach.com/dashboard/agents/research-bot"

    def test_custom_dashboard_base(self):
        v = build_variables(_event(), dashboard_base_url="https://custom.dev")
        assert v["dashboard_url"].startswith("https://custom.dev")

    def test_period_passed_through(self):
        v = build_variables(_event(), period=Period.MONTHLY)
        assert v["period"] == "monthly"


# ---------------------------------------------------------------------------
# Level / severity mapping
# ---------------------------------------------------------------------------


class TestLevelMapping:
    @pytest.mark.parametrize(
        "threshold,expected",
        [
            (0.50, "WARN"),
            (0.70, "WARN"),
            (0.89, "WARN"),
            (0.90, "ALERT"),
            (0.99, "ALERT"),
            (1.0, "CRITICAL"),
        ],
    )
    def test_threshold_to_level(self, threshold, expected):
        assert _threshold_to_level(threshold) == expected

    @pytest.mark.parametrize(
        "threshold,expected",
        [(0.50, "warning"), (0.90, "error"), (1.0, "critical")],
    )
    def test_threshold_to_severity(self, threshold, expected):
        assert _threshold_to_severity(threshold) == expected


# ---------------------------------------------------------------------------
# Period helpers
# ---------------------------------------------------------------------------


class TestPeriodHelpers:
    def test_period_label_daily(self):
        assert _period_label(Period.DAILY) == "daily"

    def test_period_label_none(self):
        assert _period_label(None) == "daily"

    def test_next_reset_daily(self):
        assert _next_reset(Period.DAILY) == "midnight UTC"

    def test_next_reset_total(self):
        assert _next_reset(Period.TOTAL) == "never"

    def test_next_reset_monthly(self):
        result = _next_reset(Period.MONTHLY)
        assert "UTC" in result


# ---------------------------------------------------------------------------
# _infer_period — budget_key → Period inference
# ---------------------------------------------------------------------------


class TestInferPeriod:
    def test_daily_key(self):
        assert _infer_period("daily:2026-03-13") == Period.DAILY

    def test_monthly_key(self):
        assert _infer_period("monthly:2026-03") == Period.MONTHLY

    def test_total_key(self):
        assert _infer_period("total") == Period.TOTAL

    def test_shared_daily_key(self):
        assert _infer_period("shared:team-daily:daily:2026-03-13") == Period.DAILY

    def test_shared_monthly_key(self):
        assert _infer_period("shared:pool:monthly:2026-03") == Period.MONTHLY

    def test_unknown_key_returns_none(self):
        assert _infer_period("unknown-format") is None

    def test_regression_daily_resets_not_never(self):
        """When no period is passed, daily budget_key must produce 'midnight UTC' not 'never'."""
        event = _event()  # budget_key="daily:2026-03-13"
        v = build_variables(event)
        assert v["period"] == "daily"
        assert v["resets_at"] == "midnight UTC"


# ---------------------------------------------------------------------------
# render / render_dict
# ---------------------------------------------------------------------------


class TestRender:
    def test_basic_render(self):
        result = render("Agent {agent_name} at {pct}%", {"agent_name": "bot", "pct": "70"})
        assert result == "Agent bot at 70%"

    def test_missing_key_preserved(self):
        result = render("Hello {unknown_key}", {})
        assert result == "Hello {unknown_key}"

    def test_partial_variables(self):
        result = render("{agent_name} - {missing}", {"agent_name": "bot"})
        assert "bot" in result
        assert "{missing}" in result


class TestRenderDict:
    def test_flat_dict(self):
        template = {"text": "Hello {agent_name}", "count": 42}
        result = render_dict(template, {"agent_name": "bot"})
        assert result["text"] == "Hello bot"
        assert result["count"] == 42

    def test_nested_dict(self):
        template = {"payload": {"summary": "{agent_name} alert"}}
        result = render_dict(template, {"agent_name": "bot"})
        assert result["payload"]["summary"] == "bot alert"

    def test_list_values(self):
        template = {"items": [{"text": "{pct}%"}, {"text": "static"}]}
        result = render_dict(template, {"pct": "70"})
        assert result["items"][0]["text"] == "70%"
        assert result["items"][1]["text"] == "static"

    def test_list_with_strings(self):
        template = {"tags": ["{level}", "agentkavach"]}
        result = render_dict(template, {"level": "WARN"})
        assert result["tags"] == ["WARN", "agentkavach"]


# ---------------------------------------------------------------------------
# Default templates structure
# ---------------------------------------------------------------------------


class TestDefaultTemplates:
    def test_slack_has_text_and_blocks(self):
        assert "text" in SLACK_DEFAULT_TEMPLATE
        assert "blocks" in SLACK_DEFAULT_TEMPLATE
        assert isinstance(SLACK_DEFAULT_TEMPLATE["blocks"], list)

    def test_email_has_subject_and_body(self):
        assert "subject" in EMAIL_DEFAULT_TEMPLATE
        assert "body" in EMAIL_DEFAULT_TEMPLATE

    def test_pagerduty_has_payload(self):
        assert "event_action" in PAGERDUTY_DEFAULT_TEMPLATE
        assert "payload" in PAGERDUTY_DEFAULT_TEMPLATE
        assert "summary" in PAGERDUTY_DEFAULT_TEMPLATE["payload"]
        assert "severity" in PAGERDUTY_DEFAULT_TEMPLATE["payload"]

    def test_webhook_has_event_fields(self):
        assert "event" in WEBHOOK_DEFAULT_TEMPLATE
        assert "level" in WEBHOOK_DEFAULT_TEMPLATE
        assert "agent" in WEBHOOK_DEFAULT_TEMPLATE

    def test_slack_renders_without_error(self):
        v = build_variables(_event())
        result = render_dict(SLACK_DEFAULT_TEMPLATE, v)
        assert "research-bot" in result["text"]
        assert "70%" in result["text"]

    def test_email_renders_without_error(self):
        v = build_variables(_event())
        subject = render(EMAIL_DEFAULT_TEMPLATE["subject"], v)
        body = render(EMAIL_DEFAULT_TEMPLATE["body"], v)
        assert "research-bot" in subject
        assert "35.00" in body

    def test_pagerduty_renders_without_error(self):
        v = build_variables(_event(threshold=1.0, spent=50.0))
        result = render_dict(PAGERDUTY_DEFAULT_TEMPLATE, v)
        assert result["payload"]["severity"] == "critical"

    def test_webhook_renders_without_error(self):
        v = build_variables(_event())
        result = render_dict(WEBHOOK_DEFAULT_TEMPLATE, v)
        assert result["agent"] == "research-bot"


# ---------------------------------------------------------------------------
# SafeFormatter
# ---------------------------------------------------------------------------


class TestSafeFormatter:
    def test_known_key(self):
        fmt = _SafeFormatter()
        assert fmt.format("{name}", name="bot") == "bot"

    def test_unknown_key_preserved(self):
        fmt = _SafeFormatter()
        assert fmt.format("{unknown}") == "{unknown}"

    def test_mixed_keys(self):
        fmt = _SafeFormatter()
        result = fmt.format("{known} {unknown}", known="yes")
        assert "yes" in result
        assert "{unknown}" in result


# ---------------------------------------------------------------------------
# Budget type formatting
# ---------------------------------------------------------------------------


class TestBudgetTypeFormatting:
    """Verify build_variables produces correct _fmt values for all budget types."""

    def test_cost_format_has_dollar(self):
        v = build_variables(_event(spent=35.0, limit=50.0), budget_type="cost")
        assert v["spent_fmt"] == "$35.00"
        assert v["budget_fmt"] == "$50.00"
        assert v["remaining_fmt"] == "$15.00"
        assert v["budget_type"] == "Cost"
        assert v["unit"] == "$"

    def test_tokens_total_format_no_dollar(self):
        v = build_variables(_event(spent=150000, limit=500000), budget_type="tokens_total")
        assert v["spent_fmt"] == "150,000"
        assert v["budget_fmt"] == "500,000"
        assert v["remaining_fmt"] == "350,000"
        assert v["budget_type"] == "Tokens (total)"
        assert v["unit"] == "tokens"

    def test_calls_format(self):
        v = build_variables(_event(spent=800, limit=1000), budget_type="calls")
        assert v["spent_fmt"] == "800"
        assert v["budget_fmt"] == "1,000"
        assert v["remaining_fmt"] == "200"
        assert v["budget_type"] == "API Calls"
        assert v["unit"] == "calls"

    def test_duration_format(self):
        # Durations render as human-readable units (s / m / h) in the
        # *_fmt variants — users want to see "45.0s" not "45,000ms" in
        # alerts. Raw variables (without _fmt) still keep the long-
        # standing :.2f numeric shape for backwards compat.
        v = build_variables(_event(spent=45000, limit=60000), budget_type="duration")
        assert v["spent_fmt"] == "45.0s"
        assert v["budget_fmt"] == "1m 0s"
        assert v["budget_type"] == "Duration"
        assert v["unit"] == "ms"

    def test_cost_subcent_precision(self):
        # Sub-cent costs must not collapse to "$0.00" — that's the
        # 2026-05-25 Slack/email alert bug user reported. Smart
        # decimals: >= $0.01 → 4 dp, >= $0.0001 → 6 dp, else 8 dp.
        v = build_variables(_event(spent=0.0007, limit=0.001), budget_type="cost")
        assert v["spent_fmt"] == "$0.000700"
        assert v["budget_fmt"] == "$0.001000"
        # Big values still display compactly.
        v2 = build_variables(_event(spent=12.34, limit=50.0), budget_type="cost")
        assert v2["spent_fmt"] == "$12.34"
        assert v2["budget_fmt"] == "$50.00"

    def test_loops_format(self):
        v = build_variables(_event(spent=18, limit=25), budget_type="loops")
        assert v["spent_fmt"] == "18"
        assert v["budget_fmt"] == "25"
        assert v["budget_type"] == "Loops"
        assert v["unit"] == "loops"

    def test_raw_values_always_two_decimals(self):
        """Raw spent/budget/remaining are always .2f regardless of type."""
        v = build_variables(_event(spent=800, limit=1000), budget_type="calls")
        assert v["spent"] == "800.00"
        assert v["budget"] == "1000.00"
        assert v["remaining"] == "200.00"

    def test_unknown_budget_type_passes_through(self):
        v = build_variables(_event(), budget_type="custom_metric")
        assert v["budget_type"] == "custom_metric"
        assert v["unit"] == ""

    def test_default_budget_type_is_cost(self):
        v = build_variables(_event())
        assert v["budget_type"] == "Cost"
        assert v["unit"] == "$"


class TestTemplateGenericity:
    """Ensure default templates use _fmt variables and are metric-agnostic."""

    def test_email_subject_has_agent_name(self):
        v = build_variables(_event())
        subject = render(EMAIL_DEFAULT_TEMPLATE["subject"], v)
        assert "research-bot" in subject

    def test_email_body_uses_fmt_not_raw(self):
        """Email body should use {spent_fmt} not ${spent}."""
        body = EMAIL_DEFAULT_TEMPLATE["body"]
        assert "{spent_fmt}" in body
        assert "${spent}" not in body

    def test_email_body_has_budget_type(self):
        body = EMAIL_DEFAULT_TEMPLATE["body"]
        assert "{budget_type}" in body

    def test_slack_text_uses_fmt(self):
        text = SLACK_DEFAULT_TEMPLATE["text"]
        assert "{budget_fmt}" in text

    def test_webhook_event_is_threshold_breach(self):
        assert WEBHOOK_DEFAULT_TEMPLATE["event"] == "threshold.breach"

    def test_email_renders_for_tokens(self):
        v = build_variables(_event(spent=150000, limit=500000), budget_type="tokens_total")
        body = render(EMAIL_DEFAULT_TEMPLATE["body"], v)
        assert "150,000" in body
        assert "500,000" in body
        assert "Tokens (total)" in body
        assert "$" not in body

    def test_email_renders_for_cost(self):
        v = build_variables(_event(spent=35.0, limit=50.0), budget_type="cost")
        body = render(EMAIL_DEFAULT_TEMPLATE["body"], v)
        assert "$35.00" in body
        assert "$50.00" in body
        assert "Cost" in body

    def test_pagerduty_renders_with_budget_type(self):
        v = build_variables(_event(threshold=1.0, spent=25, limit=25), budget_type="loops")
        result = render_dict(PAGERDUTY_DEFAULT_TEMPLATE, v)
        assert "Loops" in result["payload"]["summary"]
        assert result["payload"]["custom_details"]["type"] == "Loops"


# ---------------------------------------------------------------------------
# Slack template rendering (detailed)
# ---------------------------------------------------------------------------


class TestSlackTemplateRendering:
    """Verify SLACK_DEFAULT_TEMPLATE renders all variables correctly."""

    def test_all_variables_rendered_in_text(self):
        v = build_variables(_event(threshold=0.70, spent=35.0, limit=50.0))
        result = render_dict(SLACK_DEFAULT_TEMPLATE, v)
        text = result["text"]
        assert "research-bot" in text
        assert "70%" in text
        assert "$50.00" in text  # budget_fmt for cost type

    def test_header_block_rendered(self):
        v = build_variables(_event(threshold=0.90, spent=45.0, limit=50.0))
        result = render_dict(SLACK_DEFAULT_TEMPLATE, v)
        header = result["blocks"][0]
        assert header["type"] == "header"
        assert "research-bot" in header["text"]["text"]
        assert "ALERT" in header["text"]["text"]  # 0.90 → ALERT

    def test_section_block_rendered(self):
        v = build_variables(_event(threshold=0.70, spent=35.0, limit=50.0))
        result = render_dict(SLACK_DEFAULT_TEMPLATE, v)
        section = result["blocks"][1]
        assert section["type"] == "section"
        body = section["text"]["text"]
        assert "$35.00" in body  # spent_fmt
        assert "$50.00" in body  # budget_fmt
        assert "$15.00" in body  # remaining_fmt
        assert "daily" in body  # period
        assert "midnight UTC" in body  # resets_at (inferred from budget_key "daily:...")

    def test_actions_block_dashboard_url(self):
        v = build_variables(_event())
        result = render_dict(SLACK_DEFAULT_TEMPLATE, v)
        actions = result["blocks"][2]
        assert actions["type"] == "actions"
        button = actions["elements"][0]
        assert "agentkavach.com/dashboard/agents/research-bot" in button["url"]

    def test_critical_level_at_100_pct(self):
        v = build_variables(_event(threshold=1.0, spent=50.0, limit=50.0))
        result = render_dict(SLACK_DEFAULT_TEMPLATE, v)
        assert "CRITICAL" in result["blocks"][0]["text"]["text"]
        assert "100%" in result["text"]

    def test_custom_slack_template_renders(self):
        """A custom template fully replaces the default."""
        custom = {"text": "Alert: {agent_name} used {spent_fmt} of {budget_fmt}"}
        v = build_variables(_event())
        result = render_dict(custom, v)
        assert result["text"] == "Alert: research-bot used $35.00 of $50.00"
        assert "blocks" not in result

    def test_slack_template_with_tokens_budget_type(self):
        v = build_variables(_event(spent=150000, limit=500000), budget_type="tokens_total")
        result = render_dict(SLACK_DEFAULT_TEMPLATE, v)
        # text fallback uses {budget_fmt} (the limit) and {pct}
        assert "500,000" in result["text"]  # budget_fmt
        assert "Tokens (total)" in result["text"]
        # section block should have spent_fmt
        section_body = result["blocks"][1]["text"]["text"]
        assert "150,000" in section_body
