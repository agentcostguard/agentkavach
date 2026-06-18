"""Default alert templates and template variable rendering.

Template variables (documented in design doc §6):
    {agent_name}   — Agent identifier
    {pct}          — Budget utilization percentage (e.g. "70")
    {spent}        — Amount spent (e.g. "35.00")
    {budget}       — Budget limit (e.g. "50.00")
    {remaining}    — Remaining budget (e.g. "15.00")
    {period}       — Budget period (e.g. "daily")
    {level}        — Alert severity (WARN / ALERT / CRITICAL)
    {severity}     — PagerDuty severity (warning / error / critical)
    {resets_at}    — Next reset time (e.g. "midnight UTC")
    {dashboard_url} — Link to agent detail page

All templates use Python's str.format() — unknown keys are left as-is
via SafeFormatter so partial templates never crash.
"""

from __future__ import annotations

import string
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from agentkavach.budget import Period
from agentkavach.engine import ThresholdEvent


class _SafeFormatter(string.Formatter):
    """Formatter that leaves unknown ``{keys}`` intact instead of raising."""

    def get_value(self, key: Any, args: Any, kwargs: Any) -> Any:
        if isinstance(key, str):
            return kwargs.get(key, "{" + key + "}")
        return super().get_value(key, args, kwargs)

    def format_field(self, value: Any, format_spec: str) -> str:
        if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
            return value
        return super().format_field(value, format_spec)


_safe_fmt = _SafeFormatter()


def _threshold_to_level(threshold: float) -> str:
    """Map a threshold fraction to a human-readable level."""
    if threshold >= 1.0:
        return "CRITICAL"
    if threshold >= 0.90:
        return "ALERT"
    return "WARN"


def _threshold_to_severity(threshold: float) -> str:
    """Map a threshold fraction to a PagerDuty severity."""
    if threshold >= 1.0:
        return "critical"
    if threshold >= 0.90:
        return "error"
    return "warning"


def _period_label(period: Optional[Period]) -> str:
    if period is None:
        return "daily"
    return period.value


def _next_reset(period: Optional[Period]) -> str:
    if period is None or period is Period.TOTAL:
        return "never"
    if period is Period.DAILY:
        return "midnight UTC"
    if period is Period.MONTHLY:
        now = datetime.now(timezone.utc)
        if now.month == 12:
            return f"{now.year + 1}-01-01 UTC"
        return f"{now.year}-{now.month + 1:02d}-01 UTC"
    return "unknown"


_BUDGET_TYPE_LABELS: Dict[str, str] = {
    "cost": "Cost",
    "tokens_total": "Tokens (total)",
    "tokens_input": "Tokens (input)",
    "tokens_output": "Tokens (output)",
    "calls": "API Calls",
    "duration": "Duration",
    "loops": "Loops",
}

_BUDGET_TYPE_UNITS: Dict[str, str] = {
    "cost": "$",
    "tokens_total": "tokens",
    "tokens_input": "tokens",
    "tokens_output": "tokens",
    "calls": "calls",
    "duration": "ms",
    "loops": "loops",
}


def _format_cost(v: float) -> str:
    """Smart-precision USD formatter.

    Mirror of ``dashboard/lib/format.ts:formatCost``. A flat ``:.2f``
    renders sub-cent budgets like ``$0.001`` as ``$0.00``, which is
    exactly the bug the user hit in the 2026-05-25 prod test
    (Slack/email alerts said "Current: $0.00 / $0.00"). Decimals
    chosen by magnitude:

      >= $1.00     → 2 decimals
      >= $0.01     → 4 decimals
      >= $0.0001   → 6 decimals
      else         → 8 decimals
    """
    if v is None or v != v:  # NaN check
        return "$0.00"
    abs_v = abs(float(v))
    if abs_v == 0:
        return "$0.00"
    if abs_v >= 1:
        return f"${v:.2f}"
    if abs_v >= 0.01:
        return f"${v:.4f}"
    if abs_v >= 0.0001:
        return f"${v:.6f}"
    return f"${v:.8f}"


def _format_duration_ms(v: float) -> str:
    """Format milliseconds as ms / s / m / h."""
    if v is None or v <= 0:
        return "0ms"
    if v < 1000:
        return f"{int(round(v))}ms"
    s = v / 1000.0
    if s < 60:
        return f"{s:.1f}s"
    m = int(s // 60)
    rem_s = int(round(s % 60))
    if m < 60:
        return f"{m}m {rem_s}s"
    h = m // 60
    rem_m = m % 60
    return f"{h}h {rem_m}m"


def _format_budget_value(value: float, budget_type: str) -> str:
    """Format a value with the right shape for its budget type.

    Cost → dollar amount, counts → thousands separators, duration
    → ms/s/m/h. Used by both Slack and email templates so the
    notifications match the dashboard rendering.
    """
    if budget_type == "cost":
        return _format_cost(value)
    if budget_type == "duration":
        return _format_duration_ms(value)
    # tokens_* / calls / loops / anything else → integer count
    try:
        return f"{int(round(value)):,}"
    except (TypeError, ValueError):
        return str(value)


def _format_raw(value: float, budget_type: str) -> str:
    """Raw numeric formatter — same precision rules as ``_format_budget_value``
    but WITHOUT a unit prefix (``$`` for cost, ``ms`` for duration etc.).

    Used for the bare ``{spent}`` / ``{budget}`` / ``{remaining}``
    template variables, which historically rendered as ``35.00``
    (no ``$``) for cost AND ``800.00`` (no commas, with decimals)
    for counts. Templates that want labelled values use the
    ``_fmt`` variants instead.
    """
    if budget_type == "cost":
        if value is None or value != value:
            return "0.00"
        abs_v = abs(float(value))
        if abs_v == 0:
            return "0.00"
        if abs_v >= 1:
            return f"{value:.2f}"
        if abs_v >= 0.01:
            return f"{value:.4f}"
        if abs_v >= 0.0001:
            return f"{value:.6f}"
        return f"{value:.8f}"
    # Non-cost raw form: keep the long-standing :.2f shape so
    # existing templates / tests render unchanged. The unit-prefixed
    # _fmt variants are where smart formatting lives for non-cost.
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _infer_period(budget_key: str) -> Optional[Period]:
    """Infer the budget period from a budget_key string.

    Budget keys follow the pattern ``"daily:2026-03-13"``,
    ``"monthly:2026-03"``, ``"total"``, or
    ``"shared:name:daily:2026-03-13"``.
    """
    parts = budget_key.split(":")
    for part in parts:
        if part == "daily":
            return Period.DAILY
        if part == "monthly":
            return Period.MONTHLY
        if part == "total":
            return Period.TOTAL
    return None


def build_variables(
    event: ThresholdEvent,
    period: Optional[Period] = None,
    dashboard_base_url: str = "https://agentkavach.com",
    budget_type: str = "cost",
) -> Dict[str, str]:
    """Build the full set of template variables from a ``ThresholdEvent``.

    If *period* is not explicitly provided, it is inferred from
    ``event.budget_key`` (e.g. ``"daily:2026-03-13"`` → ``Period.DAILY``).
    """
    if period is None:
        period = _infer_period(event.budget_key)

    remaining = max(0.0, event.limit - event.spent)
    unit = _BUDGET_TYPE_UNITS.get(budget_type, "")
    label = _BUDGET_TYPE_LABELS.get(budget_type, budget_type)

    # Smart-precision formatters. {spent}/{budget}/{remaining} stay as
    # raw numeric strings ("35.00", "0.0010") — backwards compatible.
    # {spent_fmt}/{budget_fmt}/{remaining_fmt} are unit-prefixed
    # ("$35.00", "$0.0010", "120 tokens") for end-user display.
    return {
        "agent_name": event.agent_name,
        "pct": f"{event.threshold * 100:.0f}",
        "spent": _format_raw(event.spent, budget_type),
        "budget": _format_raw(event.limit, budget_type),
        "remaining": _format_raw(remaining, budget_type),
        "spent_fmt": _format_budget_value(event.spent, budget_type),
        "budget_fmt": _format_budget_value(event.limit, budget_type),
        "remaining_fmt": _format_budget_value(remaining, budget_type),
        "period": _period_label(period),
        "level": _threshold_to_level(event.threshold),
        "severity": _threshold_to_severity(event.threshold),
        "resets_at": _next_reset(period),
        "dashboard_url": f"{dashboard_base_url}/dashboard/agents/{event.agent_name}",
        "budget_type": label,
        "unit": unit,
    }


def render(template_str: str, variables: Dict[str, str]) -> str:
    """Render a template string with the given variables (safe — never raises)."""
    try:
        return _safe_fmt.format(template_str, **variables)
    except Exception:
        return template_str


def render_dict(template: Dict[str, Any], variables: Dict[str, str]) -> Dict[str, Any]:
    """Recursively render all string values in a template dict."""
    result: Dict[str, Any] = {}
    for key, value in template.items():
        if isinstance(value, str):
            result[key] = render(value, variables)
        elif isinstance(value, dict):
            result[key] = render_dict(value, variables)
        elif isinstance(value, list):
            result[key] = [
                render_dict(item, variables)
                if isinstance(item, dict)
                else render(item, variables)
                if isinstance(item, str)
                else item
                for item in value
            ]
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Default templates (from design doc §6)
# ---------------------------------------------------------------------------

SLACK_DEFAULT_TEMPLATE: Dict[str, Any] = {
    "text": "\u26a0\ufe0f AgentKavach: {agent_name} at {pct}% of {budget_fmt} {period} {budget_type} limit",
    "blocks": [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "[{level}] {agent_name} — {pct}% of {budget_type} limit",
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "*Type:* {budget_type}\n"
                    "*Current:* {spent_fmt} / {budget_fmt}\n"
                    "*Remaining:* {remaining_fmt}\n"
                    "*Period:* {period} (resets {resets_at})"
                ),
            },
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "View Dashboard \u2192"},
                    "url": "{dashboard_url}",
                }
            ],
        },
    ],
}

EMAIL_DEFAULT_TEMPLATE: Dict[str, str] = {
    "subject": "[{level}] {agent_name} — {pct}% of {budget_type} budget",
    "body": (
        'Agent "{agent_name}" has crossed the {pct}% threshold on'
        " {budget_type}.\n\n"
        "  Current:    {spent_fmt}\n"
        "  Limit:      {budget_fmt}\n"
        "  Remaining:  {remaining_fmt}\n"
        "  Period:     {period}\n"
        "  Resets:     {resets_at}\n\n"
        "View details: {dashboard_url}"
    ),
    # Clean two-column HTML body. Renders correctly in every major
    # email client — pure inline styles, no <style> block.
    "html": (
        '<div style="font-family: -apple-system, BlinkMacSystemFont, '
        '\\"Segoe UI\\", Helvetica, Arial, sans-serif; max-width: 560px; '
        'margin: 0 auto; padding: 24px; color: #1f2937;">'
        '<h2 style="margin: 0 0 8px 0; color: #b45309; font-size: 20px;">'
        "&#9888;&#65039; {level} &mdash; {agent_name}</h2>"
        '<p style="margin: 0 0 20px 0; color: #4b5563; line-height: 1.5;">'
        "Your agent <strong>{agent_name}</strong> has crossed the "
        "<strong>{pct}%</strong> threshold on <strong>{budget_type}</strong>."
        "</p>"
        '<table style="border-collapse: collapse; width: 100%; '
        'margin: 0 0 24px 0; font-size: 14px;">'
        '<tr><td style="padding: 10px 12px; border-bottom: 1px solid #e5e7eb; '
        'color: #6b7280;">Current</td>'
        '<td style="padding: 10px 12px; border-bottom: 1px solid #e5e7eb; '
        'text-align: right; font-weight: 600; color: #111827;">'
        "{spent_fmt}</td></tr>"
        '<tr><td style="padding: 10px 12px; border-bottom: 1px solid #e5e7eb; '
        'color: #6b7280;">Limit</td>'
        '<td style="padding: 10px 12px; border-bottom: 1px solid #e5e7eb; '
        'text-align: right; font-weight: 600; color: #111827;">'
        "{budget_fmt}</td></tr>"
        '<tr><td style="padding: 10px 12px; border-bottom: 1px solid #e5e7eb; '
        'color: #6b7280;">Remaining</td>'
        '<td style="padding: 10px 12px; border-bottom: 1px solid #e5e7eb; '
        'text-align: right; font-weight: 600; color: #111827;">'
        "{remaining_fmt}</td></tr>"
        '<tr><td style="padding: 10px 12px; border-bottom: 1px solid #e5e7eb; '
        'color: #6b7280;">Period</td>'
        '<td style="padding: 10px 12px; border-bottom: 1px solid #e5e7eb; '
        'text-align: right; color: #374151;">{period}</td></tr>'
        '<tr><td style="padding: 10px 12px; border-bottom: 1px solid #e5e7eb; '
        'color: #6b7280;">Resets</td>'
        '<td style="padding: 10px 12px; border-bottom: 1px solid #e5e7eb; '
        'text-align: right; color: #374151;">{resets_at}</td></tr>'
        '<tr><td style="padding: 10px 12px; color: #6b7280;">Metric</td>'
        '<td style="padding: 10px 12px; text-align: right; color: #374151;">'
        "{budget_type}</td></tr>"
        "</table>"
        '<a href="{dashboard_url}" style="display: inline-block; padding: 10px '
        "20px; background: #2563eb; color: #ffffff; text-decoration: none; "
        'border-radius: 6px; font-weight: 500;">View Dashboard &rarr;</a>'
        "</div>"
    ),
}

PAGERDUTY_DEFAULT_TEMPLATE: Dict[str, Any] = {
    "event_action": "trigger",
    "payload": {
        "summary": "AgentKavach: {agent_name} at {pct}% — {spent_fmt}/{budget_fmt} ({budget_type})",
        "severity": "{severity}",
        "source": "agentkavach",
        "custom_details": {
            "agent": "{agent_name}",
            "type": "{budget_type}",
            "current": "{spent_fmt}",
            "limit": "{budget_fmt}",
            "period": "{period}",
            "remaining": "{remaining_fmt}",
        },
    },
}

WEBHOOK_DEFAULT_TEMPLATE: Dict[str, Any] = {
    "event": "threshold.breach",
    "level": "{level}",
    "agent": "{agent_name}",
    "budget_type": "{budget_type}",
    "limit": "{budget_fmt}",
    "current": "{spent_fmt}",
    "pct": "{pct}",
    "period": "{period}",
    "remaining": "{remaining_fmt}",
}
