"""Alert configuration + local kill handling.

``ChannelConfig`` captures each alert channel (Slack, PagerDuty, webhook,
email) and its trigger threshold. Channel *delivery* is performed entirely
by the AgentKavach cloud, which aggregates spend across every agent and
fires the alert at the true combined total — so a budget shared across many
agents fires once, correctly, instead of each process evaluating only its
own slice. The only thing handled in-process is the ``kill`` channel (the
local ``on_kill`` teardown). Includes cooldown logic to prevent alert storms.
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from agentkavach.engine import ThresholdEvent

logger = logging.getLogger(__name__)

# Default cooldown period (seconds) between repeated alerts on the same
# threshold.  Prevents alert storms during rapid spend bursts.
DEFAULT_COOLDOWN_SECONDS = 300  # 5 minutes


class ChannelType(str, enum.Enum):
    """Supported alert channel types.

    Use these instead of raw strings when creating channels::

        AgentKavach.channel(ChannelType.EMAIL, threshold=0.50, to="ops@acme.com")
    """

    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    KILL = "kill"


# Strictly validated channel types. No other values are accepted.
VALID_CHANNEL_TYPES = frozenset(ct.value for ct in ChannelType)


def _normalize_channel_type(value: Union[str, ChannelType]) -> str:
    """Accept both ChannelType enum and raw strings, return the string value."""
    if isinstance(value, ChannelType):
        return value.value
    return value


@dataclass(frozen=True)
class ChannelConfig:
    """Configuration for a single alert channel with its trigger threshold.

    Created via ``AgentKavach.channel()`` factory method. The target you
    supply is synced to the AgentKavach cloud, which delivers the alert:

    - ``ChannelType.EMAIL``: ``to`` (recipient address).
    - ``ChannelType.SLACK``: ``webhook_url`` (Slack Incoming Webhook URL).
    - ``ChannelType.PAGERDUTY``: ``routing_key`` (Events API v2 routing key).
    - ``ChannelType.WEBHOOK``: ``url`` (HTTP endpoint). Optional ``secret`` for HMAC signing.
    - ``ChannelType.KILL``: No credentials. Triggers the local ``on_kill`` callback.

    The cloud must be able to reach the endpoint. For a Slack/PagerDuty/
    webhook target that is only reachable inside your network, expose a
    cloud-reachable webhook (e.g. a relay) and use ``ChannelType.WEBHOOK``.
    """

    channel_type: str
    threshold: float
    webhook_url: str = ""
    to: str = ""
    routing_key: str = ""
    url: str = ""
    secret: str = ""
    template: Optional[Dict[str, Any]] = None
    # Which budget dimension this channel's threshold evaluates against.
    # Defaults to "cost" for back-compat; YAML loader sets it from the
    # budget block (cost / tokens_total / duration).
    budget_type: str = "cost"

    def __post_init__(self) -> None:
        # Normalize ChannelType enum to string.
        if isinstance(self.channel_type, ChannelType):
            object.__setattr__(self, "channel_type", self.channel_type.value)
        if self.channel_type not in VALID_CHANNEL_TYPES:
            raise ValueError(
                f"Unknown channel type {self.channel_type!r}. "
                f"Valid types: {', '.join(sorted(VALID_CHANNEL_TYPES))}"
            )
        if not 0 < self.threshold <= 1.0:
            raise ValueError(f"Threshold must be in (0, 1.0], got {self.threshold}")
        if self.channel_type == "slack" and not self.webhook_url:
            raise ValueError("Slack channel requires 'webhook_url'")
        if self.channel_type == "email" and not self.to:
            raise ValueError("Email channel requires 'to' (recipient address)")
        if self.channel_type == "pagerduty" and not self.routing_key:
            raise ValueError("PagerDuty channel requires 'routing_key'")
        if self.channel_type == "webhook" and not self.url:
            raise ValueError("Webhook channel requires 'url'")


@dataclass(frozen=True)
class AlertRule:
    """Defines when and how to alert for a given threshold.

    Attributes:
        threshold: Spend fraction that triggers this rule (e.g. 0.70).
        channels: Channel names to dispatch to (e.g. ["email", "slack", "kill"]).
            The ``"kill"`` channel is built-in and triggers the on_kill callback.
        template: Optional custom template variables for the alert message.
    """

    threshold: float
    channels: tuple[str, ...] = ("email",)
    template: Optional[Dict[str, str]] = None
    # Which dimension this rule fires on — matched against
    # ThresholdEvent.budget_type by AlertDispatcher.dispatch.
    budget_type: str = "cost"

    def __post_init__(self) -> None:
        if not 0 < self.threshold <= 1.0:
            raise ValueError(f"Threshold must be in (0, 1.0], got {self.threshold}")
        for ch in self.channels:
            if ch not in VALID_CHANNEL_TYPES:
                raise ValueError(
                    f"Unknown channel type {ch!r} in alert rule. "
                    f"Valid types: {', '.join(sorted(VALID_CHANNEL_TYPES))}"
                )


# Default rules matching the design doc (70/90/100%).
DEFAULT_RULES: tuple[AlertRule, ...] = (
    AlertRule(threshold=0.70, channels=("email",)),
    AlertRule(threshold=0.90, channels=("email", "slack")),
    AlertRule(threshold=1.0, channels=("email", "slack", "pagerduty", "kill")),
)


@dataclass
class AlertDispatcher:
    """Evaluates threshold events and dispatches alerts through channels.

    Channel handlers are registered via ``register_channel``.  If no
    handler is registered for a channel, the alert is logged but not
    dispatched (fail-open).
    """

    rules: tuple[AlertRule, ...] = DEFAULT_RULES
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS

    _channels: Dict[str, Callable[..., None]] = field(default_factory=dict, repr=False)
    _last_fired: Dict[str, float] = field(default_factory=dict, repr=False)

    def register_channel(
        self,
        name: str,
        handler: Callable[[ThresholdEvent, Optional[Dict[str, str]]], None],
    ) -> None:
        """Register a handler function for *name* (e.g. ``"slack"``)."""
        self._channels[name] = handler

    def dispatch(self, event: ThresholdEvent) -> List[str]:
        """Evaluate *event* against rules and dispatch matching alerts.

        Returns a list of channels that were dispatched to.
        """
        dispatched: List[str] = []

        # Match on (threshold, budget_type). Rules tagged with a different
        # dimension are skipped — a cost rule must not fire on a token event.
        event_btype = getattr(event, "budget_type", "cost")
        for rule in self.rules:
            if event.threshold != rule.threshold:
                continue
            if getattr(rule, "budget_type", "cost") != event_btype:
                continue

            for channel in rule.channels:
                cooldown_key = f"{event.budget_key}:{rule.threshold}:{channel}"
                if self._in_cooldown(cooldown_key):
                    logger.debug("Skipping %s alert (cooldown active)", channel)
                    continue

                handler = self._channels.get(channel)
                if handler is None:
                    logger.info(
                        "No handler for channel %r, alert logged only: "
                        "agent=%s threshold=%.0f%% spent=$%.4f limit=$%.2f",
                        channel,
                        event.agent_name,
                        event.threshold * 100,
                        event.spent,
                        event.limit,
                    )
                    dispatched.append(channel)
                    self._record_fired(cooldown_key)
                    continue

                try:
                    handler(event, rule.template)
                    dispatched.append(channel)
                    self._record_fired(cooldown_key)
                except Exception:
                    logger.exception("Alert handler for %r failed", channel)

        return dispatched

    def rules_for_threshold(self, threshold: float) -> List[AlertRule]:
        """Return all rules matching *threshold*."""
        return [r for r in self.rules if r.threshold == threshold]

    def _in_cooldown(self, key: str) -> bool:
        last = self._last_fired.get(key)
        if last is None:
            return False
        return (time.monotonic() - last) < self.cooldown_seconds

    def _record_fired(self, key: str) -> None:
        self._last_fired[key] = time.monotonic()


# ---------------------------------------------------------------------------
# Built-in alert formatters (templates-in-code from design doc)
# ---------------------------------------------------------------------------


def format_alert_message(
    event: ThresholdEvent,
    template: Optional[Dict[str, str]] = None,
) -> str:
    """Format a human-readable alert message.

    Uses the rich template system from ``agentkavach.channels.templates``
    for full variable substitution.  Falls back to a simple format if
    the channels module is unavailable.
    """
    try:
        from agentkavach.channels.templates import build_variables, render

        variables = build_variables(event)
        if template and "body" in template:
            return render(template["body"], variables)
        return (
            f"[AgentKavach] Agent {event.agent_name!r} has reached "
            f"{variables['pct']}% of its ${event.limit:.2f} budget "
            f"(${event.spent:.4f} spent, ${variables['remaining']} remaining)"
        )
    except ImportError:
        # Fallback if channels module not available.
        pct = event.threshold * 100
        remaining = max(0.0, event.limit - event.spent)
        return (
            f"[AgentKavach] Agent {event.agent_name!r} has reached "
            f"{pct:.0f}% of its ${event.limit:.2f} budget "
            f"(${event.spent:.4f} spent, ${remaining:.4f} remaining)"
        )
