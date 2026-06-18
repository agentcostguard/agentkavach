"""Slack alert channel — sends Block Kit messages via Incoming Webhook.

The webhook URL is configured in dashboard settings (never in code).
Passed to the handler at registration time via ``SlackChannel.config``.

Usage:
    from agentkavach.channels.slack import SlackChannel

    channel = SlackChannel(webhook_url=os.environ["SLACK_WEBHOOK_URL"])
    dispatcher.register_channel("slack", channel.send)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx

from agentkavach.channels.templates import (
    SLACK_DEFAULT_TEMPLATE,
    build_variables,
    render_dict,
)
from agentkavach.engine import ThresholdEvent

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 10


class SlackChannel:
    """Slack webhook alert handler.

    Sends rich Block Kit messages to a configured Slack channel.
    The webhook URL should come from environment variables or
    dashboard settings — never hardcoded.
    """

    def __init__(self, webhook_url: str) -> None:
        if not webhook_url:
            raise ValueError("Slack webhook URL must not be empty")
        if not webhook_url.startswith("https://"):
            raise ValueError("Slack webhook URL must use HTTPS")

        self._webhook_url = webhook_url
        self._client = httpx.Client(timeout=_TIMEOUT_SECONDS)

    def send(
        self,
        event: ThresholdEvent,
        template: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send a Slack alert for the given threshold event.

        Uses *template* if provided, otherwise the default Block Kit
        template from the design doc.
        """
        variables = build_variables(event)
        tmpl = template if template else SLACK_DEFAULT_TEMPLATE
        payload = render_dict(tmpl, variables)

        try:
            resp = self._client.post(
                self._webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code != 200:
                logger.warning(
                    "Slack webhook returned %d: %s",
                    resp.status_code,
                    resp.text[:200],
                )
            else:
                logger.info(
                    "Slack alert sent for agent %s (threshold %s%%)",
                    event.agent_name,
                    int(event.threshold * 100),
                )
        except httpx.HTTPError as exc:
            logger.warning("Slack webhook request failed: %s", exc)

    def close(self) -> None:
        """Release HTTP resources."""
        self._client.close()
