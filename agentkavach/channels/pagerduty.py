"""PagerDuty alert channel — triggers incidents via Events API v2.

The routing key is configured in dashboard settings.  Passed to the
handler at registration time.

Usage:
    from agentkavach.channels.pagerduty import PagerDutyChannel

    channel = PagerDutyChannel(routing_key=os.environ["PD_ROUTING_KEY"])
    dispatcher.register_channel("pagerduty", channel.send)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx

from agentkavach.channels.templates import (
    PAGERDUTY_DEFAULT_TEMPLATE,
    build_variables,
    render_dict,
)
from agentkavach.engine import ThresholdEvent

logger = logging.getLogger(__name__)

_PD_EVENTS_URL = "https://events.pagerduty.com/v2/enqueue"
_TIMEOUT_SECONDS = 10


class PagerDutyChannel:
    """PagerDuty Events API v2 alert handler.

    The routing key must come from environment variables or dashboard
    settings — never hardcoded in application code.
    """

    def __init__(self, routing_key: str) -> None:
        if not routing_key:
            raise ValueError("PagerDuty routing key must not be empty")

        self._routing_key = routing_key
        self._client = httpx.Client(timeout=_TIMEOUT_SECONDS)

    def send(
        self,
        event: ThresholdEvent,
        template: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Trigger a PagerDuty incident for the given threshold event."""
        variables = build_variables(event)
        tmpl = template if template else PAGERDUTY_DEFAULT_TEMPLATE
        payload = render_dict(tmpl, variables)

        # Inject routing key (never from template — always from config).
        payload["routing_key"] = self._routing_key

        try:
            resp = self._client.post(
                _PD_EVENTS_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code == 202:
                logger.info(
                    "PagerDuty incident triggered for agent %s",
                    event.agent_name,
                )
            else:
                logger.warning(
                    "PagerDuty Events API returned %d: %s",
                    resp.status_code,
                    resp.text[:200],
                )
        except httpx.HTTPError as exc:
            logger.warning("PagerDuty request failed: %s", exc)

    def close(self) -> None:
        """Release HTTP resources."""
        self._client.close()
