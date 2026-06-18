"""Generic webhook alert channel — POSTs JSON to a user-configured URL.

Available on every tier. The endpoint URL (and optional signing secret) is
configured per channel in the SDK constructor or YAML config.

Signing
-------
When a ``secret`` is configured, every request is signed with HMAC-SHA256
over the raw JSON body. Two headers are added:

  * ``X-AgentKavach-Signature: sha256=<hex>`` — body signature.
  * ``X-AgentKavach-Timestamp: <unix-seconds>`` — for replay protection.

The signature format matches the verification snippet in
``dashboard/app/public/docs/alerts/page.tsx``:

    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    hmac.compare_digest(f"sha256={{expected}}", signature_header)

Customers should also reject requests whose ``X-AgentKavach-Timestamp``
is more than a few minutes old.

Usage:
    from agentkavach.channels.webhook import WebhookChannel

    channel = WebhookChannel(
        url=os.environ["ALERT_WEBHOOK_URL"],
        secret=os.environ["ALERT_WEBHOOK_SECRET"],
    )
    dispatcher.register_channel("webhook", channel.send)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from typing import Any, Dict, Optional

import httpx

from agentkavach.channels.templates import (
    WEBHOOK_DEFAULT_TEMPLATE,
    build_variables,
    render_dict,
)
from agentkavach.engine import ThresholdEvent

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 10


class WebhookChannel:
    """Generic webhook alert handler.

    POSTs a JSON payload to the configured URL.  The URL must use
    HTTPS in production — HTTP is allowed only for local development.
    """

    def __init__(
        self,
        url: str,
        secret: str = "",
    ) -> None:
        if not url:
            raise ValueError("Webhook URL must not be empty")

        self._url = url
        self._secret = secret
        self._client = httpx.Client(timeout=_TIMEOUT_SECONDS)

    def send(
        self,
        event: ThresholdEvent,
        template: Optional[Dict[str, Any]] = None,
    ) -> None:
        """POST a JSON alert to the configured webhook URL."""
        variables = build_variables(event)
        tmpl = template if template else WEBHOOK_DEFAULT_TEMPLATE
        payload = render_dict(tmpl, variables)

        # Add timestamp for dedup on the receiver side.
        payload["timestamp"] = variables.get("resets_at", "")

        # Serialize once so the signed bytes exactly match what hits the
        # wire. httpx would re-serialize via `json=`; using `content=` with
        # our own bytes guarantees signature/body parity.
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self._secret:
            # HMAC-SHA256 over the raw body. Stripe-style "sha256=<hex>"
            # so receivers can split on "=" and use hmac.compare_digest.
            signature = hmac.new(
                self._secret.encode("utf-8"),
                body,
                hashlib.sha256,
            ).hexdigest()
            headers["X-AgentKavach-Signature"] = f"sha256={signature}"
            # Receivers should reject requests whose timestamp is more
            # than ~5 min from now (replay protection).
            headers["X-AgentKavach-Timestamp"] = str(int(time.time()))

        try:
            resp = self._client.post(self._url, content=body, headers=headers)
            if resp.status_code < 300:
                logger.info(
                    "Webhook alert sent to %s for agent %s",
                    self._url,
                    event.agent_name,
                )
            else:
                logger.warning(
                    "Webhook returned %d: %s",
                    resp.status_code,
                    resp.text[:200],
                )
        except httpx.HTTPError as exc:
            logger.warning("Webhook request failed: %s", exc)

    def close(self) -> None:
        """Release HTTP resources."""
        self._client.close()
