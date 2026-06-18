"""Email alert channel — sends transactional emails via Resend API.

Emails are sent from ``no-reply@agentkavach.com``.  The Resend API key
is server-side only — the SDK dispatches email alerts through the
AgentKavach backend's alert endpoint, or directly via Resend when
configured for self-hosted deployments.

Usage:
    from agentkavach.channels.email import EmailChannel

    channel = EmailChannel(
        api_key=os.environ["RESEND_API_KEY"],
        from_email="no-reply@agentkavach.com",
        to_email="team@acme.com",
    )
    dispatcher.register_channel("email", channel.send)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx

from agentkavach.channels.templates import (
    EMAIL_DEFAULT_TEMPLATE,
    build_variables,
    render,
)
from agentkavach.engine import ThresholdEvent

logger = logging.getLogger(__name__)

_RESEND_API_URL = "https://api.resend.com/emails"
_TIMEOUT_SECONDS = 10


class EmailChannel:
    """Resend-based email alert handler.

    The API key must come from environment variables — never hardcoded.
    """

    def __init__(
        self,
        api_key: str,
        from_email: str = "no-reply@agentkavach.com",
        to_email: str = "",
        resend_template_id: Optional[str] = None,
    ) -> None:
        if not api_key:
            raise ValueError("Resend API key must not be empty")
        if not to_email:
            raise ValueError("Recipient email (to_email) must not be empty")

        self._api_key = api_key
        self._from_email = from_email
        self._to_email = to_email
        self._resend_template_id = resend_template_id
        self._client = httpx.Client(timeout=_TIMEOUT_SECONDS)

    def send(
        self,
        event: ThresholdEvent,
        template: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send an email alert for the given threshold event.

        Always renders inline subject/text/html so the layout lives in
        this repo (and is testable). The previous behavior of delegating
        to a Resend ``template_id`` produced messy field/value layouts —
        users got "0 | 100% used | Current | Limit | Remaining | Period
        | Today | Resets Tomorrow 00:00 UTC | Metric: Cost" jammed
        together — and we had no way to fix it without editing the
        template in the Resend UI.

        ``resend_template_id`` is kept on the constructor for forward
        compat (e.g. if we ever want to opt into a curated template
        again) but is no longer used by default.
        """
        variables = build_variables(event)

        tmpl = template if template else EMAIL_DEFAULT_TEMPLATE
        subject = render(tmpl.get("subject", EMAIL_DEFAULT_TEMPLATE["subject"]), variables)
        body = render(tmpl.get("body", EMAIL_DEFAULT_TEMPLATE["body"]), variables)
        html_template = tmpl.get("html") or EMAIL_DEFAULT_TEMPLATE.get("html", "")
        html_body = render(html_template, variables) if html_template else ""

        payload: Dict[str, Any] = {
            "from": self._from_email,
            "to": [self._to_email],
            "subject": subject,
            "text": body,
        }
        if html_body:
            payload["html"] = html_body

        try:
            resp = self._client.post(
                _RESEND_API_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
            if resp.status_code in (200, 201):
                logger.info(
                    "Email alert sent to %s for agent %s",
                    self._to_email,
                    event.agent_name,
                )
            else:
                logger.warning(
                    "Resend API returned %d: %s",
                    resp.status_code,
                    resp.text[:200],
                )
        except httpx.HTTPError as exc:
            logger.warning("Email send failed: %s", exc)

    def close(self) -> None:
        """Release HTTP resources."""
        self._client.close()
