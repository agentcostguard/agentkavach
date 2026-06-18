"""Unit tests for agentkavach.channels — Slack, Email, PagerDuty, Webhook handlers."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agentkavach.channels.email import EmailChannel
from agentkavach.channels.pagerduty import PagerDutyChannel
from agentkavach.channels.slack import SlackChannel
from agentkavach.channels.webhook import WebhookChannel
from agentkavach.engine import ThresholdEvent

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _event(
    threshold: float = 0.70,
    spent: float = 35.0,
    limit: float = 50.0,
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
# SlackChannel
# ---------------------------------------------------------------------------


class TestSlackChannel:
    def test_empty_url_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            SlackChannel(webhook_url="")

    def test_non_https_raises(self):
        with pytest.raises(ValueError, match="HTTPS"):
            SlackChannel(webhook_url="http://hooks.slack.com/xxx")

    def test_webhook_url_stored(self):
        with patch("agentkavach.channels.slack.httpx.Client"):
            channel = SlackChannel(webhook_url="https://hooks.slack.com/stored")
        assert channel._webhook_url == "https://hooks.slack.com/stored"

    @patch("agentkavach.channels.slack.httpx.Client")
    def test_send_posts_to_webhook(self, mock_client_cls):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        channel = SlackChannel(webhook_url="https://hooks.slack.com/test")
        channel._client = mock_client
        channel.send(_event())

        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "https://hooks.slack.com/test"
        payload = call_args[1].get("json") or call_args.kwargs.get("json")
        assert "text" in payload
        assert "test-agent" in payload["text"]

    @patch("agentkavach.channels.slack.httpx.Client")
    def test_send_default_template_has_block_kit(self, mock_client_cls):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        channel = SlackChannel(webhook_url="https://hooks.slack.com/test")
        channel._client = mock_client
        channel.send(_event())

        payload = mock_client.post.call_args.kwargs.get("json")
        assert "blocks" in payload
        assert isinstance(payload["blocks"], list)
        assert payload["blocks"][0]["type"] == "header"

    @patch("agentkavach.channels.slack.httpx.Client")
    def test_send_template_variables_rendered(self, mock_client_cls):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        channel = SlackChannel(webhook_url="https://hooks.slack.com/test")
        channel._client = mock_client
        channel.send(_event(threshold=0.90, spent=45.0, limit=50.0, agent="render-bot"))

        payload = mock_client.post.call_args.kwargs.get("json")
        # text fallback should have agent name and percentage
        assert "render-bot" in payload["text"]
        assert "90%" in payload["text"]
        # Block Kit header should also be rendered
        header_text = payload["blocks"][0]["text"]["text"]
        assert "render-bot" in header_text
        assert "ALERT" in header_text  # 0.90 → ALERT level

    @patch("agentkavach.channels.slack.httpx.Client")
    def test_send_with_custom_template(self, mock_client_cls):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        channel = SlackChannel(webhook_url="https://hooks.slack.com/test")
        channel._client = mock_client
        custom = {"text": "Custom: {agent_name} at {pct}%"}
        channel.send(_event(), template=custom)

        payload = mock_client.post.call_args.kwargs.get("json")
        assert payload["text"] == "Custom: test-agent at 70%"
        # Custom template should NOT have blocks (it replaces the default entirely)
        assert "blocks" not in payload

    @patch("agentkavach.channels.slack.httpx.Client")
    def test_non_200_response_logs_warning(self, mock_client_cls):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "invalid_payload"
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        channel = SlackChannel(webhook_url="https://hooks.slack.com/test")
        channel._client = mock_client

        with patch("agentkavach.channels.slack.logger") as mock_logger:
            channel.send(_event())
            mock_logger.warning.assert_called_once()
            args = mock_logger.warning.call_args[0]
            assert 400 in args or "400" in str(args)

    @patch("agentkavach.channels.slack.httpx.Client")
    def test_500_response_does_not_raise(self, mock_client_cls):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "internal_error"
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        channel = SlackChannel(webhook_url="https://hooks.slack.com/test")
        channel._client = mock_client
        # Should not raise — fail-open design.
        channel.send(_event())

    @patch("agentkavach.channels.slack.httpx.Client")
    def test_response_body_truncated_in_log(self, mock_client_cls):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "x" * 500  # Long error body
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        channel = SlackChannel(webhook_url="https://hooks.slack.com/test")
        channel._client = mock_client

        with patch("agentkavach.channels.slack.logger") as mock_logger:
            channel.send(_event())
            # resp.text[:200] is used in the log — verify it was truncated
            log_args = mock_logger.warning.call_args[0]
            body_arg = log_args[2]  # Third positional arg is the body text
            assert len(body_arg) == 200

    @patch("agentkavach.channels.slack.httpx.Client")
    def test_send_handles_connect_error(self, mock_client_cls):
        import httpx

        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ConnectError("fail")
        mock_client_cls.return_value = mock_client

        channel = SlackChannel(webhook_url="https://hooks.slack.com/test")
        channel._client = mock_client
        # Should not raise.
        channel.send(_event())

    @patch("agentkavach.channels.slack.httpx.Client")
    def test_send_handles_timeout_error(self, mock_client_cls):
        import httpx

        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.TimeoutException("timeout")
        mock_client_cls.return_value = mock_client

        channel = SlackChannel(webhook_url="https://hooks.slack.com/test")
        channel._client = mock_client
        # Should not raise — fail-open.
        channel.send(_event())

    @patch("agentkavach.channels.slack.httpx.Client")
    def test_close_calls_client_close(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        channel = SlackChannel(webhook_url="https://hooks.slack.com/test")
        channel._client = mock_client
        channel.close()
        mock_client.close.assert_called_once()

    @patch("agentkavach.channels.slack.httpx.Client")
    def test_200_response_logs_info(self, mock_client_cls):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        channel = SlackChannel(webhook_url="https://hooks.slack.com/test")
        channel._client = mock_client

        with patch("agentkavach.channels.slack.logger") as mock_logger:
            channel.send(_event())
            mock_logger.info.assert_called_once()

    @patch("agentkavach.channels.slack.httpx.Client")
    def test_concurrent_sends(self, mock_client_cls):
        """httpx.Client is thread-safe — verify concurrent sends don't raise."""
        import concurrent.futures

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        channel = SlackChannel(webhook_url="https://hooks.slack.com/test")
        channel._client = mock_client

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(channel.send, _event()) for _ in range(8)]
            for f in futures:
                f.result()  # Should not raise.

        assert mock_client.post.call_count == 8


# ---------------------------------------------------------------------------
# EmailChannel
# ---------------------------------------------------------------------------


class TestEmailChannel:
    def test_empty_api_key_raises(self):
        with pytest.raises(ValueError, match="API key"):
            EmailChannel(api_key="", to_email="x@y.com")

    def test_empty_to_email_raises(self):
        with pytest.raises(ValueError, match="to_email"):
            EmailChannel(api_key="re_xxx", to_email="")

    @patch("agentkavach.channels.email.httpx.Client")
    def test_send_posts_to_resend(self, mock_client_cls):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        channel = EmailChannel(api_key="re_test", to_email="team@acme.com")
        channel._client = mock_client
        channel.send(_event())

        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["to"] == ["team@acme.com"]
        assert payload["from"] == "no-reply@agentkavach.com"
        assert "test-agent" in payload["subject"]
        assert "35.00" in payload["text"]

    @patch("agentkavach.channels.email.httpx.Client")
    def test_send_uses_bearer_auth(self, mock_client_cls):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        channel = EmailChannel(api_key="re_secret_key", to_email="x@y.com")
        channel._client = mock_client
        channel.send(_event())

        headers = mock_client.post.call_args.kwargs.get("headers")
        assert headers["Authorization"] == "Bearer re_secret_key"

    @patch("agentkavach.channels.email.httpx.Client")
    def test_api_key_not_in_payload(self, mock_client_cls):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        channel = EmailChannel(api_key="re_secret", to_email="x@y.com")
        channel._client = mock_client
        channel.send(_event())

        payload = json.dumps(mock_client.post.call_args.kwargs.get("json"))
        assert "re_secret" not in payload


# ---------------------------------------------------------------------------
# PagerDutyChannel
# ---------------------------------------------------------------------------


class TestPagerDutyChannel:
    def test_empty_routing_key_raises(self):
        with pytest.raises(ValueError, match="routing key"):
            PagerDutyChannel(routing_key="")

    @patch("agentkavach.channels.pagerduty.httpx.Client")
    def test_send_posts_to_pd(self, mock_client_cls):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 202
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        channel = PagerDutyChannel(routing_key="R0key123")
        channel._client = mock_client
        channel.send(_event(threshold=1.0, spent=50.0))

        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "pagerduty.com" in call_args[0][0]
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        # Routing key injected from config, not template.
        assert payload["routing_key"] == "R0key123"
        assert payload["payload"]["severity"] == "critical"

    @patch("agentkavach.channels.pagerduty.httpx.Client")
    def test_routing_key_not_from_template(self, mock_client_cls):
        """Routing key must come from config, not from user-supplied template."""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 202
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        channel = PagerDutyChannel(routing_key="REAL_KEY")
        channel._client = mock_client
        evil_template = {
            "routing_key": "EVIL_KEY",
            "event_action": "trigger",
            "payload": {"summary": "test", "severity": "warning", "source": "x"},
        }
        channel.send(_event(), template=evil_template)

        payload = mock_client.post.call_args.kwargs.get("json")
        # The real key must override anything in the template.
        assert payload["routing_key"] == "REAL_KEY"


# ---------------------------------------------------------------------------
# WebhookChannel
# ---------------------------------------------------------------------------


class TestWebhookChannel:
    def test_empty_url_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            WebhookChannel(url="")

    @patch("agentkavach.channels.webhook.httpx.Client")
    def test_send_posts_json(self, mock_client_cls):
        import json

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        channel = WebhookChannel(url="https://example.com/hook")
        channel._client = mock_client
        channel.send(_event())

        mock_client.post.assert_called_once()
        # Phase 102: switched from `json=` to `content=` so the signed
        # bytes exactly match what hits the wire. The payload still
        # contains the rendered template.
        body = mock_client.post.call_args.kwargs.get("content")
        assert isinstance(body, (bytes, bytearray))
        payload = json.loads(body.decode("utf-8"))
        assert payload["event"] == "threshold.breach"
        assert payload["agent"] == "test-agent"

    @patch("agentkavach.channels.webhook.httpx.Client")
    def test_webhook_signature_is_hmac_sha256_hex(self, mock_client_cls):
        """Phase 102 — restored documented HMAC-SHA256 signing.

        The signature MUST equal hmac_sha256(secret, raw_body).hexdigest()
        prefixed with "sha256=", matching the verification snippet in
        ``dashboard/app/public/docs/alerts/page.tsx``.
        """
        import hashlib
        import hmac as _hmac

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        channel = WebhookChannel(url="https://example.com/hook", secret="s3cret")
        channel._client = mock_client
        channel.send(_event())

        kwargs = mock_client.post.call_args.kwargs
        body = kwargs.get("content")
        headers = kwargs.get("headers")
        expected = _hmac.new(b"s3cret", body, hashlib.sha256).hexdigest()
        assert headers["X-AgentKavach-Signature"] == f"sha256={expected}"

    @patch("agentkavach.channels.webhook.httpx.Client")
    def test_webhook_signature_changes_when_payload_changes(self, mock_client_cls):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        channel = WebhookChannel(url="https://example.com/hook", secret="s3cret")
        channel._client = mock_client

        channel.send(_event(agent="agent-A"))
        sig_a = mock_client.post.call_args.kwargs["headers"]["X-AgentKavach-Signature"]

        channel.send(_event(agent="agent-B"))
        sig_b = mock_client.post.call_args.kwargs["headers"]["X-AgentKavach-Signature"]

        assert sig_a != sig_b

    @patch("agentkavach.channels.webhook.httpx.Client")
    def test_webhook_does_not_send_raw_secret_header(self, mock_client_cls):
        """The pre-Phase-102 code shipped the raw shared secret in
        ``X-AgentKavach-Secret``. That was a real security regression and
        is now removed."""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        channel = WebhookChannel(url="https://example.com/hook", secret="s3cret")
        channel._client = mock_client
        channel.send(_event())

        headers = mock_client.post.call_args.kwargs.get("headers")
        assert "X-AgentKavach-Secret" not in headers
        # The raw secret must not appear anywhere in any header value.
        for value in headers.values():
            assert "s3cret" not in str(value)

    @patch("agentkavach.channels.webhook.httpx.Client")
    def test_webhook_includes_timestamp_header_when_signed(self, mock_client_cls):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        channel = WebhookChannel(url="https://example.com/hook", secret="s3cret")
        channel._client = mock_client
        channel.send(_event())

        headers = mock_client.post.call_args.kwargs.get("headers")
        assert "X-AgentKavach-Timestamp" in headers
        # Stored as unix-seconds.
        assert headers["X-AgentKavach-Timestamp"].isdigit()

    @patch("agentkavach.channels.webhook.httpx.Client")
    def test_no_secret_no_signature_header(self, mock_client_cls):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        channel = WebhookChannel(url="https://example.com/hook")
        channel._client = mock_client
        channel.send(_event())

        headers = mock_client.post.call_args.kwargs.get("headers")
        assert "X-AgentKavach-Signature" not in headers
        assert "X-AgentKavach-Secret" not in headers

    @patch("agentkavach.channels.webhook.httpx.Client")
    def test_handles_error(self, mock_client_cls):
        import httpx

        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ConnectError("fail")
        mock_client_cls.return_value = mock_client

        channel = WebhookChannel(url="https://example.com/hook")
        channel._client = mock_client
        # Should not raise.
        channel.send(_event())
