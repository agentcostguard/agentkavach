"""Alert channel integrations — Slack, Email, PagerDuty, Webhook.

Each channel module exposes a ``send(event, template, config)`` function
and a ``DEFAULT_TEMPLATE`` dict.  Channel handlers are registered with
the ``AlertDispatcher`` at client init time.
"""
