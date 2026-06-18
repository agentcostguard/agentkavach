"""AgentKavach client: multi-provider LLM wrapper with hard budget limits.

Drop-in wrapper for OpenAI, Anthropic, Google, and Mistral that adds budget
enforcement, cost tracking, and alert dispatch.  All LLM calls pass
through the engine's pre/post-flight checks.

Usage:
    from agentkavach import AgentKavach, Budget

    guard = AgentKavach(
        provider="openai",
        llm_key="sk-...",
        agent_name="research-bot",
        budget=Budget.daily(50),
    )
    response = guard.create(model="gpt-4o", messages=[...])
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import yaml

from agentkavach.alerts import (
    VALID_CHANNEL_TYPES,
    AlertDispatcher,
    AlertRule,
    ChannelConfig,
)
from agentkavach.budget import Budget
from agentkavach.buffer import Buffer
from agentkavach.engine import SpendEngine
from agentkavach.providers import UsageRecord
from agentkavach.providers.anthropic import calculate_cost as anthropic_cost
from agentkavach.providers.anthropic import count_tokens as anthropic_count
from agentkavach.providers.anthropic import parse_usage as anthropic_parse
from agentkavach.providers.google import calculate_cost as google_cost
from agentkavach.providers.google import count_tokens as google_count
from agentkavach.providers.google import parse_usage as google_parse
from agentkavach.providers.mistral import calculate_cost as mistral_cost
from agentkavach.providers.mistral import count_tokens as mistral_count
from agentkavach.providers.mistral import parse_usage as mistral_parse
from agentkavach.providers.openai import calculate_cost as openai_cost
from agentkavach.providers.openai import count_tokens as openai_count
from agentkavach.providers.openai import parse_usage as openai_parse
from agentkavach.sender import create_tracer_provider
from agentkavach.stream import StreamWrapper

logger = logging.getLogger(__name__)

# Phase 53: cap stored prompt text at 2 kB. Mirrored server-side in
# server/ingest.py:MAX_PROMPT_CHARS so direct API users can't bypass.
# Two layers because:
#   1. SDK cap keeps wire payloads small (cheaper Kafka, cheaper egress).
#   2. Server cap is the actual security boundary — SDK could be skipped.
MAX_PROMPT_CHARS: int = 2048
_PROMPT_TRUNCATION_SUFFIX = "... [truncated]"


def _truncate_prompt(value: Optional[str]) -> Optional[str]:
    """Truncate prompt to ``MAX_PROMPT_CHARS`` with a marker suffix."""
    if not value:
        return value
    if len(value) <= MAX_PROMPT_CHARS:
        return value
    head_len = MAX_PROMPT_CHARS - len(_PROMPT_TRUNCATION_SUFFIX)
    if head_len <= 0:
        return _PROMPT_TRUNCATION_SUFFIX[:MAX_PROMPT_CHARS]
    return value[:head_len] + _PROMPT_TRUNCATION_SUFFIX


# Provider name -> (parse_usage, calculate_cost, count_tokens) mapping.
_PROVIDER_FUNCS: Dict[str, tuple[Any, Any, Any]] = {
    "openai": (openai_parse, openai_cost, openai_count),
    "anthropic": (anthropic_parse, anthropic_cost, anthropic_count),
    "google": (google_parse, google_cost, google_count),
    "mistral": (mistral_parse, mistral_cost, mistral_count),
}


class _ChatCompletions:
    """Namespace that mirrors ``openai.chat.completions``."""

    def __init__(self, guard: AgentKavach) -> None:
        self._guard = guard

    def create(self, **kwargs: Any) -> Any:
        return self._guard._safe_call(kwargs)


class _Chat:
    """Namespace that mirrors ``openai.chat``."""

    def __init__(self, guard: AgentKavach) -> None:
        self.completions = _ChatCompletions(guard)


class _Messages:
    """Namespace that mirrors ``anthropic.messages``."""

    def __init__(self, guard: AgentKavach) -> None:
        self._guard = guard

    def create(self, **kwargs: Any) -> Any:
        return self._guard._safe_call(kwargs)


class AgentKavach:
    """Multi-provider LLM wrapper with hard budget limits.

    Intercepts all LLM calls with:
    1. Pre-flight: in-memory budget check (~0.1ms)
    2. Provider call: delegates to the real provider SDK
    3. Post-flight: record actual cost, check thresholds, buffer event

    Supports OpenAI, Anthropic, Google, and Mistral providers with a unified
    budget enforcement layer.  Multiple AgentKavach instances can share
    a single budget via ``Budget.org_budget()``.

    Fail-open: if AgentKavach internals error, the LLM call still
    proceeds.  Only ``BudgetExceededError`` propagates.
    """

    # Phase 53: emit the save_prompts privacy warning exactly once per
    # process. A long-lived app that constructs many AgentKavach
    # instances (e.g. one per agent) would otherwise spam the operator
    # with the same warning on every init.
    _save_prompts_warning_emitted: bool = False

    def __init__(
        self,
        provider: str = "openai",
        llm_key: str = "",
        agent_name: str = "default",
        budget: Optional[Budget] = None,
        org_budget: Optional[Budget] = None,
        channels: Optional[Sequence[ChannelConfig]] = None,
        on_kill: Optional[Callable[[], None]] = None,
        api_key: str = "",
        run_id: Optional[str] = None,
        # Guardrail limits (per-run).
        max_tokens_per_run: Optional[int] = None,
        max_calls_per_run: Optional[int] = None,
        max_runtime_seconds: Optional[float] = None,
        detect_loops: bool = False,
        loop_threshold: int = 3,
        save_prompts: bool = False,
        fail_on_error: bool = False,
        # Legacy params (functional, prefer llm_key + channels for new code)
        openai_api_key: str = "",
        anthropic_api_key: str = "",
        google_api_key: str = "",
        alerts: Optional[Sequence[AlertRule]] = None,
        thresholds: Optional[tuple[float, ...]] = None,
        endpoint: Optional[str] = None,
        buffer_path: Optional[str] = None,
        slack_webhook_url: str = "",
        resend_api_key: str = "",
        alert_email: str = "",
        pagerduty_routing_key: str = "",
        webhook_url: str = "",
        webhook_secret: str = "",
    ) -> None:
        # AgentKavach API key (for telemetry export). Required — the SDK
        # never reads it from the environment. Callers pass it explicitly,
        # e.g. AgentKavach(api_key=os.environ["AGENTKAVACH_API_KEY"], ...).
        self._cg_api_key = api_key
        self._save_prompts = save_prompts
        self._fail_on_error = fail_on_error
        self._provider = provider.lower()

        # Phase 53: privacy warning when prompt persistence is on. Only
        # warn once per process to avoid log noise when many agents share
        # the same opt-in config.
        if self._save_prompts and not AgentKavach._save_prompts_warning_emitted:
            logger.warning(
                "save_prompts=True — prompt text will be persisted to the "
                "AgentKavach backend (capped at 2 kB, retained up to 30 "
                "days). See https://agentkavach.com/public/docs/privacy "
                "for details."
            )
            AgentKavach._save_prompts_warning_emitted = True

        if self._provider not in _PROVIDER_FUNCS:
            raise ValueError(
                f"Unknown provider {provider!r}, must be one of: "
                + ", ".join(sorted(_PROVIDER_FUNCS))
            )

        # Resolve LLM API key from explicit arguments only: llm_key beats the
        # legacy provider-specific params. The SDK never reads it from the
        # environment — pass it yourself, e.g. llm_key=os.environ["OPENAI_API_KEY"].
        self._llm_key = llm_key or openai_api_key or anthropic_api_key or google_api_key

        # Both keys are mandatory. Missing them is a configuration mistake, so
        # fail loudly at construction rather than silently doing nothing useful.
        # (An expired or revoked key is a different case: it is still passed
        # through and the backend rejects telemetry quietly at runtime.)
        if not self._cg_api_key:
            raise ValueError(
                "api_key is required. Pass your AgentKavach key explicitly, e.g. "
                'AgentKavach(api_key=os.environ["AGENTKAVACH_API_KEY"], ...). '
                "The SDK does not read it from the environment."
            )
        if not self._llm_key:
            raise ValueError(
                "llm_key is required. Pass your provider key explicitly, e.g. "
                'AgentKavach(llm_key=os.environ["OPENAI_API_KEY"], ...). '
                "The SDK does not read it from the environment."
            )

        # Legacy attribute compat (some tests inspect these directly).
        self._openai_api_key = self._llm_key if self._provider == "openai" else ""
        self._anthropic_api_key = self._llm_key if self._provider == "anthropic" else ""
        self._google_api_key = self._llm_key if self._provider == "google" else ""

        self._budget_explicit = budget is not None

        # If the caller passed an org-level budget (e.g. Budget.org_budget()),
        # route it to org_budget so the server aggregates all agents' spend.
        if budget is not None and budget.shared_name == "__org__":
            if org_budget is None:
                org_budget = budget
            budget = None  # no per-agent budget

        if budget is None:
            budget = Budget.daily(limit=100.0)
            if self._budget_explicit:
                # Budget was provided but routed to org — don't log default msg
                pass
            else:
                logger.info("No budget specified, defaulting to $100/day")

        self._budget = budget
        self._agent_name = agent_name

        # Run tracking — auto-generate UUID if not provided.
        if run_id is None:
            import uuid

            run_id = str(uuid.uuid4())
        self._run_id = run_id

        # SDK-dispatched channels are delivered client-side at their CONFIGURED
        # thresholds, so the engine must fire at those (not just the defaults).
        # Backend-dispatched channels are evaluated + delivered server-side, so
        # their thresholds don't need to be added here.
        _engine_thresholds = thresholds or (0.70, 0.90, 1.0)
        _sdk_thresholds = {
            c.threshold for c in (channels or []) if getattr(c, "dispatch", "backend") == "sdk"
        }
        if _sdk_thresholds:
            _engine_thresholds = tuple(sorted(set(_engine_thresholds) | _sdk_thresholds))

        # Core engine: all budget math happens here.
        self._engine = SpendEngine(
            budget=budget,
            agent_name=agent_name,
            thresholds=_engine_thresholds,
            on_kill=on_kill,
            org_budget=org_budget,
            max_tokens_per_run=max_tokens_per_run,
            max_calls_per_run=max_calls_per_run,
            max_runtime_seconds=max_runtime_seconds,
            detect_loops=detect_loops,
            loop_threshold=loop_threshold,
        )

        # Build alert rules from channels (new API) or alerts (legacy).
        if channels is not None:
            rules = _build_rules_from_channels(channels)
        elif alerts is not None:
            rules = tuple(alerts)
        else:
            rules = None

        self._dispatcher = AlertDispatcher(
            rules=rules or AlertDispatcher.rules,
        )

        # Register the built-in "kill" channel.
        if on_kill is not None:
            self._dispatcher.register_channel(
                "kill",
                lambda event, template=None: on_kill(),
            )

        # Register channel handlers.
        self._channels: List[Any] = []
        # Keep the raw ChannelConfig list so the sync payload can recover the
        # email recipient even when SDK-side registration failed (e.g. backend
        # is the email sender — customer provides only `to`, never `api_key`).
        self._channel_configs: List[ChannelConfig] = list(channels) if channels else []
        if channels is not None:
            self._register_channel_configs(channels)
        else:
            self._register_channels(
                slack_webhook_url=slack_webhook_url,
                resend_api_key=resend_api_key,
                alert_email=alert_email,
                pagerduty_routing_key=pagerduty_routing_key,
                webhook_url=webhook_url,
                webhook_secret=webhook_secret,
            )

        # Buffer for durability.
        self._buffer = Buffer(path=buffer_path)

        # OTel tracer (optional, only if AgentKavach API key is present).
        self._tracer = None
        if self._cg_api_key:
            try:
                tp = create_tracer_provider(
                    api_key=self._cg_api_key,
                    agent_name=agent_name,
                    endpoint=endpoint,
                    buffer=self._buffer,
                    # Phase 59 (B3): receive backend permanent-reject
                    # signals (tier_agent_limit / daily_limit /
                    # org_budget_exceeded) so we can flip
                    # ``_backend_paused`` and stop wasting LLM spend.
                    on_backend_reject=self._handle_backend_reject,
                )
                self._tracer = tp.get_tracer("agentkavach")
                self._tracer_provider = tp
            except Exception:
                logger.warning("Failed to initialize OTel tracer, continuing without telemetry")

        # Sync config to backend (non-blocking daemon thread).
        # Same resolution logic as the OTel exporter: explicit arg >
        # AGENTKAVACH_BACKEND_URL env var > key-prefix routing. Without
        # this, _sync_config would silently no-op for keys that should
        # route via prefix (the common case for prod/dev users).
        from agentkavach.sender import resolve_backend_url

        self._endpoint = (
            endpoint
            or os.environ.get("AGENTKAVACH_BACKEND_URL")
            or (resolve_backend_url(self._cg_api_key) if self._cg_api_key else "")
        )
        if self._cg_api_key:
            self._sync_config()

        # Lazy-initialized provider clients.
        self._openai_client: Any = None
        self._anthropic_client: Any = None
        self._google_client: Any = None
        self._mistral_client: Any = None

        # Phase 59 (B1): side channel for the model name the caller
        # asked for. The new ``google-genai`` SDK uses Pydantic v2
        # response objects whose schema does NOT include a ``model``
        # field, so mutating ``resp.model`` raises ``ValueError`` (not
        # Attribute/TypeError) and the prior best-effort try/except let
        # it escape, crashing every non-streaming Gemini call. The side
        # channel avoids touching the provider response object — we
        # keep the requested model on ``self`` and ``_post_flight``
        # prefers it over the (potentially empty) parsed value.
        self._last_model_requested: Optional[str] = None

        # Phase 65: input-token count for the most recent call. Captured
        # in ``_safe_call`` before the provider call so the streaming
        # completion callback can price the input side (it only learns
        # ``output_tokens`` after the stream drains). Defaulted here for
        # callers that invoke ``_on_stream_complete`` directly (tests).
        self._last_input_tokens: int = 0

        # Phase 59 (B3): set when the backend has signalled that this
        # process should stop ingesting events for this agent (tier
        # agent-limit, org-wide daily-limit, or org-budget-exceeded
        # 429s). Subsequent ``pre_flight`` calls raise
        # ``BudgetExceededError`` BEFORE the LLM is even called, so
        # the customer doesn't keep burning provider spend on events
        # that will just be dropped server-side. Cleared on next
        # process start — operators must reactivate / upgrade / wait
        # for the budget rollover.
        self._backend_paused: bool = False
        self._backend_paused_reason: Optional[str] = None
        # Phase 125: ensure the on_kill callback fires at most once when a
        # backend-side stop (dashboard Kill button / tier reject) is detected.
        self._backend_kill_fired: bool = False

        # Public namespaces mirroring each provider's native API surface.
        self.chat = _Chat(self)
        self.messages = _Messages(self)

    # -- Public API ---------------------------------------------------------

    def create(self, **kwargs: Any) -> Any:
        """Unified API for all providers.

        Accepts the same keyword arguments as the native provider SDK
        (model, messages, contents, etc.). Routes to the correct
        provider based on the ``provider`` set at construction.
        """
        return self._safe_call(kwargs)

    def generate_content(self, *, model: str = "", contents: Any = None, **kwargs: Any) -> Any:
        """Google-style API: wraps ``client.models.generate_content``."""
        kwargs["model"] = model
        kwargs["contents"] = contents
        return self._safe_call(kwargs)

    # -- Factory methods ----------------------------------------------------

    @staticmethod
    def channel(
        channel_type: Any,
        threshold: float,
        *,
        webhook_url: str = "",
        to: str = "",
        routing_key: str = "",
        url: str = "",
        secret: str = "",
        api_key: str = "",
        template: Optional[Dict[str, Any]] = None,
        dispatch: str = "backend",
    ) -> ChannelConfig:
        """Create a ``ChannelConfig`` for inline configuration.

        Accepts ``ChannelType`` enum or raw strings::

            from agentkavach import AgentKavach, ChannelType

            guard = AgentKavach(
                channels=[
                    AgentKavach.channel(ChannelType.EMAIL, threshold=0.50,
                                      to="team@acme.com"),
                    AgentKavach.channel(ChannelType.SLACK, threshold=0.80,
                                      webhook_url="https://hooks.slack.com/..."),
                    AgentKavach.channel(ChannelType.PAGERDUTY, threshold=0.95,
                                      routing_key="R0..."),
                    AgentKavach.channel(ChannelType.KILL, threshold=1.0),
                ],
            )
        """
        return ChannelConfig(
            channel_type=channel_type,
            threshold=threshold,
            webhook_url=webhook_url,
            to=to,
            routing_key=routing_key,
            url=url,
            secret=secret,
            api_key=api_key,
            template=template,
            dispatch=dispatch,
        )

    @staticmethod
    def alert(
        threshold: float,
        channels: Optional[Sequence[str]] = None,
        template: Optional[Dict[str, Any]] = None,
    ) -> AlertRule:
        """Create an ``AlertRule`` (legacy convenience method).

        Prefer ``AgentKavach.channel()`` for new code.
        """
        return AlertRule(
            threshold=threshold,
            channels=tuple(channels) if channels else ("email",),
            template=template,
        )

    # -- Channel registration -----------------------------------------------

    def _register_channel_configs(self, channels: Sequence[ChannelConfig]) -> None:
        """Register client-side handlers for the channels the SDK delivers.

        Slack/PagerDuty/Webhook are SDK-delivered only when ``dispatch="sdk"``
        (internal / on-prem / firewalled endpoints the backend can't reach).
        With the default ``dispatch="backend"`` they are delivered by the cloud,
        so we do NOT register them here — which also means a backend-dispatched
        channel can never be double-delivered.

        Email is special: it is delivered by the backend (our Resend key) unless
        the customer supplies their own Resend ``api_key``, in which case the SDK
        sends it client-side (the legacy path). ``kill`` is registered separately
        via ``on_kill``.
        """
        for ch in channels:
            if ch.channel_type == "kill":
                continue  # Handled separately via on_kill registration.

            try:
                if ch.channel_type == "email":
                    from agentkavach.channels.email import EmailChannel

                    resend_key = ch.api_key
                    # No api_key → backend-dispatch mode (customer provides only
                    # `to`, our Resend key sends). The recipient still reaches the
                    # backend via the sync-config payload (AlertConfig.target).
                    if not resend_key:
                        continue
                    handler = EmailChannel(api_key=resend_key, to_email=ch.to)
                    self._dispatcher.register_channel("email", handler.send)
                    self._channels.append(handler)
                    continue

                # slack / pagerduty / webhook: SDK-delivered only in "sdk" mode.
                if getattr(ch, "dispatch", "backend") != "sdk":
                    continue  # backend-dispatched — the cloud delivers it.

                if ch.channel_type == "slack":
                    from agentkavach.channels.slack import SlackChannel

                    handler = SlackChannel(webhook_url=ch.webhook_url)
                    self._dispatcher.register_channel("slack", handler.send)
                    self._channels.append(handler)

                elif ch.channel_type == "pagerduty":
                    from agentkavach.channels.pagerduty import PagerDutyChannel

                    handler = PagerDutyChannel(routing_key=ch.routing_key)
                    self._dispatcher.register_channel("pagerduty", handler.send)
                    self._channels.append(handler)

                elif ch.channel_type == "webhook":
                    from agentkavach.channels.webhook import WebhookChannel

                    handler = WebhookChannel(url=ch.url, secret=ch.secret)
                    self._dispatcher.register_channel("webhook", handler.send)
                    self._channels.append(handler)

            except Exception:
                logger.warning("Failed to register %s channel", ch.channel_type, exc_info=True)

    def _register_channels(
        self,
        slack_webhook_url: str,
        resend_api_key: str,
        alert_email: str,
        pagerduty_routing_key: str,
        webhook_url: str,
        webhook_secret: str,
    ) -> None:
        """Auto-register channel handlers based on available credentials (legacy)."""
        if slack_webhook_url:
            try:
                from agentkavach.channels.slack import SlackChannel

                ch = SlackChannel(webhook_url=slack_webhook_url)
                self._dispatcher.register_channel("slack", ch.send)
                self._channels.append(ch)
            except Exception:
                logger.warning("Failed to register Slack channel", exc_info=True)

        if resend_api_key and alert_email:
            try:
                from agentkavach.channels.email import EmailChannel

                ch = EmailChannel(api_key=resend_api_key, to_email=alert_email)
                self._dispatcher.register_channel("email", ch.send)
                self._channels.append(ch)
            except Exception:
                logger.warning("Failed to register email channel", exc_info=True)

        if pagerduty_routing_key:
            try:
                from agentkavach.channels.pagerduty import PagerDutyChannel

                ch = PagerDutyChannel(routing_key=pagerduty_routing_key)
                self._dispatcher.register_channel("pagerduty", ch.send)
                self._channels.append(ch)
            except Exception:
                logger.warning("Failed to register PagerDuty channel", exc_info=True)

        if webhook_url:
            try:
                from agentkavach.channels.webhook import WebhookChannel

                ch = WebhookChannel(url=webhook_url, secret=webhook_secret)
                self._dispatcher.register_channel("webhook", ch.send)
                self._channels.append(ch)
            except Exception:
                logger.warning("Failed to register webhook channel", exc_info=True)

    # -- Properties ---------------------------------------------------------

    @property
    def engine(self) -> SpendEngine:
        """Expose the engine for inspection (spent, remaining, etc.)."""
        return self._engine

    @property
    def spent(self) -> float:
        return self._engine.spent

    @property
    def remaining(self) -> float:
        return self._engine.remaining

    @property
    def save_prompts(self) -> bool:
        """Whether prompt text is included in telemetry events."""
        return self._save_prompts

    # -- Provider clients ---------------------------------------------------

    def _get_openai_client(self) -> Any:
        if self._openai_client is None:
            from openai import OpenAI

            self._openai_client = OpenAI(api_key=self._llm_key)
        return self._openai_client

    def _get_anthropic_client(self) -> Any:
        if self._anthropic_client is None:
            from anthropic import Anthropic

            self._anthropic_client = Anthropic(api_key=self._llm_key)
        return self._anthropic_client

    def _get_google_client(self) -> Any:
        if self._google_client is None:
            # Phase 45: migrated from the deprecated ``google-generativeai``
            # package to ``google-genai`` (imported as ``google.genai``).
            # The old SDK's module-level ``configure()`` + ``GenerativeModel``
            # surface is replaced by a stateful ``Client`` whose
            # ``client.models`` namespace exposes ``generate_content``,
            # ``generate_content_stream``, and ``count_tokens``.
            from google import genai

            self._google_client = genai.Client(api_key=self._llm_key)
        return self._google_client

    def _get_mistral_client(self) -> Any:
        if self._mistral_client is None:
            from mistralai import Mistral

            self._mistral_client = Mistral(api_key=self._llm_key)
        return self._mistral_client

    # -- Call execution -----------------------------------------------------

    def _safe_call(self, kwargs: Dict[str, Any]) -> Any:
        """Execute an LLM call with budget enforcement.

        Uses native provider token counting for accurate input cost
        estimation before the call.  OpenAI counting is local (~0.1ms
        via tiktoken).  Anthropic/Google counting makes one API call
        (~100-200ms) but returns exact input tokens.

        Fail-open: pre/post-flight errors are caught and logged.
        The only exception that propagates is ``BudgetExceededError``.
        """
        # Phase 59 (B3): if the backend has signalled a permanent
        # reject (tier_agent_limit / daily_limit / org_budget_exceeded)
        # via a prior 429, fail-fast BEFORE we call the LLM. Otherwise
        # the customer keeps burning provider spend on calls whose
        # telemetry will just be dropped server-side. The reason is
        # surfaced in the exception so the caller can act on it.
        if self._backend_paused:
            from agentkavach.exceptions import BudgetExceededError

            reason = self._backend_paused_reason or "backend_rejected"
            # Phase 125: a backend stop (dashboard Kill button or tier
            # reject) should run the customer's on_kill teardown, just like
            # an in-process budget kill. Invoke it once, here on the call
            # thread (never on the telemetry thread that set the flag), and
            # fail-open so a raising callback never masks the stop.
            if not self._backend_kill_fired:
                self._backend_kill_fired = True
                if self._engine.on_kill is not None:
                    try:
                        self._engine.on_kill()
                    except Exception:
                        logger.exception("on_kill callback raised on backend stop")
            raise BudgetExceededError(
                f"AgentKavach backend rejected ingest for agent "
                f"{self._agent_name!r} (reason={reason!r}). Calls are "
                f"paused to avoid wasted LLM spend. Resolve the cause "
                f"(reactivate the agent, upgrade tier, or wait for the "
                f"daily reset) and restart the process."
            )

        model = kwargs.get("model", "unknown")
        stream = kwargs.get("stream", False)

        messages = self._extract_messages(kwargs)
        # Capture prompt text for save_prompts feature.
        self._last_prompt: Optional[str] = None
        if self._save_prompts and messages:
            try:
                self._last_prompt = messages[0].get("content", "") if messages else None
            except Exception:
                pass

        # Native token counting for accurate input cost.
        input_tokens: Optional[int] = None
        try:
            input_tokens = self._count_input_tokens(model, messages)
        except Exception:
            logger.warning("Native token counting failed, falling back to heuristic", exc_info=True)

        # Phase 65: stash the input-token count on the instance so the
        # streaming completion callback (``_on_stream_complete``) can
        # price both halves of the call. Streams only learn
        # ``output_tokens`` after the chunks have been drained — the
        # input side was already known here, before the provider call,
        # and would otherwise be lost. Without this the DB row for a
        # streaming call lands with ``cost=0`` (the bug Phase 65 fixes).
        self._last_input_tokens: int = input_tokens or 0

        try:
            self._engine.pre_flight(model, messages, input_tokens=input_tokens)
        except Exception as exc:
            from agentkavach.exceptions import BudgetExceededError, GuardrailError

            if isinstance(exc, (BudgetExceededError, GuardrailError)):
                # Trigger thresholds and on_kill before raising.
                events = self._engine.check_thresholds()
                for event in events:
                    self._dispatcher.dispatch(event)
                raise
            if self._fail_on_error:
                logger.error("Pre-flight failed with fail_on_error=True: %s", exc)
                if self._engine.on_kill is not None:
                    try:
                        self._engine.on_kill()
                    except Exception:
                        logger.exception("on_kill callback raised during fail_on_error")
                else:
                    logger.warning(
                        "fail_on_error triggered but no on_kill callback defined for agent %s "
                        "— the agent process will NOT be terminated",
                        self._engine.agent_name,
                    )
                raise
            logger.warning("Pre-flight check failed (non-budget): %s", exc)

        # Phase 123: measure the provider call so the Performance page can
        # report real per-call latency. ``time.monotonic`` is immune to
        # wall-clock adjustments. The clock starts right before the LLM
        # call and stops either when it returns (success) or raises
        # (error) — pre-flight budget/guardrail time is intentionally
        # excluded so the latency reflects the provider, not our checks.
        call_start = time.monotonic()
        try:
            response = self._call_provider(kwargs)
        except Exception:
            # Phase 123: the provider call itself failed. Emit a failure
            # event (status="error") with the measured latency so the
            # Performance page shows the failure + how long it took
            # before erroring, then RE-RAISE the original exception
            # unchanged — AgentKavach never swallows provider errors.
            # Recording is fully best-effort: any error while building or
            # emitting the failure event is swallowed so the original
            # provider exception is what reaches the caller.
            duration_ms = max(0, round((time.monotonic() - call_start) * 1000))
            try:
                self._record_failure(model, duration_ms)
            except Exception:
                logger.debug("Failure-event recording raised; ignoring", exc_info=True)
            raise

        duration_ms = max(0, round((time.monotonic() - call_start) * 1000))

        if stream:
            # Streaming latency is best-effort: the call to open the stream
            # returned quickly here, but the real cost/tokens (and the
            # event row) are finalized in ``_on_stream_complete`` after the
            # generator drains. We don't have a reliable end-to-end
            # duration for streams, so the stream event keeps duration_ms=0
            # and status defaults to success (NULL) server-side.
            return StreamWrapper(
                stream=response,
                model=model,
                engine=self._engine,
                on_complete=self._on_stream_complete,
            )

        try:
            # Pass the requested model as a fallback. Some provider
            # SDKs (notably google-genai) return immutable response
            # objects where ``resp.model = model_name`` silently fails —
            # parse_usage then reads "unknown" and pricing lookup
            # returns $0. Phase 33 fix: trust the kwargs over the response.
            self._post_flight(response, requested_model=model, duration_ms=duration_ms)
        except Exception as exc:
            from agentkavach.exceptions import GuardrailError

            if isinstance(exc, GuardrailError):
                raise
            if self._fail_on_error:
                logger.error("Post-flight failed with fail_on_error=True: %s", exc)
                if self._engine.on_kill is not None:
                    try:
                        self._engine.on_kill()
                    except Exception:
                        logger.exception("on_kill callback raised during fail_on_error")
                else:
                    logger.warning(
                        "fail_on_error triggered but no on_kill callback defined for agent %s "
                        "— the agent process will NOT be terminated",
                        self._engine.agent_name,
                    )
                raise
            logger.warning("Post-flight recording failed", exc_info=True)

        return response

    def _count_input_tokens(self, model: str, messages: List[Dict[str, str]]) -> int:
        """Count input tokens using the provider's native API.

        - OpenAI: local tiktoken (~0.1ms, no network call)
        - Anthropic: client.messages.count_tokens() (~100-200ms network call)
        - Google: model.count_tokens() (~100-200ms network call)
        """
        _, _, count_fn = _PROVIDER_FUNCS[self._provider]

        if self._provider == "openai":
            return count_fn(model, messages)
        elif self._provider == "anthropic":
            client = self._get_anthropic_client()
            return count_fn(model, messages, client=client)
        elif self._provider == "google":
            client = self._get_google_client()
            return count_fn(model, messages, client=client)
        elif self._provider == "mistral":
            return count_fn(model, messages)

        return count_fn(model, messages)

    def _extract_messages(self, kwargs: Dict[str, Any]) -> List[Dict[str, str]]:
        """Normalize provider-specific input into a messages list for pre-flight."""
        if "messages" in kwargs:
            return kwargs["messages"]
        contents = kwargs.get("contents")
        if contents is not None:
            if isinstance(contents, str):
                return [{"role": "user", "content": contents}]
            if isinstance(contents, list):
                return [{"role": "user", "content": str(c)} for c in contents]
        return []

    def _call_provider(self, kwargs: Dict[str, Any]) -> Any:
        """Dispatch to the correct provider SDK."""
        if self._provider == "openai":
            client = self._get_openai_client()
            call_kwargs = {k: v for k, v in kwargs.items() if k != "contents"}
            return client.chat.completions.create(**call_kwargs)

        if self._provider == "anthropic":
            client = self._get_anthropic_client()
            call_kwargs = {k: v for k, v in kwargs.items() if k != "contents"}
            return client.messages.create(**call_kwargs)

        if self._provider == "google":
            # Phase 45: ``google-genai`` SDK. The streaming and
            # non-streaming paths are split into two methods on
            # ``client.models``: ``generate_content`` (sync) and
            # ``generate_content_stream`` (iterator). Both accept the
            # same kwargs: ``model``, ``contents``, optional ``config``.
            client = self._get_google_client()
            model_name = kwargs.get("model", "")
            contents = kwargs.get("contents", "")
            stream = bool(kwargs.get("stream"))
            # The new SDK does not accept arbitrary generation kwargs
            # at the top level — they live under a ``config`` object.
            # Pass any caller-provided ``config`` through verbatim, and
            # drop ``stream``/``model``/``contents``/``messages`` which
            # are handled explicitly above.
            call_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in ("model", "contents", "messages", "stream")
            }
            if stream:
                # Streaming responses are iterators of chunk objects
                # whose shapes (``chunk.text``, ``chunk.usage_metadata``)
                # are identical to the old SDK — Phase 44's
                # ``_count_chunk_tokens`` already understands them.
                return client.models.generate_content_stream(
                    model=model_name, contents=contents, **call_kwargs
                )
            resp = client.models.generate_content(
                model=model_name, contents=contents, **call_kwargs
            )
            # Phase 59 (B1): record the requested model on a side
            # channel rather than mutating the provider response. The
            # new ``google-genai`` SDK uses Pydantic v2 models whose
            # schema does NOT include a ``model`` field, so assignment
            # raises ``ValueError`` (NOT AttributeError / TypeError) —
            # the prior try/except was too narrow and the exception
            # escaped, crashing every non-streaming Gemini call.
            # ``_post_flight`` reads ``self._last_model_requested``
            # and prefers it over ``usage.model`` when the parser
            # couldn't extract one.
            self._last_model_requested = model_name
            return resp

        if self._provider == "mistral":
            client = self._get_mistral_client()
            stream = bool(kwargs.get("stream"))
            # ``chat.complete`` does not accept ``stream=True``;
            # streaming requires the dedicated ``chat.stream`` method.
            call_kwargs = {k: v for k, v in kwargs.items() if k not in ("contents", "stream")}
            if stream:
                return client.chat.stream(**call_kwargs)
            return client.chat.complete(**call_kwargs)

        raise ValueError(f"Unknown provider: {self._provider}")

    def _build_event(
        self,
        *,
        agent_name: str,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        duration_ms: int,
        status: str,
        idempotency_key: str,
        partial: bool,
    ) -> Dict[str, Any]:
        """Construct one ingest-event dict.

        Phase 123: single source of truth for the event schema so the
        success path and the failure path can never drift. The field
        names match the backend ``IngestEvent`` model exactly (so a
        disk-buffer replay validates server-side). ``status`` is
        ``"success"`` or ``"error"``; ``cost``/tokens default to 0 for a
        failed call. The prompt (when ``save_prompts`` is on) is attached
        by the caller, not here, since a failed call has no useful prompt
        capture semantics beyond what the success path already does.
        """
        return {
            "agent_name": agent_name,
            "model": model,
            "provider": provider,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "duration_ms": duration_ms,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "idempotency_key": idempotency_key,
            "run_id": self._run_id,
            "partial": partial,
        }

    def _record_failure(self, model: str, duration_ms: int) -> None:
        """Phase 123: emit a status="error" event for a failed provider call.

        A failed call has no usage to price, so cost and tokens are 0.
        The event still rides the normal transport — OTel span (when a
        tracer is active) or the disk buffer (passthrough mode) — so the
        backend's events.status column records the failure and the
        Performance page's success-rate / failure-count reflect it.

        Pre-flight ``BudgetExceededError`` / ``GuardrailError`` never
        reach here: they short-circuit before the provider call, so a
        budget block is NOT mis-recorded as a provider failure.
        """
        idem_key = str(uuid.uuid4())
        failure_event = self._build_event(
            agent_name=self._agent_name,
            model=model,
            provider=self._provider,
            input_tokens=0,
            output_tokens=0,
            cost=0.0,
            duration_ms=duration_ms,
            status="error",
            idempotency_key=idem_key,
            partial=False,
        )
        # Same tracer-vs-buffer guard as the success path: when OTel is
        # active the exporter is the canonical transport, so writing to
        # the disk buffer here would double-send.
        if self._tracer is None:
            self._buffer.write(failure_event)
        self._trace_call(
            UsageRecord(
                provider=self._provider,
                model=model,
                input_tokens=0,
                output_tokens=0,
            ),
            cost=0.0,
            partial=False,
            idempotency_key=idem_key,
            duration_ms=duration_ms,
            status="error",
        )

    def _post_flight(self, response: Any, requested_model: str = "", duration_ms: int = 0) -> None:
        """Record spend, check thresholds, buffer event, trace span.

        *requested_model* is the model name the caller asked for; we
        always prefer it over whatever the provider echoes back so that
        the dashboard's Model Breakdown collapses correctly.

        Phase 86: ALWAYS overwrite ``usage.model`` with the requested
        name when present. Pre-Phase-86, OpenAI customers calling
        ``guard.create(model="gpt-4o-mini")`` saw two rows in the
        dashboard's Model Breakdown — non-streaming responses echoed
        the versioned form (``gpt-4o-mini-2024-07-18``) while stream
        chunks echoed the alias (``gpt-4o-mini``). Same underlying
        model, fragmented usage. The customer wrote ``"gpt-4o-mini"``
        in their code; that is the stable identifier they expect on
        their dashboard. Pricing still resolves correctly because the
        pricing table contains both forms with identical prices and
        ``estimate_cost`` normalizes aliases / date suffixes via
        ``_prefix_match`` regardless.

        Phase 33 (preserved): if no ``requested_model`` is available AND
        the provider returned ``"unknown"`` (e.g. Google's immutable
        response object), pricing still misses → cost = $0 → budgets
        never trip. The combined Phase 33 + Phase 86 logic below
        handles both shapes.
        """
        from dataclasses import replace

        parse_fn, cost_fn, _ = _PROVIDER_FUNCS[self._provider]
        usage = parse_fn(response)
        # Phase 59 (B1): the side channel (``_last_model_requested``)
        # carries the model name through provider calls whose response
        # objects can't be mutated (Google's Pydantic v2 response).
        # Prefer it over the ``requested_model`` arg when both are
        # present — they're normally identical, but the side channel
        # is set even for callers that bypass ``create()`` (e.g.
        # tests calling ``_post_flight`` directly after
        # ``_call_provider``).
        effective_requested = self._last_model_requested or requested_model
        if effective_requested:
            # Phase 86: always record the model the customer requested,
            # not the provider's echo. The provider echo is logged at
            # DEBUG level when it differs so operators can still see
            # which versioned snapshot served the call.
            if usage.model and usage.model != effective_requested:
                logger.debug(
                    "Provider echoed model %r for requested %r — recording "
                    "as %r so dashboard Model Breakdown stays consistent.",
                    usage.model,
                    effective_requested,
                    effective_requested,
                )
            usage = replace(usage, model=effective_requested)
        cost = cost_fn(usage)

        # Extract tool name from response for loop detection.
        tool_name = self._extract_tool_name(response)

        self._engine.post_flight(
            model=usage.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            tool_name=tool_name,
        )

        events = self._engine.check_thresholds()
        for event in events:
            self._dispatcher.dispatch(event)

        # Phase 35: align with `IngestEvent` field names so a buffer
        # replay validates server-side. Pre-fix this dict used
        # `agent`/`tokens_in`/`tokens_out` and every buffer replay
        # 422'd silently (the "Buffer replay got 422" warnings from
        # the 2026-05-25 prod test). Includes idempotency_key so the
        # server's dedup path can kick in on at-least-once redelivery.
        # Phase 59 (B2): mint ONE idempotency key per call and share it
        # between the disk buffer write and the OTel span. Pre-Phase-59
        # each path generated its own UUID, so the same logical event
        # arrived at /v1/ingest twice with different keys, server-side
        # dedup (keyed on ``idempotency_key``) couldn't catch it, and
        # every call inserted two DB rows — exactly 2× the real cost.
        # Found in the 2026-05-25 real-SDK end-to-end test: 6 anthropic
        # calls produced 12 rows. With one shared key the server rejects
        # the duplicate even if both paths somehow fire.
        idem_key = str(uuid.uuid4())

        # Phase 123: build via the shared helper so the success-event
        # schema can never drift from the failure-event schema. A
        # successful call carries status="success" and the measured
        # provider latency.
        event_data: Dict[str, Any] = self._build_event(
            agent_name=self._agent_name,
            model=usage.model,
            provider=usage.provider,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost=cost,
            duration_ms=duration_ms,
            status="success",
            idempotency_key=idem_key,
            partial=False,
        )
        if self._save_prompts and getattr(self, "_last_prompt", None):
            # Phase 53: cap at MAX_PROMPT_CHARS so the on-wire payload
            # and downstream storage stay bounded.
            event_data["prompt"] = _truncate_prompt(self._last_prompt)

        # Phase 59 (B2) — Option A: when OTel is active the exporter is
        # the canonical ingest transport (it POSTs spans → /v1/ingest).
        # Writing the same event to the disk buffer here would create a
        # second send path because the exporter replays buffered events
        # on every successful export — same call lands at /v1/ingest
        # twice. The buffer is meant to be a *retry* persistence layer
        # that the exporter owns on send-failure (see
        # AgentKavachExporter._buffer_events). Only write here in
        # passthrough/no-tracer mode where OTel isn't doing the send.
        if self._tracer is None:
            self._buffer.write(event_data)

        # Pass the shared idempotency key to the OTel path so the span
        # attribute (and therefore the ingest event) carries the SAME
        # key as the buffered copy — defence in depth in case the
        # tracer guard above is ever bypassed.
        self._trace_call(
            usage,
            cost,
            partial=False,
            idempotency_key=idem_key,
            duration_ms=duration_ms,
            status="success",
        )

    def _on_stream_complete(self, model: str, output_tokens: int, partial: bool) -> None:
        """Callback from StreamWrapper when a stream finishes.

        Phase 65: price the full call (input + output) and propagate the
        real cost into both the engine and the DB event. Pre-Phase-65 the
        engine only saw ``output_tokens`` (via
        ``SpendEngine.record_partial``) and the DB row was written with
        ``cost=0`` / ``input_tokens=0`` — streaming users could watch the
        event counter climb while their dashboard reported $0 spent,
        meaning budgets and alerts never fired. The fix:

          1. Recover ``input_tokens`` from ``_last_input_tokens`` (stashed
             in ``_safe_call`` right before the stream was opened — see
             the Phase 65 comment there).
          2. Compute the full cost via ``estimate_cost``. ``record_partial``
             already added the output-only slice to the engine's spend
             tracker; here we top it up with the input-side cost so the
             total matches what the non-streaming path would have
             recorded.
          3. Write the event row with the real ``input_tokens`` and
             ``cost`` so server-side aggregations are accurate.
        """
        events = self._engine.check_thresholds()
        for event in events:
            self._dispatcher.dispatch(event)

        from agentkavach.pricing import estimate_cost, get_price

        input_tokens = int(getattr(self, "_last_input_tokens", 0) or 0)

        # Full cost = input_tokens × input_per_1k + output_tokens × output_per_1k.
        full_cost = estimate_cost(model, input_tokens, output_tokens)
        if full_cost is None:
            # Unknown model — keep both engine and event at $0 (same
            # behaviour as ``post_flight`` / ``record_partial`` when
            # pricing lookup misses). A warning was already logged once
            # by ``record_partial`` so we don't spam.
            full_cost = 0.0

        # Reconcile engine spend: ``StreamWrapper._finalize`` already
        # added the OUTPUT-only slice via ``record_partial``. To bring
        # the engine to the same total as the non-streaming path we
        # only need to add the INPUT-only delta — adding the full cost
        # again would double-count the output. Use ``get_price`` so the
        # alias / prefix-match logic that ``estimate_cost`` runs is
        # applied consistently.
        price = get_price(model)
        if price is not None and input_tokens > 0:
            input_cost = input_tokens / 1000 * price.input_per_1k
            if input_cost > 0:
                with self._engine._lock:
                    key = self._engine.budget.key
                    self._engine._spend[key] = self._engine._spend.get(key, 0.0) + input_cost
                    if self._engine.org_budget is not None:
                        org_key = self._engine.org_budget.key
                        self._engine._spend[org_key] = (
                            self._engine._spend.get(org_key, 0.0) + input_cost
                        )
                # Re-check thresholds after the input-cost top-up so
                # budgets that crossed only because of the input slice
                # still fire alerts.
                more_events = self._engine.check_thresholds()
                for event in more_events:
                    self._dispatcher.dispatch(event)

        # Phase 59 (B2): same shared-idempotency-key + tracer-guard
        # pattern as ``_post_flight`` — see comment there for why.
        idem_key = str(uuid.uuid4())

        # Phase 123: streams report status="success" with duration_ms=0
        # (no reliable end-to-end stream latency — documented in the plan).
        stream_event: Dict[str, Any] = self._build_event(
            agent_name=self._agent_name,
            model=model,
            provider=self._provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=full_cost,
            duration_ms=0,
            status="success",
            idempotency_key=idem_key,
            partial=partial,
        )
        if self._save_prompts and getattr(self, "_last_prompt", None):
            # Phase 53: same 2 kB cap as the non-streaming path.
            stream_event["prompt"] = _truncate_prompt(self._last_prompt)
        if self._tracer is None:
            self._buffer.write(stream_event)

        self._trace_call(
            UsageRecord(
                provider=self._provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
            cost=full_cost,
            partial=partial,
            idempotency_key=idem_key,
        )

    @staticmethod
    def _extract_tool_name(response: Any) -> Optional[str]:
        """Extract tool name from a provider response, if any."""
        try:
            # OpenAI: response.choices[0].message.tool_calls[0].function.name
            choices = getattr(response, "choices", None)
            if choices:
                msg = getattr(choices[0], "message", None)
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    return getattr(tool_calls[0].function, "name", None)
            # Anthropic: response.content[0].name (tool_use block)
            content = getattr(response, "content", None)
            if isinstance(content, list):
                for block in content:
                    if getattr(block, "type", None) == "tool_use":
                        return getattr(block, "name", None)
        except Exception:
            pass
        return None

    def _trace_call(
        self,
        usage: UsageRecord,
        cost: float,
        partial: bool,
        idempotency_key: Optional[str] = None,
        duration_ms: Optional[int] = None,
        status: str = "success",
    ) -> None:
        """Record an OTel span for the call.

        Phase 35: emit ``agentkavach.idempotency_key`` per span. The
        server uses this for dedup on Kafka at-least-once redelivery —
        without it, a single LLM call's event can be counted multiple
        times if Kafka retries the message, silently inflating costs
        and tripping budgets prematurely.

        Phase 59 (B2): if *idempotency_key* is passed in, use it
        verbatim — it's the same key the caller used for the buffer
        write, so server dedup correctly rejects whichever copy arrives
        second. If omitted, mint a fresh one (preserves backward compat
        for any direct callers).
        """
        if self._tracer is None:
            return
        with self._tracer.start_as_current_span("llm.call") as span:
            span.set_attribute("gen_ai.agent.name", self._agent_name)
            span.set_attribute("gen_ai.request.model", usage.model)
            span.set_attribute("gen_ai.system", usage.provider)
            span.set_attribute("gen_ai.usage.input_tokens", usage.input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", usage.output_tokens)
            span.set_attribute("gen_ai.usage.cost", cost)
            span.set_attribute("agentkavach.partial", partial)
            span.set_attribute("agentkavach.run_id", self._run_id)
            # Phase 123: per-call outcome + measured latency. The exporter
            # (sender._span_to_event) maps ``agentkavach.status`` →
            # event["status"] so events.status records the failure, and
            # prefers ``agentkavach.duration_ms`` (the real measured
            # provider latency) over the span wall-clock when present.
            span.set_attribute("agentkavach.status", status)
            if duration_ms is not None:
                span.set_attribute("agentkavach.duration_ms", int(duration_ms))
            span.set_attribute(
                "agentkavach.idempotency_key",
                idempotency_key or str(uuid.uuid4()),
            )
            if self._save_prompts and getattr(self, "_last_prompt", None):
                # Phase 53: 2 kB cap on the OTel span attribute too —
                # otherwise an enormous prompt would balloon span size
                # and slow down ingest.
                span.set_attribute("gen_ai.prompt", _truncate_prompt(self._last_prompt) or "")

    def _handle_backend_reject(self, reason: str, body: Dict[str, Any]) -> None:
        """Phase 59 (B3): callback invoked by the OTel exporter on a
        permanent 429 (tier_agent_limit / daily_limit /
        org_budget_exceeded).

        Flips ``_backend_paused`` so the next ``_safe_call`` raises
        ``BudgetExceededError`` BEFORE the LLM is invoked — without
        this the SDK silently swallowed the 429 and the customer kept
        paying provider fees on calls whose telemetry was being
        dropped.

        ``body`` is the parsed 429 JSON body (used here only for the
        retained reason; the exporter has already logged a human-
        readable warning).
        """
        # Idempotent: if we've already paused for any reason, keep the
        # original (more useful to operators than overwriting with the
        # latest 429).
        if self._backend_paused:
            return
        self._backend_paused = True
        self._backend_paused_reason = reason

    def shutdown(self) -> None:
        """Flush pending telemetry and release resources."""
        if hasattr(self, "_tracer_provider"):
            try:
                self._tracer_provider.shutdown()
            except Exception:
                pass

        for ch in getattr(self, "_channels", []):
            close_fn = getattr(ch, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass

    # -- Config sync --------------------------------------------------------

    def _sync_config(self) -> None:
        """POST budget/alert config to the backend so alerts fire server-side.

        Runs in a daemon thread so it never blocks SDK initialization.
        All errors are caught and logged — never propagates.

        Skipped under pytest so unit tests that construct the SDK with
        synthetic keys don't fire daemon threads at the prod backend.
        """
        # Detect pytest via the env var pytest sets per-test. Belt-and-
        # suspenders: AGENTKAVACH_DISABLE_SYNC=1 also disables (useful
        # for non-pytest runners or scripts that want fail-closed safety).
        if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("AGENTKAVACH_DISABLE_SYNC"):
            return

        def _do_sync() -> None:
            import json

            payload = self._build_sync_payload()
            if not payload:
                return

            endpoint = self._endpoint.rstrip("/") if self._endpoint else ""
            if not endpoint:
                return

            self._post_sync_config(f"{endpoint}/v1/sync-config", json.dumps(payload).encode())

        t = threading.Thread(target=_do_sync, daemon=True)
        t.start()

    def _post_sync_config(self, url: str, data: bytes, attempts: int = 3) -> bool:
        """POST the sync-config payload, retrying transient failures.

        Without a retry, a single timed-out request (e.g. a cold backend)
        silently dropped the agent's budget/alert config — and the backend
        then never fired that agent's threshold alerts, with no signal to the
        customer. Retries with a short backoff. Returns True on success.
        """
        import time as _time
        import urllib.request

        last_exc = None
        for attempt in range(attempts):
            try:
                req = urllib.request.Request(
                    url,
                    data=data,
                    headers={
                        "Authorization": f"Bearer {self._cg_api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    logger.debug("Config sync response: %s", resp.status)
                return True
            except Exception as exc:  # noqa: BLE001 — best-effort, never propagates
                last_exc = exc
                if attempt < attempts - 1:
                    _time.sleep(2 * (attempt + 1))  # 2s, 4s backoff
        logger.warning("Config sync to backend failed after %d attempts: %s", attempts, last_exc)
        return False

    def _build_sync_payload(self) -> Optional[Dict[str, Any]]:
        """Build the JSON payload for /v1/sync-config from SDK state.

        Syncs **every** limit type the SDK enforces, not just cost. Pre-
        Phase 33, only cost budgets reached the server, so the dashboard
        Budgets section had no way to render token / call / duration
        caps — users saw an isolated cost bar even when their agent
        was actually being killed by ``max_calls_per_run``.
        """
        budgets: List[Dict[str, Any]] = []
        alert_configs: List[Dict[str, Any]] = []
        org_budget: Optional[Dict[str, Any]] = None

        # Cost budget (server tracks against actual ingested cost).
        if self._budget is not None:
            budgets.append(
                {
                    "budget_type": "cost",
                    "period": self._budget.period.value,
                    "limit_value": self._budget.limit,
                    "unit": "usd",
                }
            )

        # Per-run guardrails (token / call / duration / loops). These
        # are enforced client-side but synced server-side so the
        # dashboard can render the full set of agent limits.
        engine = self._engine
        if getattr(engine, "max_tokens_per_run", None):
            budgets.append(
                {
                    "budget_type": "tokens_total",
                    "period": "per_run",
                    "limit_value": float(engine.max_tokens_per_run),
                    "unit": "tokens",
                }
            )
        if getattr(engine, "max_calls_per_run", None):
            budgets.append(
                {
                    "budget_type": "calls",
                    "period": "per_run",
                    "limit_value": float(engine.max_calls_per_run),
                    "unit": "calls",
                }
            )
        if getattr(engine, "max_runtime_seconds", None):
            budgets.append(
                {
                    "budget_type": "duration",
                    "period": "per_run",
                    # Server stores duration as ms internally; convert.
                    "limit_value": float(engine.max_runtime_seconds) * 1000.0,
                    "unit": "ms",
                }
            )
        if getattr(engine, "detect_loops", False) and getattr(engine, "loop_threshold", None):
            budgets.append(
                {
                    "budget_type": "loops",
                    "period": "per_run",
                    "limit_value": float(engine.loop_threshold),
                    "unit": "loops",
                }
            )

        # Alert configs from dispatcher rules + channel configs. Each
        # rule's budget_type tags the AlertConfig so the backend evaluates
        # the threshold against the right usage axis (cost / tokens_total
        # / duration) instead of defaulting to cost.
        for rule in self._dispatcher.rules:
            for channel in rule.channels:
                mode = self._channel_dispatch_mode(channel, rule.threshold)
                ac: Dict[str, Any] = {
                    "channel": channel,
                    "threshold_pct": rule.threshold,
                    "budget_type": getattr(rule, "budget_type", "cost"),
                    "dispatch": mode,
                }
                # Only backend-dispatched channels need their endpoint synced —
                # the backend uses it to deliver. For sdk-dispatched channels the
                # SDK delivers locally, so we deliberately do NOT ship the
                # (often internal) URL/secret to the cloud; the backend just
                # records the alert for the dashboard.
                if mode == "backend":
                    target = self._resolve_channel_target(channel)
                    if target:
                        ac["target"] = target
                    if channel == "webhook":
                        secret = self._resolve_webhook_secret()
                        if secret:
                            ac["secret"] = secret
                alert_configs.append(ac)

        # Org budget
        org_b = getattr(self._engine, "org_budget", None)
        org_alert_configs: Optional[List[Dict[str, Any]]] = None
        if org_b is not None:
            org_budget = {
                "budget_type": "cost",
                "period": org_b.period.value,
                "limit_value": org_b.limit,
                "unit": "usd",
            }
            # When org budget is present, alert configs should be synced as
            # org-level so alert_check evaluates them with combined spend.
            if alert_configs:
                org_alert_configs = alert_configs
                alert_configs = []  # don't duplicate as per-agent

        if not budgets and not alert_configs and org_budget is None:
            return None

        payload: Dict[str, Any] = {
            "agent_name": self._agent_name,
            "budgets": budgets,
            "alert_configs": alert_configs,
            "org_budget": org_budget,
        }
        if org_alert_configs is not None:
            payload["org_alert_configs"] = org_alert_configs
        return payload

    def _resolve_channel_target(self, channel: str) -> Optional[str]:
        """Return the customer-owned target the backend needs to route a
        channel's alert: email recipient, slack webhook URL, pagerduty routing
        key, or webhook URL. Read from the ChannelConfig list the customer passed.
        """
        if channel == "email":
            return self._find_email_target()
        if channel == "slack":
            for cc in self._channel_configs:
                if cc.channel_type == "slack" and cc.webhook_url:
                    return cc.webhook_url
        elif channel == "pagerduty":
            for cc in self._channel_configs:
                if cc.channel_type == "pagerduty" and cc.routing_key:
                    return cc.routing_key
        elif channel == "webhook":
            for cc in self._channel_configs:
                if cc.channel_type == "webhook" and cc.url:
                    return cc.url
        return None

    def _resolve_webhook_secret(self) -> Optional[str]:
        """Return the webhook signing secret (if any) so the backend can sign
        the POST exactly like the SDK does."""
        for cc in self._channel_configs:
            if cc.channel_type == "webhook" and getattr(cc, "secret", ""):
                return cc.secret
        return None

    def _channel_dispatch_mode(self, channel: str, threshold: float) -> str:
        """Return the dispatch mode ('backend' | 'sdk') for the channel config
        matching (channel_type, threshold). Defaults to 'backend'."""
        for cc in self._channel_configs:
            if cc.channel_type == channel and cc.threshold == threshold:
                return getattr(cc, "dispatch", "backend")
        return "backend"

    def _find_email_target(self) -> Optional[str]:
        """Extract email target so the backend can route the alert.

        Prefer the registered handler (it confirms client-side dispatch
        succeeded), but fall back to the original ChannelConfig list —
        backend-dispatch mode (no api_key) skips handler registration and
        the recipient would otherwise be lost.
        """
        for ch in self._channels:
            to = getattr(ch, "_to_email", None) or getattr(ch, "to_email", None)
            if to:
                return to
        for cc in self._channel_configs:
            if cc.channel_type == "email" and cc.to:
                return cc.to
        return None

    # -- YAML config loader -------------------------------------------------

    @classmethod
    def from_yaml(
        cls,
        yaml_file: str,
        agent: Optional[str] = None,
        on_kill: Optional[Callable[[], None]] = None,
        api_key: str = "",
        llm_key: str = "",
    ) -> Dict[str, AgentKavach] | AgentKavach:
        """Load configuration from a YAML file.

        Returns a dict of ``{agent_name: AgentKavach}`` instances, or a
        single ``AgentKavach`` if *agent* is specified.

        ``api_key`` and ``llm_key`` are required (the keys are never read from
        the environment). Pass them explicitly here to apply to every agent,
        e.g. ``AgentKavach.from_yaml(path, api_key=os.environ["AGENTKAVACH_API_KEY"],
        llm_key=os.environ["OPENAI_API_KEY"])``. A per-agent ``api_key`` /
        ``llm_key`` in the YAML overrides these.
        """
        with open(yaml_file) as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ValueError(f"Invalid YAML config: expected dict, got {type(config).__name__}")

        defaults = config.get("defaults", {})
        agents_config = config.get("agents", {})
        channels_config = config.get("channels", {})
        org_budget_config = config.get("org_budget", {})

        # Parse and validate channel definitions.
        channel_defs = _parse_channel_defs(channels_config)

        # Parse org budget (applies to all agents if present).
        parsed_org_budget: Optional[Budget] = None
        if org_budget_config:
            org_limit = float(org_budget_config.get("limit", 0))
            org_period = org_budget_config.get("period", "daily")
            if org_limit > 0:
                parsed_org_budget = Budget.org_budget(limit=org_limit, period=org_period)

        # Build AgentKavach instances.
        clients: Dict[str, AgentKavach] = {}
        for agent_name, agent_config in agents_config.items():
            merged = _merge_config(defaults, agent_config)

            # Handle "budget: default" (use defaults budget).
            budget_config = merged.get("budget", {})
            if budget_config == "default":
                budget_config = defaults.get("budget", {})

            budget = _parse_budget(budget_config, agent_name)
            agent_provider = merged.get("provider", "openai")
            common = dict(
                provider=agent_provider,
                api_key=merged.get("api_key", "") or api_key,
                llm_key=merged.get("llm_key", "") or llm_key,
                budget=budget,
                org_budget=parsed_org_budget,
                on_kill=on_kill,
                agent_name=agent_name,
                save_prompts=bool(merged.get("save_prompts", False)),
                fail_on_error=bool(merged.get("fail_on_error", False)),
                # Per-run guardrails (token / call / duration / loop caps) so a
                # YAML config can express everything the inline constructor can.
                **_parse_guardrails(merged.get("guardrails", {}), agent_name),
            )

            if channels_config:
                # Modern path: a channels: section means we can build full
                # ChannelConfig objects (credentials + per-channel dispatch
                # mode), so YAML gets the same backend/sdk routing as the
                # inline channels= API. _parse_alerts still validates the
                # channel names against the defined channels.
                _parse_alerts(merged.get("alerts", []), channel_defs)
                channel_configs = _build_channel_configs_from_yaml(
                    merged.get("alerts", []), channel_defs
                )
                clients[agent_name] = cls(channels=channel_configs or None, **common)
            else:
                # Legacy path: bare alert types with no channels: section.
                alert_rules = _parse_alerts(merged.get("alerts", []), None)
                clients[agent_name] = cls(alerts=alert_rules or None, **common)

        if agent is not None:
            if agent not in clients:
                raise KeyError(f"Agent {agent!r} not found in config (available: {list(clients)})")
            return clients[agent]

        return clients


# ---------------------------------------------------------------------------
# YAML config helpers
# ---------------------------------------------------------------------------


def _merge_config(defaults: Dict, agent: Dict) -> Dict:
    """Shallow-merge agent config over defaults."""
    merged = dict(defaults)
    merged.update(agent)
    return merged


def _parse_budget(
    config: Dict | str,
    agent_name: str,
) -> Budget:
    """Parse a budget config dict into a Budget object."""
    if not config or config == "default":
        return Budget.daily(limit=100.0)

    if isinstance(config, str):
        raise ValueError(f"Unknown budget value {config!r} for agent {agent_name!r}")

    budget_type = config.get("type", "daily")
    limit = float(config.get("limit", 100.0))

    if budget_type == "daily":
        return Budget.daily(limit=limit)
    elif budget_type == "monthly":
        return Budget.monthly(limit=limit)
    elif budget_type == "total":
        return Budget.total(limit=limit)
    else:
        raise ValueError(f"Unknown budget type: {budget_type!r}")


# Per-run guardrail keys a YAML ``guardrails:`` block may set, mapped to the
# AgentKavach constructor kwargs. Mirrors the inline constructor exactly so the
# two configuration paths are at parity.
_GUARDRAIL_INT_KEYS = ("max_tokens_per_run", "max_calls_per_run", "loop_threshold")
_GUARDRAIL_FLOAT_KEYS = ("max_runtime_seconds",)
_GUARDRAIL_BOOL_KEYS = ("detect_loops",)
_GUARDRAIL_KEYS = _GUARDRAIL_INT_KEYS + _GUARDRAIL_FLOAT_KEYS + _GUARDRAIL_BOOL_KEYS


def _parse_guardrails(config: Dict, agent_name: str) -> Dict[str, Any]:
    """Parse a YAML ``guardrails:`` block into AgentKavach constructor kwargs.

    Supported keys (all optional): ``max_tokens_per_run``, ``max_calls_per_run``,
    ``loop_threshold`` (ints); ``max_runtime_seconds`` (float);
    ``detect_loops`` (bool). Unknown keys raise so typos surface loudly instead
    of silently disabling a guardrail.
    """
    if not config:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"'guardrails' for agent {agent_name!r} must be a mapping")

    unknown = set(config) - set(_GUARDRAIL_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown guardrail key(s) {sorted(unknown)} for agent {agent_name!r}. "
            f"Valid keys: {', '.join(_GUARDRAIL_KEYS)}"
        )

    out: Dict[str, Any] = {}
    for k in _GUARDRAIL_INT_KEYS:
        if k in config:
            out[k] = int(config[k])
    for k in _GUARDRAIL_FLOAT_KEYS:
        if k in config:
            out[k] = float(config[k])
    for k in _GUARDRAIL_BOOL_KEYS:
        if k in config:
            out[k] = bool(config[k])
    return out


def _parse_alerts(
    configs: List[Dict],
    channel_defs: Optional[Dict[str, Dict]] = None,
) -> List[AlertRule]:
    """Parse a list of alert config dicts into AlertRule objects.

    If *channel_defs* is provided, channel names in alerts are validated
    against the defined channels (plus the built-in ``kill`` channel).
    """
    rules: List[AlertRule] = []
    for cfg in configs:
        threshold = float(cfg.get("threshold", cfg.get("at", 0)))
        channels = list(cfg.get("channels", ["email"]))
        template = cfg.get("template")
        # Legacy: "action: kill" is equivalent to adding "kill" to channels.
        if cfg.get("action") == "kill" and "kill" not in channels:
            channels.append("kill")
        # Validate channel names and resolve them to channel TYPES. When a
        # channels: section is present, alerts reference channels by their
        # (possibly arbitrary) NAME — e.g. "team-slack" with type: slack — so we
        # validate against the defined names and resolve each to its type, since
        # AlertRule/the dispatcher operate on types. Without a channels: section
        # (legacy path), the alert already references bare channel types, so we
        # validate against VALID_CHANNEL_TYPES. "kill" is always built-in.
        resolved: List[str] = []
        for ch in channels:
            if ch == "kill":
                resolved.append("kill")
                continue
            if channel_defs is not None:
                if ch not in channel_defs:
                    raise ValueError(
                        f"Channel {ch!r} used in alerts but not defined in the "
                        f"channels section. Define it first or remove it from alerts."
                    )
                resolved.append(channel_defs[ch].get("type", ch))
            else:
                if ch not in VALID_CHANNEL_TYPES:
                    raise ValueError(
                        f"Unknown channel type {ch!r} in alert config. "
                        f"Valid types: {', '.join(sorted(VALID_CHANNEL_TYPES))}"
                    )
                resolved.append(ch)
        rules.append(
            AlertRule(
                threshold=threshold,
                channels=tuple(resolved),
                template=template,
            )
        )
    return rules


_YAML_CRED_KEY = {
    "slack": "webhook_url",
    "pagerduty": "routing_key",
    "webhook": "url",
    "email": "to",
}


def _build_channel_configs_from_yaml(
    alert_configs: List[Dict], channel_defs: Dict[str, Dict]
) -> List[ChannelConfig]:
    """Build ChannelConfig objects from YAML alerts + channel defs.

    Produces one ChannelConfig per (alert threshold × channel), carrying the
    channel's credentials and its per-channel ``dispatch`` mode. Used when a
    ``channels:`` section is present so YAML configs get the same dispatch
    routing (backend vs sdk) as the inline ``channels=`` API.
    """
    out: List[ChannelConfig] = []
    for cfg in alert_configs:
        threshold = float(cfg.get("threshold", cfg.get("at", 0)))
        channels = list(cfg.get("channels", ["email"]))
        # Which budget dimension this alert watches (cost / tokens_total /
        # duration / calls / ...). Defaults to "cost" for back-compat; required
        # to alert on a non-cost dimension from YAML (e.g. a tokens cap).
        btype = cfg.get("budget_type")
        if cfg.get("action") == "kill" and "kill" not in channels:
            channels.append("kill")
        for ch in channels:
            if ch == "kill":
                kw_kill: Dict[str, Any] = {}
                if btype is not None:
                    kw_kill["budget_type"] = btype
                out.append(ChannelConfig(channel_type="kill", threshold=threshold, **kw_kill))
                continue
            # The alert references a channel by its NAME (the key in the
            # channels: section). The real channel TYPE comes from that def's
            # `type:` field — so you can declare several channels of the same
            # type (e.g. a public webhook + an internal webhook) under distinct
            # names with independent dispatch modes.
            cdef = channel_defs.get(ch, {})
            ch_type = cdef.get("type", ch)
            kw: Dict[str, Any] = {"dispatch": cdef.get("dispatch", "backend")}
            if btype is not None:
                kw["budget_type"] = btype
            cred_key = _YAML_CRED_KEY.get(ch_type)
            if cred_key and cred_key in cdef:
                kw[cred_key] = cdef[cred_key]
            if ch_type == "webhook" and "secret" in cdef:
                kw["secret"] = cdef["secret"]
            if ch_type == "email" and "api_key" in cdef:
                kw["api_key"] = cdef["api_key"]
            template = cfg.get("template")
            if template is not None:
                kw["template"] = template
            out.append(ChannelConfig(channel_type=ch_type, threshold=threshold, **kw))
    return out


def _parse_channel_defs(channels_config: Dict) -> Dict[str, Dict]:
    """Parse and validate the top-level channels section of a YAML config.

    Each channel must have a ``type`` that is one of the valid channel types.
    Returns a dict of ``{channel_name: channel_config}``.
    """
    defs: Dict[str, Dict] = {}
    for name, cfg in channels_config.items():
        if not isinstance(cfg, dict):
            raise ValueError(f"Channel {name!r} config must be a mapping, got {type(cfg).__name__}")
        ch_type = cfg.get("type", name)
        if ch_type not in VALID_CHANNEL_TYPES:
            raise ValueError(
                f"Channel {name!r} has unknown type {ch_type!r}. "
                f"Valid types: {', '.join(sorted(VALID_CHANNEL_TYPES))}"
            )
        defs[name] = cfg
    return defs


def _resolve_channel_creds(channel_defs: Dict[str, Dict]) -> Dict[str, str]:
    """Extract channel credentials from parsed channel definitions.

    Returns kwargs suitable for passing to the AgentKavach constructor.
    """
    kwargs: Dict[str, str] = {}
    for _name, cfg in channel_defs.items():
        ch_type = cfg.get("type", _name)
        if ch_type == "slack" and "webhook_url" in cfg:
            kwargs["slack_webhook_url"] = cfg["webhook_url"]
        elif ch_type == "email":
            if "to" in cfg:
                kwargs["alert_email"] = cfg["to"]
            if "api_key" in cfg:
                kwargs["resend_api_key"] = cfg["api_key"]
        elif ch_type == "pagerduty" and "routing_key" in cfg:
            kwargs["pagerduty_routing_key"] = cfg["routing_key"]
        elif ch_type == "webhook":
            if "url" in cfg:
                kwargs["webhook_url"] = cfg["url"]
            if "secret" in cfg:
                kwargs["webhook_secret"] = cfg["secret"]
    return kwargs


def _build_rules_from_channels(channels: Sequence[ChannelConfig]) -> tuple[AlertRule, ...]:
    """Group ChannelConfig objects by (threshold, budget_type) into AlertRules.

    Channels bound to different budget dimensions don't merge — a cost @
    50% rule must stay separate from a tokens @ 50% rule so the dispatcher
    fires the right one.
    """
    grouped: Dict[Tuple[float, str], List[str]] = defaultdict(list)
    for ch in channels:
        grouped[(ch.threshold, getattr(ch, "budget_type", "cost"))].append(ch.channel_type)
    rules = []
    for threshold, btype in sorted(grouped):
        rules.append(
            AlertRule(
                threshold=threshold,
                channels=tuple(grouped[(threshold, btype)]),
                budget_type=btype,
            )
        )
    return tuple(rules)
