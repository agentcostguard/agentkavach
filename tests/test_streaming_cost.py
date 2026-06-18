"""Phase 65 regression guard: streaming events must record full cost.

Background
----------
Pre-Phase-65 a streaming ``guard.create(..., stream=True)`` call landed
in the backend ``events`` table with ``cost=0.0`` and
``input_tokens=0`` even though the per-chunk token counter (Phase 44)
extracted ``output_tokens`` correctly. The gap was in
``AgentKavach._on_stream_complete``: it discarded the input-token count
that ``_safe_call`` had already computed and wrote the event row at
$0. Net effect: customers using streaming saw their dashboards report
no spend, budgets and alerts never fired, and the cost-tracking
guarantee was silently broken for the streaming path only.

What this file asserts
----------------------
For every supported provider (OpenAI / Anthropic / Google / Mistral),
mock the provider SDK to return a stream of chunks with KNOWN token
counts, drain the wrapper, and assert:

1. The engine's recorded spend matches ``estimate_cost(model, input,
   output)`` exactly (within float epsilon). Pre-Phase-65 the engine
   only saw the output slice via ``record_partial``.

2. The event handed to the buffer (``_buffer.write``) carries the same
   non-zero cost AND the right ``input_tokens`` / ``output_tokens``.

3. The OTel span carries the SAME idempotency key as the buffer event
   (Phase 59 invariant, re-checked here to confirm the Phase 65 fix
   didn't accidentally desync the two paths).

Implementation note
-------------------
We don't spin up the FastAPI app — that's covered by
``test_sdk_no_double_write.py``. Here we only need to observe the
SDK's own bookkeeping (engine spend + buffer payload), which is
cheaper and lets the test stay tightly focused on the pricing math.

Buffers default to ``~/.agentkavach`` which leaks across tests; we
patch a per-test ``tmp_path`` in via ``buffer_path=`` so each
provider's run starts clean.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agentkavach import AgentKavach, Budget
from agentkavach.pricing import estimate_cost

# ─── Test environment (no real backend, no real LLM) ──────────────────────


_TEST_ENV = {
    # Point the (unused) backend resolver at a sentinel host so no
    # accidental DNS lookup leaves the test process.
    "AGENTKAVACH_BACKEND_URL": "http://testserver-phase65",
}


# ─── Per-provider chunk fixtures ──────────────────────────────────────────


def _openai_stream(output_tokens: int):
    """Mimic ``client.chat.completions.create(..., stream=True)`` output.

    Two chunks: one empty delta-content chunk that the heuristic
    counter rounds to 0 (because the content is ""), then a final
    ``usage`` chunk that supplies the exact count. We send empty
    content for the delta so only the final chunk contributes to the
    accumulated total — the cost assertion needs an exact value.
    """
    return iter(
        [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=""))],
                usage=None,
            ),
            SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(completion_tokens=output_tokens),
            ),
        ]
    )


def _anthropic_stream(output_tokens: int):
    """Mimic ``client.messages.create(..., stream=True)`` output."""
    return iter(
        [
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text=""),
            ),
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="end_turn"),
                usage=SimpleNamespace(output_tokens=output_tokens),
            ),
        ]
    )


def _google_stream(output_tokens: int):
    """Mimic ``client.models.generate_content_stream(...)`` output."""
    return iter(
        [
            SimpleNamespace(text="", usage_metadata=None),
            SimpleNamespace(
                text=None,
                usage_metadata=SimpleNamespace(
                    prompt_token_count=0,
                    candidates_token_count=output_tokens,
                    total_token_count=output_tokens,
                ),
            ),
        ]
    )


def _mistral_stream(output_tokens: int):
    """Mimic ``client.chat.stream(...)`` output (CompletionEvent wrap)."""
    return iter(
        [
            SimpleNamespace(
                data=SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=""))],
                    usage=None,
                )
            ),
            SimpleNamespace(
                data=SimpleNamespace(
                    choices=[],
                    usage=SimpleNamespace(completion_tokens=output_tokens, prompt_tokens=0),
                )
            ),
        ]
    )


def _wire_mock_client(provider: str, chunks: Any) -> MagicMock:
    """Return a MagicMock provider SDK that yields *chunks* for streams."""
    mock = MagicMock()
    if provider == "openai":
        mock.chat.completions.create.return_value = chunks
    elif provider == "anthropic":
        mock.messages.create.return_value = chunks
        # Pre-flight count_tokens stub — value here is irrelevant; the
        # SDK reads it into ``_last_input_tokens`` only to feed pricing,
        # and we override that on the guard before draining the stream.
        mock.messages.count_tokens.return_value = SimpleNamespace(input_tokens=999)
    elif provider == "google":
        mock.models.generate_content_stream.return_value = chunks
        mock.models.count_tokens.return_value = SimpleNamespace(total_tokens=999)
    elif provider == "mistral":
        mock.chat.stream.return_value = chunks
    return mock


# Each entry: (provider, model, stream_factory, mock_attr, kwargs_extra)
#  - mock_attr: the AgentKavach private getter to patch so the SDK
#    never imports the real provider package.
#  - kwargs_extra: provider-specific extra kwargs for ``guard.create``.
_PROVIDER_CASES = [
    ("openai", "gpt-4o", _openai_stream, "_get_openai_client", {}),
    ("anthropic", "claude-sonnet-4-0", _anthropic_stream, "_get_anthropic_client", {}),
    ("google", "gemini-2.5-flash", _google_stream, "_get_google_client", {"contents": "hi"}),
    ("mistral", "mistral-large-latest", _mistral_stream, "_get_mistral_client", {}),
]


def _build_guard(provider: str, agent: str, buffer_path: str) -> AgentKavach:
    """Construct an AgentKavach with no API key (no telemetry export).

    We omit ``api_key`` so the OTel tracer is NOT created — the SDK
    falls back to the buffer-write path which is exactly what we want
    to inspect. The Budget is generous so threshold dispatch doesn't
    interfere with the cost assertions.
    """
    return AgentKavach(
        provider=provider,
        api_key="ak_test",
        llm_key="sk-fake-llm-not-real",
        agent_name=agent,
        budget=Budget.daily(limit=1000.0),
        buffer_path=buffer_path,
    )


@pytest.mark.parametrize(
    "provider,model,stream_factory,mock_attr,kwargs_extra",
    _PROVIDER_CASES,
    ids=[p[0] for p in _PROVIDER_CASES],
)
def test_streaming_records_full_cost(
    tmp_path, provider, model, stream_factory, mock_attr, kwargs_extra
):
    """One ``guard.create(stream=True)`` records the FULL priced cost.

    Asserts: engine.spent == event.cost == estimate_cost(model, in, out)
    where ``in`` is the value the SDK captured pre-call and ``out`` is
    the value the final usage chunk carries. Pre-Phase-65 the SDK
    recorded only the output slice in engine spend and wrote
    cost=0 / input_tokens=0 on the event row — both halves of the
    invariant fail under the bug.
    """
    input_tokens = 80
    output_tokens = 250
    expected_cost = estimate_cost(model, input_tokens, output_tokens)
    assert expected_cost is not None and expected_cost > 0, (
        f"Model {model!r} missing from PRICE_TABLE — fix the fixture."
    )

    chunks = stream_factory(output_tokens)
    mock_client = _wire_mock_client(provider, chunks)

    with patch.dict(os.environ, _TEST_ENV):
        with patch(f"agentkavach.client.AgentKavach.{mock_attr}", return_value=mock_client):
            agent_name = f"phase65-{provider}-stream"
            guard = _build_guard(provider, agent_name, str(tmp_path / "buffer"))

            # Pin the input-token count the SDK would otherwise compute
            # via the provider's native counter. Native counting paths
            # for Anthropic/Google make real network calls and would
            # hit the (mocked) client in ways that vary per provider —
            # patching ``_count_input_tokens`` keeps this test focused
            # on the pricing-math regression, not on each provider's
            # token-counting code path (Phase 36 has dedicated tests
            # for that).
            with patch.object(AgentKavach, "_count_input_tokens", return_value=input_tokens):
                kwargs = {
                    "model": model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                }
                kwargs.update(kwargs_extra)
                stream = guard.create(**kwargs)
                for _ in stream:
                    pass

            # --- Engine spend: full priced cost (input + output) ----------
            assert guard._engine.spent == pytest.approx(expected_cost, abs=1e-9), (
                f"Engine spend for {provider} stream is "
                f"{guard._engine.spent!r}, expected {expected_cost!r}. "
                f"Phase 65 regression: input-token cost was dropped."
            )

            # --- Event row: full cost + real token counts -----------------
            # The buffer path is what populates the DB row when no OTel
            # tracer is attached. Read it directly off disk via the
            # buffer's own iterator — same shape the server consumes.
            events = list(guard._buffer.read_all())
            assert len(events) == 1, (
                f"Expected 1 buffered event for {provider} stream, found {len(events)}."
            )
            event = events[0]
            assert event["input_tokens"] == input_tokens, (
                f"Event input_tokens for {provider} = {event['input_tokens']!r}, "
                f"expected {input_tokens}. Phase 65 regression."
            )
            assert event["output_tokens"] == output_tokens
            assert event["cost"] == pytest.approx(expected_cost, abs=1e-9), (
                f"Event cost for {provider} stream = {event['cost']!r}, "
                f"expected {expected_cost!r}. Phase 65 regression: "
                f"the bug wrote cost=0 here."
            )
            assert event["partial"] is False
            assert event["model"] == model
            assert event["provider"] == provider
            assert event["idempotency_key"], "idempotency_key must be set per Phase 59"


def test_streaming_unknown_model_falls_back_to_zero(tmp_path):
    """Unknown model → cost stays $0 (no crash).

    This is the only legitimate $0 case for streaming. If pricing lookup
    misses (custom/private model, no ``register_price`` call) the
    engine and event both record zero rather than raising — same
    fail-soft contract as the non-streaming ``post_flight``.
    """
    unknown_model = "internal-llm-not-in-price-table"
    output_tokens = 42

    chunks = _openai_stream(output_tokens)
    mock_client = _wire_mock_client("openai", chunks)

    with patch.dict(os.environ, _TEST_ENV):
        with patch("agentkavach.client.AgentKavach._get_openai_client", return_value=mock_client):
            guard = _build_guard("openai", "phase65-unknown", str(tmp_path / "buffer"))
            with patch.object(AgentKavach, "_count_input_tokens", return_value=100):
                stream = guard.create(
                    model=unknown_model,
                    messages=[{"role": "user", "content": "hi"}],
                    stream=True,
                )
                for _ in stream:
                    pass

            assert guard._engine.spent == 0.0
            events = list(guard._buffer.read_all())
            assert len(events) == 1
            assert events[0]["cost"] == 0.0
            # The token counts are still captured even when pricing is
            # unknown — operators can see traffic exists.
            assert events[0]["input_tokens"] == 100
            assert events[0]["output_tokens"] == output_tokens


def test_streaming_partial_disconnect_still_prices(tmp_path):
    """A client-disconnected stream still records the partial cost.

    ``StreamWrapper.close()`` (called when the consumer abandons the
    generator) fires the same ``on_complete`` callback with
    ``partial=True``. The cost recorded reflects the OUTPUT tokens seen
    so far + the FULL INPUT cost (the customer paid the input price
    regardless of how much they read back). Pre-Phase-65 this case
    also wrote cost=0 — silently undercounting partial reads.
    """
    input_tokens = 50
    partial_output_tokens = 17

    chunks = _openai_stream(partial_output_tokens)
    mock_client = _wire_mock_client("openai", chunks)
    model = "gpt-4o"

    with patch.dict(os.environ, _TEST_ENV):
        with patch("agentkavach.client.AgentKavach._get_openai_client", return_value=mock_client):
            guard = _build_guard("openai", "phase65-partial", str(tmp_path / "buffer"))
            with patch.object(AgentKavach, "_count_input_tokens", return_value=input_tokens):
                stream = guard.create(
                    model=model,
                    messages=[{"role": "user", "content": "hi"}],
                    stream=True,
                )
                # Drain one chunk then close — simulates a client
                # disconnect mid-stream. The final usage chunk is
                # NEVER consumed, so output_tokens stays at whatever
                # the first chunk reported (0 in our fixture).
                next(stream)
                stream.close()

            events = list(guard._buffer.read_all())
            assert len(events) == 1
            event = events[0]
            assert event["partial"] is True
            # Output is 0 (we abandoned before the usage chunk) but
            # input cost was still paid → event.cost == input slice.
            expected_partial_cost = estimate_cost(model, input_tokens, 0)
            assert event["cost"] == pytest.approx(expected_partial_cost, abs=1e-9), (
                f"Partial stream cost = {event['cost']!r}, expected "
                f"{expected_partial_cost!r} (input-only)."
            )
            assert event["input_tokens"] == input_tokens
