"""Phase 61 CI guard against the SDK double-write bug (Phase 58/59).

Background
----------
The SDK has TWO transports that can land an event in the backend's
``events`` table for a single LLM call:

1. **Buffer path**: ``_post_flight`` calls ``self._buffer.write(event)``.
   The buffer is meant to be a *retry persistence layer* for events that
   the OTel exporter failed to deliver — the exporter drains it on the
   next successful export (``AgentKavachExporter._replay_buffer``).

2. **OTel path**: ``_post_flight`` calls ``_trace_call`` which records a
   span; the ``BatchSpanProcessor`` flushes spans to
   ``AgentKavachExporter.export()`` which POSTs them to ``/v1/ingest``.

Pre-Phase-59 the SDK wrote to BOTH paths unconditionally with DIFFERENT
``idempotency_key`` values, so:

  * OTel export sends 1 event → server inserts 1 row → success.
  * Exporter then replays the buffer → POST 1 event (different key) →
    server can't dedup → inserts a SECOND row.
  * Net result: every SDK-produced LLM call was double-counted in the
    DB, doubling reported cost and tripping budgets prematurely.

This was caught at the end of Phase 58 by a real end-to-end run that
ingested 6 anthropic calls and found 12 DB rows. The bug had shipped
because every prior test mocked the buffer XOR the OTel path — never
both running — so a real ``_post_flight`` (which exercises BOTH) was
never observed in CI.

What this file asserts
----------------------
For each provider (OpenAI / Anthropic / Google / Mistral), and for the
two cases that hit different code paths (non-streaming via
``_post_flight``; streaming via ``_on_stream_complete``):

  * Exactly **one** ``Event`` row per ``guard.create(...)`` call.
  * Exactly **one** ``IdempotencyKey`` row per call.
  * The row's ``cost`` matches the SDK's pricing-engine value to within
    1e-9 USD.

Implementation
--------------
The test spins up the FastAPI app against an in-memory SQLite database
using a ``TestClient``, then patches ``AgentKavachExporter``'s
``httpx.Client`` so that the OTel exporter's POSTs land on the
TestClient (== in-process backend) instead of a real network call. The
provider SDK clients are mocked to return canned responses with known
token counts. The buffer is given a unique tmp_path so disk artifacts
from one test don't bleed into another.

After ``guard.create(...)`` we force the OTel BatchSpanProcessor to
flush (``_tracer_provider.force_flush``) and explicitly shut it down to
guarantee any in-flight export has completed before we count DB rows.

Why ``xfail`` for now
---------------------
Phase 59 is the fix for the double-write bug. As of the commit this
test was written against (develop @ 9918b1e), the SDK is still buggy
and the assertions deliberately fail. The whole module is marked
``xfail(strict=False)`` so CI stays green while Phase 59 is in flight;
once Phase 59 lands and the fix is on develop, removing the xfail
marker becomes a one-line follow-up commit and CI will start enforcing
the invariant immediately. The ``strict=False`` is intentional so the
suite doesn't break twice: once if the bug is unfixed, once if the bug
is fixed but the xfail marker hasn't been removed yet.
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Phase 59 landed: B2 (shared idempotency key + tracer-guarded buffer
# write in ``_post_flight`` / ``_on_stream_complete``) eliminates the
# double-write. These assertions now enforce the invariant — any
# regression that re-introduces the bug will fail CI.


# ─── Backend / TestClient setup ─────────────────────────────────────────────


_TEST_ENV = {
    "AGENTKAVACH_JWT_SECRET": "test-secret-key",
    "AGENTKAVACH_JWT_ALGORITHM": "HS256",
    "AGENTKAVACH_JWT_EXPIRE_MINUTES": "60",
    # Force the SDK's resolve_backend_url() to point at our TestClient
    # base URL. The TestClient itself is what handles the POST — see
    # _patch_exporter_to_testclient below.
    "AGENTKAVACH_BACKEND_URL": "http://testserver",
}


@pytest.fixture
def backend():
    """Spin up the full FastAPI app on an in-memory SQLite DB.

    Yields ``(client, raw_api_key, org_id, session_factory)``.

    Mirrors ``tests/test_server_ingest.py``'s setup but with one
    additional knob: the raw API key starts with ``ak_dev_`` so the
    SDK's prefix-based ``resolve_backend_url`` accepts it without
    complaining (we still override via env var, but keeping the prefix
    realistic guards against accidental key-shape coupling regressions).
    """
    from server.config import get_settings
    from server.database import get_db
    from server.keys import hash_api_key
    from server.models import ApiKey, Base, Organization
    from server.ratelimit import reset_rate_limiter
    from server.writer import reset_producer

    with patch.dict(os.environ, _TEST_ENV):
        get_settings.cache_clear()
        reset_rate_limiter()
        reset_producer()

        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)

        session = Session()
        org = Organization(name="Phase61TestOrg")
        session.add(org)
        session.flush()

        # ak_dev_ prefix means resolve_backend_url returns DEV_BACKEND_URL
        # absent the env-var override (defense in depth).
        raw_key = "ak_dev_phase61_double_write_guard_abcdef0123456789"
        api_key = ApiKey(
            name="phase61-key",
            prefix=raw_key[:24],
            hashed_key=hash_api_key(raw_key),
            org_id=org.id,
        )
        session.add(api_key)
        session.commit()
        org_id = org.id
        session.close()

        def _override_db():
            db = Session()
            try:
                yield db
            finally:
                db.close()

        from server.app import create_app

        app = create_app(init_db=False)
        app.dependency_overrides[get_db] = _override_db
        client = TestClient(app)
        try:
            yield client, raw_key, org_id, Session
        finally:
            app.dependency_overrides.clear()
            engine.dispose()
            reset_rate_limiter()
            reset_producer()
            get_settings.cache_clear()


# ─── SDK → TestClient transport bridge ─────────────────────────────────────


def _patch_exporter_to_testclient(guard: Any, client: TestClient) -> None:
    """Replace the exporter's internal ``httpx.Client`` with *client*.

    The OTel ``BatchSpanProcessor`` exports on a background thread —
    we need every POST that exporter issues to land on the in-process
    FastAPI app rather than a real network call. ``TestClient`` is
    itself an ``httpx.Client`` subclass so its ``post(url, ...)`` API
    is wire-compatible with the exporter's call site.
    """
    # Walk the tracer provider to find the AgentKavachExporter instance.
    tp = guard._tracer_provider
    # SynchronousMultiSpanProcessor.span_processors holds the actual list.
    processors = list(tp._active_span_processor._span_processors)
    for proc in processors:
        exporter = getattr(proc, "span_exporter", None)
        if exporter is None:
            continue
        # Swap the underlying transport.
        try:
            exporter._client.close()
        except Exception:
            pass
        exporter._client = client


def _flush_and_shutdown(guard: Any) -> None:
    """Drain the OTel batch queue and the buffer-replay path.

    A bare ``force_flush()`` is not always enough on its own — the
    ``BatchSpanProcessor`` runs the exporter on a worker thread, and on
    the very first call the worker may still be initialising when
    ``force_flush`` returns. A short sleep + an explicit ``shutdown``
    guarantees the worker drains. We then add one more tiny sleep so
    the post-export ``_replay_buffer`` (which sends a second HTTP POST
    in the buggy code path) has a chance to land in the DB before the
    test starts counting rows. Without that sleep the test could
    accidentally pass on the buggy SDK because the second write hasn't
    arrived yet.
    """
    try:
        guard._tracer_provider.force_flush(timeout_millis=5_000)
    except Exception:
        pass
    time.sleep(0.2)
    try:
        guard.shutdown()
    except Exception:
        pass
    time.sleep(0.2)


# ─── Provider mocks ────────────────────────────────────────────────────────


def _openai_response(model: str, in_toks: int, out_toks: int) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(prompt_tokens=in_toks, completion_tokens=out_toks),
        choices=[],
    )


def _anthropic_response(model: str, in_toks: int, out_toks: int) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(input_tokens=in_toks, output_tokens=out_toks),
        content=[],
    )


def _google_response(model: str, in_toks: int, out_toks: int) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        usage_metadata=SimpleNamespace(
            prompt_token_count=in_toks,
            candidates_token_count=out_toks,
            total_token_count=in_toks + out_toks,
        ),
    )


def _mistral_response(model: str, in_toks: int, out_toks: int) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(prompt_tokens=in_toks, completion_tokens=out_toks),
    )


# Each entry: (provider, model, response_factory, mock_attr)
#  - mock_attr is the AgentKavach private getter we patch so the SDK
#    never tries to import the real provider package or hit the wire.
_PROVIDER_CASES = [
    ("openai", "gpt-4o", _openai_response, "_get_openai_client"),
    ("anthropic", "claude-sonnet-4-0", _anthropic_response, "_get_anthropic_client"),
    ("google", "gemini-2.5-flash", _google_response, "_get_google_client"),
    ("mistral", "mistral-large-latest", _mistral_response, "_get_mistral_client"),
]


def _wire_provider_mock(provider: str, response: Any):
    """Return a MagicMock provider client that returns *response*.

    Each provider has a different call surface — match the namespace
    the real SDK exposes so AgentKavach._call_provider's dispatch
    works unchanged.
    """
    mock = MagicMock()
    if provider == "openai":
        mock.chat.completions.create.return_value = response
    elif provider == "anthropic":
        mock.messages.create.return_value = response
        # count_tokens API used by SDK's pre-flight; return a stub.
        mock.messages.count_tokens.return_value = SimpleNamespace(input_tokens=100)
    elif provider == "google":
        mock.models.generate_content.return_value = response
        mock.models.count_tokens.return_value = SimpleNamespace(total_tokens=100)
    elif provider == "mistral":
        mock.chat.complete.return_value = response
    return mock


# ─── Tests ─────────────────────────────────────────────────────────────────


def _build_guard(
    client: TestClient, api_key: str, provider: str, agent: str, buffer_path: str | None = None
):
    """Construct an AgentKavach configured to use our TestClient backend.

    ``buffer_path`` MUST be a per-test unique path. The default ``Buffer()``
    resolves to the shared ``~/.agentkavach/buffer.jsonl``; under ``pytest -n
    auto`` that file is shared across xdist workers, so a *concurrent* test's
    buffered events would be replayed by this guard's exporter and land as extra
    rows — a non-deterministic cross-worker leak. An isolated tmp buffer makes
    the single-row invariant deterministic regardless of parallelism.
    """
    from agentkavach import AgentKavach, Budget

    guard = AgentKavach(
        provider=provider,
        api_key=api_key,
        llm_key="sk-fake-llm-key-not-real",
        agent_name=agent,
        budget=Budget.daily(limit=100.0),
        # endpoint= matches the env var override (belt + suspenders so the
        # exporter doesn't accidentally resolve to a real backend URL even
        # if the env override gets stripped by some other fixture).
        endpoint="http://testserver",
        buffer_path=buffer_path,
    )
    _patch_exporter_to_testclient(guard, client)
    return guard


def _assert_single_row(
    session_factory,
    *,
    org_id: str,
    agent_name: str,
    expected_cost: float,
):
    """Core invariant: 1 Event row + 1 IdempotencyKey row, correct cost."""
    from server.models import Event, IdempotencyKey

    session = session_factory()
    try:
        rows = (
            session.query(Event)
            .filter(Event.org_id == org_id, Event.agent_name == agent_name)
            .all()
        )
        assert len(rows) == 1, (
            f"Expected exactly 1 DB row for agent {agent_name!r} but found "
            f"{len(rows)} — this is the Phase 58 double-write bug."
        )
        assert abs(rows[0].cost - expected_cost) < 1e-9, (
            f"Cost mismatch: DB row has {rows[0].cost!r}, SDK reported {expected_cost!r}"
        )

        idem_rows = session.query(IdempotencyKey).filter(IdempotencyKey.org_id == org_id).all()
        # One idempotency key per logical call. The buggy double-write
        # path produces TWO keys (one for the buffer write, one for the
        # OTel span) — this assertion would catch it even if the
        # event-row assertion were ever weakened.
        assert len(idem_rows) == 1, (
            f"Expected exactly 1 idempotency_key row but found "
            f"{len(idem_rows)} — indicates two distinct send paths."
        )
    finally:
        session.close()


@pytest.mark.parametrize(
    "provider,model,response_factory,mock_attr",
    _PROVIDER_CASES,
    ids=[p[0] for p in _PROVIDER_CASES],
)
@pytest.mark.real_tracer
def test_one_call_one_row_non_streaming(
    backend, tmp_path, provider, model, response_factory, mock_attr
):
    """One non-streaming ``guard.create`` → exactly one DB row.

    Exercises ``_post_flight`` which writes to the buffer AND triggers
    the OTel span. The buggy SDK lands TWO rows; the fixed SDK lands
    ONE.
    """
    from agentkavach.providers import UsageRecord
    from agentkavach.providers import openai as openai_provider

    client, api_key, org_id, session_factory = backend

    # Canned response with stable token counts so the cost assertion is
    # deterministic regardless of provider parser quirks.
    in_toks, out_toks = 123, 45
    response = response_factory(model, in_toks, out_toks)

    # Provider-correct expected cost via the SDK's own pricing engine —
    # never hardcode dollar values; if PRICE_TABLE shifts the assertion
    # follows it.
    from agentkavach.pricing import estimate_cost

    expected_cost = estimate_cost(model, in_toks, out_toks)
    assert expected_cost is not None, f"Model {model!r} missing from PRICE_TABLE"

    # Force the env-var override INSIDE the test body too — the SDK
    # reads it at construction time and other fixtures (autouse) may
    # have reset env between fixtures.
    with patch.dict(os.environ, _TEST_ENV):
        agent_name = f"phase61-{provider}-nonstream"
        mock_client = _wire_provider_mock(provider, response)

        with patch(f"agentkavach.client.AgentKavach.{mock_attr}", return_value=mock_client):
            guard = _build_guard(
                client,
                api_key,
                provider,
                agent_name,
                buffer_path=str(tmp_path / "buffer.jsonl"),
            )

            # The actual call. Each provider's _call_provider takes
            # different kwargs, but ``messages`` is the common
            # OpenAI-style input and the only one that flows through
            # the SDK's pre-flight token counter. For Google the SDK
            # also accepts ``contents`` — pass both for safety.
            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
            }
            if provider == "google":
                kwargs["contents"] = "hi"
            guard.create(**kwargs)

            _flush_and_shutdown(guard)

    _assert_single_row(
        session_factory,
        org_id=org_id,
        agent_name=agent_name,
        expected_cost=expected_cost,
    )

    # Silence unused-import warning when ruff catches symbols we keep
    # around for clarity in the body docstrings.
    _ = (UsageRecord, openai_provider)


# Streaming tests cover OpenAI + Anthropic only. Google/Mistral
# streaming has its own chunk-shape complexity that Phase 44 fully
# tested in tests/test_streaming_providers.py — duplicating those
# here would double the surface area without exercising the
# double-write invariant any differently (the streaming write path
# is `_on_stream_complete`, which mirrors `_post_flight`'s buffer +
# OTel pattern identically across providers).
@pytest.mark.parametrize(
    "provider,model,mock_attr",
    [
        ("openai", "gpt-4o", "_get_openai_client"),
        ("anthropic", "claude-sonnet-4-0", "_get_anthropic_client"),
    ],
    ids=["openai-stream", "anthropic-stream"],
)
@pytest.mark.real_tracer
def test_one_streaming_call_one_row(backend, tmp_path, provider, model, mock_attr):
    """One streaming ``guard.create`` → exactly one DB row.

    Exercises ``_on_stream_complete`` (not ``_post_flight``). The
    streaming code path has the SAME double-write structure: buffer
    write + OTel span. A separate test guards that path because Phase
    59 must fix both call-sites — fixing only the non-streaming one
    would leave streaming users silently double-billed.
    """
    client, api_key, org_id, session_factory = backend

    # Build a tiny stream that ends with a usage chunk. Token counts
    # come out of the final ``message_delta`` (anthropic) /
    # ``stream_options.include_usage`` (openai) chunk so the SDK
    # records a non-zero output.
    if provider == "openai":
        final_chunk = SimpleNamespace(choices=[], usage=SimpleNamespace(completion_tokens=42))
        chunks = [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="hi"))],
                usage=None,
            ),
            final_chunk,
        ]
    else:  # anthropic
        chunks = [
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="hi"),
            ),
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="end_turn"),
                usage=SimpleNamespace(output_tokens=42),
            ),
        ]

    with patch.dict(os.environ, _TEST_ENV):
        agent_name = f"phase61-{provider}-stream"
        mock_client = MagicMock()
        if provider == "openai":
            mock_client.chat.completions.create.return_value = iter(chunks)
        else:
            mock_client.messages.create.return_value = iter(chunks)
            mock_client.messages.count_tokens.return_value = SimpleNamespace(input_tokens=100)

        with patch(f"agentkavach.client.AgentKavach.{mock_attr}", return_value=mock_client):
            guard = _build_guard(
                client,
                api_key,
                provider,
                agent_name,
                buffer_path=str(tmp_path / "buffer.jsonl"),
            )

            stream = guard.create(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )
            # Drain the wrapper so _on_stream_complete fires.
            for _ in stream:
                pass

            _flush_and_shutdown(guard)

    # Phase 65: streaming events now record the FULL cost
    # (input_tokens × input_per_1k + output_tokens × output_per_1k).
    # Pre-Phase-65 the SDK wrote cost=0 / input_tokens=0 for streams,
    # so a customer using streaming saw the event counter climb while
    # their dashboard showed $0 spent — budgets and alerts never
    # fired. This assertion guards the fix: any regression that
    # silently drops cost back to zero will fail CI immediately.
    from agentkavach.pricing import estimate_cost
    from server.models import Event, IdempotencyKey

    # Mocks above pin input_tokens via count_tokens stubs (Anthropic =
    # 100; OpenAI uses tiktoken on the literal "hi" prompt). The
    # output side is whatever the final usage chunk reports — 42 for
    # both providers in this test. We resolve the expected cost from
    # the SDK's own pricing engine so the assertion follows
    # PRICE_TABLE updates without manual maintenance.
    session = session_factory()
    try:
        rows = (
            session.query(Event)
            .filter(Event.org_id == org_id, Event.agent_name == agent_name)
            .all()
        )
        assert len(rows) == 1, (
            f"Expected 1 DB row for streaming {provider} but found "
            f"{len(rows)} — streaming double-write regression."
        )
        row = rows[0]
        # The fixture above sends one content-delta chunk ("hi" = 2
        # chars → 1 token via the max(1, len//4) heuristic) plus a
        # final usage chunk reporting 42 exact tokens, so the SDK
        # accumulates 43 output_tokens. Don't hardcode the
        # heuristic's output — assert the row > 0 and let the cost
        # cross-check below verify the math is self-consistent.
        assert row.output_tokens > 0
        # Cost must be > 0 AND match the SDK's pricing engine.
        # Phase 65 fixes the bug where streaming events landed at $0.
        expected_cost = estimate_cost(model, row.input_tokens, row.output_tokens)
        assert expected_cost is not None and expected_cost > 0
        assert row.cost > 0, (
            f"Streaming {provider} event recorded cost={row.cost!r} — Phase 65 "
            f"regression: streams must price the call, not write $0."
        )
        assert abs(row.cost - expected_cost) < 1e-9, (
            f"Cost mismatch for streaming {provider}: row={row.cost!r}, "
            f"pricing engine={expected_cost!r}"
        )
        idem_rows = session.query(IdempotencyKey).filter(IdempotencyKey.org_id == org_id).all()
        assert len(idem_rows) == 1, (
            f"Expected 1 idempotency_key row for streaming {provider} but "
            f"found {len(idem_rows)} — two send paths active."
        )
    finally:
        session.close()
