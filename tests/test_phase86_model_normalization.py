"""Phase 86: SDK records the model the customer REQUESTED, not the
provider's echo.

Why: OpenAI returns versioned snapshots (``gpt-4o-mini-2024-07-18``)
from non-streaming responses but aliases (``gpt-4o-mini``) from
streaming chunks. Same logical model — but the dashboard's Model
Breakdown grouped events by exact-match ``model`` string, so customers
who called ``guard.create(model="gpt-4o-mini")`` saw two separate rows
for the same agent. Cost and usage fragmented across both rows,
confusing customers into thinking they were using two different models.

Fix: ``_post_flight`` and the streaming completion callback always
record the model name the customer passed to ``guard.create()``. The
provider echo is logged at DEBUG when it differs. The pricing engine
still resolves the right price (the table has both forms with identical
prices, and ``estimate_cost`` normalizes aliases via ``_prefix_match``).

Coverage: 4 providers × {non-streaming, streaming} = 8 paths.
Plus alias-equivalence checks against the pricing table.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from agentkavach.budget import Budget
from agentkavach.client import AgentKavach
from agentkavach.pricing import estimate_cost, get_price

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _client(provider: str, tmp_path) -> AgentKavach:
    """Test-only AgentKavach: no API key (skip OTel), buffer enabled."""
    return AgentKavach(
        api_key="ak_test",  # no OTel → buffer is the event sink
        llm_key="sk-test-not-real",
        provider=provider,
        agent_name=f"test-{provider}",
        budget=Budget.daily(limit=10.0),
        buffer_path=str(tmp_path / f"buf-{provider}.jsonl"),
    )


def _openai_response(echoed_model: str) -> SimpleNamespace:
    return SimpleNamespace(
        model=echoed_model,
        usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=500),
    )


def _anthropic_response(echoed_model: str) -> SimpleNamespace:
    return SimpleNamespace(
        model=echoed_model,
        usage=SimpleNamespace(input_tokens=1000, output_tokens=500),
        content=[],
    )


def _google_response(echoed_model: str) -> SimpleNamespace:
    return SimpleNamespace(
        model=echoed_model,
        usage_metadata=SimpleNamespace(
            prompt_token_count=1000,
            candidates_token_count=500,
        ),
        choices=[],
    )


def _mistral_response(echoed_model: str) -> SimpleNamespace:
    return SimpleNamespace(
        model=echoed_model,
        usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=500),
    )


# ---------------------------------------------------------------------------
# Non-streaming: _post_flight records the REQUESTED model
# ---------------------------------------------------------------------------


class TestPostFlightUsesRequestedModel:
    """The buffered event's ``model`` field must equal the requested
    name, even when the provider echoes a different (e.g. versioned)
    form back to us."""

    def test_openai_post_flight_uses_requested_model(self, tmp_path):
        cg = _client("openai", tmp_path)
        # Provider echoes the versioned snapshot.
        resp = _openai_response(echoed_model="gpt-4o-mini-2024-07-18")
        # Customer asked for the alias.
        cg._post_flight(resp, requested_model="gpt-4o-mini")

        events = cg._buffer.read_all()
        assert len(events) == 1
        assert events[0]["model"] == "gpt-4o-mini", (
            "Phase 86: customer wrote 'gpt-4o-mini' in their code — "
            "that's what must appear in the dashboard's Model Breakdown."
        )
        # Cost must still resolve (alias and versioned form share pricing).
        assert events[0]["cost"] > 0

    def test_anthropic_post_flight_uses_requested_model(self, tmp_path):
        cg = _client("anthropic", tmp_path)
        # Provider echoes a dated snapshot.
        resp = _anthropic_response(echoed_model="claude-3-5-sonnet-20241022")
        # Customer asked for the alias.
        cg._post_flight(resp, requested_model="claude-3.5-sonnet")

        events = cg._buffer.read_all()
        assert len(events) == 1
        assert events[0]["model"] == "claude-3.5-sonnet"
        assert events[0]["cost"] > 0

    def test_google_post_flight_uses_requested_model(self, tmp_path):
        cg = _client("google", tmp_path)
        # Provider echoes a different model than requested.
        resp = _google_response(echoed_model="gemini-1.5-flash")
        cg._post_flight(resp, requested_model="gemini-2.5-flash")

        events = cg._buffer.read_all()
        assert len(events) == 1
        assert events[0]["model"] == "gemini-2.5-flash"
        assert events[0]["cost"] > 0

    def test_mistral_post_flight_uses_requested_model(self, tmp_path):
        cg = _client("mistral", tmp_path)
        resp = _mistral_response(echoed_model="mistral-large-2411")
        cg._post_flight(resp, requested_model="mistral-large-latest")

        events = cg._buffer.read_all()
        assert len(events) == 1
        assert events[0]["model"] == "mistral-large-latest"
        assert events[0]["cost"] > 0


# ---------------------------------------------------------------------------
# Streaming: _on_stream_complete records the REQUESTED model
# ---------------------------------------------------------------------------


class TestStreamCompleteUsesRequestedModel:
    """``StreamWrapper`` calls back with the model passed at construction
    time, which is always the requested name — not a value pulled from a
    chunk. Pre-Phase-86 chunks could carry the alias while the
    non-streaming response carried the versioned form (or vice versa),
    so customers saw two rows. Stream side was already correct here, but
    we lock it in with a regression test for each provider."""

    def _drain(self, cg: AgentKavach, provider: str, requested_model: str) -> None:
        """Mimic what _safe_call does when stream=True: build a
        StreamWrapper and exhaust it so the on_complete callback fires."""
        from agentkavach.stream import StreamWrapper

        cg._provider = provider
        # input_tokens needed for streaming cost calc (Phase 65).
        cg._last_input_tokens = 1000

        # One chunk with a usage tail so output_tokens > 0.
        if provider == "openai":
            chunks = [SimpleNamespace(usage=SimpleNamespace(completion_tokens=500), choices=[])]
        elif provider == "anthropic":
            chunks = [
                SimpleNamespace(usage=SimpleNamespace(output_tokens=500), choices=[], delta=None)
            ]
        elif provider == "google":
            chunks = [
                SimpleNamespace(
                    usage_metadata=SimpleNamespace(candidates_token_count=500),
                    choices=[],
                    text=None,
                    delta=None,
                )
            ]
        elif provider == "mistral":
            inner = SimpleNamespace(usage=SimpleNamespace(completion_tokens=500), choices=[])
            chunks = [SimpleNamespace(data=inner, usage=None, choices=[], delta=None, text=None)]
        else:
            raise AssertionError(provider)

        wrapper = StreamWrapper(
            stream=iter(chunks),
            model=requested_model,
            engine=cg._engine,
            on_complete=cg._on_stream_complete,
        )
        list(wrapper)  # drain → triggers _finalize → _on_stream_complete.

    def test_openai_stream_uses_requested_model(self, tmp_path):
        cg = _client("openai", tmp_path)
        self._drain(cg, "openai", requested_model="gpt-4o-mini")

        events = cg._buffer.read_all()
        assert len(events) == 1
        # Streaming side records the model passed to StreamWrapper —
        # always the requested one, never read from chunks.
        assert events[0]["model"] == "gpt-4o-mini"
        assert events[0]["output_tokens"] == 500

    def test_anthropic_stream_uses_requested_model(self, tmp_path):
        cg = _client("anthropic", tmp_path)
        self._drain(cg, "anthropic", requested_model="claude-3.5-sonnet")

        events = cg._buffer.read_all()
        assert len(events) == 1
        assert events[0]["model"] == "claude-3.5-sonnet"

    def test_google_stream_uses_requested_model(self, tmp_path):
        cg = _client("google", tmp_path)
        self._drain(cg, "google", requested_model="gemini-2.5-flash")

        events = cg._buffer.read_all()
        assert len(events) == 1
        assert events[0]["model"] == "gemini-2.5-flash"

    def test_mistral_stream_uses_requested_model(self, tmp_path):
        cg = _client("mistral", tmp_path)
        self._drain(cg, "mistral", requested_model="mistral-large-latest")

        events = cg._buffer.read_all()
        assert len(events) == 1
        assert events[0]["model"] == "mistral-large-latest"


# ---------------------------------------------------------------------------
# End-to-end via _safe_call: requested model survives the full pipeline
# ---------------------------------------------------------------------------


class TestEndToEndRequestedModelWins:
    """Drive the full ``guard.create()`` path with the OpenAI mock
    returning a versioned snapshot — assert the buffer records the alias.
    This is the exact customer-visible bug Phase 86 fixes."""

    def test_openai_create_records_requested_not_echoed(self, tmp_path):
        from unittest.mock import patch

        cg = _client("openai", tmp_path)
        echoed = _openai_response(echoed_model="gpt-4o-mini-2024-07-18")

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = echoed

        with patch.object(cg, "_get_openai_client", return_value=mock_client):
            cg.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
            )

        events = cg._buffer.read_all()
        assert len(events) == 1
        assert events[0]["model"] == "gpt-4o-mini", (
            "End-to-end customer mental model: they wrote 'gpt-4o-mini' "
            "in their code, so 'gpt-4o-mini' is what should appear in "
            "the dashboard. The OpenAI API echo of "
            "'gpt-4o-mini-2024-07-18' must NOT leak through."
        )


# ---------------------------------------------------------------------------
# Pricing-table alias equivalence (no-change verification)
# ---------------------------------------------------------------------------


class TestAliasPricingEquivalence:
    """For every (alias, versioned) pair that surfaces in real customer
    traffic, both forms must price to the same cost. Otherwise the
    Phase 86 normalization (recording the alias instead of the
    versioned form) would distort costs."""

    EQUIVALENT_PAIRS = [
        ("gpt-4o", "gpt-4o-2024-11-20"),
        ("gpt-4o-mini", "gpt-4o-mini-2024-07-18"),
        # alias-resolution pairs (alias → canonical in the price table).
        ("claude-3.5-sonnet", "claude-3-5-sonnet-20241022"),
        ("claude-3.5-haiku", "claude-3-5-haiku-20241022"),
        ("mistral-large-latest", "mistral-large-2411"),
        ("mistral-small-latest", "mistral-small-2503"),
    ]

    def test_aliases_and_versioned_forms_price_identically(self):
        for alias, versioned in self.EQUIVALENT_PAIRS:
            price_alias = get_price(alias)
            price_versioned = get_price(versioned)
            assert price_alias is not None, f"missing price for alias {alias!r}"
            assert price_versioned is not None, f"missing price for {versioned!r}"
            assert price_alias.input_per_1k == price_versioned.input_per_1k, (
                f"{alias!r} input price != {versioned!r} input price — Phase 86 "
                f"normalization would distort recorded costs."
            )
            assert price_alias.output_per_1k == price_versioned.output_per_1k, (
                f"{alias!r} output price != {versioned!r} output price."
            )

    def test_estimate_cost_matches_across_alias(self):
        """End-to-end estimate_cost agreement, defending against future
        edits to the price table that might break the equivalence."""
        for alias, versioned in self.EQUIVALENT_PAIRS:
            c_alias = estimate_cost(alias, input_tokens=1000, output_tokens=500)
            c_versioned = estimate_cost(versioned, input_tokens=1000, output_tokens=500)
            assert c_alias == c_versioned, (
                f"estimate_cost({alias!r}) = {c_alias} but "
                f"estimate_cost({versioned!r}) = {c_versioned}"
            )
