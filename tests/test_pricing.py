"""Unit tests for agentkavach.pricing — price tables, aliases, cost estimation."""

from __future__ import annotations

import pytest

from agentkavach.pricing import (
    _ALIASES,
    _CUSTOM_PRICES,
    PRICE_TABLE,
    ModelPrice,
    estimate_cost,
    get_price,
    register_price,
    resolve_model,
    supported_models,
)

# ---------------------------------------------------------------------------
# resolve_model / get_price
# ---------------------------------------------------------------------------


class TestResolveModel:
    def test_canonical_name_unchanged(self):
        assert resolve_model("gpt-4o") == "gpt-4o"

    def test_alias_resolves(self):
        assert resolve_model("claude-opus") == "claude-opus-4-6"
        assert resolve_model("gemini-pro") == "gemini-2.5-pro"

    def test_mistral_alias_resolves(self):
        assert resolve_model("mistral-large") == "mistral-large-2411"
        assert resolve_model("mistral-small") == "mistral-small-2503"
        assert resolve_model("codestral") == "codestral-latest"
        assert resolve_model("pixtral-large") == "pixtral-large-latest"
        assert resolve_model("ministral-8b") == "ministral-8b-latest"

    def test_unknown_model_returned_as_is(self):
        assert resolve_model("some-future-model") == "some-future-model"


class TestGetPrice:
    def test_known_model(self):
        price = get_price("gpt-4o")
        assert price is not None
        assert isinstance(price, ModelPrice)
        assert price.input_per_1k > 0
        assert price.output_per_1k > 0

    def test_alias_lookup(self):
        price = get_price("claude-opus")
        assert price is not None
        assert price == get_price("claude-opus-4-0")

    def test_unknown_model_returns_none(self):
        assert get_price("nonexistent-model-v99") is None


# ---------------------------------------------------------------------------
# estimate_cost
# ---------------------------------------------------------------------------


class TestEstimateCost:
    def test_basic_estimation(self):
        # gpt-4o: input=$0.0025/1k, output=$0.010/1k
        cost = estimate_cost("gpt-4o", input_tokens=1000, output_tokens=1000)
        assert cost is not None
        assert cost == pytest.approx(0.0025 + 0.010)

    def test_zero_tokens(self):
        cost = estimate_cost("gpt-4o", input_tokens=0, output_tokens=0)
        assert cost == 0.0

    def test_unknown_model_returns_none(self):
        assert estimate_cost("fake-model", input_tokens=100, output_tokens=50) is None

    def test_alias_estimation(self):
        direct = estimate_cost("claude-opus-4-0", input_tokens=500, output_tokens=200)
        via_alias = estimate_cost("claude-opus", input_tokens=500, output_tokens=200)
        assert direct == via_alias

    @pytest.mark.parametrize(
        "model",
        [m for m in PRICE_TABLE],
    )
    def test_all_models_have_positive_prices(self, model):
        price = PRICE_TABLE[model]
        assert price.input_per_1k > 0, f"{model} input price must be positive"
        assert price.output_per_1k > 0, f"{model} output price must be positive"


# ---------------------------------------------------------------------------
# supported_models
# ---------------------------------------------------------------------------


class TestSupportedModels:
    def test_includes_canonical_models(self):
        models = supported_models()
        assert "gpt-4o" in models
        assert "claude-opus-4-0" in models

    def test_includes_aliases(self):
        models = supported_models()
        for alias in _ALIASES:
            assert alias in models

    def test_returns_sorted(self):
        models = supported_models()
        assert models == sorted(models)


# ---------------------------------------------------------------------------
# ModelPrice immutability
# ---------------------------------------------------------------------------


class TestPrefixMatch:
    """Date-suffixed and snapshot models should resolve via prefix stripping."""

    def test_anthropic_date_suffix(self):
        """claude-haiku-4-5-20251001 -> claude-haiku-4-5"""
        price = get_price("claude-haiku-4-5-20251001")
        assert price is not None
        assert price == get_price("claude-haiku-4-5")

    def test_anthropic_sonnet_date_suffix(self):
        """claude-sonnet-4-20250514 -> claude-sonnet-4 -> alias -> claude-sonnet-4-6"""
        price = get_price("claude-sonnet-4-20250514")
        assert price is not None
        assert price.input_per_1k == 0.003

    def test_openai_date_suffix(self):
        """gpt-4o-2025-03-15 -> gpt-4o"""
        price = get_price("gpt-4o-2025-03-15")
        assert price is not None
        assert price == get_price("gpt-4o")

    def test_openai_mini_date_suffix(self):
        """gpt-4o-mini-2025-01-01 -> gpt-4o-mini"""
        price = get_price("gpt-4o-mini-2025-01-01")
        assert price is not None
        assert price == get_price("gpt-4o-mini")

    def test_gemini_version_suffix(self):
        """gemini-2.0-flash-001 -> gemini-2.0-flash"""
        price = get_price("gemini-2.0-flash-001")
        assert price is not None
        assert price == get_price("gemini-2.0-flash")

    def test_mistral_date_suffix(self):
        """mistral-large-2501 -> mistral-large -> alias"""
        price = get_price("mistral-large-2501")
        assert price is not None

    def test_unknown_still_returns_none(self):
        """Completely unknown models still return None."""
        assert get_price("totally-unknown-model") is None

    def test_exact_match_still_preferred(self):
        """Exact match takes priority over prefix match."""
        price = get_price("claude-3-5-sonnet-20241022")
        assert price is not None
        assert price == PRICE_TABLE["claude-3-5-sonnet-20241022"]


class TestRegisterPrice:
    """Users can register custom pricing for new/private models."""

    def test_register_and_retrieve(self):
        register_price("my-custom-model", 0.01, 0.03)
        price = get_price("my-custom-model")
        assert price is not None
        assert price.input_per_1k == 0.01
        assert price.output_per_1k == 0.03
        # Cleanup
        _CUSTOM_PRICES.pop("my-custom-model", None)

    def test_custom_overrides_builtin(self):
        """Custom price takes priority over built-in table."""
        original = get_price("gpt-4o")
        register_price("gpt-4o", 0.999, 0.999)
        overridden = get_price("gpt-4o")
        assert overridden.input_per_1k == 0.999
        # Cleanup — restore original behavior
        _CUSTOM_PRICES.pop("gpt-4o", None)
        restored = get_price("gpt-4o")
        assert restored == original

    def test_estimate_cost_with_custom(self):
        register_price("test-model", 0.01, 0.02)
        cost = estimate_cost("test-model", input_tokens=1000, output_tokens=500)
        assert cost is not None
        assert abs(cost - (0.01 + 0.01)) < 0.0001  # 1K in * 0.01 + 0.5K out * 0.02
        _CUSTOM_PRICES.pop("test-model", None)

    def test_register_price_importable(self):
        """register_price is part of the public API."""
        from agentkavach import register_price as rp

        assert callable(rp)


class TestModelPrice:
    def test_frozen(self):
        price = ModelPrice(input_per_1k=0.01, output_per_1k=0.03)
        with pytest.raises(AttributeError):
            price.input_per_1k = 0.02  # type: ignore[misc]
