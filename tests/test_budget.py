"""Unit tests for agentkavach.budget — budget types and key generation."""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from agentkavach.budget import Budget, Period, _budget_key


class TestBudgetConstructors:
    def test_daily(self):
        b = Budget.daily(limit=50.0)
        assert b.limit == 50.0
        assert b.period is Period.DAILY
        assert b.shared_name is None
        assert not b.is_shared

    def test_monthly(self):
        b = Budget.monthly(limit=500.0)
        assert b.limit == 500.0
        assert b.period is Period.MONTHLY

    def test_total(self):
        b = Budget.total(limit=1000.0)
        assert b.limit == 1000.0
        assert b.period is Period.TOTAL

    def test_shared_budget_method_removed(self):
        """Budget.shared_budget was removed in favor of org_budget."""
        assert not hasattr(Budget, "shared_budget")

    def test_negative_limit_raises(self):
        with pytest.raises(ValueError, match="positive"):
            Budget.daily(limit=-10.0)

    def test_zero_limit_raises(self):
        with pytest.raises(ValueError, match="positive"):
            Budget.daily(limit=0.0)

    def test_org_budget_creates_org_scoped_budget(self):
        b = Budget.org_budget(limit=50.0)
        assert b.limit == 50.0
        assert b.period is Period.DAILY
        assert b.shared_name == "__org__"
        assert b.is_shared

    def test_org_budget_default_period_is_daily(self):
        b = Budget.org_budget(limit=10.0)
        assert b.period is Period.DAILY

    def test_org_budget_monthly(self):
        b = Budget.org_budget(limit=500.0, period="monthly")
        assert b.limit == 500.0
        assert b.period is Period.MONTHLY
        assert b.shared_name == "__org__"

    def test_org_budget_invalid_period_raises(self):
        with pytest.raises(ValueError):
            Budget.org_budget(limit=10.0, period="weekly")

    def test_org_budget_key(self):
        b = Budget.org_budget(limit=50.0)
        assert "shared:__org__:" in b.key

    def test_frozen(self):
        b = Budget.daily(limit=50.0)
        with pytest.raises(AttributeError):
            b.limit = 100.0  # type: ignore[misc]


class TestBudgetKey:
    def test_daily_key_format(self):
        key = _budget_key(Period.DAILY)
        assert re.match(r"daily:\d{4}-\d{2}-\d{2}$", key)

    def test_monthly_key_format(self):
        key = _budget_key(Period.MONTHLY)
        assert re.match(r"monthly:\d{4}-\d{2}$", key)

    def test_total_key(self):
        assert _budget_key(Period.TOTAL) == "total"

    def test_shared_key_prefix(self):
        key = _budget_key(Period.DAILY, shared_name="team")
        assert key.startswith("shared:team:daily:")

    def test_budget_key_property(self):
        b = Budget.daily(limit=50.0)
        now = datetime.now(timezone.utc)
        expected = f"daily:{now.strftime('%Y-%m-%d')}"
        assert b.key == expected

    def test_org_budget_key_property(self):
        b = Budget.org_budget(limit=100.0)
        assert "shared:__org__:" in b.key


class TestPeriodEnum:
    def test_values(self):
        assert Period.DAILY.value == "daily"
        assert Period.MONTHLY.value == "monthly"
        assert Period.TOTAL.value == "total"

    def test_from_string(self):
        assert Period("daily") is Period.DAILY

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            Period("weekly")
