import pytest

from ops_agent.config import DEFAULT_MODEL
from ops_agent.pricing import PRICES, call_cost_usd


def test_pinned_model_has_a_price():
    assert DEFAULT_MODEL in PRICES


def test_cost_for_a_normal_call():
    assert call_cost_usd(DEFAULT_MODEL, 2000, 300) == pytest.approx(2000 * 1e-7 + 300 * 5e-7)


def test_long_prompt_tier_applies_per_call():
    assert call_cost_usd(DEFAULT_MODEL, 100_000, 10) == pytest.approx(100_000 * 5e-7 + 10 * 2.5e-6)


def test_unknown_model_has_no_price():
    assert call_cost_usd("some/other-model", 10, 10) is None
