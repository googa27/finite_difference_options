"""Discounted call bounds, including counterexamples to the retained v0 check."""

from __future__ import annotations

from math import exp

import pytest

from finite_difference_options.integrations.compiled_pde_black_scholes_route import _compiled_no_arbitrage
from finite_difference_options.validation.black_scholes_parity import (
    black_scholes_call_greeks,
    black_scholes_call_oracle,
)
from finite_difference_options.validation.fd_evidence.carry_bounds import call_carry_bounds


def _analytical_case(*, strike=1.0, rate=0.05, carry=0.0):
    value = black_scholes_call_oracle(1.0, strike, rate, 0.2, 1.0, dividend_yield=carry)
    greeks = black_scholes_call_greeks(1.0, strike, rate, 0.2, 1.0, dividend_yield=carry)
    return dict(spot=1.0, strike=strike, risk_free_rate=rate, dividend_yield=carry, maturity=1.0, value=value, **greeks)


@pytest.mark.parametrize(("strike", "rate", "carry"), [(0.5, 0.0, 0.8), (0.1, 0.0, -0.5), (1.0, -0.1, 0.2)])
def test_analytic_call_satisfies_discounted_bounds(strike, rate, carry):
    case = _analytical_case(strike=strike, rate=rate, carry=carry)
    result = call_carry_bounds(**case)
    assert all(value for key, value in result.items() if key.endswith("_ok"))
    assert result["delta_upper_bound"] == exp(-carry)
    assert result["price_lower_bound"] == max(exp(-carry) - strike * exp(-rate), 0.0)
    assert result["price_upper_bound"] == exp(-carry)
    if carry in (0.8, -0.5):
        legacy = _compiled_no_arbitrage(1.0, strike, case["value"], case["delta"], case["gamma"])
        assert not all(value for key, value in legacy.items() if key.endswith("_ok"))


@pytest.mark.parametrize(
    ("field", "value", "failed_check"),
    [
        ("value", -0.1, "value_bound_ok"),
        ("value", 1.1, "upper_bound_ok"),
        ("delta", -0.1, "delta_lower_bound_ok"),
        ("delta", 1.1, "delta_upper_bound_ok"),
        ("gamma", -0.1, "gamma_non_negative_ok"),
    ],
)
def test_each_pointwise_bound_has_a_negative_control(field, value, failed_check):
    case = _analytical_case()
    case[field] = value
    assert call_carry_bounds(**case)[failed_check] is False


@pytest.mark.parametrize(
    ("field", "value"), [("value", float("nan")), ("maturity", -1.0), ("strike", 0.0), ("dividend_yield", -1000.0)]
)
def test_invalid_or_overflowed_carry_bounds_are_refused(field, value):
    case = _analytical_case()
    case[field] = value
    with pytest.raises(ValueError):
        call_carry_bounds(**case)
