"""European-call pointwise bounds under deterministic proportional carry."""

from __future__ import annotations

from math import exp, isfinite
from typing import Any


def call_carry_bounds(
    *,
    spot: float,
    strike: float,
    value: float,
    delta: float,
    gamma: float,
    risk_free_rate: float,
    dividend_yield: float,
    maturity: float,
) -> dict[str, Any]:
    """Check discounted price/Delta bounds; this is not a full-surface proof."""
    if not all(isfinite(x) for x in (spot, strike, value, delta, gamma, risk_free_rate, dividend_yield, maturity)):
        raise ValueError("carry-bound inputs must be finite")
    if spot <= 0.0 or strike <= 0.0 or maturity < 0.0:
        raise ValueError("spot/strike must be positive and maturity nonnegative")
    try:
        carry_discount = exp(-dividend_yield * maturity)
        discounted_spot = spot * carry_discount
        discounted_strike = strike * exp(-risk_free_rate * maturity)
    except OverflowError as exc:
        raise ValueError("discounted carry bounds must remain finite") from exc
    if not all(isfinite(x) for x in (carry_discount, discounted_spot, discounted_strike)):
        raise ValueError("discounted carry bounds must remain finite")
    intrinsic = max(discounted_spot - discounted_strike, 0.0)
    return {
        "assumptions": "European call; deterministic proportional carry and fixed BSM parameters for Delta",
        "discounted_spot": discounted_spot,
        "discounted_strike": discounted_strike,
        "price_lower_bound": intrinsic,
        "price_upper_bound": discounted_spot,
        "delta_upper_bound": carry_discount,
        "value_minus_discounted_intrinsic": value - intrinsic,
        "upper_gap": discounted_spot - value,
        "value_bound_ok": value >= intrinsic - 1.0e-12,
        "upper_bound_ok": value <= discounted_spot + 1.0e-12,
        "delta_lower_bound_ok": delta >= -1.0e-12,
        "delta_upper_bound_ok": delta <= carry_discount + 1.0e-12,
        "gamma_non_negative_ok": gamma >= -1.0e-12,
        "scope": "single-point necessary bounds; not a full-surface no-arbitrage proof",
    }
