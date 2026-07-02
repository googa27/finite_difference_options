"""Executable Heston state-convention and oracle tests for FDO issue #45."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import norm

from finite_difference_options.exceptions import ValidationError
from finite_difference_options.pricing import (
    create_unified_european_call,
    create_unified_pricing_engine,
)
from finite_difference_options.processes import create_standard_heston
from finite_difference_options.validation.black_scholes_parity import (
    black_scholes_call_oracle,
)
from finite_difference_options.validation.heston_oracle import (
    HestonOracleCase,
    heston_call_oracle,
    heston_variance_boundary_benchmark,
)


def _black_scholes_call_with_dividend(
    *,
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    integrated_variance: float,
    maturity: float,
) -> float:
    volatility_time = np.sqrt(integrated_variance)
    d1 = (
        np.log(spot / strike)
        + (rate - dividend_yield) * maturity
        + 0.5 * integrated_variance
    ) / volatility_time
    d2 = d1 - volatility_time
    return float(
        spot * np.exp(-dividend_yield * maturity) * norm.cdf(d1)
        - strike * np.exp(-rate * maturity) * norm.cdf(d2)
    )


def _deterministic_heston_integrated_variance(case: HestonOracleCase) -> float:
    return (
        case.theta * case.maturity
        + (case.variance - case.theta)
        * (1.0 - np.exp(-case.kappa * case.maturity))
        / case.kappa
    )


def test_heston_log_state_coefficients_match_documented_equations() -> None:
    process = create_standard_heston(
        r=0.05,
        kappa=2.0,
        theta=0.04,
        sigma=0.3,
        rho=-0.7,
        dividend_yield=0.01,
    )
    x = np.log(100.0)
    v = 0.09
    state = np.array([x, v])

    drift = process.drift(0.0, state)
    covariance = process.covariance(0.0, state)

    assert_allclose(drift, np.array([0.05 - 0.01 - 0.5 * v, 2.0 * (0.04 - v)]))
    assert_allclose(
        covariance,
        np.array(
            [
                [v, -0.7 * 0.3 * v],
                [-0.7 * 0.3 * v, 0.3**2 * v],
            ]
        ),
    )
    assert_allclose(process.affine_covariance_form().evaluate(state)[0], covariance)


def test_heston_payoff_receives_spot_through_explicit_log_transform() -> None:
    process = create_standard_heston()
    engine = create_unified_pricing_engine(process)
    option = create_unified_european_call(strike=100.0, maturity=0.25)
    spot_grid = np.array([80.0, 100.0, 120.0])
    log_spot_grid = np.log(spot_grid)
    variance_grid = np.array([0.01, 0.04])

    terminal = engine._build_initial_condition(option, log_spot_grid, variance_grid)

    expected = np.broadcast_to(
        np.maximum(spot_grid - 100.0, 0.0).reshape(-1, 1), (3, 2)
    )
    assert_allclose(terminal, expected, atol=1e-12)
    metadata = process.factor_metadata()[0]
    assert metadata.coordinate == "log_spot"
    assert metadata.payoff_transform == "exp"


def test_heston_variance_boundary_policy_is_explicit_and_fail_closed() -> None:
    process = create_standard_heston(kappa=2.0, theta=0.04, sigma=0.3)
    benchmark = heston_variance_boundary_benchmark(process)

    assert benchmark.lower_boundary == 0.0
    assert benchmark.drift_at_lower_boundary > 0.0
    assert benchmark.min_eigenvalue_at_lower_boundary >= -1e-12
    assert benchmark.clips_interior_variance is False

    with pytest.raises(
        ValidationError, match="variance coordinate must be non-negative"
    ):
        process.covariance(0.0, np.array([np.log(100.0), -1e-4]))


def test_heston_fourier_oracle_recovers_black_scholes_limit() -> None:
    case = HestonOracleCase(
        spot=100.0,
        strike=100.0,
        rate=0.03,
        dividend_yield=0.0,
        maturity=1.0,
        variance=0.04,
        kappa=3.0,
        theta=0.04,
        vol_of_vol=1e-4,
        rho=-0.4,
    )

    heston_price = heston_call_oracle(case)
    black_scholes_price = black_scholes_call_oracle(
        spot=case.spot,
        strike=case.strike,
        rate=case.rate,
        sigma=np.sqrt(case.theta),
        maturity=case.maturity,
    )

    assert_allclose(heston_price, black_scholes_price, rtol=0.0, atol=5e-3)


def test_heston_zero_vol_of_vol_limit_preserves_dividend_yield() -> None:
    case = HestonOracleCase(
        spot=100.0,
        strike=100.0,
        rate=0.03,
        dividend_yield=0.05,
        maturity=1.0,
        variance=0.04,
        kappa=3.0,
        theta=0.04,
        vol_of_vol=0.0,
        rho=-0.4,
    )

    expected = _black_scholes_call_with_dividend(
        spot=case.spot,
        strike=case.strike,
        rate=case.rate,
        dividend_yield=case.dividend_yield,
        integrated_variance=case.theta * case.maturity,
        maturity=case.maturity,
    )

    assert_allclose(heston_call_oracle(case), expected, rtol=0.0, atol=1e-12)


def test_heston_zero_vol_of_vol_limit_uses_mean_reverting_variance_path() -> None:
    case = HestonOracleCase(
        spot=100.0,
        strike=100.0,
        rate=0.03,
        dividend_yield=0.0,
        maturity=1.0,
        variance=0.09,
        kappa=3.0,
        theta=0.04,
        vol_of_vol=0.0,
        rho=-0.4,
    )

    integrated_variance = _deterministic_heston_integrated_variance(case)
    expected = _black_scholes_call_with_dividend(
        spot=case.spot,
        strike=case.strike,
        rate=case.rate,
        dividend_yield=case.dividend_yield,
        integrated_variance=integrated_variance,
        maturity=case.maturity,
    )

    assert_allclose(heston_call_oracle(case), expected, rtol=0.0, atol=1e-12)


def test_heston_fourier_oracle_is_stable_under_tighter_integration() -> None:
    case = HestonOracleCase(
        spot=100.0,
        strike=105.0,
        rate=0.04,
        dividend_yield=0.01,
        maturity=0.75,
        variance=0.05,
        kappa=1.8,
        theta=0.04,
        vol_of_vol=0.35,
        rho=-0.6,
    )

    coarse = heston_call_oracle(case, abs_tol=1e-7, rel_tol=1e-7)
    tight = heston_call_oracle(case, abs_tol=1e-10, rel_tol=1e-10)

    assert coarse > 0.0
    assert_allclose(coarse, tight, rtol=0.0, atol=2e-4)
    assert_allclose(tight, 5.64422258051323, rtol=0.0, atol=1e-8)
