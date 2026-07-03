"""American/Bermudan obstacle LCP route tests for issue #66."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from finite_difference_options.exceptions import ValidationError
from finite_difference_options.pricing import (
    UnifiedAmericanOption,
    UnifiedBermudanOption,
    create_unified_american_call,
    create_unified_american_put,
    create_unified_bermudan_put,
    create_unified_european_call,
    create_unified_european_put,
    create_unified_pricing_engine,
)
from finite_difference_options.processes import create_black_scholes_process


def _spot_index(grid: np.ndarray, spot: float = 100.0) -> int:
    return int(np.argmin(np.abs(grid - spot)))


def _market_setup():
    process = create_black_scholes_process(mu=0.05, sigma=0.20)
    engine = create_unified_pricing_engine(process)
    grid = np.linspace(1.0e-6, 260.0, 121)
    times = np.linspace(0.0, 1.0, 81)
    return engine, grid, times


def test_american_and_bermudan_options_are_distinct_contracts() -> None:
    american = create_unified_american_put(100.0, 1.0)
    bermudan = create_unified_bermudan_put(100.0, 1.0, exercise_dates=(0.5, 1.0))

    assert isinstance(american, UnifiedAmericanOption)
    assert isinstance(bermudan, UnifiedBermudanOption)
    assert american.exercise_style == "american"
    assert bermudan.exercise_style == "bermudan"
    assert bermudan.exercise_dates == (0.5, 1.0)

    with pytest.raises(ValidationError, match="exercise_dates"):
        UnifiedBermudanOption(strike=100.0, maturity=1.0, option_type="put", exercise_dates=(0.75, 0.5))


def test_american_put_satisfies_obstacle_and_complementarity_diagnostics() -> None:
    engine, grid, times = _market_setup()
    option = create_unified_american_put(100.0, 1.0)

    prices = engine.price_option(option, grid, time_grid=times)
    payoff = option.payoff(grid)
    diagnostics = engine.solver.last_lcp_diagnostics

    assert prices.shape == (len(times), len(grid))
    assert_allclose(prices[-1], payoff, atol=1.0e-10)
    assert np.all(prices >= payoff - 5.0e-8)
    assert diagnostics.converged
    assert diagnostics.exercise_style == "american"
    assert diagnostics.max_primal_violation <= 5.0e-8
    assert diagnostics.max_dual_violation <= 5.0e-5
    assert diagnostics.max_complementarity <= 5.0e-4
    assert diagnostics.max_iterations > 0
    assert diagnostics.exercise_boundary[0] > 0.0


def test_american_put_dominates_matched_european_put_at_valuation() -> None:
    engine, grid, times = _market_setup()
    american = create_unified_american_put(100.0, 1.0)
    european = create_unified_european_put(100.0, 1.0)

    american_prices = engine.price_option(american, grid, time_grid=times)
    european_prices = engine.price_option(european, grid, time_grid=times)
    spot_idx = _spot_index(grid)

    assert american_prices[0, spot_idx] >= european_prices[0, spot_idx] - 5.0e-3
    assert american_prices[0, spot_idx] > american.payoff(grid)[spot_idx]


def test_non_dividend_american_call_matches_european_call() -> None:
    engine, grid, times = _market_setup()
    american = create_unified_american_call(100.0, 1.0)
    european = create_unified_european_call(100.0, 1.0)

    american_prices = engine.price_option(american, grid, time_grid=times)
    european_prices = engine.price_option(european, grid, time_grid=times)
    spot_idx = _spot_index(grid)

    assert abs(american_prices[0, spot_idx] - european_prices[0, spot_idx]) <= 0.75
    assert engine.solver.last_lcp_diagnostics.exercise_boundary[0] == 0.0


def test_bermudan_put_orders_between_european_and_american_values() -> None:
    engine, grid, times = _market_setup()
    european = create_unified_european_put(100.0, 1.0)
    bermudan = create_unified_bermudan_put(100.0, 1.0, exercise_dates=(0.25, 0.5, 0.75, 1.0))
    american = create_unified_american_put(100.0, 1.0)

    european_prices = engine.price_option(european, grid, time_grid=times)
    bermudan_prices = engine.price_option(bermudan, grid, time_grid=times)
    bermudan_diagnostics = engine.solver.last_lcp_diagnostics
    american_prices = engine.price_option(american, grid, time_grid=times)
    spot_idx = _spot_index(grid)

    assert european_prices[0, spot_idx] <= bermudan_prices[0, spot_idx] + 5.0e-3
    assert bermudan_prices[0, spot_idx] <= american_prices[0, spot_idx] + 5.0e-3
    assert bermudan_diagnostics.exercise_style == "bermudan"


def test_lcp_iteration_limit_failure_is_explicit() -> None:
    engine, grid, times = _market_setup()
    option = UnifiedAmericanOption(
        strike=100.0,
        maturity=1.0,
        option_type="put",
        lcp_tolerance=1.0e-14,
        lcp_max_iterations=1,
    )

    with pytest.raises(ValidationError, match="American LCP solver did not converge"):
        engine.price_option(option, grid, time_grid=times)
