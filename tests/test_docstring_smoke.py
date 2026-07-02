"""Smoke tests derived from public docstring examples.

These tests execute small pricing/greeks snippets that are presented in module
documentation and keep the documented examples runnable.
"""

import numpy as np

from finite_difference_options.processes.affine import (
    create_black_scholes_process,
    create_standard_heston,
)
from finite_difference_options.pricing import (
    create_log_grid,
    create_unified_european_call,
    create_unified_pricing_engine,
)
from finite_difference_options.greeks.base import GreeksCalculatorFactory


def test_docstring_example_black_scholes_price_smoke() -> None:
    """Smoke test for 1D Black--Scholes pricing example.

    This mirrors the usage pattern shown in ``pricing/engines/unified.py``.
    """
    process = create_black_scholes_process(mu=0.03, sigma=0.2)
    engine = create_unified_pricing_engine(process)
    option = create_unified_european_call(strike=100.0, maturity=1.0)
    grid = create_log_grid(20.0, 200.0, 51)
    times = np.linspace(0.0, 1.0, 8)

    prices = engine.price_option(option, grid, time_grid=times)
    assert prices.shape == (len(times), len(grid))
    assert np.all(np.isfinite(prices))
    assert np.all(prices >= 0.0)


def test_docstring_example_heston_price_smoke() -> None:
    """Smoke test for 2D Heston pricing example."""
    process = create_standard_heston(
        r=0.03, kappa=1.8, theta=0.05, sigma=0.35, rho=-0.35
    )
    engine = create_unified_pricing_engine(process)
    option = create_unified_european_call(strike=100.0, maturity=0.25)
    spot_grid = create_log_grid(40.0, 220.0, 17, center=100.0)
    x_grid = np.log(spot_grid)
    v_grid = np.linspace(0.01, 0.30, 8)

    time_grid = np.linspace(0.0, option.maturity, 10)

    prices = engine.price_option(option, x_grid, v_grid, time_grid=time_grid)
    assert prices.shape == (len(time_grid), len(x_grid), len(v_grid))
    assert np.all(np.isfinite(prices))
    assert np.all(prices >= 0.0)


def test_docstring_example_greeks_smoke() -> None:
    """Smoke test for Greeks calculation example."""
    process = create_black_scholes_process(mu=0.03, sigma=0.2)
    engine = create_unified_pricing_engine(process)
    option = create_unified_european_call(strike=100.0, maturity=0.5)
    s_grid = np.linspace(40.0, 200.0, 31)
    times = np.linspace(0.0, 0.5, 12)

    prices = engine.price_option(option, s_grid, time_grid=times)
    calculator = GreeksCalculatorFactory.create_calculator(process)
    greeks = calculator.calculate(prices, s_grid, time_grid=times)

    assert set(greeks).issuperset({"delta", "gamma", "theta"})
    assert greeks["delta"].shape == (len(s_grid),)
    assert greeks["gamma"].shape == (len(s_grid),)
    assert greeks["theta"].shape == prices.shape
