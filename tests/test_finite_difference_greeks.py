"""Unit tests for FiniteDifferenceGreeks."""

import pathlib
import sys


import numpy as np

from finite_difference_options.greeks import FiniteDifferenceGreeks


def test_greeks_methods_run():
    """Test that finite difference greeks methods run without errors."""
    s = np.linspace(0.0, 2.0, 5)
    t = np.linspace(0.0, 1.0, 5)

    # Create a simple grid
    grid = np.outer(np.ones_like(t), s**2)

    calc = FiniteDifferenceGreeks()

    # These should run without raising exceptions
    delta = calc.delta(grid, s)
    gamma = calc.gamma(grid, s)
    theta = calc.theta(grid, t)

    # Check that outputs have the expected shape
    assert delta.shape == grid.shape
    assert gamma.shape == grid.shape
    assert theta.shape == grid.shape
