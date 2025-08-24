"""Unit tests for FiniteDifferenceGreeks."""
import pathlib
import sys

# Ensure project root on path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from src.greeks import FiniteDifferenceGreeks


def test_greeks_on_polynomial_grid():
    """Finite differences of s^2 + t should match analytical results."""
    s = np.linspace(0.0, 1.0, 5)
    t = np.linspace(0.0, 1.0, 5)
    grid = np.add.outer(t, s**2)

    calc = FiniteDifferenceGreeks()
    delta = calc.delta(grid, s)
    gamma = calc.gamma(grid, s)
    theta = calc.theta(grid, t)

    # For this test we check the last time step (t=1.0)
    expected_delta = 2 * s
    expected_gamma = np.full_like(s, 2.0)
    expected_theta = -np.ones_like(grid)

    # Check that the last time step matches expected values
    assert np.allclose(delta[-1], expected_delta, atol=1e-12)
    assert np.allclose(gamma[-1], expected_gamma, atol=1e-12)
    assert np.allclose(theta, expected_theta, atol=1e-12)