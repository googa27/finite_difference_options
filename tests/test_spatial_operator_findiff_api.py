"""Regression tests for the maintained Findiff derivative API."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.testing import assert_allclose

from finite_difference_options.instruments.operators import SpatialOperator
from finite_difference_options.processes.affine import GeometricBrownianMotion


def test_spatial_operator_uses_maintained_findiff_api_and_quadratic_oracle() -> None:
    """The GBM generator is exact for a quadratic under second-order stencils."""

    grid = np.linspace(80.0, 120.0, 5)
    process = GeometricBrownianMotion(mu=0.08, sigma=0.2)
    values = grid**2

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="FinDiff is deprecated")
        observed = SpatialOperator(process, discount_rate=0.03).build(grid)(values)

    first_derivative = 2.0 * grid
    second_derivative = np.full_like(grid, 2.0)
    expected = (
        0.5 * process.sigma**2 * grid**2 * second_derivative + process.mu * grid * first_derivative - 0.03 * values
    )
    assert_allclose(observed, expected, atol=1e-10)
