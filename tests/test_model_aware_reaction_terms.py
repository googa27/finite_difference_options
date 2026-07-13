"""Regression tests for model-aware operators and reaction terms."""

from __future__ import annotations



import numpy as np
from numpy.testing import assert_allclose

from finite_difference_options.instruments.operators import SpatialOperator
from finite_difference_options.processes.affine import (
    GeometricBrownianMotion,
    create_standard_heston,
)
from finite_difference_options.solvers.base import FiniteDifferenceSolverAdapter


def test_spatial_operator_accepts_explicit_discount_independent_of_drift() -> None:
    grid = np.linspace(80.0, 120.0, 5)
    process = GeometricBrownianMotion(mu=0.08, sigma=0.2)
    values = np.ones_like(grid)

    explicit = SpatialOperator(process, discount_rate=0.03).build(grid)(values)
    legacy = SpatialOperator(process).build(grid)(values)

    assert_allclose(explicit[1:-1], -0.03, atol=1e-10)
    assert_allclose(legacy[1:-1], -0.08, atol=1e-10)


def test_unified_adapter_preserves_explicit_zero_discount_rate() -> None:
    process = GeometricBrownianMotion(mu=0.08, sigma=0.2)
    adapter = FiniteDifferenceSolverAdapter(process)

    class ZeroRateInstrument:
        risk_free_rate = 0.0
        discount_rate = None

    assert adapter._risk_free_rate_for(ZeroRateInstrument()) == 0.0


def test_heston_discount_is_reaction_not_drift_component() -> None:
    process = create_standard_heston(r=0.06, dividend_yield=0.02)
    states = np.array([[np.log(100.0), 0.04], [np.log(110.0), 0.09]])

    coefficients = process.evaluate_coefficients(0.0, states)
    reaction_only = process.apply_generator(
        0.0,
        states,
        value=np.ones(2),
        gradient=np.zeros((2, 2)),
        hessian=np.zeros((2, 2, 2)),
    )

    assert_allclose(coefficients.discount, [0.06, 0.06])
    assert_allclose(reaction_only, [-0.06, -0.06])
    assert_allclose(coefficients.drift[:, 0], 0.06 - 0.02 - 0.5 * states[:, 1])
