"""Banded v1 parity against retained dense arithmetic and polynomial identities."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from finite_difference_options.exceptions import ValidationError
from finite_difference_options.integrations.compiled_pde_black_scholes_route import (
    _black_scholes_matrix,
    _solve_compiled_black_scholes_grid,
)
from finite_difference_options.solvers._compiled_black_scholes import (
    BlackScholesTridiagonalOperator,
    solve_compiled_black_scholes_grid_v1,
)


@pytest.mark.parametrize("nonuniform", [False, True])
def test_operator_apply_polynomial_identity_and_dense_multi_rhs(nonuniform: bool) -> None:
    grid = 3.0 * np.linspace(0.0, 1.0, 31) ** (1.3 if nonuniform else 1.0)
    original = grid.copy()
    op = BlackScholesTridiagonalOperator.from_grid(grid, risk_free_rate=-0.03, dividend_yield=0.07, volatility=0.2)
    values = np.column_stack([np.ones_like(grid), grid, grid**2])
    actual = op.apply(values)
    assert_allclose(actual[1:-1, 0], 0.03, atol=3e-13)
    assert_allclose(actual[1:-1, 1], -0.07 * grid[1:-1], atol=3e-13)
    assert_allclose(actual[1:-1, 2], (0.2**2 - 0.03 - 2 * 0.07) * grid[1:-1] ** 2, atol=3e-12)
    dense = _black_scholes_matrix(grid, risk_free_rate=-0.03, dividend_yield=0.07, volatility=0.2)
    assert_allclose(actual, dense @ values, atol=3e-12, rtol=3e-12)
    assert_allclose(op.apply(grid), dense @ grid, atol=3e-12, rtol=3e-12)
    assert op.apply(np.empty((len(grid), 0))).shape == (len(grid), 0)
    assert_array_equal(grid, original)
    for diagonal in (op.lower, op.diagonal, op.upper):
        assert not diagonal.flags.writeable
    grid[:] = 0.0
    assert_allclose(op.apply(values), actual)


@pytest.mark.parametrize("nonuniform", [False, True])
@pytest.mark.parametrize("theta", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("rate,carry", [(0.05, 0.0), (-0.03, 0.07), (0.01, -0.1)])
def test_all_time_slices_boundaries_and_exact_dt_cache_match_dense(
    nonuniform: bool, theta: float, rate: float, carry: float
) -> None:
    grid = 3.0 * np.linspace(0.0, 1.0, 21) ** (1.2 if nonuniform else 1.0)
    times = 0.2 * np.linspace(0.0, 1.0, 51) ** (1.1 if nonuniform else 1.0)
    inputs = dict(
        spot_grid=grid,
        time_grid=times,
        strike=1.0,
        risk_free_rate=rate,
        dividend_yield=carry,
        volatility=0.2,
        theta=theta,
    )
    expected, expected_schedule, _ = _solve_compiled_black_scholes_grid(**inputs)
    actual, schedule, metadata = solve_compiled_black_scholes_grid_v1(**inputs)
    assert_allclose(actual, expected, atol=3e-12, rtol=3e-12)
    assert schedule == expected_schedule
    assert_array_equal(actual[:, [0, -1]], expected[:, [0, -1]])
    dts = [float(b - a) for a, b in zip(times[:-1], times[1:], strict=True)]
    assert metadata["factorization_count"] == len(set(dts))
    assert metadata["factorization_cache_hits"] == len(dts) - len(set(dts))
    assert metadata["exact_dt_hex"] == [dt.hex() for dt in dict.fromkeys(dts)]
    assert metadata["operator_storage_bytes"] == 3 * len(grid) * 8
    assert metadata["solution_history_bytes"] == actual.nbytes
    assert metadata["linear_solver"] == "scipy.lapack.dgttrf+dgttrs"


@pytest.mark.parametrize(
    "field,value",
    [
        ("theta", -0.1),
        ("theta", 1.1),
        ("theta", float("nan")),
        ("volatility", -0.1),
        ("risk_free_rate", float("inf")),
        ("strike", 0.0),
    ],
)
def test_invalid_coefficients_refused(field: str, value: float) -> None:
    inputs = dict(
        spot_grid=np.linspace(0, 3, 11),
        time_grid=np.linspace(0, 1, 5),
        strike=1.0,
        risk_free_rate=0.05,
        dividend_yield=0.0,
        volatility=0.2,
        theta=0.5,
    )
    inputs[field] = value
    with pytest.raises(ValidationError):
        solve_compiled_black_scholes_grid_v1(**inputs)


@pytest.mark.parametrize(
    "spot,time",
    [
        (np.linspace(0.1, 3, 11), np.linspace(0, 1, 5)),
        (np.linspace(0, 3, 11), np.linspace(0.1, 1, 5)),
        (np.array([0, 1, 1, 2, 3]), np.linspace(0, 1, 5)),
    ],
)
def test_boundary_coordinate_and_monotone_grid_contract(spot: np.ndarray, time: np.ndarray) -> None:
    with pytest.raises(ValidationError):
        solve_compiled_black_scholes_grid_v1(
            spot_grid=spot,
            time_grid=time,
            strike=1.0,
            risk_free_rate=0.05,
            dividend_yield=0.0,
            volatility=0.2,
            theta=0.5,
        )
