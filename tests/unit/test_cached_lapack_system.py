"""Independent linear-system and cache contracts for the public theta route."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from scipy.stats import norm

from finite_difference_options.exceptions import ValidationError
from finite_difference_options.solvers import BandedOperatorCache, CachedBlackScholesFiniteDifferenceSolver


def _parameters():
    return dict(
        grid=np.linspace(0.0, 400.0, 9), risk_free_rate=0.04, dividend_yield=0.01, volatility=0.2, theta=0.5, dt=0.01
    )


def _dense_uniform_theta_matrix(n, rate=0.04, dividend=0.01, volatility=0.2, theta=0.5, dt=0.01):
    """Central Black-Scholes stencil in i=S/h coordinates, independent of production assembly."""
    matrix = np.eye(n)
    for i in range(1, n - 1):
        matrix[i, i - 1] = -theta * dt * (0.5 * volatility**2 * i**2 - 0.5 * (rate - dividend) * i)
        matrix[i, i] = 1 + theta * dt * (volatility**2 * i**2 + rate)
        matrix[i, i + 1] = -theta * dt * (0.5 * volatility**2 * i**2 + 0.5 * (rate - dividend) * i)
    return matrix


@pytest.mark.parametrize("columns", [1, 4])
def test_cached_system_matches_dense_oracle_without_mutating_rhs(columns):
    system = BandedOperatorCache().get_or_build(**_parameters())
    rhs = np.random.default_rng(907).normal(size=(9, columns * 2))[:, ::2]
    if columns == 1:
        rhs = rhs[:, 0]
    before = rhs.copy()
    expected = np.linalg.solve(_dense_uniform_theta_matrix(9), rhs)
    for _ in range(3):
        actual = system.solve(rhs)
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
        np.testing.assert_array_equal(rhs, before)


@pytest.mark.parametrize(
    "rhs", [np.zeros(8), np.zeros((10, 2)), np.zeros((9, 1, 1)), np.full(9, np.nan), np.full((9, 2), np.inf)]
)
def test_invalid_rhs_fails_with_typed_validation_error(rhs):
    system = BandedOperatorCache().get_or_build(**_parameters())
    with pytest.raises(ValidationError):
        system.solve(rhs)


def test_nonfinite_factorization_is_not_cached():
    cache = BandedOperatorCache()
    parameters = _parameters()
    parameters["risk_free_rate"] = np.inf
    with pytest.raises(ValidationError):
        cache.get_or_build(**parameters)
    assert cache.info().entries == 0


@pytest.mark.parametrize(
    "field,value",
    [("risk_free_rate", 0.05), ("dividend_yield", 0.02), ("volatility", 0.3), ("theta", 1.0), ("dt", 0.02)],
)
def test_cache_invalidation_covers_each_operator_invariant(field, value):
    cache = BandedOperatorCache()
    parameters = _parameters()
    first = cache.get_or_build(**parameters)
    assert cache.get_or_build(**parameters) is first
    changed = dict(parameters, **{field: value})
    assert cache.get_or_build(**changed) is not first
    assert cache.info().misses == 2
    assert cache.info().hits == 1


def test_cache_uses_grid_contents_and_does_not_cache_rhs():
    cache = BandedOperatorCache()
    parameters = _parameters()
    first = cache.get_or_build(**parameters)
    assert cache.get_or_build(**dict(parameters, grid=parameters["grid"].copy())) is first
    grid = parameters["grid"].copy()
    grid[4] += 0.1
    assert cache.get_or_build(**dict(parameters, grid=grid)) is not first
    np.testing.assert_allclose(first.solve(2 * np.ones(9)), 2 * first.solve(np.ones(9)))


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_vanilla_solution_preserves_analytic_price_and_cache_reuse(option_type):
    solver = CachedBlackScholesFiniteDifferenceSolver()
    spots = np.linspace(0.0, 400.0, 801)
    times = np.linspace(0.0, 1.0, 401)
    parameters = dict(
        spot_grid=spots,
        time_grid=times,
        strike=100.0,
        risk_free_rate=0.04,
        dividend_yield=0.01,
        volatility=0.2,
        option_type=option_type,
    )
    first = solver.solve_european(**parameters)
    repeated = solver.solve_european(**parameters)
    d1 = (0.04 - 0.01 + 0.5 * 0.2**2) / 0.2
    d2 = d1 - 0.2
    call = 100.0 * np.exp(-0.01) * norm.cdf(d1) - 100.0 * np.exp(-0.04) * norm.cdf(d2)
    expected = call if option_type == "call" else call - 100.0 * np.exp(-0.01) + 100.0 * np.exp(-0.04)
    assert np.interp(100.0, spots, first[-1]) == pytest.approx(expected, abs=0.002)
    np.testing.assert_array_equal(first, repeated)
    assert solver.cache.info().misses == 1
    assert solver.cache.info().solves == 800
    assert solver.cache.info().hits == 799


def test_pivoted_factorization_solves_nonsingular_zero_diagonal():
    from finite_difference_options.solvers._tridiagonal import TridiagonalLU

    lower = np.array([0.0, 1.0, 1.0])
    diagonal = np.array([0.0, 2.0, 2.0])
    upper = np.array([1.0, 1.0, 0.0])
    matrix = np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]])
    factors = TridiagonalLU.factor(lower, diagonal, upper)
    np.testing.assert_allclose(factors.solve(np.eye(3)), np.linalg.inv(matrix), atol=1e-14)
    np.testing.assert_array_equal(diagonal, [0.0, 2.0, 2.0])


def test_singular_and_near_zero_pivots_raise_typed_errors():
    from finite_difference_options.solvers._tridiagonal import TridiagonalLU

    for diagonal in (np.zeros(3), np.full(3, 1e-15)):
        with pytest.raises(ValidationError, match="singular tridiagonal pivot"):
            TridiagonalLU.factor(np.zeros(3), diagonal, np.zeros(3))


def test_factor_storage_is_immutable_and_solution_does_not_alias_it():
    from finite_difference_options.solvers._tridiagonal import TridiagonalLU

    factors = TridiagonalLU.factor(np.zeros(3), np.full(3, 2.0), np.zeros(3))
    for array in (factors.lower, factors.diagonal, factors.upper, factors.second_upper, factors.pivots):
        assert not array.flags.writeable
    values = factors.solve(np.ones(3))
    values[:] = 0.0
    np.testing.assert_array_equal(factors.solve(np.ones(3)), np.full(3, 0.5))


def test_empty_rhs_batch_does_not_corrupt_native_memory():
    # Some SciPy wrappers return the right empty shape but corrupt memory that
    # fails at interpreter shutdown, so a normal in-process assertion is insufficient.
    probe = """
import numpy as np
from finite_difference_options.solvers._tridiagonal import TridiagonalLU
rng = np.random.default_rng(410937)
for _ in range(200):
    lower = rng.uniform(.3, 1.7, 17)
    upper = rng.uniform(-1.6, -.2, 17)
    diagonal = rng.uniform(-.2, .2, 17)
    diagonal[-1] += 1
    factors = TridiagonalLU.factor(lower, diagonal, upper)
    rhs = np.empty((17, 0))
    result = factors.solve(rhs)
    assert result.shape == (17, 0)
    assert result is not rhs
"""
    completed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
