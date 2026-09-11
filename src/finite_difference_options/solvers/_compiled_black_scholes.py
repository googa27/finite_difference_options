"""Opt-in float64 banded kernel for the compiled European Black-Scholes route.

The retained dense v0 arithmetic is a separate replay contract. This kernel owns
three-diagonal application and local exact-dt LU reuse; full solution history is
still O(space nodes * time nodes), and no monotonicity guarantee is inferred from
central differences alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite
from typing import Any

import numpy as np
from numpy.typing import NDArray

from finite_difference_options.exceptions import ValidationError
from ._tridiagonal import TridiagonalLU

Array = NDArray[np.float64]


@dataclass(frozen=True)
class BlackScholesTridiagonalOperator:
    """Owned, read-only row-aligned diagonals; boundary rows are zero."""

    lower: Array
    diagonal: Array
    upper: Array

    @classmethod
    def from_grid(
        cls, grid: Array, *, risk_free_rate: float, dividend_yield: float, volatility: float
    ) -> BlackScholesTridiagonalOperator:
        grid = _grid(grid, "spot", minimum_nodes=5)
        rate = _finite_scalar(risk_free_rate, "risk_free_rate")
        carry = _finite_scalar(dividend_yield, "dividend_yield")
        sigma = _finite_scalar(volatility, "volatility")
        if sigma < 0.0:
            raise ValidationError("volatility must be nonnegative")
        lower = np.zeros(len(grid), dtype=np.float64)
        diagonal = np.zeros(len(grid), dtype=np.float64)
        upper = np.zeros(len(grid), dtype=np.float64)
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            try:
                h_minus = grid[1:-1] - grid[:-2]
                h_plus = grid[2:] - grid[1:-1]
                total_h = h_minus + h_plus
                diffusion = 0.5 * sigma * sigma * grid[1:-1] ** 2
                drift = (rate - carry) * grid[1:-1]
                lower[1:-1] = diffusion * (2.0 / (h_minus * total_h)) - drift * h_plus / (h_minus * total_h)
                diagonal[1:-1] = (
                    diffusion * (-2.0 / (h_minus * h_plus)) + drift * (h_plus - h_minus) / (h_minus * h_plus) - rate
                )
                upper[1:-1] = diffusion * (2.0 / (h_plus * total_h)) + drift * h_minus / (h_plus * total_h)
            except FloatingPointError as exc:
                raise ValidationError("Black-Scholes operator coefficients must be finite") from exc
        for array in (lower, diagonal, upper):
            if not np.all(np.isfinite(array)):
                raise ValidationError("Black-Scholes operator coefficients must be finite")
            array.setflags(write=False)
        return cls(lower, diagonal, upper)

    def apply(self, values: Array) -> Array:
        """Apply without dense assembly or mutation; preserve vector/batch rank."""
        array = _float_array(values, "operator values")
        if array.ndim not in (1, 2) or array.shape[0] != len(self.diagonal):
            raise ValidationError("operator values must have shape (nodes,) or (nodes, columns)")
        if not np.all(np.isfinite(array)):
            raise ValidationError("operator values must be finite")
        diagonal = self.diagonal[:, None] if array.ndim == 2 else self.diagonal
        lower = self.lower[:, None] if array.ndim == 2 else self.lower
        upper = self.upper[:, None] if array.ndim == 2 else self.upper
        with np.errstate(over="raise", invalid="raise"):
            try:
                result = diagonal * array
                result[1:] += lower[1:] * array[:-1]
                result[:-1] += upper[:-1] * array[1:]
            except FloatingPointError as exc:
                raise ValidationError("operator application must remain finite") from exc
        return result


def solve_compiled_black_scholes_grid_v1(
    *,
    spot_grid: Array,
    time_grid: Array,
    strike: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    theta: float,
) -> tuple[Array, tuple[dict[str, float | int | str], ...], dict[str, Any]]:
    """Solve dV/dtau=L[V] using banded theta stepping and exact-interval caching.

    Spot and time-to-maturity grids begin at zero. Coefficients and both boundary
    types are fixed during this invocation, making exact float dt a sufficient
    local factor-cache key. Boundary values are evaluated afresh at every node.
    """
    spots = _grid(spot_grid, "spot", minimum_nodes=5)
    times = _grid(time_grid, "time", minimum_nodes=2)
    strike = _finite_scalar(strike, "strike")
    rate = _finite_scalar(risk_free_rate, "risk_free_rate")
    carry = _finite_scalar(dividend_yield, "dividend_yield")
    sigma = _finite_scalar(volatility, "volatility")
    theta = _finite_scalar(theta, "theta")
    if strike <= 0.0 or sigma < 0.0 or not 0.0 <= theta <= 1.0:
        raise ValidationError("strike must be positive, volatility nonnegative, and theta in [0, 1]")
    operator = BlackScholesTridiagonalOperator.from_grid(
        spots, risk_free_rate=rate, dividend_yield=carry, volatility=sigma
    )
    values = np.empty((len(times), len(spots)), dtype=np.float64)
    values[0] = np.maximum(spots - strike, 0.0)
    values[0, 0] = 0.0
    values[0, -1] = _upper_boundary(spots[-1], strike, rate, carry, 0.0)
    schedule: list[dict[str, float | int | str]] = [_boundary_record(0, times[0], times[-1], values[0, -1])]
    cache: dict[float, TridiagonalLU] = {}
    for index, (previous_time, next_time) in enumerate(zip(times[:-1], times[1:], strict=True), start=1):
        dt = float(next_time - previous_time)
        if dt not in cache:
            with np.errstate(over="raise", invalid="raise"):
                try:
                    cache[dt] = TridiagonalLU.factor(
                        -theta * dt * operator.lower, 1.0 - theta * dt * operator.diagonal, -theta * dt * operator.upper
                    )
                except FloatingPointError as exc:
                    raise ValidationError("theta-system coefficients must be finite") from exc
        with np.errstate(over="raise", invalid="raise"):
            try:
                rhs = values[index - 1] + (1.0 - theta) * dt * operator.apply(values[index - 1])
            except FloatingPointError as exc:
                raise ValidationError("theta-system RHS must be finite") from exc
        upper = _upper_boundary(spots[-1], strike, rate, carry, float(next_time))
        rhs[0], rhs[-1] = 0.0, upper
        solution = cache[dt].solve(rhs)
        solution[0], solution[-1] = 0.0, upper
        values[index] = solution
        schedule.append(_boundary_record(index, next_time, times[-1], upper))
    return (
        values,
        tuple(schedule),
        {
            "operator_sign_convention": "dV/dtau = L[V]",
            "drift": "(r-q) S dV/dS",
            "diffusion": "0.5 sigma^2 S^2 d2V/dS2",
            "reaction": "-r V",
            "matrix_shape": (len(spots), len(spots)),
            "nonzero_count": sum(int(np.count_nonzero(a)) for a in (operator.lower, operator.diagonal, operator.upper)),
            "linear_solver": "scipy.lapack.dgttrf+dgttrs",
            "operator_application": "row_aligned_three_diagonals",
            "dtype": "float64",
            "factorization_count": len(cache),
            "factorization_cache_hits": len(times) - 1 - len(cache),
            "solve_count": len(times) - 1,
            "exact_dt_hex": [dt.hex() for dt in cache],
            "cache_scope": "one invocation; fixed grid, coefficients, theta, dtype and Dirichlet rows",
            "operator_storage_bytes": sum(a.nbytes for a in (operator.lower, operator.diagonal, operator.upper)),
            "factor_storage_bytes": sum(
                a.nbytes
                for factor in cache.values()
                for a in (factor.lower, factor.diagonal, factor.upper, factor.second_upper, factor.pivots)
            ),
            "solution_history_bytes": values.nbytes,
        },
    )


def _float_array(value: Array, name: str) -> Array:
    if np.iscomplexobj(value):
        raise ValidationError(f"{name} must be real")
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{name} must be a real numeric array") from exc


def _grid(value: Array, name: str, *, minimum_nodes: int) -> Array:
    grid = _float_array(value, name)
    if grid.ndim != 1 or len(grid) < minimum_nodes:
        raise ValidationError(f"{name} grid must be a vector with at least {minimum_nodes} nodes")
    if not np.all(np.isfinite(grid)) or grid[0] != 0.0:
        raise ValidationError(f"{name} grid must be finite and begin at zero")
    if np.any(np.diff(grid) <= 0.0):
        raise ValidationError(f"{name} grid must be strictly increasing")
    return grid.copy()


def _finite_scalar(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value):
        raise ValidationError(f"{name} must be a finite real number")
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{name} must be a finite real number") from exc
    if not isfinite(scalar):
        raise ValidationError(f"{name} must be a finite real number")
    return scalar


def _upper_boundary(spot: float, strike: float, rate: float, carry: float, tau: float) -> float:
    try:
        value = max(float(spot) * exp(-carry * tau) - strike * exp(-rate * tau), 0.0)
    except OverflowError as exc:
        raise ValidationError("boundary values must remain finite") from exc
    if not isfinite(value):
        raise ValidationError("boundary values must remain finite")
    return value


def _boundary_record(index: int, tau: float, maturity: float, upper: float) -> dict[str, float | int | str]:
    return {
        "step_index": index,
        "tau": float(tau),
        "calendar_time": float(maturity - tau),
        "lower": 0.0,
        "upper": float(upper),
        "source": "terminal_payoff_boundary" if index == 0 else "compiled_boundary_expression",
    }
