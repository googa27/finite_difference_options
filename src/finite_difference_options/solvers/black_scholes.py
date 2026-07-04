"""Cached banded one-dimensional Black--Scholes finite-difference solver.

This module provides a small explicit solver for validated public-synthetic
Black--Scholes-style routes.  It intentionally consumes scalar coefficients,
monotone grids and typed option metadata supplied by callers; it does not infer
product semantics beyond the requested vanilla call/put payoff and boundary
formula.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from finite_difference_options.exceptions import ValidationError

Array = NDArray[np.float64]
OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class OperatorCacheInfo:
    """Snapshot of operator/factorization cache activity."""

    hits: int
    misses: int
    entries: int
    solves: int

    @property
    def reuse_count(self) -> int:
        """Number of factorized systems reused from cache."""

        return self.hits

    def as_dict(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "entries": self.entries,
            "solves": self.solves,
            "reuse_count": self.reuse_count,
        }


@dataclass
class BandedOperatorCache:
    """Cache theta-system tridiagonal operators and Thomas factorizations.

    The cache key includes the grid bytes, coefficients, theta, time step and
    dtype so changing any numerical invariant rebuilds the operator and
    factorization.  Cached objects are immutable snapshots from the caller's
    perspective; RHS values and boundary values are never cached.
    """

    max_entries: int = 32
    _systems: dict[tuple[object, ...], "_CachedThetaSystem"] = field(default_factory=dict, init=False, repr=False)
    hits: int = field(default=0, init=False)
    misses: int = field(default=0, init=False)
    solves: int = field(default=0, init=False)

    def get_or_build(
        self,
        *,
        grid: Array,
        risk_free_rate: float,
        dividend_yield: float,
        volatility: float,
        theta: float,
        dt: float,
    ) -> "_CachedThetaSystem":
        key = _theta_system_key(
            grid=grid,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
            theta=theta,
            dt=dt,
        )
        cached = self._systems.get(key)
        if cached is not None:
            self.hits += 1
            return cached

        self.misses += 1
        system = _CachedThetaSystem.build(
            grid=grid,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
            theta=theta,
            dt=dt,
        )
        if len(self._systems) >= self.max_entries:
            oldest = next(iter(self._systems))
            del self._systems[oldest]
        self._systems[key] = system
        return system

    def record_solve(self) -> None:
        self.solves += 1

    def clear(self) -> None:
        self._systems.clear()
        self.hits = 0
        self.misses = 0
        self.solves = 0

    def info(self) -> OperatorCacheInfo:
        return OperatorCacheInfo(
            hits=self.hits,
            misses=self.misses,
            entries=len(self._systems),
            solves=self.solves,
        )


@dataclass
class CachedBlackScholesFiniteDifferenceSolver:
    """Crank--Nicolson/theta Black--Scholes solver with cached banded systems."""

    theta: float = 0.5
    cache: BandedOperatorCache = field(default_factory=BandedOperatorCache)
    last_cache_info: OperatorCacheInfo = field(default_factory=lambda: OperatorCacheInfo(0, 0, 0, 0), init=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.theta <= 1.0 or not np.isfinite(self.theta):
            raise ValidationError("theta must be finite and lie in [0, 1]")

    def solve_european(
        self,
        *,
        spot_grid: Array,
        time_grid: Array,
        strike: float,
        risk_free_rate: float,
        volatility: float,
        option_type: OptionType = "call",
        dividend_yield: float = 0.0,
    ) -> Array:
        """Return values in forward-tau order: payoff at row 0, valuation at row -1."""

        grid = _validate_spot_grid(spot_grid)
        tau_grid = _validate_time_grid(time_grid)
        if not np.isfinite(strike) or strike <= 0.0:
            raise ValidationError("strike must be finite and positive")
        if not np.isfinite(risk_free_rate) or not np.isfinite(dividend_yield):
            raise ValidationError("risk_free_rate and dividend_yield must be finite")
        if dividend_yield < 0.0:
            raise ValidationError("dividend_yield must be non-negative")
        if not np.isfinite(volatility) or volatility <= 0.0:
            raise ValidationError("volatility must be finite and positive")
        if option_type not in {"call", "put"}:
            raise ValidationError("option_type must be 'call' or 'put'")

        values = np.empty((len(tau_grid), len(grid)), dtype=np.float64)
        values[0] = _payoff(grid, strike=strike, option_type=option_type)
        values[0, 0], values[0, -1] = _boundary_values(
            grid,
            strike=strike,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            tau=0.0,
            option_type=option_type,
        )
        time_steps = np.diff(tau_grid)
        uniform_dt = float(time_steps[0])
        use_uniform_dt = bool(np.allclose(time_steps, uniform_dt, rtol=1.0e-12, atol=1.0e-15))

        for row, (tau_prev, tau_next) in enumerate(zip(tau_grid[:-1], tau_grid[1:], strict=True), start=1):
            dt = uniform_dt if use_uniform_dt else float(tau_next - tau_prev)
            system = self.cache.get_or_build(
                grid=grid,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                volatility=volatility,
                theta=self.theta,
                dt=dt,
            )
            rhs = system.apply_rhs(values[row - 1])
            lower, upper = _boundary_values(
                grid,
                strike=strike,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                tau=float(tau_next),
                option_type=option_type,
            )
            rhs[0] = lower
            rhs[-1] = upper
            values[row] = system.solve(rhs)
            self.cache.record_solve()

        self.last_cache_info = self.cache.info()
        return values


@dataclass(frozen=True)
class _CachedThetaSystem:
    """Cached tridiagonal theta-system plus its Thomas factorization."""

    rhs_lower: Array
    rhs_diag: Array
    rhs_upper: Array
    factor_lower: Array
    factor_diag: Array
    factor_c_prime: Array

    @classmethod
    def build(
        cls,
        *,
        grid: Array,
        risk_free_rate: float,
        dividend_yield: float,
        volatility: float,
        theta: float,
        dt: float,
    ) -> "_CachedThetaSystem":
        op_lower, op_diag, op_upper = _black_scholes_tridiagonal_operator(
            grid,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
        )
        lhs_lower = -theta * dt * op_lower
        lhs_diag = 1.0 - theta * dt * op_diag
        lhs_upper = -theta * dt * op_upper
        rhs_lower = (1.0 - theta) * dt * op_lower
        rhs_diag = 1.0 + (1.0 - theta) * dt * op_diag
        rhs_upper = (1.0 - theta) * dt * op_upper

        # Dirichlet rows are applied by boundary algebra every time step.
        lhs_lower = lhs_lower.copy()
        lhs_diag = lhs_diag.copy()
        lhs_upper = lhs_upper.copy()
        rhs_lower = rhs_lower.copy()
        rhs_diag = rhs_diag.copy()
        rhs_upper = rhs_upper.copy()
        lhs_lower[0] = lhs_upper[0] = lhs_lower[-1] = lhs_upper[-1] = 0.0
        lhs_diag[0] = lhs_diag[-1] = 1.0
        rhs_lower[0] = rhs_upper[0] = rhs_lower[-1] = rhs_upper[-1] = 0.0
        rhs_diag[0] = rhs_diag[-1] = 1.0

        factor_diag, factor_c_prime = _factor_tridiagonal(lhs_lower, lhs_diag, lhs_upper)
        return cls(
            rhs_lower=rhs_lower,
            rhs_diag=rhs_diag,
            rhs_upper=rhs_upper,
            factor_lower=lhs_lower,
            factor_diag=factor_diag,
            factor_c_prime=factor_c_prime,
        )

    def apply_rhs(self, values: Array) -> Array:
        rhs = self.rhs_diag * values
        rhs[1:] += self.rhs_lower[1:] * values[:-1]
        rhs[:-1] += self.rhs_upper[:-1] * values[1:]
        return rhs

    def solve(self, rhs: Array) -> Array:
        return _solve_factored_tridiagonal(self.factor_lower, self.factor_diag, self.factor_c_prime, rhs)


def _black_scholes_tridiagonal_operator(
    grid: Array,
    *,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
) -> tuple[Array, Array, Array]:
    n = len(grid)
    lower = np.zeros(n, dtype=np.float64)
    diag = np.zeros(n, dtype=np.float64)
    upper = np.zeros(n, dtype=np.float64)
    drift = risk_free_rate - dividend_yield
    variance = volatility * volatility
    for idx in range(1, n - 1):
        s_i = float(grid[idx])
        h_minus = float(grid[idx] - grid[idx - 1])
        h_plus = float(grid[idx + 1] - grid[idx])
        d1_minus = -h_plus / (h_minus * (h_minus + h_plus))
        d1_center = (h_plus - h_minus) / (h_minus * h_plus)
        d1_plus = h_minus / (h_plus * (h_minus + h_plus))
        d2_minus = 2.0 / (h_minus * (h_minus + h_plus))
        d2_center = -2.0 / (h_minus * h_plus)
        d2_plus = 2.0 / (h_plus * (h_minus + h_plus))
        diffusion_scale = 0.5 * variance * s_i * s_i
        drift_scale = drift * s_i
        lower[idx] = diffusion_scale * d2_minus + drift_scale * d1_minus
        diag[idx] = diffusion_scale * d2_center + drift_scale * d1_center - risk_free_rate
        upper[idx] = diffusion_scale * d2_plus + drift_scale * d1_plus
    return lower, diag, upper


def _factor_tridiagonal(lower: Array, diag: Array, upper: Array) -> tuple[Array, Array]:
    n = len(diag)
    factor_diag = np.empty(n, dtype=np.float64)
    c_prime = np.zeros(n, dtype=np.float64)
    factor_diag[0] = diag[0]
    if abs(factor_diag[0]) <= 1.0e-14:
        raise ValidationError("singular tridiagonal pivot at row 0")
    c_prime[0] = upper[0] / factor_diag[0] if n > 1 else 0.0
    for idx in range(1, n):
        factor_diag[idx] = diag[idx] - lower[idx] * c_prime[idx - 1]
        if abs(factor_diag[idx]) <= 1.0e-14:
            raise ValidationError(f"singular tridiagonal pivot at row {idx}")
        if idx < n - 1:
            c_prime[idx] = upper[idx] / factor_diag[idx]
    return factor_diag, c_prime


def _solve_factored_tridiagonal(lower: Array, factor_diag: Array, c_prime: Array, rhs: Array) -> Array:
    values = np.asarray(rhs, dtype=np.float64)
    if values.ndim == 1:
        return _solve_factored_tridiagonal_vector(lower, factor_diag, c_prime, values)
    if values.ndim != 2:
        raise ValidationError("tridiagonal RHS must be one- or two-dimensional")
    d_prime = np.empty_like(values, dtype=np.float64)
    d_prime[0, :] = values[0, :] / factor_diag[0]
    for idx in range(1, values.shape[0]):
        d_prime[idx, :] = (values[idx, :] - lower[idx] * d_prime[idx - 1, :]) / factor_diag[idx]
    solution = np.empty_like(values, dtype=np.float64)
    solution[-1, :] = d_prime[-1, :]
    for idx in range(values.shape[0] - 2, -1, -1):
        solution[idx, :] = d_prime[idx, :] - c_prime[idx] * solution[idx + 1, :]
    return solution


def _solve_factored_tridiagonal_vector(lower: Array, factor_diag: Array, c_prime: Array, rhs: Array) -> Array:
    n = len(rhs)
    d_prime = np.empty(n, dtype=np.float64)
    d_prime[0] = rhs[0] / factor_diag[0]
    for idx in range(1, n):
        d_prime[idx] = (rhs[idx] - lower[idx] * d_prime[idx - 1]) / factor_diag[idx]
    solution = np.empty(n, dtype=np.float64)
    solution[-1] = d_prime[-1]
    for idx in range(n - 2, -1, -1):
        solution[idx] = d_prime[idx] - c_prime[idx] * solution[idx + 1]
    return solution


def _payoff(grid: Array, *, strike: float, option_type: OptionType) -> Array:
    if option_type == "call":
        return np.maximum(grid - strike, 0.0)
    return np.maximum(strike - grid, 0.0)


def _boundary_values(
    grid: Array,
    *,
    strike: float,
    risk_free_rate: float,
    dividend_yield: float,
    tau: float,
    option_type: OptionType,
) -> tuple[float, float]:
    if option_type == "call":
        return 0.0, float(max(grid[-1] * np.exp(-dividend_yield * tau) - strike * np.exp(-risk_free_rate * tau), 0.0))
    return float(strike * np.exp(-risk_free_rate * tau)), 0.0


def _validate_spot_grid(spot_grid: Array) -> Array:
    grid = np.asarray(spot_grid, dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 5:
        raise ValidationError("spot_grid must be one-dimensional with at least five nodes")
    if not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0):
        raise ValidationError("spot_grid must be finite and strictly increasing")
    if grid[0] < 0.0:
        raise ValidationError("spot_grid lower boundary must be non-negative")
    return grid


def _validate_time_grid(time_grid: Array) -> Array:
    grid = np.asarray(time_grid, dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 2:
        raise ValidationError("time_grid must be one-dimensional with at least two nodes")
    if not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0):
        raise ValidationError("time_grid must be finite and strictly increasing")
    if not np.isclose(grid[0], 0.0, rtol=0.0, atol=1.0e-12):
        raise ValidationError("time_grid must start at zero transformed time")
    return grid


def _theta_system_key(
    *,
    grid: Array,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    theta: float,
    dt: float,
) -> tuple[object, ...]:
    contiguous = np.ascontiguousarray(grid, dtype=np.float64)
    return (
        "black_scholes_theta_system_v1",
        contiguous.shape,
        str(contiguous.dtype),
        sha256(contiguous.tobytes()).hexdigest(),
        float(risk_free_rate),
        float(dividend_yield),
        float(volatility),
        float(theta),
        float(dt),
    )


__all__ = [
    "BandedOperatorCache",
    "CachedBlackScholesFiniteDifferenceSolver",
    "OperatorCacheInfo",
]
