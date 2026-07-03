"""Finite difference solvers and time-stepping schemes.

Core PDE stepping primitives used by :mod:`finite_difference_options.solvers.base`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import findiff as fd
import numpy as np
from findiff import BoundaryConditions, FinDiff, PDE
from numpy.typing import NDArray


class TimeStepper(ABC):
    """Abstract base class for one-step time integration schemes."""

    @abstractmethod
    def step(
        self,
        u: NDArray[np.float64],
        operator: fd.FinDiff,
        bc: BoundaryConditions,
        dt: float,
    ) -> NDArray[np.float64]:
        """Advance the solution ``u`` by one time step of size ``dt``."""


@dataclass
class ThetaMethod(TimeStepper):
    """Generic :math:`\theta`-method time integrator.

    ``theta=0`` gives explicit Euler, ``theta=0.5`` Crank--Nicolson,
    ``theta=1`` implicit Euler.
    """

    theta: float

    def step(
        self,
        u: NDArray[np.float64],
        operator: fd.FinDiff,
        bc: BoundaryConditions,
        dt: float,
    ) -> NDArray[np.float64]:
        a_matrix = fd.Identity() - self.theta * dt * operator
        b_matrix = fd.Identity() + (1 - self.theta) * dt * operator
        rhs = b_matrix(u)
        pde = PDE(lhs=a_matrix, rhs=rhs, bcs=bc)
        return pde.solve()


class ExplicitEuler(ThetaMethod):
    """Explicit Euler scheme (:math:`\theta=0`)."""

    def __init__(self) -> None:  # pragma: no cover - simple initialiser
        super().__init__(theta=0.0)


class CrankNicolson(ThetaMethod):
    """Crank--Nicolson scheme (:math:`\theta=0.5`)."""

    def __init__(self) -> None:  # pragma: no cover - simple initialiser
        super().__init__(theta=0.5)


@dataclass(frozen=True)
class ThetaSubstepRecord:
    """One realised theta-method substep in a solver run.

    ``base_step_index`` refers to the external time-grid interval.
    ``dt_fraction`` is relative to that interval's ``dt``.  The record is kept
    small so it can be copied into route provenance or solver evidence.
    """

    base_step_index: int
    substep_index: int
    theta: float
    dt_fraction: float
    label: str


@dataclass
class RannacherCrankNicolson(TimeStepper):
    """Rannacher startup followed by Crank--Nicolson.

    Financial option payoffs have kinks at strikes.  Starting Crank--Nicolson
    directly from such data underdamps high-frequency modes and contaminates
    near-strike Greeks.  This stepper replaces the first one or two external
    time-grid intervals with two Backward-Euler half-steps each, then switches to
    Crank--Nicolson.
    """

    implicit_euler_half_steps: int = 4
    theta_after_startup: float = 0.5

    def __post_init__(self) -> None:
        allowed = {0, 2, 4}
        if self.implicit_euler_half_steps not in allowed:
            raise ValueError(
                "implicit_euler_half_steps must be one of 0, 2, or 4 so every "
                "startup interval is fully advanced by BE half-steps"
            )
        if not 0.0 <= self.theta_after_startup <= 1.0:
            raise ValueError("theta_after_startup must lie in [0, 1]")
        self.reset()

    def reset(self) -> None:
        """Clear realised schedule state before a new PDE solve."""

        self._base_step_index = 0
        self._schedule: list[ThetaSubstepRecord] = []

    @property
    def schedule(self) -> tuple[ThetaSubstepRecord, ...]:
        """Return the realised schedule from the most recent solve."""

        return tuple(self._schedule)

    def schedule_summary(self) -> str:
        """Return a concise human-readable schedule summary."""

        return (
            f"{self.implicit_euler_half_steps} BE half-steps, "
            f"then theta={self.theta_after_startup}"
        )

    def step(
        self,
        u: NDArray[np.float64],
        operator: fd.FinDiff,
        bc: BoundaryConditions,
        dt: float,
    ) -> NDArray[np.float64]:
        """Advance one external time interval using the configured schedule."""

        value = u
        base_step = self._base_step_index
        startup_base_steps = self.implicit_euler_half_steps // 2

        if base_step < startup_base_steps:
            for local_substep in range(2):
                record = ThetaSubstepRecord(
                    base_step_index=base_step,
                    substep_index=local_substep,
                    theta=1.0,
                    dt_fraction=0.5,
                    label="rannacher_be_half_step",
                )
                self._schedule.append(record)
                value = ThetaMethod(theta=1.0).step(
                    value,
                    operator,
                    bc,
                    dt * record.dt_fraction,
                )
        else:
            record = ThetaSubstepRecord(
                base_step_index=base_step,
                substep_index=0,
                theta=self.theta_after_startup,
                dt_fraction=1.0,
                label=(
                    "crank_nicolson"
                    if self.theta_after_startup == 0.5
                    else "theta_step"
                ),
            )
            self._schedule.append(record)
            value = ThetaMethod(theta=self.theta_after_startup).step(
                value,
                operator,
                bc,
                dt,
            )

        self._base_step_index += 1
        return value


@dataclass(frozen=True)
class LCPLevelDiagnostics:
    """Complementarity diagnostics for one obstacle time step."""

    time_index: int
    tau: float
    iterations: int
    converged: bool
    max_update: float
    primal_violation: float
    dual_violation: float
    complementarity: float
    active_nodes: int
    exercise_boundary: float
    reason: str


@dataclass(frozen=True)
class LCPDiagnostics:
    """Aggregated diagnostics for an American/Bermudan LCP solve."""

    exercise_style: str
    levels: tuple[LCPLevelDiagnostics, ...]
    tolerance: float
    relaxation: float
    max_iterations: int

    @property
    def converged(self) -> bool:
        """Whether every projected solve converged."""

        return all(level.converged for level in self.levels)

    @property
    def max_primal_violation(self) -> float:
        """Maximum obstacle violation ``max(payoff - value, 0)``."""

        return max((level.primal_violation for level in self.levels), default=0.0)

    @property
    def max_dual_violation(self) -> float:
        """Maximum negative dual residual violation."""

        return max((level.dual_violation for level in self.levels), default=0.0)

    @property
    def max_complementarity(self) -> float:
        """Maximum componentwise complementarity product."""

        return max((level.complementarity for level in self.levels), default=0.0)

    @property
    def exercise_boundary(self) -> tuple[float, ...]:
        """Exercise-boundary estimate per diagnosed time level."""

        return tuple(level.exercise_boundary for level in self.levels)


@dataclass
class ProjectedSORLCP:
    """Transparent one-dimensional Black-Scholes obstacle solver.

    The solver advances the transformed forward time ``tau = T - t`` and solves
    at each exercise date the LCP

    ``value >= payoff``, ``A value - rhs >= 0`` and
    ``(value-payoff) * (A value-rhs) = 0``.
    """

    theta: float = 1.0
    tolerance: float = 1.0e-8
    max_iterations: int = 10_000
    relaxation: float = 1.2
    fail_on_nonconvergence: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.theta <= 1.0:
            raise ValueError("theta must lie in (0, 1] for the projected LCP solver")
        if self.tolerance <= 0.0 or not np.isfinite(self.tolerance):
            raise ValueError("tolerance must be finite and positive")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if not 0.0 < self.relaxation < 2.0:
            raise ValueError("relaxation must lie in (0, 2)")
        self.last_diagnostics = LCPDiagnostics(
            exercise_style="american",
            levels=(),
            tolerance=self.tolerance,
            relaxation=self.relaxation,
            max_iterations=self.max_iterations,
        )

    def solve_black_scholes(
        self,
        *,
        spot_grid: NDArray[np.float64],
        payoff: NDArray[np.float64],
        time_grid: NDArray[np.float64],
        strike: float,
        option_type: str,
        risk_free_rate: float,
        dividend_yield: float,
        volatility: float,
        exercise_style: str,
        exercise_dates: tuple[float, ...] = (),
    ) -> NDArray[np.float64]:
        """Return values ordered by increasing transformed time ``tau``."""

        grid = self._validate_grid(spot_grid)
        payoff_array = np.asarray(payoff, dtype=np.float64)
        if payoff_array.shape != grid.shape:
            raise ValueError("payoff must have the same shape as spot_grid")
        tau_grid = self._validate_time_grid(time_grid)
        option_kind = option_type.lower()
        if option_kind not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")
        if exercise_style not in {"american", "bermudan"}:
            raise ValueError("exercise_style must be 'american' or 'bermudan'")
        if not np.isfinite(strike) or strike <= 0.0:
            raise ValueError("strike must be finite and positive")
        if not np.isfinite(risk_free_rate) or not np.isfinite(dividend_yield):
            raise ValueError("risk_free_rate and dividend_yield must be finite")
        if not np.isfinite(volatility) or volatility <= 0.0:
            raise ValueError("volatility must be finite and positive")

        maturity = float(tau_grid[-1])
        exercise_tau = self._exercise_times_in_tau(exercise_style, exercise_dates, maturity)
        operator = self._black_scholes_operator(
            grid,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
        )

        values = np.empty((len(tau_grid), len(grid)), dtype=np.float64)
        values[0] = payoff_array
        levels: list[LCPLevelDiagnostics] = []

        for step_index, (tau_prev, tau_next) in enumerate(
            zip(tau_grid[:-1], tau_grid[1:], strict=True),
            start=1,
        ):
            dt = float(tau_next - tau_prev)
            a_matrix, b_matrix = self._theta_matrices(operator, dt)
            rhs = b_matrix @ values[step_index - 1]
            lower, upper = self._boundary_values(
                grid,
                strike=strike,
                option_type=option_kind,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                tau=float(tau_next),
            )
            self._apply_dirichlet_rows(a_matrix, rhs, lower, upper)
            exercise_allowed = self._exercise_allowed(
                exercise_style,
                float(tau_prev),
                float(tau_next),
                exercise_tau,
            )
            if exercise_allowed:
                solution, level = self._psor(
                    a_matrix,
                    rhs,
                    payoff_array,
                    grid,
                    tau=float(tau_next),
                    time_index=step_index,
                    initial_guess=values[step_index - 1],
                )
            else:
                solution = np.linalg.solve(a_matrix, rhs)
                solution[0] = lower
                solution[-1] = upper
                level = self._diagnose_level(
                    a_matrix,
                    solution,
                    rhs,
                    payoff_array,
                    grid,
                    tau=float(tau_next),
                    time_index=step_index,
                    iterations=0,
                    converged=True,
                    max_update=0.0,
                    reason="linear_continuation_step",
                )
            values[step_index] = solution
            levels.append(level)

        self.last_diagnostics = LCPDiagnostics(
            exercise_style=exercise_style,
            levels=tuple(levels),
            tolerance=self.tolerance,
            relaxation=self.relaxation,
            max_iterations=self.max_iterations,
        )
        if self.fail_on_nonconvergence and not self.last_diagnostics.converged:
            worst = max(levels, key=lambda level: level.max_update, default=None)
            detail = "unknown"
            if worst is not None:
                detail = f"tau={worst.tau:.6g}, iterations={worst.iterations}, update={worst.max_update:.3e}"
            from finite_difference_options.exceptions import ValidationError

            raise ValidationError(f"American LCP solver did not converge ({detail})")
        return values

    @staticmethod
    def _validate_grid(spot_grid: NDArray[np.float64]) -> NDArray[np.float64]:
        grid = np.asarray(spot_grid, dtype=np.float64)
        if grid.ndim != 1 or len(grid) < 5:
            raise ValueError("American LCP requires a one-dimensional grid with at least five nodes")
        if not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0):
            raise ValueError("American LCP grid must be finite and strictly increasing")
        if grid[0] < 0.0:
            raise ValueError("spot grid lower boundary must be non-negative")
        return grid

    @staticmethod
    def _validate_time_grid(time_grid: NDArray[np.float64]) -> NDArray[np.float64]:
        grid = np.asarray(time_grid, dtype=np.float64)
        if grid.ndim != 1 or len(grid) < 2:
            raise ValueError("time_grid must be one-dimensional with at least two points")
        if not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0):
            raise ValueError("time_grid must be finite and strictly increasing")
        if not np.isclose(grid[0], 0.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("time_grid must start at zero transformed time")
        return grid

    @staticmethod
    def _black_scholes_operator(
        grid: NDArray[np.float64],
        *,
        risk_free_rate: float,
        dividend_yield: float,
        volatility: float,
    ) -> NDArray[np.float64]:
        n = len(grid)
        operator = np.zeros((n, n), dtype=np.float64)
        drift = risk_free_rate - dividend_yield
        variance = volatility * volatility
        for i in range(1, n - 1):
            s_i = grid[i]
            h_minus = grid[i] - grid[i - 1]
            h_plus = grid[i + 1] - grid[i]
            d1_minus = -h_plus / (h_minus * (h_minus + h_plus))
            d1_center = (h_plus - h_minus) / (h_minus * h_plus)
            d1_plus = h_minus / (h_plus * (h_minus + h_plus))
            d2_minus = 2.0 / (h_minus * (h_minus + h_plus))
            d2_center = -2.0 / (h_minus * h_plus)
            d2_plus = 2.0 / (h_plus * (h_minus + h_plus))
            diffusion_scale = 0.5 * variance * s_i * s_i
            drift_scale = drift * s_i
            operator[i, i - 1] = diffusion_scale * d2_minus + drift_scale * d1_minus
            operator[i, i] = diffusion_scale * d2_center + drift_scale * d1_center - risk_free_rate
            operator[i, i + 1] = diffusion_scale * d2_plus + drift_scale * d1_plus
        return operator

    def _theta_matrices(
        self,
        operator: NDArray[np.float64],
        dt: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        identity = np.eye(operator.shape[0], dtype=np.float64)
        a_matrix = identity - self.theta * dt * operator
        b_matrix = identity + (1.0 - self.theta) * dt * operator
        return a_matrix, b_matrix

    @staticmethod
    def _boundary_values(
        grid: NDArray[np.float64],
        *,
        strike: float,
        option_type: str,
        risk_free_rate: float,
        dividend_yield: float,
        tau: float,
    ) -> tuple[float, float]:
        if option_type == "put":
            return float(strike), 0.0
        lower = 0.0
        continuation = grid[-1] * np.exp(-dividend_yield * tau) - strike * np.exp(-risk_free_rate * tau)
        intrinsic = grid[-1] - strike
        return lower, float(max(continuation, intrinsic, 0.0))

    @staticmethod
    def _apply_dirichlet_rows(
        a_matrix: NDArray[np.float64],
        rhs: NDArray[np.float64],
        lower: float,
        upper: float,
    ) -> None:
        a_matrix[0, :] = 0.0
        a_matrix[0, 0] = 1.0
        rhs[0] = lower
        a_matrix[-1, :] = 0.0
        a_matrix[-1, -1] = 1.0
        rhs[-1] = upper

    @staticmethod
    def _exercise_times_in_tau(
        exercise_style: str,
        exercise_dates: tuple[float, ...],
        maturity: float,
    ) -> tuple[float, ...]:
        if exercise_style == "american":
            return ()
        if not exercise_dates:
            raise ValueError("Bermudan exercise requires exercise_dates")
        taus = []
        previous = -np.inf
        for date in exercise_dates:
            if not np.isfinite(date) or date <= 0.0 or date > maturity:
                raise ValueError("exercise_dates must lie in (0, maturity]")
            if date <= previous:
                raise ValueError("exercise_dates must be strictly increasing")
            previous = date
            tau = maturity - float(date)
            if tau > 1.0e-12:
                taus.append(tau)
        return tuple(sorted(taus))

    @staticmethod
    def _exercise_allowed(
        exercise_style: str,
        tau_prev: float,
        tau_next: float,
        exercise_tau: tuple[float, ...],
    ) -> bool:
        if exercise_style == "american":
            return True
        return any(tau_prev < tau <= tau_next + 1.0e-12 for tau in exercise_tau)

    def _psor(
        self,
        a_matrix: NDArray[np.float64],
        rhs: NDArray[np.float64],
        payoff: NDArray[np.float64],
        grid: NDArray[np.float64],
        *,
        tau: float,
        time_index: int,
        initial_guess: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], LCPLevelDiagnostics]:
        x = np.maximum(np.asarray(initial_guess, dtype=np.float64).copy(), payoff)
        x[0] = max(rhs[0], payoff[0])
        x[-1] = max(rhs[-1], payoff[-1])
        converged = False
        max_update = np.inf
        reason = "iteration_limit"
        iteration_count = 0
        for iteration in range(1, self.max_iterations + 1):
            iteration_count = iteration
            old = x.copy()
            for i in range(1, len(x) - 1):
                lower_sum = float(a_matrix[i, :i] @ x[:i])
                upper_sum = float(a_matrix[i, i + 1 :] @ old[i + 1 :])
                raw = (rhs[i] - lower_sum - upper_sum) / a_matrix[i, i]
                relaxed = old[i] + self.relaxation * (raw - old[i])
                x[i] = max(payoff[i], relaxed)
            x[0] = max(rhs[0], payoff[0])
            x[-1] = max(rhs[-1], payoff[-1])
            max_update = float(np.max(np.abs(x - old)))
            scale = 1.0 + float(np.max(np.abs(x)))
            if max_update <= self.tolerance * scale:
                converged = True
                reason = "projected_sor_converged"
                break
            if not np.all(np.isfinite(x)):
                reason = "non_finite_iterate"
                break
        level = self._diagnose_level(
            a_matrix,
            x,
            rhs,
            payoff,
            grid,
            tau=tau,
            time_index=time_index,
            iterations=iteration_count,
            converged=converged,
            max_update=max_update,
            reason=reason,
        )
        return x, level

    @staticmethod
    def _diagnose_level(
        a_matrix: NDArray[np.float64],
        value: NDArray[np.float64],
        rhs: NDArray[np.float64],
        payoff: NDArray[np.float64],
        grid: NDArray[np.float64],
        *,
        tau: float,
        time_index: int,
        iterations: int,
        converged: bool,
        max_update: float,
        reason: str,
    ) -> LCPLevelDiagnostics:
        residual = a_matrix @ value - rhs
        interior = slice(1, -1)
        primal = np.maximum(payoff[interior] - value[interior], 0.0)
        dual = np.maximum(-residual[interior], 0.0)
        complementarity = np.abs((value[interior] - payoff[interior]) * residual[interior])
        active = np.where((value <= payoff + 1.0e-7) & (payoff > 1.0e-10))[0]
        boundary = 0.0
        if len(active):
            interior_active = active[(active > 0) & (active < len(grid) - 1)]
            if len(interior_active):
                boundary = float(grid[int(np.max(interior_active))])
        return LCPLevelDiagnostics(
            time_index=time_index,
            tau=tau,
            iterations=iterations,
            converged=converged,
            max_update=float(max_update),
            primal_violation=float(np.max(primal, initial=0.0)),
            dual_violation=float(np.max(dual, initial=0.0)),
            complementarity=float(np.max(complementarity, initial=0.0)),
            active_nodes=int(len(active)),
            exercise_boundary=boundary,
            reason=reason,
        )


class PDESolver(ABC):
    """Abstract base class for finite-difference PDE engines.

    Concrete solvers should map a generator and boundary-condition object into an
    array of temporal solutions.
    """

    @abstractmethod
    def solve(
        self,
        *,
        generator: FinDiff,
        boundary_conditions: BoundaryConditions,
        initial_conditions: NDArray[np.float64],
        time_grid: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Solve PDE coefficients starting from ``initial_conditions``.

        Parameters
        ----------
        generator
            Spatial differential operator from ``findiff``.
        boundary_conditions
            Boundary condition container.
        initial_conditions
            Values at the first time slice.
        time_grid
            Monotone time grid in calendar time.
        """


@dataclass
class FiniteDifferenceSolver(PDESolver):
    """Finite difference PDE solver using a supplied time-stepper.

    The implementation assumes a single spatial axis and fixed time-step size in
    ``time_grid``.  ``last_step_schedule`` records any realised substep schedule
    exposed by the configured stepper.
    """

    time_stepper: TimeStepper

    def __post_init__(self) -> None:
        self.last_step_schedule: tuple[ThetaSubstepRecord, ...] = ()

    def solve(
        self,
        *,
        generator: FinDiff,
        boundary_conditions: BoundaryConditions,
        initial_conditions: NDArray[np.float64],
        time_grid: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if hasattr(self.time_stepper, "reset"):
            self.time_stepper.reset()  # type: ignore[attr-defined]

        dt = time_grid[1] - time_grid[0]
        n_time_steps = len(time_grid)
        n_spatial_points = len(initial_conditions)

        values = np.empty((n_time_steps, n_spatial_points))
        values[0] = initial_conditions

        for i in range(n_time_steps - 1):
            values[i + 1] = self.time_stepper.step(
                values[i], generator, boundary_conditions, dt
            )

        self.last_step_schedule = tuple(getattr(self.time_stepper, "schedule", ()))

        return values


def create_default_solver() -> PDESolver:
    """Return a default Crank--Nicolson finite difference solver.

    Equivalent to ``FiniteDifferenceSolver(theta=0.5)``.
    """

    return FiniteDifferenceSolver(time_stepper=ThetaMethod(theta=0.5))


__all__ = [
    "PDESolver",
    "FiniteDifferenceSolver",
    "TimeStepper",
    "ThetaMethod",
    "ThetaSubstepRecord",
    "ExplicitEuler",
    "CrankNicolson",
    "RannacherCrankNicolson",
    "LCPLevelDiagnostics",
    "LCPDiagnostics",
    "ProjectedSORLCP",
    "create_default_solver",
]
