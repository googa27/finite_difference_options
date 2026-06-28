"""Finite difference solvers and time-stepping schemes.

Core PDE stepping primitives used by :mod:`src.solvers.base`.
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
    "create_default_solver",
]
