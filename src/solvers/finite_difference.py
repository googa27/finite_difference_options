"""Finite difference solvers and time-stepping schemes."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import findiff as fd
import numpy as np
from findiff import BoundaryConditions, FinDiff, PDE
from numpy.typing import NDArray


class TimeStepper(ABC):
    """Abstract base class for time stepping schemes."""

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
    """Generic :math:`\theta`-method time integrator."""

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


class PDESolver(ABC):
    """Abstract base class for PDE solving engines."""

    @abstractmethod
    def solve(
        self,
        *,
        generator: FinDiff,
        boundary_conditions: BoundaryConditions,
        initial_conditions: NDArray[np.float64],
        time_grid: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Solve the PDE with given conditions."""


@dataclass
class FiniteDifferenceSolver(PDESolver):
    """Finite difference PDE solver using a supplied time-stepper."""

    time_stepper: TimeStepper

    def solve(
        self,
        *,
        generator: FinDiff,
        boundary_conditions: BoundaryConditions,
        initial_conditions: NDArray[np.float64],
        time_grid: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        dt = time_grid[1] - time_grid[0]
        n_time_steps = len(time_grid)
        n_spatial_points = len(initial_conditions)

        values = np.empty((n_time_steps, n_spatial_points))
        values[0] = initial_conditions

        for i in range(n_time_steps - 1):
            values[i + 1] = self.time_stepper.step(
                values[i], generator, boundary_conditions, dt
            )

        return values


def create_default_solver() -> PDESolver:
    """Return a Crank--Nicolson finite difference solver."""

    return FiniteDifferenceSolver(time_stepper=ThetaMethod(theta=0.5))


__all__ = [
    "PDESolver",
    "FiniteDifferenceSolver",
    "TimeStepper",
    "ThetaMethod",
    "ExplicitEuler",
    "CrankNicolson",
    "create_default_solver",
]
