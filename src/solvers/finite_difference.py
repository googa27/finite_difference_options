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
    ``time_grid``.
    """

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
    """Return a default Crank--Nicolson finite difference solver.

    Equivalent to ``FiniteDifferenceSolver(theta=0.5)``.
    """

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
