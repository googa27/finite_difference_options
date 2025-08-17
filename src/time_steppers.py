"""Time stepping schemes for finite difference PDE solvers."""
from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
import findiff as fd
from findiff import PDE


class TimeStepper(ABC):
    """Abstract base class for time stepping schemes."""

    @abstractmethod
    def step(
        self,
        u: NDArray[np.float64],
        operator: fd.FinDiff,
        bc,
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
        bc,
        dt: float,
    ) -> NDArray[np.float64]:
        A = fd.Identity() - self.theta * dt * operator
        B = fd.Identity() + (1 - self.theta) * dt * operator
        rhs = B(u)
        pde = PDE(lhs=A, rhs=rhs, bcs=bc)
        return pde.solve()


class ExplicitEuler(ThetaMethod):
    """Explicit Euler scheme (:math:`\theta=0`)."""

    def __init__(self) -> None:  # pragma: no cover - trivial
        super().__init__(theta=0.0)


class CrankNicolson(ThetaMethod):
    """Crank--Nicolson scheme (:math:`\theta=0.5`)."""

    def __init__(self) -> None:  # pragma: no cover - trivial
        super().__init__(theta=0.5)
