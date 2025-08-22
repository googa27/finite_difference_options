"""Legacy PDE models for backward compatibility.

This module maintains the original PDEModel interface for backward compatibility
while internally using the new modular architecture with PDESolver and PricingEngine.
New code should use the PricingEngine directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from findiff import BoundaryConditions, FinDiff

from .models import GeometricBrownianMotion, Market
from .options import EuropeanOption
from .spatial_operator import SpatialOperator
from .time_steppers import TimeStepper, ThetaMethod
from .instruments import Instrument
from .pde_solver import FiniteDifferenceSolver


class PDEModel(ABC):
    """Abstract base class for PDE pricing models.
    
    This class is maintained for backward compatibility. New implementations
    should use the PricingEngine and PDESolver architecture directly.
    """

    time_stepper: TimeStepper

    @abstractmethod
    def generator(self, s: NDArray[np.float64]) -> FinDiff:
        """Return the discretised generator on the spatial grid."""
        ...

    @abstractmethod
    def payoff(
        self, s: NDArray[np.float64], option: EuropeanOption | None
    ) -> NDArray[np.float64]:
        """Return payoff at maturity for the spatial grid."""
        ...

    @abstractmethod
    def boundary_conditions(
        self, s: NDArray[np.float64], option: EuropeanOption | None
    ) -> BoundaryConditions:
        """Return boundary conditions for the spatial grid."""
        ...

    def price(
        self,
        option: EuropeanOption | None,
        s: NDArray[np.float64],
        t: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return grid with instrument values.
        
        Uses the new PDESolver internally for consistency.
        """
        solver = FiniteDifferenceSolver(time_stepper=self.time_stepper)
        
        generator = self.generator(s)
        boundary_conditions = self.boundary_conditions(s, option)
        initial_conditions = self.payoff(s, option)
        
        return solver.solve(
            generator=generator,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            time_grid=t,
        )



@dataclass
class BlackScholesPDE(PDEModel):
    """Price European options by solving the Black–Scholes PDE."""

    instrument: Instrument
    theta: float = 0.5  # retained for backward compatibility
    time_stepper: TimeStepper | None = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Initialise default time stepper if not supplied."""
        if self.time_stepper is None:
            self.time_stepper = ThetaMethod(self.theta)
        else:  # keep ``theta`` in sync for backward compatibility
            self.theta = getattr(self.time_stepper, "theta", self.theta)

    def generator(self, s: NDArray[np.float64]) -> FinDiff:
        """Return the Black--Scholes infinitesimal generator."""

        return self.instrument.generator(s)

    def payoff(
        self, s: NDArray[np.float64], option: EuropeanOption | None = None
    ) -> NDArray[np.float64]:
        """Return option payoff at maturity."""

        return self.instrument.payoff(s)

    def boundary_conditions(
        self, s: NDArray[np.float64], option: EuropeanOption | None = None
    ) -> BoundaryConditions:
        """Return model-specific boundary conditions."""

        return self.instrument.boundary_conditions(s)


@dataclass
class CallableBondPDEModel(PDEModel):
    """PDE model for pricing simple callable bonds.

    The implementation is intentionally simplified: the underlying short rate
    follows a geometric Brownian motion and the bond may be called at a fixed
    price at any time.
    """

    face_value: float
    call_price: float
    market: Market
    model: GeometricBrownianMotion
    _maturity: float
    time_stepper: TimeStepper = field(default_factory=lambda: ThetaMethod(0.5))

    @property
    def maturity(self) -> float:
        return self._maturity

    @property
    def strike(self) -> Optional[float]:
        return None

    def generator(self, s: NDArray[np.float64]) -> FinDiff:
        """Return the infinitesimal generator for the short rate."""

        return SpatialOperator(self.model).build(s)

    def payoff(
        self, s: NDArray[np.float64], option: EuropeanOption | None = None
    ) -> NDArray[np.float64]:
        """Face value paid at maturity."""

        return np.full_like(s, min(self.face_value, self.call_price))

    def boundary_conditions(
        self, s: NDArray[np.float64], option: EuropeanOption | None = None
    ) -> BoundaryConditions:
        """Dirichlet boundaries enforcing the call price."""

        bc = BoundaryConditions(s.shape)
        # Value is zero when the bond price approaches zero
        bc[0] = 0.0
        # Bond cannot exceed the call price
        bc[-1] = self.call_price
        return bc

    def price(
        self,
        s: NDArray[np.float64],
        t: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return grid of callable bond prices.

        After each time step the value is capped at the call price to emulate
        the early redemption feature.
        """
        solver = FiniteDifferenceSolver(time_stepper=self.time_stepper)
        
        generator = self.generator(s)
        boundary_conditions = self.boundary_conditions(s)
        initial_conditions = self.payoff(s)
        
        values = solver.solve(
            generator=generator,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            time_grid=t,
        )
        
        # Apply call constraint after each time step
        values = np.minimum(values, self.call_price)
        return values
