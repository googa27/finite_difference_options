"""Finite difference pricer using :mod:`findiff`'s PDE solver."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .models import GeometricBrownianMotion, Market
from .options import EuropeanOption
from .boundary_conditions import BlackScholesBoundaryBuilder
from .spatial_operator import SpatialOperator
from .time_steppers import TimeStepper, ThetaMethod


@dataclass
class BlackScholesPDE:
    """Price European options by solving the Black–Scholes PDE."""

    model: GeometricBrownianMotion
    market: Market
    theta: float = 0.5  # retained for backward compatibility
    boundary_builder: BlackScholesBoundaryBuilder = field(
        default_factory=BlackScholesBoundaryBuilder
    )
    time_stepper: TimeStepper | None = None

    def __post_init__(self) -> None:
        """Initialise default time stepper if not supplied."""
        if self.time_stepper is None:
            self.time_stepper = ThetaMethod(self.theta)
        else:  # keep ``theta`` in sync for backward compatibility
            self.theta = getattr(self.time_stepper, "theta", self.theta)

    def price(
        self,
        option: EuropeanOption,
        s: NDArray[np.float64],
        t: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return grid with option values.

        Parameters
        ----------
        option:
            Option contract to price.
        s, t:
            Spatial and temporal grids.
        """
        dt = t[1] - t[0]
        L = SpatialOperator(self.model).build(s)

        values = np.empty((len(t), len(s)))
        values[0] = option.payoff(s)

        bc = self.boundary_builder.build(s, option)
        for i in range(len(t) - 1):
            values[i + 1] = self.time_stepper.step(values[i], L, bc, dt)
        return values

