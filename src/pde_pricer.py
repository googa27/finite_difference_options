"""Finite difference pricer using :mod:`findiff`'s PDE solver."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import findiff as fd
from findiff import PDE

from .models import GeometricBrownianMotion, Market
from .options import EuropeanOption
from .boundary_conditions import BlackScholesBoundaryBuilder


@dataclass
class BlackScholesPDE:
    """Price European options by solving the Black–Scholes PDE."""

    model: GeometricBrownianMotion
    market: Market
    theta: float = 0.5  # 0.5 corresponds to Crank–Nicolson
    boundary_builder: BlackScholesBoundaryBuilder = field(
        default_factory=BlackScholesBoundaryBuilder
    )

    def price(self, option: EuropeanOption, s: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Return grid with option values.

        Parameters
        ----------
        option:
            Option contract to price.
        s, t:
            Spatial and temporal grids.
        """
        dt = t[1] - t[0]
        L = self.model.generator(s)
        A = fd.Identity() - self.theta * dt * L
        B = fd.Identity() + (1 - self.theta) * dt * L

        values = np.empty((len(t), len(s)))
        values[0] = option.payoff(s)

        bc = self.boundary_builder.build(s, option)
        for i in range(len(t) - 1):
            rhs = B(values[i])
            pde = PDE(lhs=A, rhs=rhs, bcs=bc)
            values[i + 1] = pde.solve()
        return values

