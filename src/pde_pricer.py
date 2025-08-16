"""Finite difference pricer using :mod:`findiff`'s PDE solver."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import findiff as fd
from findiff import PDE, BoundaryConditions

from .models import GeometricBrownianMotion, Market
from .options import EuropeanOption, EuropeanCall, EuropeanPut


@dataclass
class BlackScholesPDE:
    """Price European options by solving the Black–Scholes PDE."""

    model: GeometricBrownianMotion
    market: Market
    theta: float = 0.5  # 0.5 corresponds to Crank–Nicolson

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

        bc = self._boundary_conditions(s, option)
        for i in range(len(t) - 1):
            rhs = B(values[i])
            pde = PDE(lhs=A, rhs=rhs, bcs=bc)
            values[i + 1] = pde.solve()
        return values

    def _boundary_conditions(self, s: np.ndarray, option: EuropeanOption) -> BoundaryConditions:
        """Return boundary conditions for the spatial grid."""
        ds = s[1] - s[0]
        bc = BoundaryConditions(s.shape)
        d1 = fd.FinDiff(0, ds, 1)
        d2 = fd.FinDiff(0, ds, 2)

        # Left boundary: gamma equals zero
        bc[0] = d2, 0.0

        # Right boundary depends on option type
        if isinstance(option, EuropeanCall):
            bc[-1] = d1, 1.0  # delta -> 1
        elif isinstance(option, EuropeanPut):
            bc[-1] = d1, -1.0  # delta -> -1
        else:
            bc[-1] = d2, 0.0
        return bc
