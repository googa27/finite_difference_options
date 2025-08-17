"""Utilities for constructing boundary conditions.

This module provides helpers to build boundary conditions for the
Black--Scholes PDE solver.  The implementation lives in its own module to
respect the single responsibility principle.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from findiff import BoundaryConditions, FinDiff

from .options import EuropeanOption, EuropeanCall, EuropeanPut


@dataclass
class BlackScholesBoundaryBuilder:
    """Factory for boundary conditions in the Black--Scholes model.

    The builder creates first- or second-order boundary conditions depending
    on the option type:

    * Left boundary: gamma equals zero.
    * Right boundary: delta tends to +1 for calls and -1 for puts.
    """

    def build(self, s: np.ndarray, option: EuropeanOption) -> BoundaryConditions:
        """Return boundary conditions for the spatial grid.

        Parameters
        ----------
        s:
            Spatial grid for the underlying asset price.
        option:
            Option contract whose payoff determines the boundary behaviour.
        """
        ds = s[1] - s[0]
        bc = BoundaryConditions(s.shape)
        d1 = FinDiff(0, ds, 1)
        d2 = FinDiff(0, ds, 2)

        # Gamma equals zero on the left boundary for stability
        bc[0] = d2, 0.0

        # Right boundary: delta approaches 1 for calls and -1 for puts.
        if isinstance(option, EuropeanCall):
            bc[-1] = d1, 1.0
        elif isinstance(option, EuropeanPut):
            bc[-1] = d1, -1.0
        else:
            bc[-1] = d2, 0.0

        return bc
