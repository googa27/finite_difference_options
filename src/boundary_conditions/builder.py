"""Utilities for constructing boundary conditions.

This module provides helpers to build boundary conditions for the
Black--Scholes PDE solver.  The implementation is separated from core
solver code to keep BC policy explicit and testable.

These builders currently cover univariate spatial boundaries only and are not a
full taxonomy of generic multidimensional boundary strategies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from findiff import BoundaryConditions, FinDiff

if TYPE_CHECKING:
    from ..options import EuropeanOption


@dataclass
class BlackScholesBoundaryBuilder:
    """Build boundary conditions for standard univariate Black--Scholes grids.

    The convention used is:

    * Left boundary (``s=0``): second derivative (gamma) set to zero.
    * Right boundary: first derivative (delta) equals 1 for calls, 0 for puts.

    Unknown option contract types are defaulted to second-derivative zero at the
    right boundary (Neumann-like fallback).
    """

    def build(
        self,
        s: NDArray[np.float64],
        option: EuropeanOption,
    ) -> BoundaryConditions:
        """Return boundary conditions for the spatial grid.

        Parameters
        ----------
        s:
            Spatial grid for the underlying asset price.
        option:
            Option contract whose payoff determines the boundary behaviour.

        Returns
        -------
        BoundaryConditions
            Findiff boundary-condition container configured for the chosen option.
        """
        ds = s[1] - s[0]
        bc = BoundaryConditions(s.shape)
        d1 = FinDiff(0, ds, 1)
        d2 = FinDiff(0, ds, 2)

        # Gamma equals zero on the left boundary for stability
        bc[0] = d2, 0.0

        # Right boundary: delta approaches 1 for calls and 0 for puts.
        # Use isinstance check with string literal to avoid direct import
        if option.__class__.__name__ == "EuropeanCall":
            bc[-1] = d1, 1.0
        elif option.__class__.__name__ == "EuropeanPut":
            bc[-1] = d1, 0.0
        else:
            bc[-1] = d2, 0.0

        return bc
