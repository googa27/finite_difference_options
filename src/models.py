"""Core financial models used by the PDE pricer.

This module defines simple object oriented representations of the
market environment and the underlying stochastic process.  The
implementation follows the single responsibility principle: each
class encapsulates one concept only.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import findiff as fd


@dataclass
class Market:
    """Risk neutral market with constant interest rate."""

    rate: float

    def discount(self, t: float) -> float:
        """Return discount factor for maturity ``t``."""
        return np.exp(-self.rate * t)


@dataclass
class GeometricBrownianMotion:
    """Geometric Brownian Motion under the risk neutral measure."""

    rate: float
    sigma: float

    @property
    def diffusion(self) -> float:
        """Convenient shortcut for ``sigma^2 / 2``."""
        return 0.5 * self.sigma ** 2

    def generator(self, s: NDArray[np.float64]) -> fd.FinDiff:
        """Return the discretised infinitesimal generator.

        Parameters
        ----------
        s:
            Spatial grid for the asset price.
        """
        ds = s[1] - s[0]
        d1 = fd.FinDiff(0, ds, 1)
        d2 = fd.FinDiff(0, ds, 2)

        # Black–Scholes generator L = 0.5*sigma^2*s^2*d2 + r*s*d1 - r*I
        return (
            fd.Coef(self.diffusion * s ** 2) * d2
            + fd.Coef(self.rate * s) * d1
            - self.rate * fd.Identity()
        )
