"""Finite difference option pricer using findiff.pde.

This module provides object oriented classes to price European
options under the Black--Scholes model using finite difference
methods.  The implementation follows SOLID principles:

* ``Market`` holds market information such as the risk free rate.
* ``EuropeanOption`` defines the contract parameters and payoff.
* ``FiniteDifferencePricer`` performs the pricing using the
  Crank--Nicolson theta scheme and ``findiff.pde`` for solving the
  spatial equation at each time step.

The code is intentionally simple and heavily commented to serve as a
reference implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import findiff as fd
import findiff.pde as pde


@dataclass
class Market:
    """Market environment with a constant risk free rate."""

    r: float  # Risk free interest rate

    def discount(self, t: float) -> float:
        """Discount factor for maturity ``t``."""
        return np.exp(-self.r * t)


@dataclass
class EuropeanOption:
    """European vanilla option contract."""

    strike: float  # Strike price
    maturity: float  # Time to maturity
    is_call: bool = True  # ``True`` for call, ``False`` for put

    def payoff(self, s: np.ndarray) -> np.ndarray:
        """Option payoff at maturity for asset prices ``s``."""
        if self.is_call:
            return np.maximum(s - self.strike, 0.0)
        return np.maximum(self.strike - s, 0.0)


class FiniteDifferencePricer:
    """Price European options using a finite difference scheme.

    Parameters
    ----------
    market:
        Market environment containing the risk free rate.
    sigma:
        Volatility of the underlying asset.
    s_max:
        Maximum asset price considered on the grid.
    ns:
        Number of spatial grid points.
    nt:
        Number of time steps.
    theta:
        Theta parameter of the theta-scheme (0.5 for Crank-Nicolson).
    """

    def __init__(
        self,
        market: Market,
        sigma: float,
        s_max: float,
        ns: int,
        nt: int,
        theta: float = 0.5,
    ) -> None:
        self.market = market
        self.sigma = sigma
        self.s_max = s_max
        self.ns = ns
        self.nt = nt
        self.theta = theta

    def price(self, option: EuropeanOption, s0: float) -> float:
        """Return the option price for the initial asset price ``s0``."""
        # Spatial grid for the asset price
        s = np.linspace(0.0, self.s_max, self.ns)
        ds = s[1] - s[0]

        # Time grid running from 0 to maturity
        t = np.linspace(0.0, option.maturity, self.nt)
        dt = t[1] - t[0]

        # Differential operator for the Black--Scholes PDE
        L = (
            fd.Coef(0.5 * self.sigma ** 2 * s ** 2) * fd.FinDiff(0, ds, 2)
            + fd.Coef(self.market.r * s) * fd.FinDiff(0, ds, 1)
            - self.market.r * fd.Identity()
        )

        # Theta-scheme matrices
        A = fd.Identity() - dt * self.theta * L
        B = fd.Identity() + dt * (1 - self.theta) * L

        # Initial condition equals payoff at maturity
        v = option.payoff(s)

        # Iterate backwards in time
        for j in range(self.nt - 1):
            # Boundary conditions at the new time level t[j+1]
            tau = option.maturity - t[j + 1]
            bc = fd.BoundaryConditions(s.shape)
            if option.is_call:
                bc[0] = 0.0
                bc[-1] = s[-1] - option.strike * self.market.discount(tau)
            else:
                bc[0] = option.strike * self.market.discount(tau)
                bc[-1] = 0.0

            rhs = B(v)
            v = pde.PDE(A, rhs, bc).solve()

        # Interpolate to obtain price at s0
        return float(np.interp(s0, s, v))


__all__ = ["Market", "EuropeanOption", "FiniteDifferencePricer"]
