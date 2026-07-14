"""Canonical manufactured solution for FD PDE-consistency evidence."""

from __future__ import annotations

import numpy as np


def manufactured_u(grid: np.ndarray, tau: float, alpha: float) -> np.ndarray:
    """Evaluate the canonical smooth manufactured solution."""

    return np.exp(alpha * tau) * (1.0 + 0.2 * grid + 0.05 * grid**3)


def manufactured_source(
    grid: np.ndarray,
    tau: float,
    alpha: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
) -> np.ndarray:
    """Evaluate the exact source f = u_tau - L[u] for the canonical problem."""

    u = manufactured_u(grid, tau, alpha)
    exponential = np.exp(alpha * tau)
    u_s = exponential * (0.2 + 0.15 * grid**2)
    u_ss = exponential * (0.3 * grid)
    operator = 0.5 * volatility * volatility * grid * grid * u_ss + (rate - dividend_yield) * grid * u_s - rate * u
    return alpha * u - operator


__all__ = ["manufactured_source", "manufactured_u"]
