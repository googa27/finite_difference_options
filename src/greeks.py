"""Interfaces for computing option Greeks from price grids."""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


class GreeksCalculator(ABC):
    """Abstract base class for computing option Greeks."""

    @abstractmethod
    def delta(
        self, grid: NDArray[np.float64], s: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return Delta values across the price grid."""

    @abstractmethod
    def gamma(
        self, grid: NDArray[np.float64], s: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return Gamma values across the price grid."""

    @abstractmethod
    def theta(
        self, grid: NDArray[np.float64], t: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return Theta values across the price grid."""


class FiniteDifferenceGreeks(GreeksCalculator):
    """Compute option Greeks using finite difference approximations.

    The price grid is assumed to have shape ``(len(t), len(s))`` where the first
    axis represents time to maturity and the second axis the underlying asset
    price.  Central differences are used in the interior of the grid and
    one–sided second order approximations are applied at the boundaries.
    """

    def _first_derivative(
        self, grid: NDArray[np.float64], spacing: NDArray[np.float64], axis: int
    ) -> NDArray[np.float64]:
        """Return first derivative along ``axis`` using central differences."""

        return np.gradient(grid, spacing, axis=axis, edge_order=2)

    def _second_derivative(
        self, grid: NDArray[np.float64], spacing: NDArray[np.float64], axis: int
    ) -> NDArray[np.float64]:
        """Return second derivative along ``axis`` using central differences."""

        first = np.gradient(grid, spacing, axis=axis, edge_order=2)
        return np.gradient(first, spacing, axis=axis, edge_order=2)

    def delta(
        self, grid: NDArray[np.float64], s: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return Delta of the option across the grid."""

        return self._first_derivative(grid, s, axis=1)

    def gamma(
        self, grid: NDArray[np.float64], s: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return Gamma of the option across the grid."""

        return self._second_derivative(grid, s, axis=1)

    def theta(
        self, grid: NDArray[np.float64], t: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return Theta of the option across the grid.

        The grid is defined with the time to maturity as the first axis.  Theta
        (the derivative with respect to calendar time) is therefore the negative
        derivative with respect to time to maturity.
        """

        return -self._first_derivative(grid, t, axis=0)
