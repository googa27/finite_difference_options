"""Finite difference implementations of Greeks calculators."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.exceptions import ValidationError
from .base import GreeksCalculator


class FiniteDifferenceGreeks(GreeksCalculator):
    """Compute option Greeks using finite difference approximations.

    The price grid is assumed to have shape ``(len(t), len(s))`` where the first
    axis represents time to maturity and the second axis the underlying asset
    price.  Central differences are used in the interior of the grid and
    one–sided second order approximations are applied at the boundaries.
    """

    def _validate_price_grid(self, grid: NDArray[np.float64]) -> None:
        """Validate that a price grid is provided."""
        if grid.ndim < 1:
            raise ValidationError("Prices array must have at least 1 dimension")

    def _validate_asset_grid(self, grid: NDArray[np.float64]) -> None:
        """Validate the asset price grid used for spatial derivatives."""
        if grid.ndim != 1:
            raise ValidationError("Asset price grid must be one-dimensional")
        if grid.shape[0] < 2:
            raise ValidationError("Asset price grid must have at least 2 points")

    def _validate_asset_grid_for_gamma(self, grid: NDArray[np.float64]) -> None:
        """Validate grid for second derivatives with respect to the asset price."""
        self._validate_asset_grid(grid)
        if grid.shape[0] < 3:
            raise ValidationError(
                "Asset price grid must have at least 3 points for second derivative"
            )

    def _validate_time_grid(self, grid: NDArray[np.float64]) -> None:
        """Validate the time grid used for theta calculations."""
        if grid.ndim != 1:
            raise ValidationError("Time grid must be one-dimensional")
        if grid.shape[0] < 2:
            raise ValidationError("Time grid must have at least 2 points")

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

        self._validate_price_grid(grid)
        self._validate_asset_grid(s)
        axis = 1 if grid.ndim > 1 else 0
        return self._first_derivative(grid, s, axis=axis)

    def gamma(
        self, grid: NDArray[np.float64], s: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return Gamma of the option across the grid."""
        self._validate_price_grid(grid)
        self._validate_asset_grid_for_gamma(s)
        axis = 1 if grid.ndim > 1 else 0
        return self._second_derivative(grid, s, axis=axis)

    def theta(
        self, grid: NDArray[np.float64], t: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return Theta (time decay) across the grid."""
        self._validate_price_grid(grid)
        self._validate_time_grid(t)
        if grid.shape[0] != t.shape[0]:
            raise ValidationError(
                "Time grid must match the first dimension of the prices array"
            )
        return -self._first_derivative(grid, t, axis=0)
