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
