"""Plotting interfaces for option pricing results."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class Plotter(ABC):
    """Abstract interface for rendering option pricing grids."""

    @abstractmethod
    def heatmap(
        self,
        grid: NDArray[np.float64],
        s: NDArray[np.float64],
        t: NDArray[np.float64],
    ) -> Any:
        """Return a 2-D heatmap figure."""

    @abstractmethod
    def surface(
        self,
        grid: NDArray[np.float64],
        s: NDArray[np.float64],
        t: NDArray[np.float64],
    ) -> Any:
        """Return a 3-D surface figure."""
