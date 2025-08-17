"""Plotting interfaces for option pricing results."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
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


class MatplotlibSeabornPlotter(Plotter):
    """Plotter implementation using Matplotlib and Seaborn."""

    def heatmap(
        self,
        grid: NDArray[np.float64],
        s: NDArray[np.float64],
        t: NDArray[np.float64],
    ) -> Figure:
        """Return a heatmap figure of option values.

        Parameters
        ----------
        grid:
            2-D array of option values where rows correspond to time and
            columns to asset prices.
        s:
            Discretized asset prices.
        t:
            Discretized times to maturity.
        """

        fig, ax = plt.subplots()
        sns.heatmap(
            grid,
            xticklabels=np.round(s, 2),
            yticklabels=np.round(t, 2),
            ax=ax,
        )
        ax.set_xlabel("Asset price")
        ax.set_ylabel("Time")
        return fig

    def surface(
        self,
        grid: NDArray[np.float64],
        s: NDArray[np.float64],
        t: NDArray[np.float64],
    ) -> Figure:
        """Return a 3-D surface plot of option values."""

        s_mesh, t_mesh = np.meshgrid(s, t)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(s_mesh, t_mesh, grid, cmap="viridis")
        ax.set_xlabel("Asset price")
        ax.set_ylabel("Time")
        ax.set_zlabel("Option value")
        return fig
