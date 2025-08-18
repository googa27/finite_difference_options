"""Plotting base interfaces and Matplotlib backend."""
from __future__ import annotations

from dataclasses import dataclass
from abc import ABC
from typing import Any, Dict, Iterable, Optional, Protocol, Set

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from numpy.typing import NDArray


@dataclass
class PlotOptions:
    """Unified plotting options used by both backends.

    Grids are assumed shaped (t, s). Heatmaps default to x=Time, y=Asset price.
    """

    x_label: str = "Time"
    y_label: str = "Asset price"
    cmap: str = "viridis"
    elev: float | None = None
    azim: float | None = None
    figsize: tuple[float, float] | None = (6.0, 4.0)
    height: int | None = None
    vmin: float | None = None
    vmax: float | None = None
    secondary_keys: Optional[Iterable[str]] = None
    colorbar_label: Optional[str] = "Option value"


class Plotter(Protocol):
    """Structural protocol for rendering option pricing grids."""

    def heatmap(
        self,
        grid: NDArray[np.float64],
        s: NDArray[np.float64],
        t: NDArray[np.float64],
        *,
        opts: PlotOptions,
    ) -> Any: ...

    def surface(
        self,
        grid: NDArray[np.float64],
        s: NDArray[np.float64],
        t: NDArray[np.float64],
        *,
        opts: PlotOptions,
    ) -> Any: ...

    def line1d(
        self,
        x: NDArray[np.float64],
        series: Dict[str, NDArray[np.float64]],
        *,
        opts: PlotOptions,
    ) -> Any: ...


class BasePlotter(ABC):
    """Optional abstract base with reusable helpers for plotters."""

    @staticmethod
    def _subset_indices(n: int, k: int) -> np.ndarray:
        if n <= 0:
            return np.array([], dtype=int)
        k = max(1, min(k, n))
        return np.linspace(0, n - 1, k).astype(int)

    @staticmethod
    def _apply_colorbar_label(hm, label: Optional[str]) -> None:
        if not label:
            return
        try:
            cbar = hm.collections[0].colorbar  # type: ignore[attr-defined]
            if cbar is not None:
                cbar.set_label(label)
        except Exception:
            # Be resilient if seaborn changes internals
            pass

    def _format_heatmap_ticks(
        self,
        ax,
        *,
        s: NDArray[np.float64],
        t: NDArray[np.float64],
        grid_T_shape: tuple[int, int],
        n_xticks: int = 6,
        n_yticks: int = 6,
    ) -> None:
        n_rows, n_cols = grid_T_shape
        if n_cols > 0 and n_xticks > 0:
            x_idx = self._subset_indices(n_cols, min(n_xticks, n_cols))
            ax.set_xticks(x_idx + 0.5)
            ax.set_xticklabels([f"{t[i]:.2f}" for i in x_idx], rotation=0)
        if n_rows > 0 and n_yticks > 0:
            y_idx = self._subset_indices(n_rows, min(n_yticks, n_rows))
            ax.set_yticks(y_idx + 0.5)
            ax.set_yticklabels([f"{s[i]:.2f}" for i in y_idx], rotation=0)


class MatplotlibSeabornPlotter(BasePlotter):
    """Plotter implementation using Matplotlib and Seaborn."""

    def heatmap(
        self,
        grid: NDArray[np.float64],
        s: NDArray[np.float64],
        t: NDArray[np.float64],
        *,
        opts: PlotOptions,
    ) -> Figure:
        """Return a heatmap figure of option values.

        Grids are shaped (t, s); this plots x=t and y=s.
        """

        fig, ax = plt.subplots()

        # Draw heatmap with x=time (columns) and y=asset (rows)
        hm = sns.heatmap(
            grid.T,
            xticklabels=False,
            yticklabels=False,
            ax=ax,
            cmap=opts.cmap,
            vmin=opts.vmin,
            vmax=opts.vmax,
        )
        # Colorbar and ticks
        self._apply_colorbar_label(hm, opts.colorbar_label)
        self._format_heatmap_ticks(ax, s=s, t=t, grid_T_shape=grid.T.shape)

        ax.set_aspect("auto")
        ax.set_xlabel(opts.x_label)
        ax.set_ylabel(opts.y_label)
        fig.tight_layout()
        return fig

    def surface(
        self,
        grid: NDArray[np.float64],
        s: NDArray[np.float64],
        t: NDArray[np.float64],
        *,
        opts: PlotOptions,
    ) -> Figure:
        """Return a 3-D surface plot of option values."""

        s_mesh, t_mesh = np.meshgrid(s, t)
        fig = plt.figure(figsize=opts.figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(
            s_mesh,
            t_mesh,
            grid,
            cmap=opts.cmap,
            linewidth=0.0,
            antialiased=True,
            edgecolor="none",
        )
        if opts.elev is not None or opts.azim is not None:
            ax.view_init(elev=opts.elev if opts.elev is not None else 30, azim=opts.azim if opts.azim is not None else -60)
        ax.set_xlabel("Asset price")
        ax.set_ylabel("Time")
        ax.set_zlabel("Option value")
        return fig

    def line1d(
        self,
        x: NDArray[np.float64],
        series: Dict[str, NDArray[np.float64]],
        *,
        opts: PlotOptions,
    ) -> Figure:
        fig, ax = plt.subplots(figsize=opts.figsize or (6.0, 3.5))
        sec_set: Set[str] = set(opts.secondary_keys or [])
        sec_ax = None
        for name, y in series.items():
            if name in sec_set:
                if sec_ax is None:
                    sec_ax = ax.twinx()
                sec_ax.plot(x, y, label=name, linewidth=2, linestyle="--")
            else:
                ax.plot(x, y, label=name, linewidth=2)
        # Zero reference line
        ax.axhline(0.0, color="gray", linewidth=1.0, alpha=0.6)
        ax.set_xlabel(opts.x_label)
        ax.set_ylabel("Value")
        if sec_ax is not None:
            sec_ax.set_ylabel("Greeks")
            # Combine legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = sec_ax.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
        else:
            ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
