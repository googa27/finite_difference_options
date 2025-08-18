"""Streamlit-friendly plotting defaults & helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

from .base import PlotOptions
from .colors import DEFAULT_SEQUENTIAL, DEFAULT_DIVERGING

DEFAULT_PLOT_HEIGHT: int = 420


def surface_figsize_from_height(height: int) -> Tuple[float, float]:
    """Map pixel height to a reasonable (w, h) inches for Matplotlib 3D."""
    return (height / 100) * 1.3, (height / 100)


def line_figsize_from_height(height: int) -> Tuple[float, float]:
    """Map pixel height to a reasonable (w, h) inches for line plots."""
    return (height / 100) * 1.3, (height / 100)


@dataclass
class PlotDefaults:
    """Centralized defaults with helpers to build PlotOptions."""

    cmap: str = DEFAULT_SEQUENTIAL
    diverging_cmap: str = DEFAULT_DIVERGING
    colorbar_label: Optional[str] = "Option value"
    height: int = DEFAULT_PLOT_HEIGHT
    elev: Optional[float] = 25
    azim: Optional[float] = -60

    def heatmap(self, *, x_label: str = "Time", y_label: str = "Asset price", **overrides) -> PlotOptions:
        # Allow overrides for fields we set explicitly to avoid duplicates.
        cmap = overrides.pop("cmap", self.cmap)
        height = overrides.pop("height", self.height)
        cbar = overrides.pop("colorbar_label", self.colorbar_label)
        return PlotOptions(
            x_label=x_label,
            y_label=y_label,
            cmap=cmap,
            height=height,
            colorbar_label=cbar,
            **overrides,
        )

    def surface(self, **overrides) -> PlotOptions:
        # Respect an explicitly provided figsize override; otherwise derive from height.
        # Allow overrides for fields we set explicitly to avoid duplicates.
        fs = overrides.pop("figsize", surface_figsize_from_height(self.height))
        cmap = overrides.pop("cmap", self.cmap)
        height = overrides.pop("height", self.height)
        elev = overrides.pop("elev", self.elev)
        azim = overrides.pop("azim", self.azim)
        cbar = overrides.pop("colorbar_label", self.colorbar_label)
        return PlotOptions(
            cmap=cmap,
            elev=elev,
            azim=azim,
            height=height,
            figsize=fs,
            colorbar_label=cbar,
            **overrides,
        )

    def line(self, *, x_label: str = "Asset price", y_label: str = "Option value", **overrides) -> PlotOptions:
        # Respect an explicitly provided figsize override; otherwise derive from height.
        # Allow overrides for fields we set explicitly to avoid duplicates.
        fs = overrides.pop("figsize", line_figsize_from_height(self.height))
        height = overrides.pop("height", self.height)
        cbar = overrides.pop("colorbar_label", self.colorbar_label)
        return PlotOptions(
            x_label=x_label,
            y_label=y_label,
            height=height,
            figsize=fs,
            colorbar_label=cbar,
            **overrides,
        )
