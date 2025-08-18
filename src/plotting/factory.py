"""Backend factory for plotters with lazy imports."""
from __future__ import annotations

from typing import Literal

from .base import MatplotlibSeabornPlotter, Plotter


def get_plotter(backend: Literal["matplotlib", "plotly"]) -> Plotter:
    if backend == "plotly":
        from .plotly_backend import PlotlyPlotter

        return PlotlyPlotter()
    return MatplotlibSeabornPlotter()

