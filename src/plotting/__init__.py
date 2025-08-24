"""Unified plotting package.

This package contains plotting functionality for the unified pricing framework.
"""
from .base import PlotOptions, Plotter, MatplotlibSeabornPlotter, BasePlotter
from .factory import get_plotter
from .colors import map_matplotlib_to_plotly, DEFAULT_SEQUENTIAL, DEFAULT_DIVERGING, symmetric_bounds

# Try to import PlotlyPlotter if Plotly is available
try:
    from .plotly_backend import PlotlyPlotter
except ImportError:
    PlotlyPlotter = None

__all__ = [
    "PlotOptions",
    "Plotter",
    "BasePlotter",
    "MatplotlibSeabornPlotter",
    "get_plotter",
    "PlotlyPlotter",
    "map_matplotlib_to_plotly",
    "DEFAULT_SEQUENTIAL",
    "DEFAULT_DIVERGING",
    "symmetric_bounds",
]