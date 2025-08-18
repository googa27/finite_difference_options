import numpy as np

from src.plotting.factory import get_plotter
from src.plotting.base import PlotOptions


def _dummy_grid():
    s = np.linspace(0, 2, 10)
    t = np.linspace(0, 1, 8)
    # simple separable function
    grid = np.outer(t, np.sin(s))
    return s, t, grid


def test_matplotlib_heatmap_and_line1d_smoke():
    s, t, grid = _dummy_grid()
    p = get_plotter("matplotlib")
    fig = p.heatmap(grid=grid, s=s, t=t, opts=PlotOptions())
    assert fig is not None
    series = {"Price": grid[len(t) // 2]}
    fig2 = p.line1d(x=s, series=series, opts=PlotOptions(x_label="Asset price", y_label="Value"))
    assert fig2 is not None


def test_plotly_heatmap_and_line1d_smoke():
    s, t, grid = _dummy_grid()
    p = get_plotter("plotly")
    fig = p.heatmap(grid=grid, s=s, t=t, opts=PlotOptions())
    # Plotly Figure duck-typing: has to_plotly_json
    assert hasattr(fig, "to_plotly_json")
    series = {"Price": grid[len(t) // 2]}
    fig2 = p.line1d(x=s, series=series, opts=PlotOptions(x_label="Asset price", y_label="Value"))
    assert hasattr(fig2, "to_plotly_json")
