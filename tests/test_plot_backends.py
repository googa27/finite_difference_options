from src.plotting.factory import get_plotter
from src.plotting.base import MatplotlibSeabornPlotter


def test_get_plotter_matplotlib():
    p = get_plotter("matplotlib")
    assert isinstance(p, MatplotlibSeabornPlotter)


def test_get_plotter_plotly():
    p = get_plotter("plotly")
    # Import here to avoid import-time cost when not needed
    from src.plotting.plotly_backend import PlotlyPlotter

    assert isinstance(p, PlotlyPlotter)
