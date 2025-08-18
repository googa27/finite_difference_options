"""Plotly-based plotter implementation for interactive visuals."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Set

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

from .base import Plotter, PlotOptions
from .colors import map_matplotlib_to_plotly


class PlotlyPlotter(Plotter):
    """Plotter using Plotly for interactive visuals."""

    def heatmap(
        self,
        grid: NDArray[np.float64],
        s: NDArray[np.float64],
        t: NDArray[np.float64],
        *,
        opts: PlotOptions,
    ) -> Any:
        fig = go.Figure(
            data=
            [
                go.Heatmap(
                    z=grid.T,
                    x=t,
                    y=s,
                    colorscale=map_matplotlib_to_plotly(opts.cmap),
                    zmin=opts.vmin,
                    zmax=opts.vmax,
                    colorbar=dict(title=opts.colorbar_label or ""),
                )
            ]
        )
        fig.update_layout(
            xaxis_title=opts.x_label,
            yaxis_title=opts.y_label,
            height=opts.height or 450,
            margin=dict(l=60, r=20, t=30, b=50),
        )
        return fig

    def surface(
        self,
        grid: NDArray[np.float64],
        s: NDArray[np.float64],
        t: NDArray[np.float64],
        *,
        opts: PlotOptions,
    ) -> Any:
        fig = go.Figure(
            data=[
                go.Surface(z=grid, x=s, y=t, colorscale=map_matplotlib_to_plotly(opts.cmap), showscale=True)
            ]
        )

        elev = 25.0 if opts.elev is None else float(opts.elev)
        azim = -60.0 if opts.azim is None else float(opts.azim)
        r = 1.8
        rad_elev = np.deg2rad(elev)
        rad_azim = np.deg2rad(azim)
        eye = dict(
            x=float(r * np.cos(rad_azim) * np.cos(rad_elev)),
            y=float(r * np.sin(rad_azim) * np.cos(rad_elev)),
            z=float(r * np.sin(rad_elev)),
        )

        fig.update_layout(
            scene=dict(
                xaxis_title="Asset price",
                yaxis_title="Time",
                zaxis_title="Option value",
                camera=dict(eye=eye),
            ),
            height=opts.height or 450,
            margin=dict(l=0, r=0, t=0, b=0),
        )
        return fig

    def line1d(
        self,
        x: NDArray[np.float64],
        series: Dict[str, NDArray[np.float64]],
        *,
        opts: PlotOptions,
    ) -> Any:
        from plotly.subplots import make_subplots

        sec_set: Set[str] = set(opts.secondary_keys or [])
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        for name, y in series.items():
            use_secondary = name in sec_set
            fig.add_trace(
                go.Scatter(x=x, y=y, mode="lines", name=name),
                secondary_y=use_secondary,
            )
        fig.update_xaxes(title_text=opts.x_label)
        fig.update_yaxes(title_text="Value", secondary_y=False)
        if sec_set:
            fig.update_yaxes(title_text="Greeks", secondary_y=True)
        fig.add_hline(y=0.0, line_color="gray", line_width=1, opacity=0.6)
        fig.update_layout(
            height=opts.height or 400,
            margin=dict(l=50, r=60, t=20, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig
