"""Streamlit UI for exploring Black–Scholes PDE grids.

Architecture overview:
- Pricing returns a GridResult where all grids are shaped (t, s).
- Plotting uses a backend-agnostic Plotter Protocol with PlotOptions.
- Backends are selected via get_plotter("matplotlib"|"plotly").
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st
import numpy as np
import seaborn as sns

# Ensure repository root is on the Python path when executed via Streamlit.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover - runtime path fix
    sys.path.append(str(ROOT))

from src.option_pricer import OptionPricer, GridResult  # noqa: E402
from src.plotting.factory import get_plotter  # noqa: E402
from src.plotting.colors import DEFAULT_DIVERGING, symmetric_bounds  # noqa: E402
from src.plotting.config_manager import (
    surface_figsize_from_height,
    line_figsize_from_height,
    PlottingConfigManager,
)
from src.models import GeometricBrownianMotion, Market # noqa: E402
from src.options import EuropeanCall, EuropeanPut # noqa: E402
from src.pde_pricer import CallableBondPDEModel # noqa: E402
from src.instruments import Instrument # noqa: E402

try:  # noqa: E402
    import plotly  # type: ignore
    PLOTLY_AVAILABLE = True
except Exception:  # noqa: E402
    plotly = None  # type: ignore
    PLOTLY_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


st.set_page_config(
    page_title="Black–Scholes PDE Pricer",
    page_icon="📈",
    layout="wide",
)
sns.set_theme(style="white", context="talk")


@st.cache_data(show_spinner=False)
def _compute_grid_cached(
    *,
    instrument: Instrument,
    s_max: float,
    s_steps: int,
    t_steps: int,
    return_greeks: bool,
) -> GridResult:
    pricer = OptionPricer(instrument=instrument)
    return pricer.compute_grid(
        s_max=s_max,
        s_steps=int(s_steps),
        t_steps=int(t_steps),
        return_greeks=return_greeks,
    )


def build_sidebar() -> dict:
    """Build sidebar controls and return parameters dict."""
    # Defaults for session state to persist selections across reruns
    defaults = dict(
        rate=0.05,
        sigma=0.2,
        strike=1.0,
        maturity=1.0,
        option_type="Call",
        s_max=3.0,
        s_steps=120,
        t_steps=120,
        backend_label="Plotly" if PLOTLY_AVAILABLE else "Matplotlib",
        cmap="viridis",
        plot_height=420,
        elev=25,
        azim=-60,
        live=True,
        show_delta=True,
        show_gamma=True,
        show_theta=False,
        instrument_type="European Option",
        face_value=100.0,
        call_price=105.0,
    )
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)
    st.header("Parameters")

    instrument_type = st.radio("Instrument Type", ["European Option", "Callable Bond"], key="instrument_type")

    params = {}
    if instrument_type == "European Option":
        params["rate"] = st.slider("Interest rate r", min_value=0.0, max_value=0.2, step=0.005, key="rate")
        params["sigma"] = st.slider("Volatility σ", min_value=0.05, max_value=1.0, step=0.01, key="sigma")
        params["strike"] = st.slider("Strike K", min_value=0.1, max_value=5.0, step=0.1, key="strike")
        params["maturity"] = st.slider("Maturity T (years)", min_value=0.1, max_value=5.0, step=0.1, key="maturity")
        params["option_type"] = st.radio("Option type", ["Call", "Put"], horizontal=True, key="option_type")
    elif instrument_type == "Callable Bond":
        params["face_value"] = st.slider("Face Value", min_value=50.0, max_value=200.0, step=5.0, key="face_value")
        params["call_price"] = st.slider("Call Price", min_value=50.0, max_value=200.0, step=5.0, key="call_price")
        params["maturity"] = st.slider("Maturity T (years)", min_value=0.1, max_value=5.0, step=0.1, key="maturity")
        params["rate"] = st.slider("Interest rate r (for bond model)", min_value=0.0, max_value=0.2, step=0.005, key="rate")
        params["sigma"] = st.slider("Volatility σ (for bond model)", min_value=0.01, max_value=0.1, step=0.005, key="sigma")

    params["instrument_type"] = instrument_type

    with st.expander("Grid settings", expanded=False):
        params["s_max"] = st.slider("Max stock price Sₘₐₓ", min_value=1.0, max_value=10.0, step=0.5, key="s_max")
        params["s_steps"] = st.slider("Price steps Nₛ", min_value=20, max_value=400, step=10, key="s_steps")
        params["t_steps"] = st.slider("Time steps Nₜ", min_value=20, max_value=400, step=10, key="t_steps")

    with st.expander("Plot options", expanded=False):
        backends = ["Matplotlib"]
        if PLOTLY_AVAILABLE:
            backends.insert(0, "Plotly")
        params["backend_label"] = st.selectbox("Backend", backends, key="backend_label")
        params["backend"] = "plotly" if params["backend_label"].startswith("Plotly") else "matplotlib"
        if not PLOTLY_AVAILABLE and params["backend"] == "plotly":
            st.caption("Plotly not installed — falling back to Matplotlib.")
            params["backend"] = "matplotlib"
        params["cmap"] = st.selectbox(
            "Colormap",
            ["viridis", "magma", "plasma", "cividis", "inferno", "RdBu", "coolwarm"],
            key="cmap",
        )
        params["plot_height"] = st.slider("Plot height (px)", min_value=300, max_value=800, step=10, key="plot_height")
        params["elev"] = None
        params["azim"] = None
        if params["backend"] == "matplotlib":
            params["elev"] = st.slider("3D elevation (deg)", min_value=0, max_value=80, key="elev")
            params["azim"] = st.slider("3D azimuth (deg)", min_value=-180, max_value=180, key="azim")
        params["live"] = st.toggle("Live update on change", key="live")
        params["recompute"] = st.button("Recompute", disabled=params["live"])

    return params


def main() -> None:
    """Run the Streamlit application with a modern layout."""
    st.title("Black–Scholes PDE Option Pricer")

    # Sidebar
    with st.sidebar:
        params = build_sidebar()

    # Compute in real time (cached) when parameters change
    def maybe_compute() -> GridResult:
        if params["instrument_type"] == "European Option":
            model = GeometricBrownianMotion(rate=params["rate"], sigma=params["sigma"])
            option_cls = EuropeanCall if params["option_type"] == "Call" else EuropeanPut
            instrument = option_cls(strike=params["strike"], maturity=params["maturity"], model=model)
        elif params["instrument_type"] == "Callable Bond":
            market = Market(rate=params["rate"])
            model = GeometricBrownianMotion(rate=params["rate"], sigma=params["sigma"])
            instrument = CallableBondPDEModel(
                face_value=params["face_value"],
                call_price=params["call_price"],
                market=market,
                model=model,
                _maturity=params["maturity"],
            )
        else:
            raise ValueError("Unknown instrument type")

        return _compute_grid_cached(
            instrument=instrument,
            s_max=params["s_max"],
            s_steps=params["s_steps"],
            t_steps=params["t_steps"],
            return_greeks=True,
        )

    if params["live"] or params["recompute"]:
        with st.spinner("Computing grid..."):
            data = maybe_compute()
    else:
        st.info("Live update disabled — click Recompute to update.")
        data = maybe_compute()

    s, t, grid = data.s, data.t, data.values
    # Prepare plotting defaults instance to build PlotOptions consistently
    plot_def = PlottingConfigManager(cmap=params["cmap"], height=params["plot_height"], elev=params["elev"], azim=params["azim"])
    delta, gamma, theta = data.delta, data.gamma, data.theta

    # Header metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Grid size", f"{grid.shape[1]} × {grid.shape[0]}")
    m2.metric("ΔS", f"{(s[1]-s[0]):.3f}")
    m3.metric("ΔT", f"{(t[1]-t[0]):.3f}")

    tabs = st.tabs(["Surface", "Heatmap", "Greeks", "1D Slice"])

    with tabs[0]:
        st.subheader("Value Surface")
        plotter = get_plotter(params["backend"])  # reuse factory
        inches = surface_figsize_from_height(params["plot_height"]) 
        opts = plot_def.surface(figsize=inches)
        fig3d = plotter.surface(grid=grid, s=s, t=t, opts=opts)
        if params["backend"] == "plotly":
            st.plotly_chart(fig3d, use_container_width=True)
        else:
            st.pyplot(fig3d, use_container_width=True)

    with tabs[1]:
        st.subheader("Value Heatmap")
        # Prefer Time on X-axis and Asset price on Y-axis
        plotter = get_plotter(params["backend"])  # one instance per draw
        opts = plot_def.heatmap(x_label="Time", y_label="Asset price")
        fig = plotter.heatmap(grid=grid, s=s, t=t, opts=opts)
        if params["backend"] == "plotly":
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.pyplot(fig, use_container_width=True)

    with tabs[3]:
        st.subheader("1D Slice: Price and Greeks vs Asset Price")
        default_time = float(t[len(t)//2])
        if "time_val" not in st.session_state:
            st.session_state.time_val = default_time
        time_val = st.slider(
            "Time (T)",
            min_value=float(t[0]),
            max_value=float(t[-1]),
            step=float(max(t[-1] - t[0], 1e-6) / 100),
            key="time_val",
        )
        # Find nearest time index
        t_idx = int(np.argmin(np.abs(t - time_val)))
        show_delta = st.checkbox("Show Delta", key="show_delta")
        show_gamma = st.checkbox("Show Gamma", key="show_gamma")
        show_theta = st.checkbox("Show Theta", key="show_theta")

        series = {"Price": grid[t_idx, :]}
        if show_delta:
            series["Delta"] = delta[t_idx, :]
        if show_gamma:
            series["Gamma"] = gamma[t_idx, :]
        if show_theta:
            series["Theta"] = theta[t_idx, :]

        plotter = get_plotter(params["backend"])  # reuse factory
        inches = line_figsize_from_height(params["plot_height"]) 
        opts = plot_def.line(x_label="Asset price", y_label="Option value", figsize=inches, secondary_keys=[k for k in series.keys() if k != "Price"])  # noqa: E501
        fig1d = plotter.line1d(x=s, series=series, opts=opts)
        if params["backend"] == "plotly":
            st.plotly_chart(fig1d, use_container_width=True)
        else:
            st.pyplot(fig1d, use_container_width=True)

    with tabs[2]:
        st.subheader("Greeks 3D Surfaces and Heatmaps")
        use_diverging = st.checkbox("Use diverging colormap (RdBu) and symmetric scaling", value=True)
        greeks_cmap = DEFAULT_DIVERGING if use_diverging else params["cmap"]

        # 3D surfaces first to showcase visuals
        show_3d = st.checkbox("Show 3D surfaces", value=True)
        if show_3d:
            g3d_cols = st.columns(3)
            with g3d_cols[0]:
                st.caption("Delta 3D")
                plotter = get_plotter(params["backend"])  # reuse factory
                inches = surface_figsize_from_height(params["plot_height"]) 
                opts = plot_def.surface(cmap=greeks_cmap, height=int(params["plot_height"] * 0.9), figsize=inches)
                fig_d3 = plotter.surface(grid=delta, s=s, t=t, opts=opts)
                if params["backend"] == "plotly":
                    st.plotly_chart(fig_d3, use_container_width=True)
                else:
                    st.pyplot(fig_d3, use_container_width=True)
            with g3d_cols[1]:
                st.caption("Gamma 3D")
                plotter = get_plotter(params["backend"])  # reuse factory
                inches = surface_figsize_from_height(params["plot_height"]) 
                opts = plot_def.surface(cmap=greeks_cmap, height=int(params["plot_height"] * 0.9), figsize=inches)
                fig_g3 = plotter.surface(grid=gamma, s=s, t=t, opts=opts)
                if params["backend"] == "plotly":
                    st.plotly_chart(fig_g3, use_container_width=True)
                else:
                    st.pyplot(fig_g3, use_container_width=True)
            with g3d_cols[2]:
                st.caption("Theta 3D")
                plotter = get_plotter(params["backend"])  # reuse factory
                inches = surface_figsize_from_height(params["plot_height"]) 
                opts = plot_def.surface(cmap=greeks_cmap, height=int(params["plot_height"] * 0.9), figsize=inches)
                fig_t3 = plotter.surface(grid=theta, s=s, t=t, opts=opts)
                if params["backend"] == "plotly":
                    st.plotly_chart(fig_t3, use_container_width=True)
                else:
                    st.pyplot(fig_t3, use_container_width=True)

        st.markdown("---")
        st.subheader("Greeks Heatmaps")
        gcols = st.columns(3)
        # Delta
        with gcols[0]:
            st.caption("Delta")
            plotter = get_plotter(params["backend"])  # reuse factory
            if use_diverging:
                vmin, vmax = symmetric_bounds(delta)
            else:
                vmin = float(np.nanmin(delta)) if delta is not None else None
                vmax = float(np.nanmax(delta)) if delta is not None else None
            opts = plot_def.heatmap(
                x_label="Time",
                y_label="Asset price",
                cmap=greeks_cmap,
                height=int(params["plot_height"] * 0.9),
                vmin=vmin,
                vmax=vmax,
            )
            fig_d = plotter.heatmap(grid=delta, s=s, t=t, opts=opts)
            if params["backend"] == "plotly":
                st.plotly_chart(fig_d, use_container_width=True)
            else:
                st.pyplot(fig_d, use_container_width=True)
        # Gamma
        with gcols[1]:
            st.caption("Gamma")
            plotter = get_plotter(params["backend"])  # reuse factory
            if use_diverging:
                vmin, vmax = symmetric_bounds(gamma)
            else:
                vmin = float(np.nanmin(gamma)) if gamma is not None else None
                vmax = float(np.nanmax(gamma)) if gamma is not None else None
            opts = plot_def.heatmap(
                x_label="Time",
                y_label="Asset price",
                cmap=greeks_cmap,
                height=int(params["plot_height"] * 0.9),
                vmin=vmin,
                vmax=vmax,
            )
            fig_g = plotter.heatmap(grid=gamma, s=s, t=t, opts=opts)
            if params["backend"] == "plotly":
                st.plotly_chart(fig_g, use_container_width=True)
            else:
                st.pyplot(fig_g, use_container_width=True)
        # Theta
        with gcols[2]:
            st.caption("Theta")
            plotter = get_plotter(params["backend"])  # reuse factory
            if use_diverging:
                vmin, vmax = symmetric_bounds(theta)
            else:
                vmin = float(np.nanmin(theta)) if theta is not None else None
                vmax = float(np.nanmax(theta)) if theta is not None else None
            opts = plot_def.heatmap(
                x_label="Time",
                y_label="Asset price",
                cmap=greeks_cmap,
                height=int(params["plot_height"] * 0.9),
                vmin=vmin,
                vmax=vmax,
            )
            fig_th = plotter.heatmap(grid=theta, s=s, t=t, opts=opts)
            if params["backend"] == "plotly":
                st.plotly_chart(fig_th, use_container_width=True)
            else:
                st.pyplot(fig_th, use_container_width=True)


if __name__ == "__main__":
    main()
