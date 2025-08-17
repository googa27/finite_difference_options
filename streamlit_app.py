"""Streamlit demo for the Black--Scholes PDE option pricer."""
from __future__ import annotations

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from src.models import Market, GeometricBrownianMotion
from src.options import EuropeanCall, EuropeanPut
from src.pde_pricer import BlackScholesPDE


def compute_grid(
    rate: float,
    sigma: float,
    strike: float,
    maturity: float,
    option_type: str,
    s_max: float,
    s_steps: int,
    t_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return asset and time grids with option values."""
    market = Market(rate=rate)
    model = GeometricBrownianMotion(rate=rate, sigma=sigma)
    option_cls = EuropeanCall if option_type == "Call" else EuropeanPut
    option = option_cls(strike=strike)
    s = np.linspace(0, s_max, s_steps)
    t = np.linspace(0, maturity, t_steps)
    pricer = BlackScholesPDE(model=model, market=market)
    values = pricer.price(option=option, s=s, t=t)
    return s, t, values


def main() -> None:
    """Run the Streamlit application."""
    st.title("Black–Scholes PDE Option Pricer")
    rate = st.number_input("Interest rate", value=0.05)
    sigma = st.number_input("Volatility", value=0.2)
    strike = st.number_input("Strike", value=1.0)
    maturity = st.number_input("Maturity", value=1.0)
    option_type = st.selectbox("Option type", ["Call", "Put"])
    s_max = st.number_input("Max stock price", value=3.0)
    s_steps = st.number_input("Price steps", value=100, min_value=10, step=10)
    t_steps = st.number_input("Time steps", value=100, min_value=10, step=10)

    if st.button("Compute"):
        s, t, grid = compute_grid(
            rate=rate,
            sigma=sigma,
            strike=strike,
            maturity=maturity,
            option_type=option_type,
            s_max=s_max,
            s_steps=int(s_steps),
            t_steps=int(t_steps),
        )
        fig, ax = plt.subplots()
        sns.heatmap(
            grid,
            xticklabels=np.round(s, 2),
            yticklabels=np.round(t, 2),
            ax=ax,
        )
        ax.set_xlabel("Asset price")
        ax.set_ylabel("Time")
        st.pyplot(fig)


if __name__ == "__main__":
    main()
