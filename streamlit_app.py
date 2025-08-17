"""Streamlit demo for the Black--Scholes PDE option pricer."""
from __future__ import annotations

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from src.option_pricer import OptionPricer


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
        pricer = OptionPricer(rate=rate, sigma=sigma)
        s, t, grid = pricer.compute_grid(
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
