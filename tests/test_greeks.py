"""Tests for finite difference Greek calculations."""
import pathlib
import sys

# Ensure project root on path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.stats import norm

from src.pricing import OptionPricer
from src.processes.affine import GeometricBrownianMotion
from src.instruments.base import EuropeanCall


def bs_call_greeks(
    s: float, k: float, r: float, sigma: float, T: float
) -> tuple[float, float, float]:
    """Return analytical Delta, Gamma and Theta for a call option."""

    from math import log, sqrt, exp

    d1 = (log(s / k) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (s * sigma * sqrt(T))
    theta = (
        -s * norm.pdf(d1) * sigma / (2 * sqrt(T))
        - r * k * exp(-r * T) * norm.cdf(d2)
    )
    return delta, gamma, theta


def test_finite_difference_greeks_match_analytical():
    rate = 0.05
    sigma = 0.2
    T = 1.0
    K = 1.0
    S_max = 3.0
    ns = 300
    nt = 300

    model = GeometricBrownianMotion(mu=rate, sigma=sigma)
    instrument = EuropeanCall(strike=K, maturity=T, model=model)
    pricer = OptionPricer(instrument=instrument)
    s, t, values, delta, gamma, theta = pricer.compute_grid(
        s_max=S_max,
        s_steps=ns,
        t_steps=nt,
        return_greeks=True,
    )

    idx = np.searchsorted(s, 1.0)
    delta_a, gamma_a, theta_a = bs_call_greeks(1.0, K, rate, sigma, T)

    assert abs(delta[-1, idx] - delta_a) < 1e-2
    assert abs(gamma[-1, idx] - gamma_a) < 2e-2
    assert abs(theta[-1, idx] - theta_a) < 1e-2