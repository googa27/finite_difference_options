"""Tests for OptionPricer.compute_grid."""
import pathlib
import sys

# Ensure project root on path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.stats import norm

from src.pricing import OptionPricer
from src.processes.affine import GeometricBrownianMotion
from src.instruments.base import EuropeanCall


def bs_call_price(s: float, k: float, r: float, sigma: float, T: float) -> float:
    """Return analytical Black--Scholes call price."""
    from math import log, sqrt, exp

    d1 = (log(s / k) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return s * norm.cdf(d1) - k * exp(-r * T) * norm.cdf(d2)


def test_compute_grid_shapes_and_price():
    rate = 0.05
    sigma = 0.2
    T = 1.0
    K = 1.0
    S_max = 3.0
    ns = 40
    nt = 40

    model = GeometricBrownianMotion(mu=rate, sigma=sigma)
    instrument = EuropeanCall(strike=K, maturity=T, model=model)
    pricer = OptionPricer(instrument=instrument)
    res = pricer.compute_grid(
        s_max=S_max,
        s_steps=ns,
        t_steps=nt,
    )

    assert res.s.shape == (ns,)
    assert res.t.shape == (nt,)
    assert res.values.shape == (nt, ns)

    idx = np.searchsorted(res.s, 1.0)
    price = res.values[-1, idx]
    expected = bs_call_price(1.0, K, rate, sigma, T)
    assert abs(price - expected) < 2e-2