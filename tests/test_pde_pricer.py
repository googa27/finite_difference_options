"""Tests for the finite difference Black--Scholes pricer."""
import pathlib
import sys

# Ensure project root on path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from src.models import Market, GeometricBrownianMotion
from src.options import EuropeanCall, EuropeanPut
from src.pde_pricer import BlackScholesPDE
from src.time_steppers import ExplicitEuler


def black_scholes_call(s, k, r, sigma, T):
    """Analytical Black--Scholes formula for calls."""
    from math import log, sqrt, exp
    from scipy.stats import norm

    d1 = (log(s / k) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return s * norm.cdf(d1) - k * exp(-r * T) * norm.cdf(d2)


def black_scholes_put(s, k, r, sigma, T):
    """Analytical Black--Scholes formula for puts."""
    from math import log, sqrt, exp
    from scipy.stats import norm

    d1 = (log(s / k) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return k * exp(-r * T) * norm.cdf(-d2) - s * norm.cdf(-d1)


def test_call_matches_analytical():
    rate = 0.05
    sigma = 0.2
    T = 1.0
    K = 1.0
    S_max = 3.0
    ns = 100
    nt = 100

    s = np.linspace(0.0, S_max, ns)
    t = np.linspace(0.0, T, nt)

    market = Market(rate=rate)
    model = GeometricBrownianMotion(rate=rate, sigma=sigma)
    option = EuropeanCall(strike=K)

    pricer = BlackScholesPDE(model=model, market=market)
    grid = pricer.price(option, s, t)

    idx = np.searchsorted(s, 1.0)
    pde_price = grid[-1, idx]
    analytical = black_scholes_call(1.0, K, rate, sigma, T)

    assert abs(pde_price - analytical) < 1e-2


def test_put_matches_analytical():
    rate = 0.05
    sigma = 0.2
    T = 1.0
    K = 1.0
    S_max = 3.0
    ns = 100
    nt = 100

    s = np.linspace(0.0, S_max, ns)
    t = np.linspace(0.0, T, nt)

    market = Market(rate=rate)
    model = GeometricBrownianMotion(rate=rate, sigma=sigma)
    option = EuropeanPut(strike=K)

    pricer = BlackScholesPDE(model=model, market=market)
    grid = pricer.price(option, s, t)

    idx = np.searchsorted(s, 1.0)
    pde_price = grid[-1, idx]
    analytical = black_scholes_put(1.0, K, rate, sigma, T)

    assert abs(pde_price - analytical) < 1e-2


def test_call_matches_analytical_explicit_euler():
    rate = 0.05
    sigma = 0.2
    T = 1.0
    K = 1.0
    S_max = 3.0
    ns = 50
    nt = 4000

    s = np.linspace(0.0, S_max, ns)
    t = np.linspace(0.0, T, nt)

    market = Market(rate=rate)
    model = GeometricBrownianMotion(rate=rate, sigma=sigma)
    option = EuropeanCall(strike=K)

    pricer = BlackScholesPDE(
        model=model, market=market, time_stepper=ExplicitEuler()
    )
    grid = pricer.price(option, s, t)

    idx = np.searchsorted(s, 1.0)
    pde_price = grid[-1, idx]
    analytical = black_scholes_call(1.0, K, rate, sigma, T)

    assert abs(pde_price - analytical) < 5e-2
