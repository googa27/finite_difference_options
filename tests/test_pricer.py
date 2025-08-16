"""Tests for the finite difference option pricer."""
import sys
from pathlib import Path
import numpy as np
import scipy.stats as spst

# Ensure the src package is on the import path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import Market, EuropeanOption, FiniteDifferencePricer


def bs_price(is_call: bool, s0: float, k: float, r: float, sigma: float, T: float) -> float:
    """Analytical Black--Scholes price for European options."""
    d1 = (np.log(s0 / k) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call:
        return s0 * spst.norm.cdf(d1) - k * np.exp(-r * T) * spst.norm.cdf(d2)
    return k * np.exp(-r * T) * spst.norm.cdf(-d2) - s0 * spst.norm.cdf(-d1)


def test_call_price_matches_black_scholes():
    market = Market(r=0.05)
    option = EuropeanOption(strike=100, maturity=1.0, is_call=True)
    pricer = FiniteDifferencePricer(market, sigma=0.2, s_max=200, ns=200, nt=200)
    price = pricer.price(option, s0=100)
    analytic = bs_price(True, 100, 100, 0.05, 0.2, 1.0)
    assert abs(price - analytic) < 1e-1


def test_put_price_matches_black_scholes():
    market = Market(r=0.05)
    option = EuropeanOption(strike=100, maturity=1.0, is_call=False)
    pricer = FiniteDifferencePricer(market, sigma=0.2, s_max=200, ns=200, nt=200)
    price = pricer.price(option, s0=100)
    analytic = bs_price(False, 100, 100, 0.05, 0.2, 1.0)
    assert abs(price - analytic) < 1e-1
