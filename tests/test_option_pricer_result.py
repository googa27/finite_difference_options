import numpy as np

from src.option_pricer import OptionPricer, GridResult


def test_compute_grid_returns_namedtuple_and_shapes():
    pricer = OptionPricer(rate=0.05, sigma=0.2)
    res = pricer.compute_grid(
        maturity=1.0,
        s_max=2.0,
        s_steps=20,
        t_steps=30,
        strike=1.0,
        option_type="Call",
        return_greeks=True,
    )
    assert isinstance(res, GridResult)
    assert res.values.shape == (30, 20)
    assert res.s.shape == (20,)
    assert res.t.shape == (30,)
    assert res.delta is not None and res.delta.shape == res.values.shape
    assert res.gamma is not None and res.gamma.shape == res.values.shape
    assert res.theta is not None and res.theta.shape == res.values.shape


def test_compute_grid_without_greeks_sets_none():
    pricer = OptionPricer(rate=0.05, sigma=0.2)
    res = pricer.compute_grid(
        maturity=0.5,
        s_max=1.5,
        s_steps=10,
        t_steps=12,
        strike=1.0,
        option_type="Put",
        return_greeks=False,
    )
    assert res.delta is None
    assert res.gamma is None
    assert res.theta is None
