from src.option_pricer import OptionPricer, GridResult
from src.processes.affine import GeometricBrownianMotion
from src.instruments.base import EuropeanCall, EuropeanPut


def test_compute_grid_returns_namedtuple_and_shapes():
    model = GeometricBrownianMotion(mu=0.05, sigma=0.2)
    instrument = EuropeanCall(strike=1.0, maturity=1.0, model=model)
    pricer = OptionPricer(instrument=instrument)
    res = pricer.compute_grid(
        s_max=2.0,
        s_steps=20,
        t_steps=30,
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
    model = GeometricBrownianMotion(mu=0.05, sigma=0.2)
    instrument = EuropeanPut(strike=1.0, maturity=0.5, model=model)
    pricer = OptionPricer(instrument=instrument)
    res = pricer.compute_grid(
        s_max=1.5,
        s_steps=10,
        t_steps=12,
        return_greeks=False,
    )
    assert res.delta is None
    assert res.gamma is None
    assert res.theta is None