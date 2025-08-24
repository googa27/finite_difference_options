"""Tests for callable bond pricing grid."""

import pathlib
import sys

# Ensure project root on path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from src.option_pricer import OptionPricer
from src.pde_pricer import CallableBondPDEModel
from src.market import Market
from src.processes import GeometricBrownianMotion


def test_callable_bond_grid_respects_call_price_and_face_value() -> None:
    market = Market(rate=0.03)
    short_rate = GeometricBrownianMotion(mu=0.03, sigma=0.01)
    bond_model = CallableBondPDEModel(
        face_value=100.0,
        call_price=105.0,
        market=market,
        model=short_rate,
        _maturity=1.0,
    )
    pricer = OptionPricer(instrument=bond_model)
    res = pricer.compute_grid(
        s_max=150.0,
        s_steps=50,
        t_steps=50,
    )

    assert np.all(res.values <= bond_model.call_price)
    assert np.allclose(res.values[0], bond_model.face_value)
