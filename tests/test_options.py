"""Tests for option base class and subclasses."""
import pathlib
import sys

# Ensure project root on path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from src.options import EuropeanOption, EuropeanCall, EuropeanPut
from src.models import GeometricBrownianMotion


def test_base_class_is_abstract():
    """EuropeanOption cannot be instantiated directly."""
    model = GeometricBrownianMotion(rate=0.05, sigma=0.2)
    with pytest.raises(TypeError):
        EuropeanOption(strike=1.0, maturity=1.0, model=model)


def test_subclasses_behave_correctly():
    """Subclasses inherit from EuropeanOption and compute correct payoffs."""
    s = np.array([0.5, 1.0, 1.5])
    model = GeometricBrownianMotion(rate=0.05, sigma=0.2)
    call = EuropeanCall(strike=1.0, maturity=1.0, model=model)
    put = EuropeanPut(strike=1.0, maturity=1.0, model=model)

    assert issubclass(EuropeanCall, EuropeanOption)
    assert issubclass(EuropeanPut, EuropeanOption)
    assert isinstance(call, EuropeanOption)
    assert isinstance(put, EuropeanOption)

    np.testing.assert_allclose(call.payoff(s), np.array([0.0, 0.0, 0.5]))
    np.testing.assert_allclose(put.payoff(s), np.array([0.5, 0.0, 0.0]))
