"""Tests for option base class and subclasses."""

import pathlib
import sys

# Ensure project root on path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.processes.affine import GeometricBrownianMotion
from src.instruments.base import EuropeanOption, EuropeanCall, EuropeanPut


def test_base_class_can_be_instantiated():
    """EuropeanOption can be instantiated directly."""
    model = GeometricBrownianMotion(mu=0.05, sigma=0.2)
    # This should work since it has a payoff method
    # (though it's not very useful without a concrete payoff implementation)
    option = EuropeanOption(strike=1.0, maturity=1.0, model=model)
    assert option.strike == 1.0
    assert option.maturity == 1.0


def test_subclasses_behave_correctly():
    """Subclasses inherit from EuropeanOption and compute correct payoffs."""
    s = np.array([0.5, 1.0, 1.5])
    model = GeometricBrownianMotion(mu=0.05, sigma=0.2)
    call = EuropeanCall(strike=1.0, maturity=1.0, model=model)
    put = EuropeanPut(strike=1.0, maturity=1.0, model=model)

    assert issubclass(EuropeanCall, EuropeanOption)
    assert issubclass(EuropeanPut, EuropeanOption)
    assert isinstance(call, EuropeanOption)
    assert isinstance(put, EuropeanOption)

    np.testing.assert_allclose(call.payoff(s), np.array([0.0, 0.0, 0.5]))
    np.testing.assert_allclose(put.payoff(s), np.array([0.5, 0.0, 0.0]))