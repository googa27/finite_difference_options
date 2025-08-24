"""Tests for boundary condition construction."""
import pathlib
import sys

# Ensure project root on path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from findiff import FinDiff
from src.boundary_conditions import BlackScholesBoundaryBuilder
from src.options import EuropeanCall, EuropeanPut
from src.models import GeometricBrownianMotion


def test_call_boundary_conditions():
    s = np.linspace(0.0, 1.0, 5)
    model = GeometricBrownianMotion(mu=0.05, sigma=0.2)
    option = EuropeanCall(strike=1.0, maturity=1.0, model=model)
    bc = BlackScholesBoundaryBuilder().build(s, option)

    ds = s[1] - s[0]
    shape = s.shape

    expected_left = FinDiff(0, ds, 2).matrix(shape).toarray()[0]
    expected_right = FinDiff(0, ds, 1).matrix(shape).toarray()[-1]

    lhs = bc.lhs.toarray()

    assert np.allclose(lhs[0], expected_left)
    assert np.allclose(lhs[-1], expected_right)
    assert np.allclose(bc.rhs.toarray().ravel()[0], 0.0)
    assert np.allclose(bc.rhs.toarray().ravel()[-1], 1.0)


def test_put_boundary_conditions():
    s = np.linspace(0.0, 1.0, 5)
    from src.models import GeometricBrownianMotion
    model = GeometricBrownianMotion(mu=0.05, sigma=0.2)
    option = EuropeanPut(strike=1.0, maturity=1.0, model=model)
    bc = BlackScholesBoundaryBuilder().build(s, option)

    ds = s[1] - s[0]
    shape = s.shape

    expected_left = FinDiff(0, ds, 2).matrix(shape).toarray()[0]
    expected_right = FinDiff(0, ds, 1).matrix(shape).toarray()[-1]

    lhs = bc.lhs.toarray()

    assert np.allclose(lhs[0], expected_left)
    assert np.allclose(lhs[-1], expected_right)
    assert np.allclose(bc.rhs.toarray().ravel()[0], 0.0)
    assert np.allclose(bc.rhs.toarray().ravel()[-1], 0.0)
