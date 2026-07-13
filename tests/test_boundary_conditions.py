"""Tests for typed, model-aware boundary condition construction."""

from __future__ import annotations

import math
from types import SimpleNamespace


import numpy as np
import pytest
from numpy.testing import assert_allclose

from finite_difference_options.boundary_conditions import (
    BlackScholesBoundaryBuilder,
    HestonBoundaryBuilder,
)
from finite_difference_options.exceptions import BoundaryConditionError
from finite_difference_options.instruments.base import EuropeanCall, EuropeanPut
from finite_difference_options.processes.affine import GeometricBrownianMotion


def _rhs_values(boundary_conditions):
    return boundary_conditions.rhs.toarray().ravel()


def test_call_boundary_conditions_include_strike_rate_carry_and_maturity() -> None:
    grid = np.linspace(0.0, 200.0, 5)
    option = EuropeanCall(
        strike=100.0,
        maturity=1.5,
        model=GeometricBrownianMotion(mu=0.03, sigma=0.2),
    )
    builder = BlackScholesBoundaryBuilder()

    resolution = builder.resolve(
        grid,
        option,
        risk_free_rate=0.05,
        dividend_yield=0.02,
    )
    bc = builder.build(
        grid,
        option,
        risk_free_rate=0.05,
        dividend_yield=0.02,
    )

    expected_upper = grid[-1] * math.exp(-0.02 * 1.5) - option.strike * math.exp(
        -0.05 * 1.5
    )
    rhs = _rhs_values(bc)
    assert_allclose(rhs[0], 0.0)
    assert_allclose(rhs[-1], expected_upper)
    assert resolution.specs[0].kind == "dirichlet"
    assert (
        resolution.specs[1].expression == "V(Smax,tau)=Smax*exp(-q*tau)-K*exp(-r*tau)"
    )
    assert resolution.discount_source == "explicit"


def test_put_boundary_conditions_include_discounted_strike() -> None:
    grid = np.linspace(0.0, 200.0, 5)
    option = EuropeanPut(
        strike=100.0,
        maturity=2.0,
        model=GeometricBrownianMotion(mu=0.03, sigma=0.2),
    )

    bc = BlackScholesBoundaryBuilder().build(grid, option, risk_free_rate=0.07)

    rhs = _rhs_values(bc)
    assert_allclose(rhs[0], option.strike * math.exp(-0.07 * option.maturity))
    assert_allclose(rhs[-1], 0.0)


def test_unsupported_or_ambiguous_boundary_set_fails_closed() -> None:
    grid = np.linspace(0.0, 200.0, 5)
    unsupported = SimpleNamespace(option_type="digital", strike=100.0, maturity=1.0)

    with pytest.raises(BoundaryConditionError, match="only vanilla call/put"):
        BlackScholesBoundaryBuilder(allow_legacy_mu_rate=False).build(
            grid,
            unsupported,
            risk_free_rate=0.05,
        )


def test_heston_boundary_specs_are_coordinate_and_model_aware() -> None:
    option = SimpleNamespace(option_type="call", strike=100.0, maturity=1.0)
    resolution = HestonBoundaryBuilder().resolve(
        np.log(np.array([25.0, 100.0, 400.0])),
        np.array([0.0, 0.04, 0.25]),
        option,
        risk_free_rate=0.05,
        dividend_yield=0.01,
    )

    payload = [spec.as_dict() for spec in resolution.specs]
    assert [item["coordinate"] for item in payload] == [
        "log_spot",
        "log_spot",
        "variance",
        "variance",
    ]
    assert payload[2]["kind"] == "degenerate"
    assert payload[3]["kind"] == "extrapolated"
    assert resolution.risk_free_rate == 0.05
    assert resolution.dividend_yield == 0.01
