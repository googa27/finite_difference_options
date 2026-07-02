"""Tests for schedule-aware callable bond pricing."""

from __future__ import annotations

import numpy as np
import pytest

from finite_difference_options.exceptions import PricingError
from finite_difference_options.models import Market
from finite_difference_options.pricing import OptionPricer
from finite_difference_options.pricing.engines import (
    CallableBondPDEModel,
    CallScheduleEntry,
)
from finite_difference_options.processes import OrnsteinUhlenbeck


def _short_rate_model() -> OrnsteinUhlenbeck:
    return OrnsteinUhlenbeck(kappa=0.4, theta=0.04, sigma=0.01)


def _callable_bond(*, call_price: float = 101.0) -> CallableBondPDEModel:
    return CallableBondPDEModel(
        face_value=100.0,
        call_price=call_price,
        market=Market(rate=0.04),
        model=_short_rate_model(),
        _maturity=2.0,
        coupon_rate=0.05,
        coupon_times=(0.5, 1.0, 1.5, 2.0),
        call_schedule=(CallScheduleEntry(time=1.0, price=call_price),),
    )


def test_callable_bond_requires_explicit_call_schedule() -> None:
    with pytest.raises(PricingError, match="provide explicit call_schedule"):
        CallableBondPDEModel(
            face_value=100.0,
            call_price=101.0,
            market=Market(rate=0.04),
            model=_short_rate_model(),
            _maturity=2.0,
        )


def test_callable_bond_applies_call_only_on_contractual_call_date() -> None:
    bond_model = _callable_bond(call_price=101.0)
    rate_grid = np.linspace(0.0, 0.12, 25)
    tau_grid = np.array([0.0, 1.0, 2.0])

    callable_values = bond_model.price_grid(rate_grid, tau_grid)
    diagnostics = bond_model.last_exercise_diagnostics
    straight_values = bond_model.straight_bond_grid(rate_grid, tau_grid)

    # Maturity value is redemption plus final coupon; it is not globally capped
    # by the call price because the call is contractual at t=1 only.
    assert np.allclose(callable_values[0], 102.5)
    assert np.all(callable_values[0] > bond_model.normalized_call_schedule[0].price)

    call_row = callable_values[1]
    same_date_coupon = 2.5
    assert np.all(call_row <= bond_model.normalized_call_schedule[0].price + same_date_coupon)
    assert np.any(call_row > bond_model.normalized_call_schedule[0].price)
    assert np.any(straight_values[1] > call_row)
    assert np.all(callable_values[-1] <= straight_values[-1] + 1e-12)

    assert len(diagnostics) == 1
    assert diagnostics[0].time == 1.0
    assert diagnostics[0].exercised_nodes > 0
    assert diagnostics[0].settlement_value == 101.0


def test_high_call_price_and_empty_schedule_reproduce_straight_bond() -> None:
    rate_grid = np.linspace(0.0, 0.12, 25)
    tau_grid = np.array([0.0, 1.0, 2.0])

    high_call_bond = _callable_bond(call_price=1_000.0)
    callable_grid = high_call_bond.price_grid(rate_grid, tau_grid)
    high_call_diagnostics = high_call_bond.last_exercise_diagnostics
    assert np.allclose(
        callable_grid,
        high_call_bond.straight_bond_grid(rate_grid, tau_grid),
    )
    assert high_call_diagnostics[0].exercised_nodes == 0
    assert high_call_bond.last_exercise_diagnostics == ()

    noncallable_bond = CallableBondPDEModel(
        face_value=100.0,
        call_price=None,
        market=Market(rate=0.04),
        model=_short_rate_model(),
        _maturity=2.0,
        coupon_rate=0.05,
        coupon_times=(0.5, 1.0, 1.5, 2.0),
    )
    assert np.allclose(
        noncallable_bond.price_grid(rate_grid, tau_grid),
        noncallable_bond.straight_bond_grid(rate_grid, tau_grid),
    )
    assert noncallable_bond.last_exercise_diagnostics == ()


def test_settlement_time_shortens_valuation_horizon() -> None:
    bond_model = CallableBondPDEModel(
        face_value=100.0,
        call_price=None,
        market=Market(rate=0.04),
        model=_short_rate_model(),
        _maturity=2.0,
        settlement_time=1.0,
    )
    rate_grid = np.array([0.05, 0.06])
    tau_grid = np.array([0.0, 1.0])

    values = bond_model.price_grid(rate_grid, tau_grid)
    priced = OptionPricer(instrument=bond_model).compute_grid(
        s_max=0.06,
        s_steps=2,
        t_steps=2,
    )

    assert np.isclose(values[-1, 0], 100.0 * np.exp(-0.05 * 1.0))
    assert priced.t.tolist() == [0.0, 1.0]


def test_clean_call_schedule_adds_accrued_interest_to_exercise_value() -> None:
    bond_model = CallableBondPDEModel(
        face_value=100.0,
        call_price=100.0,
        market=Market(rate=0.04),
        model=_short_rate_model(),
        _maturity=1.0,
        coupon_rate=0.06,
        coupon_times=(1.0,),
        call_schedule=(CallScheduleEntry(time=0.5, price=100.0, quote_convention="clean"),),
    )
    rate_grid = np.linspace(0.0, 0.10, 21)
    tau_grid = np.array([0.0, 0.5, 1.0])

    values = bond_model.price_grid(rate_grid, tau_grid)

    assert np.all(values[1] <= 103.0)
    assert bond_model.last_exercise_diagnostics[0].settlement_value == 103.0


def test_clean_call_on_coupon_date_does_not_double_count_coupon() -> None:
    bond_model = CallableBondPDEModel(
        face_value=100.0,
        call_price=100.0,
        market=Market(rate=0.04),
        model=_short_rate_model(),
        _maturity=2.0,
        coupon_rate=0.06,
        coupon_times=(1.0, 2.0),
        call_schedule=(CallScheduleEntry(time=1.0, price=100.0, quote_convention="clean"),),
    )
    rate_grid = np.array([0.0, 0.04, 0.08])
    tau_grid = np.array([0.0, 1.0, 2.0])

    values = bond_model.price_grid(rate_grid, tau_grid)

    assert np.isclose(values[1, 0], 106.0)
    assert np.all(values[1] <= 106.0)
    assert bond_model.accrued_interest(1.0) == 0.0
    assert bond_model.last_exercise_diagnostics[0].settlement_value == 100.0


def test_option_pricer_dispatches_callable_bond_price_grid() -> None:
    bond_model = _callable_bond(call_price=101.0)
    pricer = OptionPricer(instrument=bond_model)

    res = pricer.compute_grid(
        s_max=0.12,
        s_steps=25,
        t_steps=3,
    )

    assert res.s.shape == (25,)
    assert res.t.tolist() == [0.0, 1.0, 2.0]
    assert res.values.shape == (3, 25)
    assert np.allclose(res.values, bond_model.price_grid(res.s, res.t))
    assert res.delta is None
    assert res.gamma is None
    assert res.theta is None
