"""Factor-role and payoff compatibility tests for FDO issues #45/#62."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
import importlib

from src.exceptions import ValidationError
from src.pricing import (
    create_linear_grid,
    create_log_grid,
    create_spread_call,
    create_unified_european_call,
    create_unified_pricing_engine,
)
from src.pricing.instruments.options import (
    create_unified_basket_call,
)
from src.processes import create_black_scholes_process, create_cir_process, create_sabr_model, create_standard_heston


process_base = importlib.import_module("src.processes.base")
options_module = importlib.import_module("src.pricing.instruments.options")


def test_heston_factor_metadata_identifies_spot_and_variance() -> None:
    assert hasattr(process_base, "FactorRole")
    factor_role = process_base.FactorRole
    process = create_standard_heston()

    factors = process.factor_metadata()

    assert [factor.role for factor in factors] == [
        factor_role.TRADABLE_SPOT,
        factor_role.VARIANCE,
    ]
    assert [factor.name for factor in factors] == ["spot", "variance"]
    assert factors[0].asset_id == "spot"
    assert factors[1].asset_id is None


def test_process_factor_roles_distinguish_spots_rates_and_volatility() -> None:
    factor_role = process_base.FactorRole

    gbm_roles = [factor.role for factor in create_black_scholes_process(0.05, 0.2).factor_metadata()]
    cir_roles = [factor.role for factor in create_cir_process(2.0, 0.04, 0.2).factor_metadata()]
    sabr_roles = [factor.role for factor in create_sabr_model(0.3, 0.7, -0.2).factor_metadata()]

    assert gbm_roles == [factor_role.TRADABLE_SPOT]
    assert cir_roles == [factor_role.SHORT_RATE]
    assert sabr_roles == [factor_role.TRADABLE_SPOT, factor_role.VOLATILITY]


def test_heston_european_payoff_broadcasts_only_the_spot_factor() -> None:
    process = create_standard_heston()
    engine = create_unified_pricing_engine(process)
    option = create_unified_european_call(100.0, 0.25)
    s_grid = create_log_grid(80.0, 120.0, 7, center=100.0)
    v_grid = create_linear_grid(0.01, 0.25, 5)
    time_grid = np.linspace(0.0, option.maturity, 4)

    prices = engine.price_option(option, s_grid, v_grid, time_grid=time_grid)

    expected_terminal = np.broadcast_to(
        np.maximum(s_grid - option.strike, 0.0).reshape(-1, 1),
        (len(s_grid), len(v_grid)),
    )
    assert prices.shape == (len(time_grid), len(s_grid), len(v_grid))
    assert_allclose(prices[-1], expected_terminal)


def test_basket_payoff_cannot_consume_heston_variance_factor() -> None:
    process = create_standard_heston()
    engine = create_unified_pricing_engine(process)
    option = create_unified_basket_call(
        strikes=np.array([100.0, 100.0]),
        weights=np.array([0.5, 0.5]),
        maturity=0.25,
    )
    s_grid = create_log_grid(80.0, 120.0, 7, center=100.0)
    v_grid = create_linear_grid(0.01, 0.25, 5)

    with pytest.raises(ValidationError, match="tradable spot.*variance"):
        engine.price_option(option, s_grid, v_grid, time_grid=np.linspace(0.0, 0.25, 4))


def test_standard_basket_has_one_strike_and_hand_calculated_payoff() -> None:
    assert hasattr(options_module, "StandardBasketOption")
    assert hasattr(options_module, "create_standard_basket_call")
    standard_basket_option = options_module.StandardBasketOption
    create_standard_basket_call = options_module.create_standard_basket_call

    option = create_standard_basket_call(
        strike=104.0,
        weights=np.array([0.6, 0.4]),
        maturity=1.0,
        asset_ids=("equity_a", "equity_b"),
    )

    assert isinstance(option, standard_basket_option)
    assert option.product_type == "standard_basket"
    assert option.strike == 104.0
    assert option.asset_ids == ("equity_a", "equity_b")

    s1_grid = np.array([90.0, 100.0, 110.0])
    s2_grid = np.array([100.0, 120.0])

    payoff = option.payoff(s1_grid, s2_grid)

    expected = np.array([[0.0, 0.0], [0.0, 4.0], [2.0, 10.0]])
    assert_allclose(payoff, expected)


def test_standard_basket_preserves_tensor_grid_for_equal_length_axes() -> None:
    create_standard_basket_call = options_module.create_standard_basket_call
    option = create_standard_basket_call(
        strike=100.0,
        weights=np.array([0.5, 0.5]),
        maturity=1.0,
    )

    payoff = option.payoff(np.array([90.0, 110.0]), np.array([100.0, 120.0]))

    expected = np.array([[0.0, 5.0], [5.0, 15.0]])
    assert payoff.shape == (2, 2)
    assert_allclose(payoff, expected)


def test_standard_basket_one_leg_broadcasts_over_heston_variance() -> None:
    process = create_standard_heston()
    engine = create_unified_pricing_engine(process)
    create_standard_basket_call = options_module.create_standard_basket_call
    option = create_standard_basket_call(strike=100.0, weights=np.array([1.0]), maturity=0.25)
    s_grid = create_log_grid(80.0, 120.0, 7, center=100.0)
    v_grid = create_linear_grid(0.01, 0.25, 5)
    time_grid = np.linspace(0.0, option.maturity, 4)

    prices = engine.price_option(option, s_grid, v_grid, time_grid=time_grid)

    expected_terminal = np.broadcast_to(
        np.maximum(s_grid - option.strike, 0.0).reshape(-1, 1),
        (len(s_grid), len(v_grid)),
    )
    assert prices.shape == (len(time_grid), len(s_grid), len(v_grid))
    assert_allclose(prices[-1], expected_terminal)


def test_asset_id_mapping_fails_closed_before_pricing() -> None:
    process = create_black_scholes_process(0.05, 0.2)
    engine = create_unified_pricing_engine(process)
    create_standard_basket_call = options_module.create_standard_basket_call
    option = create_standard_basket_call(
        strike=100.0,
        weights=np.array([1.0]),
        maturity=0.25,
        asset_ids=("not_the_process_spot",),
    )
    grid = create_log_grid(80.0, 120.0, 7, center=100.0)

    with pytest.raises(ValidationError, match="asset_id"):
        engine.price_option(option, grid, time_grid=np.linspace(0.0, 0.25, 4))


def test_spread_option_has_separate_identity_and_non_normalized_coefficients() -> None:
    option = create_spread_call(
        strike=2.0,
        weights=np.array([1.0, -0.75]),
        maturity=1.0,
        asset_ids=("equity_a", "equity_b"),
    )

    assert option.product_type == "spread"
    payoff = option.payoff(np.array([100.0, 110.0]), np.array([90.0, 100.0, 120.0]))
    expected = np.array([[30.5, 23.0, 8.0], [40.5, 33.0, 18.0]])
    assert_allclose(payoff, expected)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"strike": np.nan, "weights": np.array([1.0]), "maturity": 1.0}, "finite"),
        ({"strike": np.inf, "weights": np.array([1.0]), "maturity": 1.0}, "finite"),
        ({"strike": 100.0, "weights": np.array([1.0]), "maturity": np.nan}, "finite"),
        ({"strike": 100.0, "weights": np.array([1.0]), "maturity": np.inf}, "finite"),
        ({"strike": 100.0, "weights": np.array([]), "maturity": 1.0}, "nonempty"),
        ({"strike": 100.0, "weights": np.array([0.5, np.nan]), "maturity": 1.0}, "finite"),
        ({"strike": 100.0, "weights": np.array([[0.5, 0.5]]), "maturity": 1.0}, "one-dimensional"),
        ({"strike": 100.0, "weights": np.array([0.5, 0.25]), "maturity": 1.0}, "sum"),
        (
            {
                "strike": 100.0,
                "weights": np.array([0.5, 0.5]),
                "maturity": 1.0,
                "basket_currency": "USD",
                "asset_currencies": ("USD", "CLP"),
            },
            "cross-currency",
        ),
        (
            {"strike": 100.0, "weights": np.array([0.5, 0.5]), "maturity": 1.0, "asset_ids": ("a",)},
            "asset_ids",
        ),
    ],
)
def test_standard_basket_validation_fails_before_payoff_allocation(kwargs: dict, match: str) -> None:
    assert hasattr(options_module, "StandardBasketOption")
    standard_basket_option = options_module.StandardBasketOption

    with pytest.raises(ValidationError, match=match):
        standard_basket_option(**kwargs)
