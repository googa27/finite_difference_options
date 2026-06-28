"""Requested-state interpolation diagnostics tests for API issue #99."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import API_SCHEMA_VERSION, app
from src.instruments.base import EuropeanCall, EuropeanPut
from src.pricing import OptionPricer
from src.processes.affine import GeometricBrownianMotion


def _payload(**overrides: object) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": "black_scholes",
        "process": "geometric_brownian_motion",
        "payoff_family": "vanilla_european",
        "exercise_style": "european",
        "underlying_factor_role": "tradable_spot",
        "option_type": "Call",
        "spot": 137.5,
        "strike": 100.0,
        "maturity": 0.5,
        "rate": 0.03,
        "sigma": 0.2,
        "s_max": 250.0,
        "s_steps": 31,
        "t_steps": 21,
    }
    payload.update(overrides)
    return payload


def _expected_samples(payload: dict[str, object]) -> dict[str, float]:
    model = GeometricBrownianMotion(
        mu=float(payload["rate"]), sigma=float(payload["sigma"])
    )
    instrument_cls = EuropeanCall if payload["option_type"] == "Call" else EuropeanPut
    instrument = instrument_cls(
        strike=float(payload["strike"]),
        maturity=float(payload["maturity"]),
        model=model,
    )
    res = OptionPricer(instrument=instrument).compute_grid(
        s_max=float(payload["s_max"]),
        s_steps=int(payload["s_steps"]),
        t_steps=int(payload["t_steps"]),
        return_greeks=True,
    )
    assert res.delta is not None and res.gamma is not None and res.theta is not None
    spot = float(payload["spot"])
    strike = float(payload["strike"])
    return {
        "price_at_spot": float(np.interp(spot, res.s, res.values[-1])),
        "price_at_strike": float(np.interp(strike, res.s, res.values[-1])),
        "delta_at_spot": float(np.interp(spot, res.s, res.delta[-1])),
        "gamma_at_spot": float(np.interp(spot, res.s, res.gamma[-1])),
        "theta_at_spot": float(np.interp(spot, res.s, res.theta[-1])),
    }


def _sampling(body: dict[str, Any]) -> dict[str, Any]:
    assert body["schema_version"] == API_SCHEMA_VERSION
    sampling = body["metadata"]["sampling"]
    assert isinstance(sampling, dict)
    return sampling


@pytest.mark.parametrize("option_type", ["Call", "Put"])
def test_price_and_greeks_sample_the_same_requested_spot_for_calls_and_puts(
    option_type: str,
) -> None:
    payload = _payload(option_type=option_type, spot=137.5, strike=100.0)
    client = TestClient(app)

    price = client.post("/price", json=payload)
    greeks = client.post("/greeks", json=payload)

    assert price.status_code == 200
    assert greeks.status_code == 200
    expected = _expected_samples(payload)
    price_body = price.json()
    greeks_body = greeks.json()
    assert price_body["price"] == pytest.approx(expected["price_at_spot"], abs=1e-12)
    assert price_body["price"] != pytest.approx(expected["price_at_strike"], abs=1e-3)
    assert greeks_body["delta"] == pytest.approx(expected["delta_at_spot"], abs=1e-12)
    assert greeks_body["gamma"] == pytest.approx(expected["gamma_at_spot"], abs=1e-12)
    assert greeks_body["theta"] == pytest.approx(expected["theta_at_spot"], abs=1e-12)
    assert _sampling(price_body) == _sampling(greeks_body)


def test_sampling_metadata_documents_linear_interpolation_location() -> None:
    response = TestClient(app).post(
        "/price", json=_payload(spot=137.5, s_max=250.0, s_steps=31)
    )

    assert response.status_code == 200
    sampling = _sampling(response.json())
    assert sampling["method"] == "linear_interpolation"
    assert sampling["bounded_policy"] == "reject_outside_grid"
    assert sampling["requested_spot"] == 137.5
    assert sampling["lower_index"] == 16
    assert sampling["upper_index"] == 17
    assert sampling["lower_spot"] == pytest.approx(250.0 * 16 / 30)
    assert sampling["upper_spot"] == pytest.approx(250.0 * 17 / 30)
    assert sampling["interpolation_weight"] == pytest.approx(0.5)
    assert sampling["extrapolated"] is False


def test_sampling_metadata_documents_exact_grid_node_location() -> None:
    response = TestClient(app).post(
        "/price", json=_payload(spot=125.0, s_max=250.0, s_steps=11)
    )

    assert response.status_code == 200
    sampling = _sampling(response.json())
    assert sampling["method"] == "grid_node"
    assert sampling["requested_spot"] == 125.0
    assert sampling["lower_index"] == sampling["upper_index"] == 5
    assert sampling["lower_spot"] == sampling["upper_spot"] == 125.0
    assert sampling["interpolation_weight"] == 0.0
    assert sampling["extrapolated"] is False


def test_requested_spot_outside_grid_is_rejected_by_bounded_policy_before_solving() -> (
    None
):
    response = TestClient(app).post("/price", json=_payload(spot=260.0, s_max=250.0))

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert body["metadata"]["route_maturity"] == "experimental"
    assert "spot must lie inside the spatial grid" in str(body["detail"])
