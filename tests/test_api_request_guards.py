"""API request validation and bounded-output tests for issue #53."""
from __future__ import annotations

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError as PydanticValidationError

from api.main import OptionRequest, app
from src.instruments.base import EuropeanCall
from src.pricing import OptionPricer
from src.processes.affine import GeometricBrownianMotion


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "option_type": "Call",
        "spot": 150.0,
        "strike": 100.0,
        "maturity": 0.5,
        "rate": 0.03,
        "sigma": 0.2,
        "s_max": 250.0,
        "s_steps": 101,
        "t_steps": 51,
    }
    payload.update(overrides)
    return payload


def _expected_greeks_at_spot(payload: dict[str, object]) -> tuple[float, float, float]:
    model = GeometricBrownianMotion(mu=float(payload["rate"]), sigma=float(payload["sigma"]))
    option = EuropeanCall(
        strike=float(payload["strike"]),
        maturity=float(payload["maturity"]),
        model=model,
    )
    res = OptionPricer(instrument=option).compute_grid(
        s_max=float(payload["s_max"]),
        s_steps=int(payload["s_steps"]),
        t_steps=int(payload["t_steps"]),
        return_greeks=True,
    )
    spot = float(payload["spot"])
    return (
        float(np.interp(spot, res.s, res.delta[-1])),
        float(np.interp(spot, res.s, res.gamma[-1])),
        float(np.interp(spot, res.s, res.theta[-1])),
    )


def test_price_endpoint_rejects_unknown_option_type_instead_of_defaulting_to_put() -> None:
    client = TestClient(app)

    response = client.post("/price", json=_payload(option_type="Digital"))

    assert response.status_code == 422
    assert "option_type" in str(response.json()["detail"])


def test_price_endpoint_returns_scalar_at_explicit_spot_without_full_grid_by_default() -> None:
    client = TestClient(app)

    response = client.post("/price", json=_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == "fd-api-v1"
    assert body["spot"] == 150.0
    assert math.isfinite(body["price"])
    assert body["grid"] is None
    assert "values" not in body


def test_greeks_endpoint_samples_explicit_spot_not_strike() -> None:
    client = TestClient(app)
    payload = _payload()

    response = client.post("/greeks", json=payload)

    assert response.status_code == 200
    body = response.json()
    expected_delta, expected_gamma, expected_theta = _expected_greeks_at_spot(payload)
    assert body["spot"] == payload["spot"]
    assert body["delta"] == pytest.approx(expected_delta, abs=1e-12)
    assert body["gamma"] == pytest.approx(expected_gamma, abs=1e-12)
    assert body["theta"] == pytest.approx(expected_theta, abs=1e-12)

    strike_delta, _, _ = _expected_greeks_at_spot({**payload, "spot": payload["strike"]})
    assert abs(body["delta"] - strike_delta) > 1e-3


def test_request_budget_rejects_oversized_grids_before_solver_allocation() -> None:
    with pytest.raises(PydanticValidationError, match="node budget"):
        OptionRequest(**_payload(s_steps=1001, t_steps=101))


@pytest.mark.parametrize("field", ["spot", "strike", "maturity", "rate", "sigma", "s_max"])
@pytest.mark.parametrize("bad_value", [math.inf, math.nan])
def test_request_rejects_nonfinite_numeric_fields(field: str, bad_value: float) -> None:
    kwargs = _payload(**{field: bad_value})

    with pytest.raises(PydanticValidationError):
        OptionRequest(**kwargs)


def test_pde_solution_full_grid_requires_explicit_opt_in() -> None:
    client = TestClient(app)

    response = client.post("/pde_solution", json=_payload(s_steps=21, t_steps=21))

    assert response.status_code == 400
    assert "include_full_grid" in response.json()["detail"]


def test_pde_solution_full_grid_is_bounded_when_explicitly_requested() -> None:
    client = TestClient(app)

    response = client.post("/pde_solution", json=_payload(include_full_grid=True, s_steps=21, t_steps=21))

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == "fd-api-v1"
    assert len(body["s"]) == 21
    assert len(body["t"]) == 21
    assert len(body["prices"]) == 21
