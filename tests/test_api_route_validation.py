"""API model/payoff/process fail-closed validation tests for issue #98."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import API_SCHEMA_VERSION, app


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": "black_scholes",
        "process": "geometric_brownian_motion",
        "payoff_family": "vanilla_european",
        "exercise_style": "european",
        "underlying_factor_role": "tradable_spot",
        "option_type": "Call",
        "spot": 100.0,
        "strike": 100.0,
        "maturity": 0.25,
        "rate": 0.03,
        "sigma": 0.2,
        "s_max": 250.0,
        "s_steps": 31,
        "t_steps": 21,
    }
    payload.update(overrides)
    return payload


def _assert_unsupported(
    response, *, route: str, reason_fragment: str
) -> dict[str, object]:
    assert response.status_code == 501
    body = response.json()
    assert body["schema_version"] == API_SCHEMA_VERSION
    assert body["error"]["code"] == "unsupported_route"
    assert body["error"]["route"] == route
    assert body["metadata"]["route_maturity"] == "unsupported"
    assert body["detail"]["capability_status"] == "unsupported"
    assert reason_fragment in body["detail"]["reason"]
    assert body["detail"]["supported_contract"]["model"] == "black_scholes"
    return body


def test_supported_black_scholes_vanilla_contract_is_explicit_and_still_prices() -> (
    None
):
    response = TestClient(app).post("/price", json=_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == API_SCHEMA_VERSION
    solver = body["metadata"]["solver"]
    assert solver["model"] == "black_scholes"
    assert solver["process"] == "geometric_brownian_motion"
    assert solver["payoff_family"] == "vanilla_european"
    assert solver["exercise_style"] == "european"


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("model", "merton_jump_diffusion"),
        ("process", "unknown_process"),
        ("payoff_family", "quanto_cliquet"),
        ("exercise_style", "bermudan"),
        ("underlying_factor_role", "inflation_index"),
    ],
)
def test_unknown_model_payoff_process_enum_values_are_validation_errors(
    field: str,
    bad_value: str,
) -> None:
    response = TestClient(app).post("/price", json=_payload(**{field: bad_value}))

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert field in str(body["detail"])


@pytest.mark.parametrize(
    ("overrides", "reason_fragment"),
    [
        ({"model": "heston", "process": "heston"}, "model heston is not enabled"),
        ({"process": "heston"}, "process heston is incompatible"),
        ({"payoff_family": "basket"}, "payoff family basket is incompatible"),
        ({"payoff_family": "digital"}, "payoff family digital is incompatible"),
        ({"exercise_style": "american"}, "exercise style american is incompatible"),
        (
            {"underlying_factor_role": "variance"},
            "factor role variance is incompatible",
        ),
    ],
)
def test_known_but_unsupported_route_contracts_fail_closed_before_pricing(
    overrides: dict[str, object],
    reason_fragment: str,
) -> None:
    response = TestClient(app).post("/price", json=_payload(**overrides))

    body = _assert_unsupported(
        response, route="/price", reason_fragment=reason_fragment
    )
    assert (
        body["detail"]["requested_contract"]["model"] == _payload(**overrides)["model"]
    )


def test_unsupported_contracts_fail_closed_on_greeks_and_full_pde_routes() -> None:
    client = TestClient(app)

    greeks = client.post("/greeks", json=_payload(model="heston", process="heston"))
    pde = client.post(
        "/pde_solution",
        json=_payload(model="heston", process="heston", include_full_grid=True),
    )

    _assert_unsupported(
        greeks, route="/greeks", reason_fragment="model heston is not enabled"
    )
    _assert_unsupported(
        pde, route="/pde_solution", reason_fragment="model heston is not enabled"
    )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("correlation", 1.5),
        ("correlation", -1.5),
        ("variance", -0.01),
        ("long_run_variance", -0.01),
        ("mean_reversion", 0.0),
        ("vol_of_vol", -0.01),
    ],
)
def test_model_specific_scalar_constraints_are_validated(
    field: str, bad_value: float
) -> None:
    response = TestClient(app).post(
        "/price", json=_payload(model="heston", **{field: bad_value})
    )

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert field in str(body["detail"])


def test_supported_black_scholes_route_rejects_unused_heston_parameters() -> None:
    response = TestClient(app).post("/price", json=_payload(variance=0.04))

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert "black_scholes route does not accept model-specific fields" in str(
        body["detail"]
    )


def test_openapi_exposes_explicit_model_payoff_process_enums() -> None:
    schema = TestClient(app).get("/openapi.json").json()
    components = schema["components"]["schemas"]

    for enum_name in [
        "PricingModel",
        "ProcessType",
        "PayoffFamily",
        "ExerciseStyle",
        "FactorRole",
    ]:
        assert enum_name in components

    option_request = components["OptionRequest"]["properties"]
    assert option_request["model"]["$ref"].endswith("/PricingModel")
    assert option_request["process"]["$ref"].endswith("/ProcessType")
    assert option_request["payoff_family"]["$ref"].endswith("/PayoffFamily")
    assert option_request["exercise_style"]["$ref"].endswith("/ExerciseStyle")
    assert option_request["underlying_factor_role"]["$ref"].endswith("/FactorRole")
