"""Stable API schema and OpenAPI contract tests for issue #97."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

import api.main as api_main
from api.main import API_SCHEMA_VERSION, app


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
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


def _assert_base_success_contract(body: dict[str, Any], route: str) -> None:
    assert body["schema_version"] == API_SCHEMA_VERSION
    assert isinstance(body["request_id"], str) and body["request_id"]
    assert isinstance(body["run_id"], str) and body["run_id"].startswith("fd-run-")
    metadata = body["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["route"] == route
    assert metadata["route_maturity"] == "experimental"
    assert metadata["warnings"]
    assert metadata["units"]["time"] == "years"
    assert metadata["units"]["rate"] == "annual_decimal"
    assert metadata["solver"]["engine"] == "OptionPricer"
    assert metadata["solver"]["grid_nodes"] == 31 * 21
    assert metadata["convergence"]["status"] == "not_assessed"


def _assert_error_envelope(
    body: dict[str, Any], *, code: str, route: str, status: int
) -> None:
    assert body["schema_version"] == API_SCHEMA_VERSION
    assert isinstance(body["request_id"], str) and body["request_id"]
    assert isinstance(body["run_id"], str) and body["run_id"].startswith("fd-run-")
    error = body["error"]
    assert error["code"] == code
    assert error["route"] == route
    assert error["http_status"] == status
    assert body["metadata"]["route"] == route


def test_successful_price_greeks_and_pde_routes_share_versioned_metadata_contract() -> (
    None
):
    client = TestClient(app)

    price = client.post(
        "/price", json=_payload(), headers={"X-Request-ID": "req-schema-97"}
    )
    greeks = client.post("/greeks", json=_payload())
    pde = client.post("/pde_solution", json=_payload(include_full_grid=True))

    assert price.status_code == 200
    assert price.json()["request_id"] == "req-schema-97"
    _assert_base_success_contract(price.json(), "/price")

    assert greeks.status_code == 200
    _assert_base_success_contract(greeks.json(), "/greeks")
    assert {"delta", "gamma", "theta"}.issubset(greeks.json())

    assert pde.status_code == 200
    _assert_base_success_contract(pde.json(), "/pde_solution")
    assert {"prices", "delta", "gamma", "theta"}.issubset(pde.json())


def test_validation_errors_are_machine_readable_and_preserve_stable_ids() -> None:
    client = TestClient(app)

    response = client.post(
        "/price",
        json=_payload(option_type="Digital"),
        headers={"X-Request-ID": "bad-option-type"},
    )

    assert response.status_code == 422
    body = response.json()
    _assert_error_envelope(body, code="validation_error", route="/price", status=422)
    assert body["request_id"] == "bad-option-type"
    assert "option_type" in str(body["detail"])
    assert body["metadata"]["route_maturity"] == "experimental"


def test_unhandled_internal_errors_are_machine_readable_without_leaking_tracebacks(
    monkeypatch,
) -> None:
    client = TestClient(app, raise_server_exceptions=False)

    def fail_solver(*args: object, **kwargs: object) -> object:
        raise RuntimeError("synthetic solver failure with private internals")

    monkeypatch.setattr(api_main, "_compute_grid", fail_solver)
    response = client.post(
        "/price",
        json=_payload(),
        headers={"X-Request-ID": "internal-failure"},
    )

    assert response.status_code == 500
    body = response.json()
    _assert_error_envelope(body, code="internal_error", route="/price", status=500)
    assert body["request_id"] == "internal-failure"
    assert body["error"]["message"] == "Internal server error"
    assert body["detail"] == {"exception_type": "RuntimeError"}
    assert "private internals" not in str(body)


def test_unsupported_and_bad_request_routes_use_the_same_error_envelope() -> None:
    client = TestClient(app)

    bad_request = client.post("/pde_solution", json=_payload())
    unsupported = client.post("/reports/crif", json=[])

    assert bad_request.status_code == 400
    bad_body = bad_request.json()
    _assert_error_envelope(
        bad_body,
        code="bad_request",
        route="/pde_solution",
        status=400,
    )
    assert "include_full_grid" in bad_body["error"]["message"]

    assert unsupported.status_code == 501
    unsupported_body = unsupported.json()
    _assert_error_envelope(
        unsupported_body,
        code="unsupported_route",
        route="/reports/crif",
        status=501,
    )
    assert unsupported_body["detail"]["capability_status"] in {
        "scaffold",
        "unsupported",
    }


def test_openapi_schema_contains_reviewed_v1_response_error_components() -> None:
    schema = TestClient(app).get("/openapi.json").json()
    components = schema["components"]["schemas"]

    required_components = {
        "PriceResponse",
        "GreeksResponse",
        "FullPDEResponse",
        "ErrorResponse",
        "ErrorBody",
        "ResponseMetadata",
        "SolverMetadata",
        "ConvergenceDiagnostics",
        "UnitMetadata",
        "RouteWarning",
    }
    assert required_components.issubset(components)

    for name in ("PriceResponse", "GreeksResponse", "FullPDEResponse"):
        properties = components[name]["properties"]
        assert {"schema_version", "request_id", "run_id", "metadata"}.issubset(
            properties
        )

    error_properties = components["ErrorResponse"]["properties"]
    assert {
        "schema_version",
        "request_id",
        "run_id",
        "metadata",
        "error",
        "detail",
    }.issubset(error_properties)

    success_refs = {
        "/price": "PriceResponse",
        "/greeks": "GreeksResponse",
        "/pde_solution": "FullPDEResponse",
    }
    for path, model_name in success_refs.items():
        response_schema = schema["paths"][path]["post"]["responses"]["200"]["content"][
            "application/json"
        ]["schema"]
        assert response_schema["$ref"].endswith(f"/{model_name}")

    for path in [
        "/price",
        "/greeks",
        "/pde_solution",
        "/reports/crif",
        "/reports/cuso",
        "/reports/basel",
        "/reports/frtb",
    ]:
        operation = schema["paths"][path]["post"]
        assert "422" in operation["responses"]
        assert operation["responses"]["422"]["content"]["application/json"]["schema"][
            "$ref"
        ].endswith("/ErrorResponse")

    for path in ["/reports/crif", "/reports/cuso", "/reports/basel", "/reports/frtb"]:
        assert schema["paths"][path]["post"]["responses"]["501"]["content"][
            "application/json"
        ]["schema"]["$ref"].endswith("/ErrorResponse")
