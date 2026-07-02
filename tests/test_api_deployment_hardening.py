"""Deployment hardening black-box tests for API issue #101."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import finite_difference_options.api.main as api_main
from finite_difference_options.api.main import (
    API_SCHEMA_VERSION,
    DeploymentSecurityPolicy,
    app,
)


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


@pytest.fixture(autouse=True)
def _isolated_security_policy(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(
        api_main,
        "DEFAULT_API_SECURITY_POLICY",
        DeploymentSecurityPolicy(rate_limit_requests=0),
    )
    api_main._RATE_LIMITER.reset()
    yield
    api_main._RATE_LIMITER.reset()


def _assert_error(body: dict[str, Any], *, code: str, route: str, status: int) -> None:
    assert body["schema_version"] == API_SCHEMA_VERSION
    assert body["error"]["code"] == code
    assert body["error"]["route"] == route
    assert body["error"]["http_status"] == status
    assert body["metadata"]["route"] == route


def _cors_security_test_app() -> FastAPI:
    test_app = FastAPI()

    @test_app.middleware("http")
    async def _security(http_request, call_next):
        return await api_main._apply_deployment_security(http_request, call_next)

    api_main._configure_cors(test_app, DeploymentSecurityPolicy(local_dev_cors=True))

    @test_app.post("/price")
    def _synthetic_price() -> dict[str, str]:
        return {"status": "ok"}

    return test_app


def test_invalid_auth_required_env_value_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FDO_API_AUTH_REQUIRED", "treu")

    with pytest.raises(ValueError, match="FDO_API_AUTH_REQUIRED"):
        DeploymentSecurityPolicy.from_env()


def test_default_cors_policy_rejects_unlisted_browser_origins() -> None:
    response = TestClient(app).options(
        "/price",
        headers={
            "Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_local_development_cors_is_explicit_opt_in() -> None:
    test_app = FastAPI()
    api_main._configure_cors(test_app, DeploymentSecurityPolicy(local_dev_cors=True))

    @test_app.post("/price")
    def _synthetic_price() -> dict[str, str]:
        return {"status": "ok"}

    response = TestClient(test_app).options(
        "/price",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_allowed_browser_origin_can_read_auth_error_envelopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_main,
        "DEFAULT_API_SECURITY_POLICY",
        DeploymentSecurityPolicy(
            local_dev_cors=True,
            auth_required=True,
            api_key="test-secret",
            rate_limit_requests=0,
        ),
    )

    response = TestClient(_cors_security_test_app()).post(
        "/price",
        json=_payload(),
        headers={"Origin": "http://localhost:5173"},
    )

    assert response.status_code == 401
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    _assert_error(response.json(), code="auth_required", route="/price", status=401)


def test_authentication_configuration_fails_closed_without_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_main,
        "DEFAULT_API_SECURITY_POLICY",
        DeploymentSecurityPolicy(
            auth_required=True, api_key=None, rate_limit_requests=0
        ),
    )

    response = TestClient(app).post("/price", json=_payload())

    assert response.status_code == 503
    body = response.json()
    _assert_error(body, code="auth_misconfigured", route="/price", status=503)
    assert "test-secret" not in str(body)


def test_authentication_rejects_missing_and_bad_credentials_then_allows_bearer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_main,
        "DEFAULT_API_SECURITY_POLICY",
        DeploymentSecurityPolicy(
            auth_required=True,
            api_key="test-secret",
            rate_limit_requests=0,
        ),
    )
    client = TestClient(app)

    missing = client.post("/price", json=_payload())
    bad = client.post(
        "/price", json=_payload(), headers={"Authorization": "Bearer wrong"}
    )
    good = client.post(
        "/price", json=_payload(), headers={"Authorization": "Bearer test-secret"}
    )

    assert missing.status_code == 401
    _assert_error(missing.json(), code="auth_required", route="/price", status=401)
    assert bad.status_code == 403
    _assert_error(bad.json(), code="auth_invalid", route="/price", status=403)
    assert good.status_code == 200
    assert "test-secret" not in str(missing.json())
    assert "test-secret" not in str(bad.json())
    assert "test-secret" not in str(good.json())


def test_constant_time_key_compare_handles_non_ascii_values() -> None:
    assert api_main._constant_time_key_equal("clave-ñ", "clave-ñ")
    assert not api_main._constant_time_key_equal("inválido", "test-secret")


def test_process_local_rate_limit_returns_bounded_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_main,
        "DEFAULT_API_SECURITY_POLICY",
        DeploymentSecurityPolicy(rate_limit_requests=1, rate_limit_window_seconds=60.0),
    )
    api_main._RATE_LIMITER.reset()
    client = TestClient(app)

    first = client.post("/price", json=_payload())
    second = client.post("/price", json=_payload())

    assert first.status_code == 200
    assert second.status_code == 429
    body = second.json()
    _assert_error(body, code="rate_limit_exceeded", route="/price", status=429)
    assert body["detail"]["resource"]["limit"] == "rate_limit_requests"
    assert body["detail"]["resource"]["scope"] == "process_local_demo"


def test_failed_authentication_attempts_consume_rate_limit_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_main,
        "DEFAULT_API_SECURITY_POLICY",
        DeploymentSecurityPolicy(
            auth_required=True,
            api_key="test-secret",
            rate_limit_requests=1,
            rate_limit_window_seconds=60.0,
        ),
    )
    api_main._RATE_LIMITER.reset()
    client = TestClient(app)

    first = client.post("/price", json=_payload())
    second = client.post("/price", json=_payload())

    assert first.status_code == 401
    _assert_error(first.json(), code="auth_required", route="/price", status=401)
    assert second.status_code == 429
    _assert_error(second.json(), code="rate_limit_exceeded", route="/price", status=429)


def test_black_box_lifecycle_failures_use_stable_error_envelopes() -> None:
    client = TestClient(app)

    malformed = client.post(
        "/price",
        content="{not-json",
        headers={"content-type": "application/json"},
    )
    oversized = client.post(
        "/pde_solution",
        json=_payload(include_full_grid=True, max_output_nodes=1),
    )
    unsupported = client.post("/reports/crif", json=[])

    assert malformed.status_code == 422
    _assert_error(malformed.json(), code="validation_error", route="/price", status=422)
    assert oversized.status_code == 422
    _assert_error(
        oversized.json(),
        code="resource_limit_exceeded",
        route="/pde_solution",
        status=422,
    )
    assert unsupported.status_code == 501
    _assert_error(
        unsupported.json(), code="unsupported_route", route="/reports/crif", status=501
    )
