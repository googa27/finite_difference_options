"""Runtime resource-control tests for API issue #100."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from fastapi.testclient import TestClient

import finite_difference_options.api.main as api_main
from finite_difference_options.api.main import API_SCHEMA_VERSION, app


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
        "maturity": 0.5,
        "rate": 0.03,
        "sigma": 0.2,
        "s_max": 250.0,
        "s_steps": 21,
        "t_steps": 21,
    }
    payload.update(overrides)
    return payload


@contextmanager
def _acquire_all_solve_slots() -> Iterator[None]:
    acquired = 0
    try:
        for _ in range(api_main.DEFAULT_API_RESOURCE_POLICY.max_concurrent_solves):
            assert api_main._SOLVE_SEMAPHORE.acquire(blocking=False)
            acquired += 1
        yield
    finally:
        for _ in range(acquired):
            api_main._SOLVE_SEMAPHORE.release()


def test_success_metadata_includes_configured_resource_budget() -> None:
    response = TestClient(app).post("/price", json=_payload(timeout_seconds=0.5))

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == API_SCHEMA_VERSION
    budget = body["metadata"]["resource_budget"]
    assert budget["policy_name"] == "local_demo"
    assert budget["state_dimensions"] == 1
    assert budget["compute_nodes"] == 21 * 21
    assert budget["output_nodes"] == 1
    assert budget["estimated_response_bytes"] == 8
    assert budget["timeout_seconds"] == 0.5
    assert (
        budget["max_concurrent_solves"]
        == api_main.DEFAULT_API_RESOURCE_POLICY.max_concurrent_solves
    )


def test_full_grid_price_budget_counts_scalar_price_output() -> None:
    response = TestClient(app).post(
        "/price",
        json=_payload(include_full_grid=True, max_response_bytes=8_000_000),
    )

    assert response.status_code == 200
    body = response.json()
    budget = body["metadata"]["resource_budget"]
    assert budget["output_nodes"] == 1 + 21 + 21 + 21 * 21
    assert budget["estimated_response_bytes"] == budget["output_nodes"] * 8


def test_scalar_output_budget_does_not_constrain_compute_budget() -> None:
    response = TestClient(app).post("/price", json=_payload(max_output_nodes=1))

    assert response.status_code == 200
    body = response.json()
    budget = body["metadata"]["resource_budget"]
    assert budget["compute_nodes"] == 21 * 21
    assert budget["max_compute_nodes"] >= budget["compute_nodes"]
    assert budget["max_output_nodes"] == 1
    assert budget["output_nodes"] == 1


def test_output_response_size_budget_rejects_full_grid_before_solver(
    monkeypatch,
) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> object:
        raise AssertionError("solver should not run after output budget rejection")

    monkeypatch.setattr(api_main, "_compute_grid", fail_if_called)

    response = TestClient(app).post(
        "/price",
        json=_payload(include_full_grid=True, max_response_bytes=128),
    )

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "resource_limit_exceeded"
    assert body["detail"]["resource"]["limit"] == "max_response_bytes"
    assert body["detail"]["resource"]["estimated_response_bytes"] > 128


def test_concurrency_limit_returns_bounded_error_without_running_solver(
    monkeypatch,
) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> object:
        raise AssertionError("solver should not run without a concurrency slot")

    monkeypatch.setattr(api_main, "_compute_grid", fail_if_called)
    client = TestClient(app)

    with _acquire_all_solve_slots():
        response = client.post("/price", json=_payload())

    assert response.status_code == 429
    body = response.json()
    assert body["error"]["code"] == "concurrency_limit_exceeded"
    assert body["detail"]["resource"]["limit"] == "max_concurrent_solves"
    assert body["metadata"]["route"] == "/price"


def test_cooperative_timeout_path_is_deterministic_and_bounded(monkeypatch) -> None:
    ticks = iter([0.0, 0.0, 10.0])
    monkeypatch.setattr(api_main, "_monotonic", lambda: next(ticks, 10.0))

    response = TestClient(app).post(
        "/price",
        json=_payload(timeout_seconds=0.001),
    )

    assert response.status_code == 504
    body = response.json()
    assert body["error"]["code"] == "request_timeout"
    assert body["error"]["source"] == "http_exception"
    assert body["detail"]["stage"] == "after_solve"
    assert body["detail"]["timeout_seconds"] == 0.001


def test_pre_solve_timeout_releases_concurrency_slot(monkeypatch) -> None:
    ticks = iter([0.0, 10.0])
    monkeypatch.setattr(api_main, "_monotonic", lambda: next(ticks, 10.0))

    response = TestClient(app).post(
        "/price",
        json=_payload(timeout_seconds=0.001),
    )

    assert response.status_code == 504
    body = response.json()
    assert body["error"]["code"] == "request_timeout"
    assert body["detail"]["stage"] == "before_solve"

    assert api_main._SOLVE_SEMAPHORE.acquire(blocking=False)
    api_main._SOLVE_SEMAPHORE.release()


def test_dimension_compute_budget_rejects_unsupported_state_dimension_before_solver() -> (
    None
):
    response = TestClient(app).post("/price", json=_payload(state_dimensions=2))

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert "state dimension budget" in str(body["detail"])
