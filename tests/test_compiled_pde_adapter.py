"""Compiled pde_ir.v0 adapter regressions for issue #141."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from finite_difference_options.cli.main import app
from finite_difference_options.integrations import compiled_pde_adapter as adapter
from finite_difference_options.integrations.compiled_pde_adapter import (
    CompiledPDEAdapterError,
    load_compiled_pde_json,
    packaged_compiled_black_scholes_fixture,
    packaged_compiled_black_scholes_fixture_path,
    screen_compiled_pde_payload,
    solve_compiled_pde_payload,
)
from finite_difference_options.validation.benchmark_registry import (
    run_registered_benchmark,
)


def _fixture() -> dict[str, Any]:
    with packaged_compiled_black_scholes_fixture_path() as fixture_path:
        return load_compiled_pde_json(fixture_path)


def test_packaged_fixture_path_loads_same_payload() -> None:
    assert _fixture() == packaged_compiled_black_scholes_fixture()


def test_compiled_pde_fixture_screens_and_solves_preserving_identity() -> None:
    payload = _fixture()

    screen = screen_compiled_pde_payload(payload)
    result = solve_compiled_pde_payload(payload)

    assert screen.supported
    assert screen.diagnostics == ()
    assert screen.route["dimension"] == 1
    assert (
        screen.route["source_ir_canonical_hash"]
        == "sha256:5ab53779a5e322284a6cb18b22302c119f22bc740659aedf1c07823529d68a47"
    )
    assert screen.route["compiled_hash"] == "sha256:970088e5dcb16535edfd230bfe992ea7eb68aede901c7b543682b39f1a5ac32e"
    assert screen.route["measure"] == "Q"
    assert screen.route["numeraire"] == {
        "currency": "USD",
        "kind": "money_market_account",
    }
    assert screen.route["time_orientation"] == "backward"
    assert screen.route["boundary_conditions"] == ("asymptotic", "dirichlet")
    assert result.passed
    assert result.problem_id == "public-synthetic.compiled-pde.black-scholes-call.v0"
    assert result.values["price"] == pytest.approx(result.values["oracle_price"], abs=5.0e-4)
    assert result.evidence["source_ir_canonical_hash"] == screen.route["source_ir_canonical_hash"]
    assert result.evidence["compiled_hash"] == screen.route["compiled_hash"]
    assert result.evidence["units"]["value"] == {
        "currency": "USD",
        "dimension": "money",
        "scale": "absolute",
    }
    assert result.diagnostics["fallbacks"] == ()
    schedule = result.evidence["boundary_schedule_applied"]
    assert schedule["source"] == "compiled_route_explicit_schedule"
    assert schedule["tau_grid_count"] == 200
    assert schedule["applied"][0] == {
        "step_index": 0,
        "tau": 0.0,
        "calendar_time": 1.0,
        "lower": 0.0,
        "upper": 2.0,
        "source": "terminal_payoff_boundary",
    }
    assert schedule["applied"][-1]["upper"] == pytest.approx(3.0 - 1.0 * 2.718281828459045**-0.05)
    assert result.diagnostics["resource_controls"]["boundary_rebuilt_each_time_step"] == "true"


def test_compiled_pde_route_does_not_invoke_legacy_fixture_or_mu_boundary_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("compiled PDE route must not call legacy BS fixture or model.mu boundary fallback")

    monkeypatch.setattr(
        "finite_difference_options.validation.black_scholes_parity.run_public_black_scholes_parity_fixture",
        fail_if_called,
    )
    monkeypatch.setattr(
        "finite_difference_options.boundary_conditions.builder.BlackScholesBoundaryBuilder._risk_free_rate",
        fail_if_called,
    )

    result = solve_compiled_pde_payload(_fixture())

    assert result.passed
    assert result.values == pytest.approx(
        {
            "price": 0.10461512422686384,
            "oracle_price": 0.10450583572185568,
            "delta": 0.6359686349407326,
            "reference_delta": 0.6368306511756191,
            "gamma": 1.8711922912959007,
            "reference_gamma": 1.8762017345846895,
        },
        abs=1.0e-14,
    )


def test_compiled_pde_registered_benchmark_executes_real_adapter() -> None:
    result = run_registered_benchmark("VQPW-FD-COMPILED-PDE-BS-CALL-V0")

    assert result.passed
    assert float(result.metrics["price_abs"]) <= 5.0e-4
    assert result.invariants == {
        "source_ir_canonical_hash": True,
        "compiled_operator_hash": True,
        "measure_numeraire_units": True,
        "time_orientation": True,
        "boundary_semantics": True,
        "price_abs_tolerance_ok": True,
    }


@pytest.mark.parametrize(
    ("mutation", "code"),
    (
        (
            lambda payload: payload.__setitem__("private_terms", {"customer_id": "forbidden"}),
            "compiled_pde.unknown_field",
        ),
        (
            lambda payload: payload.__setitem__("privacy_class", "private"),
            "compiled_pde.privacy_unsupported",
        ),
        (
            lambda payload: payload["source_pde_ir"]["state_variables"].append(
                deepcopy(payload["source_pde_ir"]["state_variables"][0])
            ),
            "compiled_pde.dimension_unsupported",
        ),
        (
            lambda payload: payload["source_pde_ir"]["boundary_conditions"][0].__setitem__("kind", "periodic"),
            "compiled_pde.boundary_unsupported",
        ),
        (
            lambda payload: payload["source_pde_ir"]["boundary_conditions"][1].__setitem__("expression", "V = S - K"),
            "compiled_pde.source_hash_mismatch",
        ),
        (
            lambda payload: payload["source_pde_ir"].__setitem__("time_orientation", "forward"),
            "compiled_pde.time_orientation_unsupported",
        ),
        (
            lambda payload: payload["solver_plan"].__setitem__("exercise_style", "american"),
            "compiled_pde.exercise_unsupported",
        ),
        (
            lambda payload: payload["solver_plan"]["requested_outputs"].append("vega"),
            "compiled_pde.output_unsupported",
        ),
        (
            lambda payload: payload["compiled_operator_result"]["compiled_operator"].__setitem__(
                "compiled_hash", "sha256:" + "0" * 64
            ),
            "compiled_pde.compiled_hash_mismatch",
        ),
    ),
)
def test_compiled_pde_mutations_fail_before_numerical_solve(
    mutation: Any, code: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _fixture()
    mutation(payload)

    def fail_if_called(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("numerical solver should not run for unsupported compiled PDE payload")

    monkeypatch.setattr(
        "finite_difference_options.validation.black_scholes_parity.run_public_black_scholes_parity_fixture",
        fail_if_called,
    )

    screen = screen_compiled_pde_payload(payload)
    assert not screen.supported
    assert code in {diagnostic["code"] for diagnostic in screen.diagnostics}
    with pytest.raises(CompiledPDEAdapterError):
        solve_compiled_pde_payload(payload)


def test_compiled_pde_screen_rejects_nested_non_mappings_before_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _fixture()
    payload["source_pde_ir"] = []

    def fail_if_called(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("screen must not build a route for invalid nested objects")

    monkeypatch.setattr(adapter, "_route", fail_if_called)

    screen = screen_compiled_pde_payload(payload)

    assert not screen.supported
    assert screen.route == {}
    assert {(diagnostic["code"], diagnostic["path"], diagnostic["observed"]) for diagnostic in screen.diagnostics} >= {
        ("compiled_pde.object_type", "payload.source_pde_ir", "array")
    }


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        (
            lambda payload: payload["source_pde_ir"].__setitem__("boundary_conditions", "not-an-array"),
            ("compiled_pde.list_type", "source_pde_ir.boundary_conditions", "string"),
        ),
        (
            lambda payload: payload["solver_plan"].__setitem__("requested_outputs", {"value": True}),
            ("compiled_pde.list_type", "solver_plan.requested_outputs", "object"),
        ),
        (
            lambda payload: payload["source_pde_ir"].__setitem__("boundary_conditions", ["not-an-object"]),
            ("compiled_pde.object_type", "source_pde_ir.boundary_conditions[0]", "string"),
        ),
    ),
)
def test_compiled_pde_screen_reports_typed_nested_section_diagnostics(
    mutation: Any,
    expected: tuple[str, str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _fixture()
    mutation(payload)

    def fail_if_called(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("screen must not build a route before validation succeeds")

    monkeypatch.setattr(adapter, "_route", fail_if_called)

    screen = screen_compiled_pde_payload(payload)

    assert not screen.supported
    assert screen.route == {}
    assert expected in {
        (diagnostic["code"], diagnostic["path"], diagnostic["observed"]) for diagnostic in screen.diagnostics
    }


def test_compiled_pde_cli_screen_and_solve_emit_deterministic_json(
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    out_path = tmp_path / "result.json"
    evidence_path = tmp_path / "evidence.json"

    with packaged_compiled_black_scholes_fixture_path() as fixture_path:
        screen = runner.invoke(app, ["qps", "screen", str(fixture_path), "--json"])
        solve = runner.invoke(
            app,
            [
                "qps",
                "solve",
                str(fixture_path),
                "--out",
                str(out_path),
                "--evidence",
                str(evidence_path),
            ],
        )

    assert screen.exit_code == 0, screen.output
    screen_payload = json.loads(screen.output)
    assert screen_payload["supported"] is True
    assert (
        screen_payload["route"]["compiled_hash"]
        == "sha256:970088e5dcb16535edfd230bfe992ea7eb68aede901c7b543682b39f1a5ac32e"
    )
    assert solve.exit_code == 0, solve.output
    result_payload = json.loads(out_path.read_text(encoding="utf-8"))
    evidence_payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert result_payload["status"] == "passed"
    assert result_payload["values"]["price"] == pytest.approx(result_payload["values"]["oracle_price"], abs=5.0e-4)
    assert evidence_payload["compiled_hash"] == result_payload["evidence"]["compiled_hash"]
    assert json.loads(json.dumps(result_payload, sort_keys=True)) == result_payload
