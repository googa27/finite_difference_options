"""Compiled pde_ir.v0 adapter regressions for issue #141."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from finite_difference_options.cli.main import app
from finite_difference_options.integrations.compiled_pde_adapter import (
    CompiledPDEAdapterError,
    load_compiled_pde_json,
    packaged_compiled_black_scholes_fixture,
    screen_compiled_pde_payload,
    solve_compiled_pde_payload,
)
from finite_difference_options.validation.benchmark_registry import run_registered_benchmark

FIXTURE_PATH = Path("tests/fixtures/compiled_pde/black_scholes_call_v0.json")


def _fixture() -> dict[str, Any]:
    return load_compiled_pde_json(FIXTURE_PATH)


def test_packaged_and_public_compiled_pde_fixtures_are_identical() -> None:
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
    assert screen.route["numeraire"] == {"currency": "USD", "kind": "money_market_account"}
    assert screen.route["time_orientation"] == "backward"
    assert screen.route["boundary_conditions"] == ("asymptotic", "dirichlet")
    assert result.passed
    assert result.problem_id == "public-synthetic.compiled-pde.black-scholes-call.v0"
    assert result.values["price"] == pytest.approx(result.values["oracle_price"], abs=5.0e-4)
    assert result.evidence["source_ir_canonical_hash"] == screen.route["source_ir_canonical_hash"]
    assert result.evidence["compiled_hash"] == screen.route["compiled_hash"]
    assert result.evidence["units"]["value"] == {"currency": "USD", "dimension": "money", "scale": "absolute"}
    assert result.diagnostics["fallbacks"] == ()


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
        (lambda payload: payload.__setitem__("privacy_class", "private"), "compiled_pde.privacy_unsupported"),
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
            lambda payload: payload["source_pde_ir"].__setitem__("time_orientation", "forward"),
            "compiled_pde.time_orientation_unsupported",
        ),
        (
            lambda payload: payload["solver_plan"].__setitem__("exercise_style", "american"),
            "compiled_pde.exercise_unsupported",
        ),
        (lambda payload: payload["solver_plan"]["requested_outputs"].append("vega"), "compiled_pde.output_unsupported"),
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


def test_compiled_pde_cli_screen_and_solve_emit_deterministic_json(tmp_path: Path) -> None:
    runner = CliRunner()
    out_path = tmp_path / "result.json"
    evidence_path = tmp_path / "evidence.json"

    screen = runner.invoke(app, ["qps", "screen", str(FIXTURE_PATH), "--json"])
    solve = runner.invoke(
        app,
        [
            "qps",
            "solve",
            str(FIXTURE_PATH),
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
