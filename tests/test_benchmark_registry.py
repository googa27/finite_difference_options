"""Executable tests for the FD benchmark registry (#49)."""

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import replace

import pytest

from finite_difference_options.validation.benchmark_registry import (
    BenchmarkCase,
    BenchmarkRegistryError,
    registry_as_dict,
    registry_by_id,
    run_registered_benchmark,
    validate_benchmark_registry,
    write_benchmark_result_json,
    write_registry_json,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
CAPABILITY_MATRIX = ROOT / "docs" / "CAPABILITY_MATRIX.md"
STATIC_REGISTRY = ROOT / "tests" / "fixtures" / "fd_benchmark_registry_v1.json"
EVIDENCE_ID_RE = re.compile(r"`([A-Z0-9][A-Z0-9-]+-V\d+)`")


def _capability_matrix_evidence_ids() -> set[str]:
    text = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    ids: set[str] = set()
    for line in text.splitlines():
        if not line.startswith("| ") or "Evidence / benchmark ID" in line:
            continue
        ids.update(EVIDENCE_ID_RE.findall(line))
    return ids


def test_default_benchmark_registry_is_valid_and_versioned() -> None:
    registry = validate_benchmark_registry()
    ids = [case.benchmark_id for case in registry]

    assert len(ids) == len(set(ids))
    assert "BS-CALL-PARITY-V0" in ids
    assert "ADI-OPERATOR-SPLIT-V0" in ids
    assert all(case.benchmark_id.endswith("-V0") for case in registry)
    assert all(case.issue_refs for case in registry)


def test_capability_matrix_evidence_ids_are_registry_rows() -> None:
    matrix_ids = _capability_matrix_evidence_ids()
    registry_ids = set(registry_by_id())

    assert matrix_ids
    assert matrix_ids <= registry_ids


def test_registry_rows_reference_existing_fixture_paths() -> None:
    for case in validate_benchmark_registry():
        for fixture_path in case.fixture_paths:
            assert (
                ROOT / fixture_path
            ).exists(), f"{case.benchmark_id} references missing fixture {fixture_path}"


def test_static_registry_fixture_matches_generated_payload(
    tmp_path: pathlib.Path,
) -> None:
    generated = json.loads(json.dumps(registry_as_dict()))
    cached = json.loads(STATIC_REGISTRY.read_text(encoding="utf-8"))

    assert cached == generated

    written = tmp_path / "registry.json"
    write_registry_json(written)
    assert json.loads(written.read_text(encoding="utf-8")) == generated


def test_black_scholes_registered_benchmark_executes_real_runner(
    tmp_path: pathlib.Path,
) -> None:
    artifact = tmp_path / "bs_result.json"
    result = run_registered_benchmark("BS-CALL-PARITY-V0", artifact_path=artifact)

    assert result.passed
    assert result.metrics["price_abs"] <= 5.0e-4
    assert result.metrics["delta_abs"] <= 5.0e-2
    assert result.metrics["gamma_abs"] <= 2.0e-2
    assert result.invariants["price_abs_tolerance_ok"]
    assert result.invariants["delta_abs_tolerance_ok"]
    assert result.invariants["gamma_abs_tolerance_ok"]
    assert result.evidence["fixture_id"] == "public-synthetic.black-scholes-call.v0"
    assert result.evidence["route_id"] == "fd.black_scholes_1d.crank_nicolson"
    assert all(result.invariants.values())
    assert json.loads(artifact.read_text(encoding="utf-8")) == json.loads(
        json.dumps(result.as_dict())
    )

    explicit_artifact = tmp_path / "explicit_result.json"
    write_benchmark_result_json(explicit_artifact, result)
    assert json.loads(explicit_artifact.read_text(encoding="utf-8")) == json.loads(
        json.dumps(result.as_dict())
    )


def test_black_scholes_runner_fails_when_declared_greek_tolerance_is_violated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from finite_difference_options.validation import black_scholes_parity

    report = black_scholes_parity.run_public_black_scholes_parity_fixture()
    bad_report = replace(
        report,
        errors={
            **report.errors,
            "delta_abs": 10.0
            * registry_by_id()["BS-CALL-PARITY-V0"].tolerances[1].threshold,
        },
    )
    monkeypatch.setattr(
        black_scholes_parity,
        "run_public_black_scholes_parity_fixture",
        lambda: bad_report,
    )

    result = run_registered_benchmark("BS-CALL-PARITY-V0")

    assert not result.passed
    assert result.invariants["price_abs_tolerance_ok"]
    assert result.invariants["delta_abs_tolerance_ok"] is False
    assert result.invariants["gamma_abs_tolerance_ok"]


def test_black_scholes_runner_fails_when_declared_invariant_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from finite_difference_options.validation import black_scholes_parity

    report = black_scholes_parity.run_public_black_scholes_parity_fixture()
    incomplete_report = replace(
        report,
        no_arbitrage={
            key: value
            for key, value in report.no_arbitrage.items()
            if key != "gamma_non_negative_ok"
        },
    )
    monkeypatch.setattr(
        black_scholes_parity,
        "run_public_black_scholes_parity_fixture",
        lambda: incomplete_report,
    )

    result = run_registered_benchmark("BS-CALL-PARITY-V0")

    assert not result.passed
    assert result.invariants["gamma_non_negative_ok"] is False


def test_qps_runner_fails_when_declared_price_tolerance_is_violated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from finite_difference_options.validation import black_scholes_parity

    report = black_scholes_parity.run_public_black_scholes_parity_fixture()
    bad_final_row = replace(report.observations[-1], abs_error=1.0)
    bad_report = replace(
        report,
        observations=(*report.observations[:-1], bad_final_row),
        errors={**report.errors, "price_abs": 1.0},
    )
    monkeypatch.setattr(
        black_scholes_parity,
        "run_public_black_scholes_parity_fixture",
        lambda: bad_report,
    )

    result = run_registered_benchmark("QPS-VANILLA-CALL-V0")

    assert not result.passed
    assert result.invariants["price_abs_tolerance_ok"] is False


def test_validated_route_parity_benchmarks_execute_real_runners() -> None:
    qps = run_registered_benchmark("QPS-VANILLA-CALL-V0")
    heston_limit = run_registered_benchmark("HESTON-BS-LIMIT-V0")

    assert qps.passed
    assert qps.invariants == {
        "schema_version": True,
        "problem_hash": True,
        "typed_boundary": True,
        "calendar_time_orientation": True,
        "price_abs_tolerance_ok": True,
    }
    assert qps.metrics["typed_boundary_count"] == 2

    assert heston_limit.passed
    assert heston_limit.invariants["limit_price_matches_black_scholes"]
    assert float(heston_limit.metrics["price_abs"]) <= float(
        heston_limit.metrics["threshold"]
    )


def test_metadata_only_benchmarks_fail_closed_when_run_directly() -> None:
    with pytest.raises(BenchmarkRegistryError) as excinfo:
        run_registered_benchmark("ADI-OPERATOR-SPLIT-V0")

    assert "has no executable runner" in str(excinfo.value)


def test_registry_validation_rejects_duplicate_oracleless_validated_case() -> None:
    valid = registry_by_id()["BS-CALL-PARITY-V0"]
    invalid = BenchmarkCase(
        benchmark_id=valid.benchmark_id,
        title="invalid duplicate",
        family="route_parity",
        status="validated",
        route_id="fd.invalid",
        model="invalid",
        instrument="invalid",
        state_convention="invalid",
        grid_family="invalid",
        time_schedule="invalid",
        oracle=valid.oracle.__class__(
            kind="none", source="none", independence="none", notes="none"
        ),
        tolerances=(),
        invariants=("unexecuted_parity",),
    )

    with pytest.raises(BenchmarkRegistryError) as excinfo:
        validate_benchmark_registry((valid, invalid))

    errors = "\n".join(excinfo.value.errors)
    assert "duplicate benchmark_id" in errors
    assert "claims validated without an oracle" in errors
    assert "validated route-parity benchmark" in errors
