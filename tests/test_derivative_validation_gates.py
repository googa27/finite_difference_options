"""Derivative convergence and stability gates for Issue #58."""

from __future__ import annotations

import json
import pathlib

from finite_difference_options.validation.benchmark_registry import run_registered_benchmark
from finite_difference_options.validation.greek_derivative_gates import (
    GREEK_DERIVATIVE_VALIDATION_BENCHMARK_ID,
    run_greek_derivative_validation,
    write_greek_derivative_validation_artifact,
)


def test_greek_derivative_validation_matrix_passes_pr_gate() -> None:
    report = run_greek_derivative_validation(mode="pr")

    assert report.passed
    assert report.benchmark_id == GREEK_DERIVATIVE_VALIDATION_BENCHMARK_ID
    assert len(report.matrix) >= 12
    assert report.metrics["max_delta_abs_error"] <= report.thresholds.max_delta_abs_error
    assert report.metrics["max_gamma_abs_error"] <= report.thresholds.max_gamma_abs_error
    assert report.metrics["max_finest_to_middle_error_ratio"] <= report.thresholds.max_finest_to_middle_error_ratio
    assert report.metrics["strike_alignment_delta_diff_abs"] <= report.thresholds.strike_alignment_delta_diff_abs
    assert report.metrics["strike_alignment_gamma_diff_abs"] <= report.thresholds.strike_alignment_gamma_diff_abs
    assert report.metrics["rannacher_gamma_roughness_ratio"] <= report.thresholds.rannacher_gamma_roughness_ratio
    assert report.invariants["expiry_kink_rejected"]
    assert report.invariants["runtime_recorded"]

    first_case = report.matrix[0]
    assert first_case["grid_family"] == "strike_centered_nonuniform"
    assert first_case["finest"]["delta"]["reference_abs_error"] < first_case["middle"]["delta"]["reference_abs_error"]
    assert first_case["finest"]["gamma"]["reference_abs_error"] < first_case["middle"]["gamma"]["reference_abs_error"]
    assert first_case["finest"]["delta"]["diagnostics"]["coordinate_spacing"] == "nonuniform"


def test_greek_derivative_validation_artifact_is_serializable(tmp_path: pathlib.Path) -> None:
    report = run_greek_derivative_validation(mode="pr")
    artifact = tmp_path / "fd-greek-derivative-validation.json"

    write_greek_derivative_validation_artifact(artifact, report)

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "finite-difference-greek-validation/v0"
    assert payload["benchmark_id"] == GREEK_DERIVATIVE_VALIDATION_BENCHMARK_ID
    assert payload["passed"] is True
    assert payload["metrics"] == report.metrics
    assert payload["artifact_kind"] == "pr-fast-matrix"
    assert len(payload["matrix"]) == len(report.matrix)
    assert payload["runtime_seconds"] >= 0.0


def test_registered_derivative_validation_benchmark_executes_and_writes_artifact(
    tmp_path: pathlib.Path,
) -> None:
    artifact = tmp_path / "registered-fd-greek-validation.json"

    result = run_registered_benchmark(
        GREEK_DERIVATIVE_VALIDATION_BENCHMARK_ID,
        artifact_path=artifact,
    )

    assert result.passed
    assert result.metrics["max_delta_abs_error"] <= 1.0e-3
    assert result.metrics["max_gamma_abs_error"] <= 3.0e-4
    assert result.metrics["benchmark_cases"] >= 12
    assert result.invariants["nonuniform_delta_converged"]
    assert result.invariants["nonuniform_gamma_converged"]
    assert result.invariants["strike_alignment_bounded"]
    assert result.invariants["rannacher_smooths_kinked_gamma"]
    assert json.loads(artifact.read_text(encoding="utf-8"))["benchmark_id"] == result.benchmark_id
