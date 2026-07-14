"""Issue #142 Black-Scholes verification evidence tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from finite_difference_options.cli.main import app
from finite_difference_options.validation.fd_verification import (
    FD_BS_VERIFICATION_BENCHMARK_ID,
    FDVerificationError,
    run_fd_bs_verification_benchmark,
    validate_fd_bs_verification_bundle,
)


def test_fd_bs_001_evidence_has_oracle_greek_residual_and_hash_gates() -> None:
    bundle = run_fd_bs_verification_benchmark()

    assert bundle["benchmark_id"] == FD_BS_VERIFICATION_BENCHMARK_ID
    assert bundle["evidence"]["status"] == "passed"
    assert bundle["config"]["source_ir_canonical_hash"].startswith("sha256:")
    assert bundle["config"]["compiled_hash"].startswith("sha256:")
    assert set(bundle["evidence"]["hashes"]) == {
        "request_hash",
        "config_hash",
        "convention_hash",
        "result_hash",
        "evidence_hash",
    }
    finest = bundle["results"]["full_refinement"]["rows"][-1]
    tolerances = bundle["results"]["tolerances"]
    assert finest["price_abs"] <= tolerances["price_abs"]
    assert finest["delta_abs"] <= tolerances["delta_abs"]
    assert finest["gamma_abs"] <= tolerances["gamma_abs"]
    assert finest["payoff_linf"] <= tolerances["payoff_linf"]
    assert finest["pde_residual_linf"] <= tolerances["pde_residual_linf"]
    assert finest["boundary_linf"] <= tolerances["boundary_linf"]
    assert len(bundle["results"]["spatial_refinement"]["rows"]) == 3
    assert len(bundle["results"]["temporal_refinement"]["rows"]) == 3
    assert bundle["results"]["manufactured_solution"]["min_observed_residual_order"] > 1.9
    validate_fd_bs_verification_bundle(bundle)


def test_fd_bs_001_validation_recomputes_truth_and_rejects_tampering() -> None:
    bundle = run_fd_bs_verification_benchmark()
    tampered = copy.deepcopy(bundle)
    tampered["results"]["full_refinement"]["rows"][-1]["price"] = 0.0

    with pytest.raises(FDVerificationError, match="hash mismatch|results do not match"):
        validate_fd_bs_verification_bundle(tampered)


def test_fd_options_validation_run_benchmark_writes_artifact(tmp_path: Path) -> None:
    out = tmp_path / "fd-verification.json"
    result = CliRunner().invoke(
        app,
        ["validation", "run-benchmark", "fd-bs-001", "--out", str(out)],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "passed"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["benchmark_id"] == "fd-bs-001"
    assert payload["evidence"]["status"] == "passed"
    validate_fd_bs_verification_bundle(payload)
