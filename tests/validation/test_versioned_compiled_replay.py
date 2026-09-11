"""Replay selection must refuse unavailable implementations before numerical work."""

from __future__ import annotations

from copy import deepcopy

import pytest

from finite_difference_options.validation import fd_verification as verification


def test_unknown_recorded_code_version_refused_before_numerical_work(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = deepcopy(verification.run_fd_bs_verification_benchmark())
    bundle["provenance"]["code_version"] = "unknown-implementation"
    bundle["evidence"]["hashes"] = verification._hashes_for_bundle(bundle)

    def must_not_run() -> None:
        pytest.fail("unknown replay provenance must not choose the current numerical implementation")

    monkeypatch.setattr(verification, "run_fd_bs_verification_benchmark", must_not_run)
    with pytest.raises(verification.FDVerificationError, match="replay.*version|version.*replay"):
        verification.validate_fd_bs_verification_bundle(bundle)


@pytest.fixture(scope="module")
def v1_bundle():
    return verification.run_fd_bs_verification_benchmark_v1()


@pytest.mark.parametrize("field", ["evidence", "results", "provenance", "request", "config", "convention"])
def test_malformed_sections_refuse_before_recompute(v1_bundle, monkeypatch, field):
    bundle = deepcopy(v1_bundle)
    bundle[field] = []
    monkeypatch.setattr(
        verification, "run_fd_bs_verification_benchmark_v1", lambda: pytest.fail("invalid bundle dispatched")
    )
    with pytest.raises(verification.FDVerificationError):
        verification.validate_fd_bs_verification_bundle(bundle)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), complex(1, 0), object()])
def test_non_json_or_nonfinite_evidence_refused(v1_bundle, monkeypatch, value):
    bundle = deepcopy(v1_bundle)
    bundle["results"]["full_refinement"]["rows"][-1]["price"] = value
    monkeypatch.setattr(
        verification, "run_fd_bs_verification_benchmark_v1", lambda: pytest.fail("invalid bundle dispatched")
    )
    with pytest.raises(verification.FDVerificationError):
        verification.validate_fd_bs_verification_bundle(bundle)


@pytest.mark.parametrize("value", [[], None, "sha256:bad"])
def test_malformed_hash_map_refuses_before_recompute(v1_bundle, monkeypatch, value):
    bundle = deepcopy(v1_bundle)
    bundle["evidence"]["hashes"] = value
    monkeypatch.setattr(
        verification, "run_fd_bs_verification_benchmark_v1", lambda: pytest.fail("invalid hash map dispatched")
    )
    with pytest.raises(verification.FDVerificationError):
        verification.validate_fd_bs_verification_bundle(bundle)


@pytest.mark.parametrize(
    "field", ["python", "numpy", "scipy", "method", "dtype", "implementation_sha256", "thread_environment"]
)
def test_changed_runtime_or_implementation_refuses_before_solve(v1_bundle, monkeypatch, field):
    bundle = deepcopy(v1_bundle)
    bundle["provenance"]["numerical_runtime"][field] = "unavailable"
    bundle["evidence"]["hashes"] = verification._hashes_for_bundle(bundle)
    monkeypatch.setattr(
        verification, "run_fd_bs_verification_benchmark_v1", lambda: pytest.fail("unavailable runtime dispatched")
    )
    with pytest.raises(verification.FDVerificationError, match="runtime/implementation is unavailable"):
        verification.validate_fd_bs_verification_bundle(bundle)


@pytest.mark.parametrize("version", ["v2", [], None])
def test_unknown_schema_refuses_before_solve(v1_bundle, monkeypatch, version):
    bundle = deepcopy(v1_bundle)
    bundle["schema_version"] = version
    monkeypatch.setattr(
        verification, "run_fd_bs_verification_benchmark_v1", lambda: pytest.fail("unknown schema dispatched")
    )
    with pytest.raises(verification.FDVerificationError, match="schema_version"):
        verification.validate_fd_bs_verification_bundle(bundle)


@pytest.mark.parametrize("mutation", ["price", "operator_metadata", "status", "request", "extra"])
def test_rehashed_v1_tampering_is_not_numerical_evidence(v1_bundle, mutation):
    bundle = deepcopy(v1_bundle)
    if mutation == "price":
        bundle["results"]["full_refinement"]["rows"][-1]["price"] += 1.0e-4
    elif mutation == "operator_metadata":
        bundle["results"]["full_refinement"]["rows"][-1]["numerical_operator"]["factorization_count"] = 0
    elif mutation == "status":
        bundle["evidence"]["status"] = "failed"
    elif mutation == "request":
        bundle["request"]["route_id"] = "fd.compiled_pde.black_scholes_call_v0"
    else:
        bundle["unverified_extra_claim"] = "passed"
    bundle["evidence"]["hashes"] = verification._hashes_for_bundle(bundle)
    with pytest.raises(verification.FDVerificationError):
        verification.validate_fd_bs_verification_bundle(bundle)


def test_v1_json_roundtrip_recomputes_and_keeps_v0_defaults(v1_bundle, tmp_path):
    import json

    assert v1_bundle["schema_version"].endswith("/v1")
    assert v1_bundle["evidence"]["status"] == "passed"
    assert v1_bundle["request"]["versioned_benchmark_id"] == "FD-BS-001-V1"
    verification.validate_fd_bs_verification_bundle(json.loads(json.dumps(v1_bundle)))
    path = tmp_path / "v1.json"
    written = verification.write_fd_bs_verification_json_v1(path)
    assert json.loads(path.read_text()) == json.loads(json.dumps(written))
    assert verification.run_fd_bs_verification_benchmark()["schema_version"].endswith("/v0")


def test_v1_public_compiled_route_is_explicit_and_refuses_non_fixture():
    from finite_difference_options.integrations import (
        CompiledPDEAdapterError,
        packaged_compiled_black_scholes_fixture,
        solve_compiled_pde_payload,
        solve_compiled_pde_payload_v1,
    )

    payload = packaged_compiled_black_scholes_fixture()
    old = solve_compiled_pde_payload(payload)
    new = solve_compiled_pde_payload_v1(payload)
    assert old.schema_version.endswith("/v0")
    assert new.schema_version.endswith("/v1")
    assert old.status == new.status == "passed"
    assert abs(new.values["price"] - old.values["price"]) <= 3.0e-12
    payload["privacy_class"] = "private"
    with pytest.raises(CompiledPDEAdapterError):
        solve_compiled_pde_payload_v1(payload)
