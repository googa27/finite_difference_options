"""Pinares fixed-price proxy FD compatibility and fail-closed tests."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from finite_difference_options.contracts import (
    DEFAULT_FD_CAPABILITY_MANIFEST,
    FDRouteRequest,
    UnsupportedReason,
    UnsupportedRouteError,
    diagnose_unsupported_route,
)
from finite_difference_options.integrations.haircut_backend import create_backend
from finite_difference_options.validation.benchmark_registry import run_registered_benchmark
from finite_difference_options.validation.pinares_fixed_price_proxy import (
    PINARES_FAIL_CLOSED_BENCHMARK_ID,
    PINARES_FIXED_PRICE_PROXY_BENCHMARK_ID,
    PINARES_FIXED_PRICE_PROXY_PROBLEM_ID,
    PINARES_FIXED_PRICE_PROXY_ROUTE_ID,
    PINARES_QPS_CONTRACT_BENCHMARK_ID,
    PinaresFixedPriceProxyCase,
    build_pinares_fd_provider_evidence_manifest,
    public_pinares_fixed_price_problem_spec,
    public_pinares_full_deal_unsupported_problem_spec,
    run_public_pinares_fixed_price_proxy_fixture,
)

pytestmark = pytest.mark.usefixtures("haircut_public_solver_seam")

FIXTURE_DIR = pathlib.Path(__file__).resolve().parent / "fixtures"
QPS_FIXTURE = FIXTURE_DIR / "quant_problem_specs" / "pinares_fixed_price_proxy.json"
RESULT_FIXTURE = FIXTURE_DIR / "pinares_fd_fixed_price_proxy_v1.json"
PROVIDER_MANIFEST_FIXTURE = FIXTURE_DIR / "pinares_fd_provider_evidence_manifest_v1.json"


def _load_qps_fixture() -> dict[str, Any]:
    payload = json.loads(QPS_FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_pinares_fixed_price_problem_spec_maps_to_supported_fd_route() -> None:
    payload = public_pinares_fixed_price_problem_spec()
    request = FDRouteRequest.from_quant_problem_spec(payload)

    assert payload["problem_id"] == PINARES_FIXED_PRICE_PROXY_PROBLEM_ID
    assert payload["problem_hash"] == "publicsyntheticpinares001"
    assert payload["privacy_class"] == "public_synthetic"
    assert request.source_schema_version == "quant-problem-spec/v0"
    assert request.backend_id == "finite_difference_options.fd_backend.v0"
    assert request.dimension == 1
    assert request.grid_type == "uniform"
    assert request.pde_terms == ("drift", "diffusion", "reaction")
    assert request.boundary_conditions == ("dirichlet", "neumann")
    assert request.requested_outputs == ("value", "delta", "gamma")
    assert request.measure == "Q*"
    assert request.numeraire == "UF_money_market_account_proxy"
    assert request.units["S"] == "UF"
    assert request.valuation_date == "2026-06-30"
    assert request.time_domain == "[0, 1]"
    assert diagnose_unsupported_route(request) == ()


def test_capability_manifest_discloses_pinares_error_budgets_and_unsupported_terms() -> None:
    manifest = DEFAULT_FD_CAPABILITY_MANIFEST

    assert manifest.feature_support["pinares_fixed_price_proxy"] == "validated"
    assert manifest.feature_support["jump_integral"] == "unsupported"
    assert manifest.feature_support["hjb_control"] == "unsupported"
    assert manifest.feature_support["rofr_full_family_contract"] == "unsupported"
    assert manifest.error_budgets["pinares_fixed_price_proxy_price_abs_uf"] == 1.0
    assert manifest.error_budgets["pinares_fixed_price_proxy_delta_abs"] == 1.0e-3
    assert manifest.resource_controls["pinares_fixed_price_proxy_max_s_steps"] == 180
    assert "theta" in manifest.stability_controls


def test_pinares_fixed_price_proxy_runs_against_analytical_survival_scaled_oracle() -> None:
    report = run_public_pinares_fixed_price_proxy_fixture()

    assert report.converged
    assert report.evidence.fixture_id == "public-synthetic.pinares-fixed-price-proxy.v1"
    assert report.evidence.route_id == PINARES_FIXED_PRICE_PROXY_ROUTE_ID
    assert report.evidence.measure == "Q*"
    assert report.evidence.numeraire == "UF_money_market_account_proxy"
    assert report.evidence.units["value"] == "UF"
    assert report.final_abs_error_uf <= 1.0
    assert report.errors["delta_abs"] <= 1.0e-3
    assert report.errors["gamma_abs"] <= 5.0e-6
    assert report.price_uf == pytest.approx(report.oracle_price_uf, abs=1.0)
    assert report.oracle_price_uf == pytest.approx(report.case.survival_probability * report.base_report.oracle_price)
    assert report.no_arbitrage["value_bound_ok"]
    assert report.no_arbitrage["upper_bound_ok"]
    assert len(report.convergence_table()) == 3


def test_pinares_zero_survival_probability_is_valid_and_does_not_divide_by_zero() -> None:
    case = PinaresFixedPriceProxyCase(survival_probability=0.0)
    report = run_public_pinares_fixed_price_proxy_fixture(case=case, grid_levels=((40, 80),))

    assert case.unscaled_black_scholes_tolerance_uf() == case.price_abs_tolerance_uf
    assert report.converged
    assert report.oracle_price_uf == 0.0
    assert report.price_uf == 0.0
    assert report.delta == 0.0
    assert report.gamma == 0.0
    assert report.errors["price_abs"] == 0.0
    assert report.no_arbitrage["survival_scale_ok"]


def test_pinares_problem_spec_uses_requested_grid_levels_in_resource_controls() -> None:
    grid_levels = ((24, 48), (36, 72))
    payload = public_pinares_fixed_price_problem_spec(grid_levels=grid_levels)
    report = run_public_pinares_fixed_price_proxy_fixture(grid_levels=grid_levels)

    assert payload["solver_plan"]["resource_controls"] == {
        "deterministic": "true",
        "grid_levels": 2,
        "max_s_steps": 36,
        "max_t_steps": 72,
    }
    assert (
        report.as_dict()["problem_spec"]["solver_plan"]["resource_controls"]
        == payload["solver_plan"]["resource_controls"]
    )


def test_pinares_static_fixtures_match_generated_contracts() -> None:
    qps_cached = _load_qps_fixture()
    result_cached = json.loads(RESULT_FIXTURE.read_text(encoding="utf-8"))
    provider_manifest_cached = json.loads(PROVIDER_MANIFEST_FIXTURE.read_text(encoding="utf-8"))
    report = run_public_pinares_fixed_price_proxy_fixture()

    assert qps_cached == public_pinares_fixed_price_problem_spec()
    assert result_cached == json.loads(json.dumps(report.as_dict()))
    assert provider_manifest_cached == build_pinares_fd_provider_evidence_manifest(report)
    assert qps_cached["solver_plan"]["error_budgets"] == {
        "delta_abs": 1.0e-3,
        "gamma_abs": 5.0e-6,
        "price_abs_uf": 1.0,
    }
    assert result_cached["unsupported_scope"]["rofr"].startswith("unsupported")


def test_pinares_fd_provider_evidence_manifest_reports_dashboard_sidecar_fields() -> None:
    report = run_public_pinares_fixed_price_proxy_fixture()
    manifest = build_pinares_fd_provider_evidence_manifest(report)

    assert manifest["schema"] == "pinares.provider_evidence_manifest.v1"
    assert manifest["producer"] == "finite_difference_options"
    assert manifest["privacy_class"] == "public_synthetic"
    assert manifest["issue_refs"] == ["googa27/finite_difference_options#135"]
    assert manifest["evidence_class"] == "deterministic_proxy_not_full_family_contract_valuation"
    assert manifest["capability_manifest"]["backend_id"] == DEFAULT_FD_CAPABILITY_MANIFEST.backend_id
    assert manifest["capability_manifest"]["contract_version"] == DEFAULT_FD_CAPABILITY_MANIFEST.contract_version
    assert manifest["route"]["route_id"] == PINARES_FIXED_PRICE_PROXY_ROUTE_ID
    assert manifest["route"]["method_kind"] == "finite_difference"
    assert manifest["route"]["boundary_convention"] == "Dirichlet at S=0; linear-growth far-field at S_max"
    assert manifest["resource_controls"]["max_s_steps"] == 180
    assert manifest["performance_sidecar"]["runtime"]["seconds"] is None
    assert manifest["performance_sidecar"]["operator_factorization_cache"] == (
        "enabled_for_public_black_scholes_and_pinares_proxy"
    )
    assert manifest["parity_metrics"]["price_abs_uf"] <= manifest["error_budgets"]["price_abs_uf"]
    assert manifest["unsupported_routes"]["full_family_contract"] == "fail_closed"


def test_pinares_registered_benchmarks_execute_and_fail_closed() -> None:
    parity = run_registered_benchmark(PINARES_FIXED_PRICE_PROXY_BENCHMARK_ID)
    qps = run_registered_benchmark(PINARES_QPS_CONTRACT_BENCHMARK_ID)
    fail_closed = run_registered_benchmark(PINARES_FAIL_CLOSED_BENCHMARK_ID)

    assert parity.passed
    assert float(parity.metrics["price_abs"]) <= 1.0
    assert float(parity.metrics["delta_abs"]) <= 1.0e-3
    assert float(parity.metrics["gamma_abs"]) <= 5.0e-6
    assert parity.evidence["route_id"] == PINARES_FIXED_PRICE_PROXY_ROUTE_ID
    assert qps.passed
    assert qps.invariants["fd_route_supported"]
    assert fail_closed.passed
    assert fail_closed.invariants["rofr_not_executed"]
    assert fail_closed.invariants["full_family_contract_rejected"]
    assert fail_closed.invariants["unsupported_terms_reported"]


def test_pinares_public_synthetic_payload_screens_and_solves_through_backend_adapter() -> None:
    backend = create_backend()
    payload = public_pinares_fixed_price_problem_spec()

    screen = backend.screen(payload)
    solved = backend.solve(payload)

    assert screen.supported
    assert screen.diagnostics == ()
    assert screen.request["boundary_conditions"] == ("dirichlet", "neumann")
    assert solved.passed
    assert solved.problem_id == PINARES_FIXED_PRICE_PROXY_PROBLEM_ID
    assert solved.benchmark_ids == (
        PINARES_FIXED_PRICE_PROXY_BENCHMARK_ID,
        PINARES_QPS_CONTRACT_BENCHMARK_ID,
    )
    assert solved.values["price"] == pytest.approx(solved.values["oracle_price"], abs=1.0)
    assert solved.evidence["privacy_class"] == "public_synthetic"
    assert solved.diagnostics["fallbacks"] == ()


def test_pinares_full_deal_and_rofr_payload_fail_closed_without_numbers() -> None:
    backend = create_backend()
    payload = public_pinares_full_deal_unsupported_problem_spec()
    request = FDRouteRequest.from_quant_problem_spec(payload)
    diagnostics = diagnose_unsupported_route(request)
    reasons = {diagnostic.reason for diagnostic in diagnostics}

    assert UnsupportedReason.UNSUPPORTED_DIMENSION in reasons
    assert UnsupportedReason.UNSUPPORTED_TERM in reasons
    assert UnsupportedReason.UNSUPPORTED_EXERCISE in reasons
    assert UnsupportedReason.UNSUPPORTED_OUTPUT in reasons
    assert "solution" not in payload
    assert "values" not in payload

    screen = backend.screen(payload)
    assert not screen.supported
    assert {diagnostic["reason"] for diagnostic in screen.diagnostics} >= {
        "unsupported_dimension",
        "unsupported_pde_term",
        "unsupported_exercise_style",
    }
    with pytest.raises(UnsupportedRouteError):
        backend.solve(payload)


def test_pinares_public_fixture_is_exact_shape_gated_before_execution() -> None:
    backend = create_backend()
    payload = _load_qps_fixture()
    payload["mathematical_problem"] = {
        **payload["mathematical_problem"],
        "terminal_payoff": {
            **payload["mathematical_problem"]["terminal_payoff"],
            "expression": "max(S - K, 0)",
        },
    }

    screen = backend.screen(payload)

    assert not screen.supported
    assert screen.diagnostics[0]["reason"] == "unsupported_benchmark"
    with pytest.raises(UnsupportedRouteError, match="validated public-synthetic executable benchmark"):
        backend.solve(payload)
