"""FD capability-manifest and unsupported-route diagnostics tests."""

from __future__ import annotations

import json
import pathlib

import pytest


from finite_difference_options.contracts import (  # noqa: E402
    DEFAULT_FD_CAPABILITY_MANIFEST,
    FDRouteRequest,
    UnsupportedReason,
    UnsupportedRouteError,
    diagnose_unsupported_route,
    ensure_route_supported,
)

FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures" / "quant_problem_specs"


def _supported_payload() -> dict[str, object]:
    return {
        "schema_version": "quant-problem-spec/v0",
        "valuation_context": {
            "measure": "risk_neutral",
            "numeraire": "money_market_account",
            "units": {"underlying": "USD", "time": "ACT/365F"},
            "valuation_date": "2026-01-02",
            "maturity_date": "2027-01-02",
        },
        "mathematical_problem": {
            "dimension": 1,
            "pde_terms": ["drift", "diffusion", "reaction"],
            "boundary_conditions": ["dirichlet", "neumann"],
            "exercise_style": "european",
        },
        "solver_plan": {
            "grid_type": "log_uniform",
            "requested_outputs": ["value", "delta", "gamma"],
            "stability_controls": ["theta"],
        },
    }


def test_default_manifest_declares_fd_support_with_diagnosed_american_lcp_without_jumps() -> None:
    manifest = DEFAULT_FD_CAPABILITY_MANIFEST

    assert manifest.backend_id == "finite_difference_options.fd_backend.v0"
    assert manifest.contract_version == "0.1.0"
    assert manifest.supported_dimensions == (1, 2, 3)
    assert {"european", "bermudan", "american"} <= set(manifest.exercise_styles)
    assert "exercise_boundary" in manifest.outputs
    assert "jump_integral" not in manifest.pde_terms
    assert "rannacher" in manifest.stability_controls
    assert "policy_iteration_lcp" not in manifest.stability_controls
    assert {"measure", "numeraire", "units", "valuation_date", "maturity_date"} <= set(manifest.required_conventions)
    assert manifest.feature_support["pinares_fixed_price_proxy"] == "validated"
    assert manifest.feature_support["obstacle_lcp"] == "validated"
    assert manifest.feature_support["american_black_scholes_lcp"] == "validated"
    assert manifest.feature_support["jump_integral"] == "unsupported"
    assert manifest.feature_support["hjb_control"] == "unsupported"
    assert manifest.error_budgets["pinares_fixed_price_proxy_price_abs_uf"] == 1.0


def test_american_lcp_capability_is_gated_to_one_dimensional_black_scholes_routes() -> None:
    payload = _supported_payload()
    math_problem = dict(payload["mathematical_problem"])  # type: ignore[arg-type]
    payload["mathematical_problem"] = {
        **math_problem,
        "exercise_style": "american",
    }
    one_dimensional = FDRouteRequest.from_quant_problem_spec(payload)
    assert diagnose_unsupported_route(one_dimensional) == ()

    payload["mathematical_problem"] = {
        **math_problem,
        "dimension": 2,
        "pde_terms": ["drift", "diffusion", "reaction", "mixed_derivative"],
        "exercise_style": "american",
    }
    multidimensional = FDRouteRequest.from_quant_problem_spec(payload)
    diagnostics = diagnose_unsupported_route(multidimensional)

    assert any(
        diagnostic.reason == UnsupportedReason.UNSUPPORTED_EXERCISE
        and "multidimensional/ADI" in diagnostic.message
        for diagnostic in diagnostics
    )


def test_policy_iteration_lcp_control_fails_closed_until_implemented() -> None:
    payload = _supported_payload()
    solver_plan = dict(payload["solver_plan"])  # type: ignore[arg-type]
    payload["solver_plan"] = {
        **solver_plan,
        "stability_controls": ["policy_iteration_lcp"],
    }
    request = FDRouteRequest.from_quant_problem_spec(payload)

    diagnostics = diagnose_unsupported_route(request)

    assert any(
        diagnostic.reason == UnsupportedReason.UNSUPPORTED_STABILITY_CONTROL
        and diagnostic.value == "policy_iteration_lcp"
        for diagnostic in diagnostics
    )


def test_quant_problem_spec_mapping_preserves_conventions_and_outputs() -> None:
    request = FDRouteRequest.from_quant_problem_spec(_supported_payload())

    assert request.dimension == 1
    assert request.grid_type == "log_uniform"
    assert request.pde_terms == ("drift", "diffusion", "reaction")
    assert request.boundary_conditions == ("dirichlet", "neumann")
    assert request.requested_outputs == ("value", "delta", "gamma")
    assert request.measure == "risk_neutral"
    assert request.numeraire == "money_market_account"
    assert request.units == {"underlying": "USD", "time": "ACT/365F"}
    assert request.valuation_date == "2026-01-02"
    assert request.maturity_date == "2027-01-02"
    assert diagnose_unsupported_route(request) == ()


def test_haircut_engine_vanilla_call_fixture_maps_to_supported_fd_route() -> None:
    """Consume the same public QuantProblemSpec fixture that Haircut Engine validates."""

    payload = json.loads((FIXTURE_DIR / "vanilla_call.json").read_text())

    request = FDRouteRequest.from_quant_problem_spec(payload)

    assert request.source_schema_version == "quant-problem-spec/v0"
    assert request.dimension == 1
    assert request.grid_type == "uniform"
    assert request.pde_terms == ("drift", "diffusion", "reaction")
    assert request.boundary_conditions == ("dirichlet", "neumann")
    assert request.boundary_details == {"S=0": "0", "S=S_max": "linear growth"}
    assert request.requested_outputs == ("value", "delta", "gamma")
    assert request.measure == "risk_neutral_money_market"
    assert request.numeraire == "money_market_account_CLP"
    assert request.units["S"] == "CLP"
    assert request.valuation_date == "2026-06-30"
    assert request.time_domain == "[0, 1]"
    assert diagnose_unsupported_route(request) == ()


def test_pinares_fixed_price_problem_fixture_maps_to_supported_fd_route() -> None:
    """Pinares publishes the problem; FD only chooses grid/time controls."""

    payload = json.loads((FIXTURE_DIR / "pinares_fixed_price_proxy.json").read_text())

    request = FDRouteRequest.from_quant_problem_spec(payload)

    assert payload["problem_id"] == "pinares.fixed_price_option_proxy.v1"
    assert payload["problem_hash"] == "publicsyntheticpinares001"
    assert request.source_schema_version == "quant-problem-spec/v0"
    assert request.backend_id == "finite_difference_options.fd_backend.v0"
    assert request.dimension == 1
    assert request.grid_type == "uniform"
    assert request.pde_terms == ("drift", "diffusion", "reaction")
    assert request.boundary_conditions == ("dirichlet", "neumann")
    assert request.boundary_details == {"S=0": "dirichlet", "S=S_max": "linear_growth"}
    assert request.requested_outputs == ("value", "delta", "gamma")
    assert request.measure == "Q*"
    assert request.numeraire == "UF_money_market_account_proxy"
    assert request.units["S"] == "UF"
    assert request.valuation_date == "2026-06-30"
    assert request.time_domain == "[0, 1]"
    assert diagnose_unsupported_route(request) == ()


def test_wrong_backend_id_fails_closed_for_fd_route() -> None:
    payload = json.loads((FIXTURE_DIR / "pinares_fixed_price_proxy.json").read_text())
    payload["solver_plan"] = {
        **payload["solver_plan"],
        "backend_id": "finite_element_options.fem_backend.v0",
    }
    request = FDRouteRequest.from_quant_problem_spec(payload)

    diagnostics = diagnose_unsupported_route(request)
    assert any(
        diagnostic.reason == UnsupportedReason.UNSUPPORTED_BACKEND
        and diagnostic.field == "backend_id"
        and diagnostic.value == "finite_element_options.fem_backend.v0"
        for diagnostic in diagnostics
    )


def test_unsupported_terms_dimensions_boundaries_and_exercise_fail_closed() -> None:
    payload = _supported_payload()
    payload["mathematical_problem"] = {
        "dimension": 4,
        "pde_terms": ["drift", "jump_integral", "hjb_control"],
        "boundary_conditions": ["free_boundary"],
        "exercise_style": "swing",
    }
    request = FDRouteRequest.from_quant_problem_spec(payload)

    diagnostics = diagnose_unsupported_route(request)
    reasons = {diagnostic.reason for diagnostic in diagnostics}

    assert UnsupportedReason.UNSUPPORTED_DIMENSION in reasons
    assert UnsupportedReason.UNSUPPORTED_TERM in reasons
    assert UnsupportedReason.UNSUPPORTED_BOUNDARY in reasons
    assert UnsupportedReason.UNSUPPORTED_EXERCISE in reasons
    assert {diagnostic.field for diagnostic in diagnostics} >= {
        "dimension",
        "pde_terms",
        "boundary_conditions",
        "exercise_style",
    }
    with pytest.raises(UnsupportedRouteError, match="FD backend supports dimensions") as exc_info:
        ensure_route_supported(request)
    assert exc_info.value.diagnostics == diagnostics


def test_missing_measure_numeraire_units_and_dates_are_actionable_diagnostics() -> None:
    payload = _supported_payload()
    payload["valuation_context"] = {}
    request = FDRouteRequest.from_quant_problem_spec(payload)

    diagnostics = diagnose_unsupported_route(request)
    missing = {
        diagnostic.field for diagnostic in diagnostics if diagnostic.reason == UnsupportedReason.MISSING_CONVENTION
    }

    assert missing == {
        "measure",
        "numeraire",
        "units",
        "valuation_date",
        "maturity_date",
    }
    assert all("missing or empty" in diagnostic.message for diagnostic in diagnostics)


def test_unsupported_outputs_and_stability_controls_do_not_silently_downgrade() -> None:
    payload = _supported_payload()
    payload["solver_plan"] = {
        "grid_type": "adaptive_sparse_grid",
        "requested_outputs": ["value", "vega", "early_exercise_premium"],
        "stability_controls": ["rannacher", "semismooth_newton_lcp"],
    }
    request = FDRouteRequest.from_quant_problem_spec(payload)

    diagnostics = diagnose_unsupported_route(request)
    by_field = {diagnostic.field: diagnostic for diagnostic in diagnostics}

    assert by_field["grid_type"].reason == UnsupportedReason.UNSUPPORTED_GRID
    assert {diagnostic.value for diagnostic in diagnostics if diagnostic.field == "requested_outputs"} == {
        "vega",
        "early_exercise_premium",
    }
    assert {diagnostic.value for diagnostic in diagnostics if diagnostic.field == "stability_controls"} == {
        "semismooth_newton_lcp",
    }
