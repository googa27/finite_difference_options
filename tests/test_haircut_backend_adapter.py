"""Haircut backend adapter contract tests."""

from __future__ import annotations

import ast
from dataclasses import replace
import importlib.metadata as metadata
import json
import pathlib

import pytest

from finite_difference_options.contracts import DEFAULT_FD_CAPABILITY_MANIFEST, UnsupportedRouteError
from finite_difference_options.integrations import haircut_protocol
from finite_difference_options.integrations.haircut_backend import ContractMajorMismatchError, create_backend

pytestmark = pytest.mark.usefixtures("haircut_public_solver_seam")

FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures" / "quant_problem_specs"


def _vanilla_payload() -> dict[str, object]:
    payload = json.loads((FIXTURE_DIR / "vanilla_call.json").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _executable_payload() -> dict[str, object]:
    from finite_difference_options.validation.black_scholes_parity import (
        public_black_scholes_problem_spec,
    )

    return public_black_scholes_problem_spec()


def _section(payload: dict[str, object], name: str) -> dict[str, object]:
    section = payload[name]
    assert isinstance(section, dict)
    return section


def test_haircut_backend_identity_and_manifest_use_haircut_public_shapes() -> None:
    from haircut.solvers.backend_protocol import BackendIdentity, BackendMaturity
    from haircut.solvers.contracts import BackendCapabilityManifest, MethodMaturity

    backend = create_backend()

    assert isinstance(backend.identity, BackendIdentity)
    assert backend.identity.distribution_name == "finite-difference-options"
    assert backend.identity.implementation_id == "finite_difference_options.fd_backend.v0"
    assert str(backend.identity.contract_version) == backend.manifest.contract_version
    assert backend.identity.maturity is BackendMaturity.VALIDATED
    assert backend.identity.build_metadata["entry_point_group"] == "haircut.solver_backends"
    assert backend.identity.build_metadata["issue_refs"] == (
        "googa27/finite_difference_options#59,googa27/finite_difference_options#140,googa27/haircut-engine#217"
    )

    manifest = backend.capability_manifest
    assert isinstance(manifest, BackendCapabilityManifest)
    assert manifest.backend_id == "finite_difference_options.fd_backend.v0"
    assert manifest.contract_version == backend.manifest.contract_version
    assert len(manifest.methods) == 1
    method = manifest.methods[0]
    assert method.backend_id == manifest.backend_id
    assert method.maturity is MethodMaturity.VALIDATED
    assert method.fallback_policy == "fail_closed_no_fallback"
    assert "value" in method.output_types

    fd_manifest = backend.fd_capability_manifest()
    assert fd_manifest["backend_id"] == "finite_difference_options.fd_backend.v0"
    assert fd_manifest["status"] == "validated"


def test_haircut_backend_imports_only_public_solver_protocol_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    original_import_module = haircut_protocol.importlib.import_module

    def _spy_import_module(name: str):
        if name.startswith("haircut"):
            seen.append(name)
        return original_import_module(name)

    monkeypatch.setattr(haircut_protocol.importlib, "import_module", _spy_import_module)

    create_backend()

    assert set(seen) <= {"haircut.solvers.backend_protocol", "haircut.solvers.contracts"}


def test_haircut_backend_fails_closed_on_contract_major_mismatch() -> None:
    manifest = replace(DEFAULT_FD_CAPABILITY_MANIFEST, contract_version="1.0.0")

    with pytest.raises(ContractMajorMismatchError, match="contract major mismatch"):
        create_backend(manifest=manifest, expected_contract_version="0.1.0")


def test_built_distribution_declares_exactly_one_canonical_haircut_entry_point() -> None:
    canonical = metadata.entry_points(group="haircut.solver_backends")
    legacy = metadata.entry_points(group="haircut_engine.solver_backends")

    matches = [
        ep
        for ep in canonical
        if ep.name == "finite_difference_options"
        and ep.value == "finite_difference_options.integrations.haircut_backend:create_backend"
    ]
    assert len(matches) == 1
    assert not [ep for ep in legacy if ep.name == "finite_difference_options"]


def test_haircut_backend_screen_preserves_quant_problem_spec_without_solving() -> None:
    backend = create_backend()
    result = backend.screen(_executable_payload())

    assert result.supported
    assert result.diagnostics == ()
    assert result.request["source_schema_version"] == "quant-problem-spec/v0"
    assert result.request["measure"] == "risk_neutral"
    assert result.request["numeraire"] == "money_market_account"
    assert result.request["units"]["underlying"] == "synthetic_currency"
    assert result.request["valuation_date"] == "2026-01-02"
    assert result.request["requested_outputs"] == ("value", "delta", "gamma")


def test_haircut_backend_screen_fails_closed_before_operator_work() -> None:
    backend = create_backend()
    payload = _vanilla_payload()
    math_section = _section(payload, "mathematical_problem")
    payload["mathematical_problem"] = {
        **math_section,
        "pde_terms": ["drift", "diffusion", "jump_integral"],
        "exercise_style": "swing",
    }

    result = backend.screen(payload)

    assert not result.supported
    assert {diagnostic["field"] for diagnostic in result.diagnostics} >= {
        "pde_terms",
        "exercise_style",
    }
    assert {diagnostic["reason"] for diagnostic in result.diagnostics} >= {
        "unsupported_pde_term",
        "unsupported_exercise_style",
    }


def test_haircut_backend_screen_fails_closed_for_malformed_dimension() -> None:
    backend = create_backend()
    payload = _vanilla_payload()
    math_section = _section(payload, "mathematical_problem")
    payload["mathematical_problem"] = {**math_section, "dimension": "auto"}

    result = backend.screen(payload)

    assert not result.supported
    assert any(
        diagnostic["field"] == "dimension" and diagnostic["reason"] == "unsupported_dimension"
        for diagnostic in result.diagnostics
    )


def test_haircut_backend_screen_rejects_supported_route_without_executable_fixture() -> None:
    backend = create_backend()
    payload = _vanilla_payload()
    payload["problem_id"] = "external-private-supported-looking-problem"
    payload["solver_plan"] = {
        **_section(payload, "solver_plan"),
        "benchmark_ids": ["PRIVATE-BENCHMARK-V0"],
    }
    payload["result_bundle"] = {"benchmark_ids": ["PRIVATE-BENCHMARK-V0"]}
    payload["financial_graph"] = {}

    result = backend.screen(payload)

    assert not result.supported
    assert result.diagnostics == (
        {
            "reason": "unsupported_benchmark",
            "field": "benchmark_ids",
            "value": "PRIVATE-BENCHMARK-V0",
            "supported": (
                "BS-FD-ORACLE-V0",
                "PINARES-FD-FIXED-PRICE-PROXY-V0",
                "PINARES-QPS-FIXED-PRICE-PROXY-V0",
                "QPS-BS-CALL-PUBLIC-V0",
            ),
            "message": (
                "FD backend adapter currently executes only registered public-synthetic "
                "benchmarks; register a fixture before claiming solve support."
            ),
        },
    )


def test_haircut_backend_results_are_json_serializable() -> None:
    backend = create_backend()
    screen = backend.screen(_executable_payload())
    solved = backend.solve(_executable_payload())

    screen_payload = json.loads(json.dumps(screen.as_dict()))
    solve_payload = json.loads(json.dumps(solved.as_dict()))

    assert screen_payload["status"] == "supported"
    assert solve_payload["status"] == "passed"
    assert solve_payload["diagnostics"]["requested_benchmark_ids"] == [
        "BS-FD-ORACLE-V0",
        "QPS-BS-CALL-PUBLIC-V0",
    ]


def test_haircut_backend_module_keeps_validation_runners_lazy() -> None:
    source = pathlib.Path("src/finite_difference_options/integrations/haircut_backend.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_imports = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module is not None]

    assert not [
        node.module for node in top_level_imports if node.module.startswith("finite_difference_options.validation")
    ]


def test_haircut_backend_solve_executes_only_validated_public_synthetic_fixture() -> None:
    backend = create_backend()

    result = backend.solve(_executable_payload())

    assert result.passed
    assert result.problem_id == "public-synthetic.black-scholes-call.v0"
    assert result.benchmark_ids == ("BS-CALL-PARITY-V0", "QPS-VANILLA-CALL-V0")
    assert result.values["price"] == pytest.approx(result.values["oracle_price"], abs=1.0e-2)
    assert result.diagnostics["fallbacks"] == ()
    assert result.evidence["privacy_class"] == "public_synthetic"
    assert result.evidence["valuation_date"] == result.request["valuation_date"]
    assert result.evidence["numeraire"] == result.request["numeraire"]
    assert result.evidence["units"] == result.request["units"]
    assert result.request["boundary_conditions"] == ("dirichlet",)


def test_haircut_backend_rejects_public_benchmark_on_private_payload() -> None:
    backend = create_backend()
    payload = _executable_payload()
    payload["problem_id"] = "external-private-supported-looking-problem"
    payload["privacy_class"] = "private"

    screen = backend.screen(payload)
    assert not screen.supported
    assert screen.diagnostics[0]["reason"] == "unsupported_benchmark"

    with pytest.raises(UnsupportedRouteError, match="validated public-synthetic executable benchmark"):
        backend.solve(payload)


def test_haircut_backend_rejects_private_payload_even_with_public_fixture_ids() -> None:
    backend = create_backend()
    payload = _executable_payload()
    payload["privacy_class"] = "private"

    screen = backend.screen(payload)
    assert not screen.supported
    assert screen.diagnostics[0]["reason"] == "unsupported_benchmark"

    with pytest.raises(UnsupportedRouteError, match="validated public-synthetic executable benchmark"):
        backend.solve(payload)


def test_haircut_backend_rejects_mutated_public_fixture_fields() -> None:
    backend = create_backend()
    payload = _executable_payload()
    math_section = _section(payload, "mathematical_problem")
    payload["mathematical_problem"] = {
        **math_section,
        "state_variables": ["S", "v"],
    }

    screen = backend.screen(payload)
    assert not screen.supported
    assert screen.diagnostics[0]["reason"] == "unsupported_benchmark"

    with pytest.raises(UnsupportedRouteError, match="validated public-synthetic executable benchmark"):
        backend.solve(payload)


def test_haircut_backend_solve_rejects_supported_screen_without_executable_fixture() -> None:
    backend = create_backend()
    payload = _vanilla_payload()
    payload["problem_id"] = "external-private-supported-looking-problem"
    payload["privacy_class"] = "private"
    solver_plan = _section(payload, "solver_plan")
    payload["solver_plan"] = {
        **solver_plan,
        "benchmark_ids": ["PRIVATE-BENCHMARK-V0"],
    }
    payload["result_bundle"] = {"benchmark_ids": ["PRIVATE-BENCHMARK-V0"]}
    payload["financial_graph"] = {}

    with pytest.raises(UnsupportedRouteError, match="validated public-synthetic executable benchmark"):
        backend.solve(payload)
