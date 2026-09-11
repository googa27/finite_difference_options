"""Fail-closed adapter for exact public-synthetic compiled ``pde_ir.v0`` fixtures."""

from __future__ import annotations
import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any, cast
from finite_difference_options.contracts import DEFAULT_FD_CAPABILITY_MANIFEST
from finite_difference_options.integrations.compiled_pde_black_scholes_route import _run_compiled_black_scholes_route
from finite_difference_options.integrations.haircut_protocol import installed_distribution_version
from finite_difference_options.integrations.public_fixture_identity import matches_exact_public_fixture
from ._compiled_pde_contracts import (
    CompiledPDEStatus as CompiledPDEStatus,
    CompiledPDESolveStatus as CompiledPDESolveStatus,
    FIXTURE_SCHEMA_VERSION as FIXTURE_SCHEMA_VERSION,
    SOURCE_PDE_IR_SCHEMA_ID as SOURCE_PDE_IR_SCHEMA_ID,
    SOURCE_PDE_IR_SCHEMA_VERSION as SOURCE_PDE_IR_SCHEMA_VERSION,
    COMPILED_OPERATOR_SCHEMA_ID as COMPILED_OPERATOR_SCHEMA_ID,
    COMPILED_OPERATOR_SCHEMA_VERSION as COMPILED_OPERATOR_SCHEMA_VERSION,
    EXPECTED_SOURCE_IR_HASH as EXPECTED_SOURCE_IR_HASH,
    EXPECTED_COMPILED_HASH as EXPECTED_COMPILED_HASH,
    EXPECTED_PROBLEM_ID as EXPECTED_PROBLEM_ID,
    EXPECTED_SOURCE_PROBLEM_ID as EXPECTED_SOURCE_PROBLEM_ID,
    EXPECTED_FORMULATION_ID as EXPECTED_FORMULATION_ID,
    EXPECTED_BOUNDARY_KINDS as EXPECTED_BOUNDARY_KINDS,
    EXPECTED_OUTPUTS as EXPECTED_OUTPUTS,
    _PACKAGED_FIXTURE as _PACKAGED_FIXTURE,
    _COMPILED_ROUTE_NUMERICS as _COMPILED_ROUTE_NUMERICS,
    CompiledPDEDiagnostic as CompiledPDEDiagnostic,
    CompiledPDEScreeningResult as CompiledPDEScreeningResult,
    CompiledPDESolveResult as CompiledPDESolveResult,
    CompiledPDEAdapterError as CompiledPDEAdapterError,
)
from ._compiled_pde_validation import (
    _validate_source_ir as _validate_source_ir,
    _validate_compiled_operator as _validate_compiled_operator,
    _validate_solver_plan as _validate_solver_plan,
    _source_hash as _source_hash,
    _compiled_hash as _compiled_hash,
    _sha256_ref as _sha256_ref,
    _state_units as _state_units,
    _dict_at as _dict_at,
    _list_at as _list_at,
    _list_of_dicts_at as _list_of_dicts_at,
    _check_allowed_keys as _check_allowed_keys,
    _expect as _expect,
    _diag as _diag,
    _list as _list,
    _list_of_dicts as _list_of_dicts,
    _json_type_name as _json_type_name,
    _is_json_value as _is_json_value,
    _reject_non_finite_json as _reject_non_finite_json,
)


def load_compiled_pde_json(path: str | Path) -> dict[str, Any]:
    """Load a strict JSON object from ``path`` for adapter screening."""

    try:
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle, parse_constant=_reject_non_finite_json)
    except (OSError, ValueError) as exc:
        raise CompiledPDEAdapterError(
            (
                _diag(
                    "compiled_pde.json_invalid",
                    "compiled PDE input is not strict JSON",
                    "payload",
                ),
            )
        ) from exc
    if type(payload) is not dict:
        raise CompiledPDEAdapterError(
            (
                _diag(
                    "compiled_pde.payload_type",
                    "compiled PDE input must be a JSON object",
                    "payload",
                ),
            )
        )
    return cast(dict[str, Any], payload)


def packaged_compiled_black_scholes_fixture() -> dict[str, Any]:
    """Return the packaged exact public-synthetic compiled Black--Scholes fixture."""

    text = _packaged_compiled_black_scholes_fixture_resource().read_text(encoding="utf-8")
    payload = json.loads(text, parse_constant=_reject_non_finite_json)
    if type(payload) is not dict:  # pragma: no cover - package-data corruption guard
        raise CompiledPDEAdapterError(
            (
                _diag(
                    "compiled_pde.payload_type",
                    "packaged fixture is not a JSON object",
                    "payload",
                ),
            )
        )
    return cast(dict[str, Any], payload)


def _packaged_compiled_black_scholes_fixture_resource() -> Traversable:
    return resources.files("finite_difference_options.validation.fixtures").joinpath(_PACKAGED_FIXTURE)


@contextmanager
def packaged_compiled_black_scholes_fixture_path() -> Iterator[Path]:
    """Yield a filesystem path for the packaged compiled PDE fixture."""

    with resources.as_file(_packaged_compiled_black_scholes_fixture_resource()) as path:
        yield path


def screen_compiled_pde_payload(
    payload: Mapping[str, Any],
) -> CompiledPDEScreeningResult:
    """Validate and map a compiled PDE fixture into the native FD route envelope."""

    diagnostics = _validate(payload)
    route = _route(payload) if not diagnostics else {}
    return CompiledPDEScreeningResult(
        status="unsupported" if diagnostics else "supported",
        supported=not diagnostics,
        diagnostics=tuple(item.as_dict() for item in diagnostics),
        route=route,
    )


def solve_compiled_pde_payload(payload: Mapping[str, Any]) -> CompiledPDESolveResult:
    """Execute the exact validated compiled PDE fixture using maintained FD infrastructure."""

    return _solve_compiled_pde_version(payload, numerical_version="v0")


def solve_compiled_pde_payload_v1(payload: Mapping[str, Any]) -> CompiledPDESolveResult:
    """Opt into the versioned float64 banded route for the exact public fixture."""
    return _solve_compiled_pde_version(payload, numerical_version="v1")


def _solve_compiled_pde_version(payload: Mapping[str, Any], *, numerical_version: str) -> CompiledPDESolveResult:
    diagnostics = _validate(payload)
    if diagnostics:
        raise CompiledPDEAdapterError(diagnostics)

    route = _route(payload)
    report = (
        _run_compiled_black_scholes_route(route)
        if numerical_version == "v0"
        else _run_compiled_black_scholes_route(route, numerical_version=numerical_version)
    )
    values = {
        "price": report["price"],
        "oracle_price": report["oracle_price"],
        "delta": report["delta"],
        "reference_delta": report["reference_delta"],
        "gamma": report["gamma"],
        "reference_gamma": report["reference_gamma"],
    }
    diagnostics_payload = {
        "errors": report["errors"],
        "no_arbitrage": report["no_arbitrage"],
        "convergence": report["convergence"],
        "resource_controls": report["resource_controls"],
        "operator": report["operator"],
        "time_schedule": report["time_schedule"],
        "fallbacks": (),
        "unsupported_route_diagnostics": (),
    }
    evidence = {
        "adapter_schema_version": FIXTURE_SCHEMA_VERSION,
        "source_schema_version": SOURCE_PDE_IR_SCHEMA_VERSION,
        "compiled_schema_version": COMPILED_OPERATOR_SCHEMA_VERSION,
        "route_id": "fd.compiled_pde.black_scholes_call_" + numerical_version,
        "backend_id": DEFAULT_FD_CAPABILITY_MANIFEST.backend_id,
        "code_version": installed_distribution_version(),
        "config_hash": report["config_hash"],
        "fixture_id": EXPECTED_PROBLEM_ID,
        "seed": None,
        "source_ir_canonical_hash": route["source_ir_canonical_hash"],
        "compiled_hash": route["compiled_hash"],
        "problem_id": EXPECTED_PROBLEM_ID,
        "source_problem_id": route["source_problem_id"],
        "formulation_id": route["formulation_id"],
        "measure": route["measure"],
        "numeraire": route["numeraire"],
        "time_orientation": route["time_orientation"],
        "units": route["units"],
        "boundary_conditions": route["boundary_conditions"],
        "boundary_schedule_applied": report["boundary_schedule_applied"],
        "boundary_assumptions": report["boundary_assumptions"],
        "valuation_date": None,
        "maturity_date": None,
        "privacy_class": "public_synthetic",
        "resource_controls": report["resource_controls"],
        "status": "passed" if report["converged"] else "failed",
    }
    if numerical_version == "v1":
        from finite_difference_options.validation.fd_evidence.replay_identity import v1_runtime_identity

        evidence["numerical_runtime"] = v1_runtime_identity()
    return CompiledPDESolveResult(
        schema_version="finite-difference-options.compiled-pde-solve-result/" + numerical_version,
        backend_id=DEFAULT_FD_CAPABILITY_MANIFEST.backend_id,
        status="passed" if report["converged"] else "failed",
        problem_id=EXPECTED_PROBLEM_ID,
        values=values,
        diagnostics=diagnostics_payload,
        evidence=evidence,
        route=route,
    )


def _validate(payload: Mapping[str, Any]) -> tuple[CompiledPDEDiagnostic, ...]:
    diagnostics: list[CompiledPDEDiagnostic] = []
    if type(payload) is not dict:
        return (
            _diag(
                "compiled_pde.payload_type",
                "payload must be an exact JSON object",
                "payload",
            ),
        )
    if not _is_json_value(payload):
        return (
            _diag(
                "compiled_pde.json_type",
                "payload must contain only finite built-in JSON values",
                "payload",
            ),
        )

    root = cast(dict[str, Any], payload)
    _check_allowed_keys(
        diagnostics,
        root,
        {
            "artifact_manifest",
            "compiled_operator_result",
            "privacy_class",
            "problem_id",
            "schema_version",
            "solver_plan",
            "source_pde_ir",
        },
        "payload",
    )
    _expect(
        diagnostics,
        root.get("schema_version"),
        FIXTURE_SCHEMA_VERSION,
        "schema_version",
        "compiled_pde.schema_unsupported",
    )
    _expect(
        diagnostics,
        root.get("privacy_class"),
        "public_synthetic",
        "privacy_class",
        "compiled_pde.privacy_unsupported",
    )
    _expect(
        diagnostics,
        root.get("problem_id"),
        EXPECTED_PROBLEM_ID,
        "problem_id",
        "compiled_pde.problem_unsupported",
    )

    source = _dict_at(diagnostics, root, "source_pde_ir", "payload.source_pde_ir")
    compiled_result = _dict_at(
        diagnostics,
        root,
        "compiled_operator_result",
        "payload.compiled_operator_result",
    )
    solver = _dict_at(diagnostics, root, "solver_plan", "payload.solver_plan")
    compiled = (
        _dict_at(
            diagnostics,
            compiled_result,
            "compiled_operator",
            "payload.compiled_operator_result.compiled_operator",
        )
        if compiled_result
        else {}
    )

    _validate_source_ir(diagnostics, source)
    _validate_compiled_operator(diagnostics, compiled_result, compiled, source)
    _validate_solver_plan(diagnostics, solver)
    if not diagnostics and not matches_exact_public_fixture(root, packaged_compiled_black_scholes_fixture()):
        diagnostics.append(
            _diag(
                "compiled_pde.exact_fixture_mismatch",
                "only the exact public-synthetic compiled Black-Scholes fixture is executable",
                "payload",
                EXPECTED_PROBLEM_ID,
                str(root.get("problem_id")),
            )
        )
    return tuple(diagnostics)


def _route(payload: Mapping[str, Any]) -> dict[str, Any]:
    source = cast(Mapping[str, Any], payload.get("source_pde_ir", {}))
    compiled = cast(
        Mapping[str, Any],
        cast(Mapping[str, Any], payload.get("compiled_operator_result", {})).get("compiled_operator", {}),
    )
    boundaries = _list_of_dicts(source.get("boundary_conditions"))
    terminal = cast(Mapping[str, Any], source.get("terminal_condition", {}))
    numerics = dict(_COMPILED_ROUTE_NUMERICS)
    numerics["domain"] = dict(cast(dict[str, float], _COMPILED_ROUTE_NUMERICS["domain"]))
    numerics["grid_levels"] = tuple(cast(tuple[tuple[int, int], ...], _COMPILED_ROUTE_NUMERICS["grid_levels"]))
    return {
        "backend_id": DEFAULT_FD_CAPABILITY_MANIFEST.backend_id,
        "dimension": len(_list(source.get("state_variables"))),
        "problem_id": payload.get("problem_id"),
        "source_problem_id": source.get("problem_id"),
        "formulation_id": source.get("formulation_id"),
        "source_ir_canonical_hash": source.get("canonical_hash"),
        "compiled_hash": compiled.get("compiled_hash"),
        "measure": source.get("measure"),
        "numeraire": source.get("numeraire"),
        "time_orientation": source.get("time_orientation"),
        "operator_sign_convention": cast(Mapping[str, Any], source.get("operator", {})).get("sign_convention"),
        "boundary_conditions": tuple(sorted(str(item.get("kind")) for item in boundaries)),
        "boundary_details": {str(item.get("boundary_id")): dict(item) for item in boundaries},
        "units": {
            "value": terminal.get("unit"),
            "state": _state_units(source),
            "boundary": [item.get("unit") for item in boundaries],
        },
        "numerics": numerics,
        "compiler_evidence": compiled.get("compiler_evidence"),
    }


__all__ = [
    "CompiledPDEAdapterError",
    "CompiledPDEDiagnostic",
    "CompiledPDEScreeningResult",
    "CompiledPDESolveResult",
    "load_compiled_pde_json",
    "packaged_compiled_black_scholes_fixture",
    "packaged_compiled_black_scholes_fixture_path",
    "screen_compiled_pde_payload",
    "solve_compiled_pde_payload",
    "solve_compiled_pde_payload_v1",
]
