"""Private strict-JSON and exact source/compiler/solver-plan validation."""

from __future__ import annotations
import json
from collections.abc import Mapping
from hashlib import sha256
from typing import Any, cast
from finite_difference_options.contracts import DEFAULT_FD_CAPABILITY_MANIFEST
from ._compiled_pde_contracts import (
    SOURCE_PDE_IR_SCHEMA_ID,
    SOURCE_PDE_IR_SCHEMA_VERSION,
    COMPILED_OPERATOR_SCHEMA_ID,
    COMPILED_OPERATOR_SCHEMA_VERSION,
    EXPECTED_SOURCE_IR_HASH,
    EXPECTED_COMPILED_HASH,
    EXPECTED_PROBLEM_ID,
    EXPECTED_SOURCE_PROBLEM_ID,
    EXPECTED_FORMULATION_ID,
    EXPECTED_BOUNDARY_KINDS,
    EXPECTED_OUTPUTS,
    CompiledPDEDiagnostic,
)


def _validate_source_ir(diagnostics: list[CompiledPDEDiagnostic], source: Mapping[str, Any]) -> None:
    _expect(
        diagnostics,
        source.get("schema_id"),
        SOURCE_PDE_IR_SCHEMA_ID,
        "source_pde_ir.schema_id",
        "compiled_pde.source_schema_unsupported",
    )
    _expect(
        diagnostics,
        source.get("schema_version"),
        SOURCE_PDE_IR_SCHEMA_VERSION,
        "source_pde_ir.schema_version",
        "compiled_pde.source_schema_unsupported",
    )
    _expect(
        diagnostics,
        source.get("problem_id"),
        EXPECTED_SOURCE_PROBLEM_ID,
        "source_pde_ir.problem_id",
        "compiled_pde.source_problem_unsupported",
    )
    _expect(
        diagnostics,
        source.get("formulation_id"),
        EXPECTED_FORMULATION_ID,
        "source_pde_ir.formulation_id",
        "compiled_pde.formulation_unsupported",
    )
    _expect(
        diagnostics,
        source.get("formulation_kind"),
        "pde",
        "source_pde_ir.formulation_kind",
        "compiled_pde.formulation_unsupported",
    )
    _expect(
        diagnostics,
        source.get("privacy_class"),
        "public-synthetic",
        "source_pde_ir.privacy_class",
        "compiled_pde.privacy_unsupported",
    )
    _expect(
        diagnostics,
        source.get("measure"),
        "Q",
        "source_pde_ir.measure",
        "compiled_pde.measure_unsupported",
    )
    _expect(
        diagnostics,
        source.get("time_orientation"),
        "backward",
        "source_pde_ir.time_orientation",
        "compiled_pde.time_orientation_unsupported",
    )
    if _source_hash(source) != source.get("canonical_hash"):
        diagnostics.append(
            _diag(
                "compiled_pde.source_hash_mismatch",
                "source pde_ir canonical_hash does not match payload",
                "source_pde_ir.canonical_hash",
            )
        )
    _expect(
        diagnostics,
        source.get("canonical_hash"),
        EXPECTED_SOURCE_IR_HASH,
        "source_pde_ir.canonical_hash",
        "compiled_pde.source_hash_unsupported",
    )
    state_variables = _list_at(diagnostics, source, "state_variables", "source_pde_ir.state_variables")
    if state_variables is not None and len(state_variables) != 1:
        diagnostics.append(
            _diag(
                "compiled_pde.dimension_unsupported",
                "compiled adapter supports only the exact 1D state",
                "source_pde_ir.state_variables",
            )
        )
    boundary_items = _list_of_dicts_at(
        diagnostics,
        source,
        "boundary_conditions",
        "source_pde_ir.boundary_conditions",
    )
    if boundary_items is not None:
        boundary_kinds = tuple(sorted(str(item.get("kind")) for item in boundary_items))
    else:
        boundary_kinds = ()
    if boundary_items is not None and boundary_kinds != EXPECTED_BOUNDARY_KINDS:
        diagnostics.append(
            _diag(
                "compiled_pde.boundary_unsupported",
                "compiled adapter supports only exact dirichlet/asymptotic BS boundaries",
                "source_pde_ir.boundary_conditions",
                str(EXPECTED_BOUNDARY_KINDS),
                str(boundary_kinds),
            )
        )


def _validate_compiled_operator(
    diagnostics: list[CompiledPDEDiagnostic],
    result: Mapping[str, Any],
    compiled: Mapping[str, Any],
    source: Mapping[str, Any],
) -> None:
    _expect(
        diagnostics,
        result.get("accepted"),
        True,
        "compiled_operator_result.accepted",
        "compiled_pde.compiler_refusal",
    )
    _expect(
        diagnostics,
        compiled.get("schema_id"),
        COMPILED_OPERATOR_SCHEMA_ID,
        "compiled_operator.schema_id",
        "compiled_pde.compiled_schema_unsupported",
    )
    _expect(
        diagnostics,
        compiled.get("schema_version"),
        COMPILED_OPERATOR_SCHEMA_VERSION,
        "compiled_operator.schema_version",
        "compiled_pde.compiled_schema_unsupported",
    )
    _expect(
        diagnostics,
        compiled.get("source_ir_canonical_hash"),
        source.get("canonical_hash"),
        "compiled_operator.source_ir_canonical_hash",
        "compiled_pde.source_hash_mismatch",
    )
    if _compiled_hash(compiled) != compiled.get("compiled_hash"):
        diagnostics.append(
            _diag(
                "compiled_pde.compiled_hash_mismatch",
                "compiled operator hash does not match payload",
                "compiled_operator.compiled_hash",
            )
        )
    _expect(
        diagnostics,
        compiled.get("compiled_hash"),
        EXPECTED_COMPILED_HASH,
        "compiled_operator.compiled_hash",
        "compiled_pde.compiled_hash_unsupported",
    )
    evidence = compiled.get("compiler_evidence") if isinstance(compiled.get("compiler_evidence"), Mapping) else {}
    _expect(
        diagnostics,
        cast(Mapping[str, Any], evidence).get("compiler_version"),
        "pde_ir_symbolic_compiler.v0",
        "compiled_operator.compiler_evidence.compiler_version",
        "compiled_pde.compiler_unsupported",
    )


def _validate_solver_plan(diagnostics: list[CompiledPDEDiagnostic], solver: Mapping[str, Any]) -> None:
    _expect(
        diagnostics,
        solver.get("backend_id"),
        DEFAULT_FD_CAPABILITY_MANIFEST.backend_id,
        "solver_plan.backend_id",
        "compiled_pde.backend_unsupported",
    )
    _expect(
        diagnostics,
        solver.get("exercise_style"),
        "european",
        "solver_plan.exercise_style",
        "compiled_pde.exercise_unsupported",
    )
    _expect(
        diagnostics,
        solver.get("grid_type"),
        "uniform",
        "solver_plan.grid_type",
        "compiled_pde.grid_unsupported",
    )
    requested_outputs = _list_at(diagnostics, solver, "requested_outputs", "solver_plan.requested_outputs")
    if requested_outputs is not None:
        outputs = tuple(sorted(str(item) for item in requested_outputs))
    else:
        outputs = ()
    if requested_outputs is not None and outputs != EXPECTED_OUTPUTS:
        diagnostics.append(
            _diag(
                "compiled_pde.output_unsupported",
                "compiled adapter supports exactly value/delta/gamma outputs",
                "solver_plan.requested_outputs",
                str(EXPECTED_OUTPUTS),
                str(outputs),
            )
        )


def _source_hash(source: Mapping[str, Any]) -> str:
    payload = dict(source)
    payload.pop("canonical_hash", None)
    return _sha256_ref(payload)


def _compiled_hash(compiled: Mapping[str, Any]) -> str:
    payload = dict(compiled)
    payload.pop("compiled_hash", None)
    return _sha256_ref(payload)


def _sha256_ref(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return f"sha256:{sha256(encoded.encode('utf-8')).hexdigest()}"


def _state_units(source: Mapping[str, Any]) -> list[Any]:
    return [item.get("unit") for item in _list_of_dicts(source.get("state_variables"))]


def _dict_at(
    diagnostics: list[CompiledPDEDiagnostic],
    mapping: Mapping[str, Any],
    key: str,
    path: str | None = None,
) -> Mapping[str, Any]:
    diagnostic_path = path or key
    value = mapping.get(key)
    if type(value) is dict:
        return cast(Mapping[str, Any], value)
    if key in mapping:
        diagnostics.append(
            _diag(
                "compiled_pde.object_type",
                f"{diagnostic_path} must be a JSON object",
                diagnostic_path,
                "object",
                _json_type_name(value),
            )
        )
        return {}
    diagnostics.append(
        _diag(
            "compiled_pde.object_missing",
            f"{diagnostic_path} must be a JSON object",
            diagnostic_path,
            "object",
            "missing",
        )
    )
    return {}


def _list_at(
    diagnostics: list[CompiledPDEDiagnostic],
    mapping: Mapping[str, Any],
    key: str,
    path: str,
) -> list[Any] | None:
    value = mapping.get(key)
    if type(value) is list:
        return cast(list[Any], value)
    diagnostics.append(
        _diag(
            "compiled_pde.list_type",
            f"{path} must be a JSON array",
            path,
            "array",
            _json_type_name(value) if key in mapping else "missing",
        )
    )
    return None


def _list_of_dicts_at(
    diagnostics: list[CompiledPDEDiagnostic],
    mapping: Mapping[str, Any],
    key: str,
    path: str,
) -> list[dict[str, Any]] | None:
    values = _list_at(diagnostics, mapping, key, path)
    if values is None:
        return None
    items: list[dict[str, Any]] = []
    for index, item in enumerate(values):
        if type(item) is dict:
            items.append(cast(dict[str, Any], item))
        else:
            diagnostics.append(
                _diag(
                    "compiled_pde.object_type",
                    f"{path}[{index}] must be a JSON object",
                    f"{path}[{index}]",
                    "object",
                    _json_type_name(item),
                )
            )
    return items


def _check_allowed_keys(
    diagnostics: list[CompiledPDEDiagnostic],
    mapping: Mapping[str, Any],
    allowed: set[str],
    path: str,
) -> None:
    for key in mapping:
        if key not in allowed:
            diagnostics.append(
                _diag(
                    "compiled_pde.unknown_field",
                    "unknown fields are rejected before solve",
                    f"{path}.{key}",
                )
            )


def _expect(
    diagnostics: list[CompiledPDEDiagnostic],
    observed: object,
    expected: object,
    path: str,
    code: str,
) -> None:
    if observed != expected or type(observed) is not type(expected):
        diagnostics.append(_diag(code, f"unsupported value at {path}", path, str(expected), str(observed)))


def _diag(
    code: str,
    message: str,
    path: str,
    expected: str | None = None,
    observed: str | None = None,
) -> CompiledPDEDiagnostic:
    return CompiledPDEDiagnostic(code=code, message=message, path=path, expected=expected, observed=observed)


def _list(value: object) -> list[Any]:
    return value if type(value) is list else []


def _list_of_dicts(value: object) -> list[dict[str, Any]]:
    return [cast(dict[str, Any], item) for item in _list(value) if type(item) is dict]


def _json_type_name(value: object) -> str:
    if type(value) is dict:
        return "object"
    if type(value) is list:
        return "array"
    if type(value) is str:
        return "string"
    if type(value) is bool:
        return "boolean"
    if type(value) in {int, float}:
        return "number"
    if value is None:
        return "null"
    return type(value).__name__


def _is_json_value(value: object) -> bool:
    try:
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return False
    return type(value) in {dict, list, str, int, float, bool, type(None)} or value is None


def _reject_non_finite_json(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")
