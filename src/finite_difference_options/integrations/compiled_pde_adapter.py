"""Fail-closed adapter for exact public-synthetic compiled ``pde_ir.v0`` fixtures."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any, Literal, cast


from finite_difference_options.contracts import DEFAULT_FD_CAPABILITY_MANIFEST
from finite_difference_options.integrations.compiled_pde_black_scholes_route import (
    _run_compiled_black_scholes_route,
)
from finite_difference_options.integrations.haircut_protocol import _distribution_version
from finite_difference_options.integrations.public_fixture_identity import (
    matches_exact_public_fixture,
)

CompiledPDEStatus = Literal["supported", "unsupported"]
CompiledPDESolveStatus = Literal["passed", "failed"]

FIXTURE_SCHEMA_VERSION = "finite-difference-options.compiled-pde-adapter-fixture/v0"
SOURCE_PDE_IR_SCHEMA_ID = "financial_problem_formulations.pde_ir.v0"
SOURCE_PDE_IR_SCHEMA_VERSION = "pde_ir.v0"
COMPILED_OPERATOR_SCHEMA_ID = "financial_problem_formulations.pde_ir.compiled_symbolic_operator.v0"
COMPILED_OPERATOR_SCHEMA_VERSION = "compiled_symbolic_operator.v0"
EXPECTED_SOURCE_IR_HASH = "sha256:5ab53779a5e322284a6cb18b22302c119f22bc740659aedf1c07823529d68a47"
EXPECTED_COMPILED_HASH = "sha256:970088e5dcb16535edfd230bfe992ea7eb68aede901c7b543682b39f1a5ac32e"
EXPECTED_PROBLEM_ID = "public-synthetic.compiled-pde.black-scholes-call.v0"
EXPECTED_SOURCE_PROBLEM_ID = "black_scholes_call_public_synthetic"
EXPECTED_FORMULATION_ID = "black_scholes_call_pde_v0"
EXPECTED_BOUNDARY_KINDS = ("asymptotic", "dirichlet")
EXPECTED_OUTPUTS = ("delta", "gamma", "value")
_PACKAGED_FIXTURE = "compiled_pde_black_scholes_call_v0.json"
_COMPILED_ROUTE_NUMERICS = {
    "spot": 1.0,
    "strike": 1.0,
    "risk_free_rate": 0.05,
    "dividend_yield": 0.0,
    "volatility": 0.2,
    "maturity": 1.0,
    "domain": {"s_min": 0.0, "s_max": 3.0, "t_min": 0.0, "t_max": 1.0},
    "grid_levels": ((40, 40), (80, 120), (120, 200)),
    "theta": 0.5,
    "tolerances": {"price": 5.0e-4, "delta": 1.0e-3, "gamma": 8.0e-3},
}


@dataclass(frozen=True)
class CompiledPDEDiagnostic:
    """Stable unsupported-route diagnostic for compiled PDE payloads."""

    code: str
    message: str
    path: str
    expected: str | None = None
    observed: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return asdict(self)


@dataclass(frozen=True)
class CompiledPDEScreeningResult:
    """Screening result emitted before numerical work."""

    status: CompiledPDEStatus
    supported: bool
    diagnostics: tuple[dict[str, str | None], ...]
    route: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompiledPDESolveResult:
    """Deterministic solve/result bundle for the exact compiled fixture."""

    schema_version: str
    backend_id: str
    status: CompiledPDESolveStatus
    problem_id: str
    values: dict[str, float]
    diagnostics: dict[str, Any]
    evidence: dict[str, Any]
    route: dict[str, Any]

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class CompiledPDEAdapterError(ValueError):
    """Raised when a compiled PDE payload is unsupported before solve."""

    def __init__(self, diagnostics: Sequence[CompiledPDEDiagnostic]) -> None:
        self.diagnostics = tuple(diagnostics)
        super().__init__("; ".join(f"{item.code}: {item.message}" for item in self.diagnostics))


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

    diagnostics = _validate(payload)
    if diagnostics:
        raise CompiledPDEAdapterError(diagnostics)

    route = _route(payload)
    report = _run_compiled_black_scholes_route(route)
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
        "route_id": "fd.compiled_pde.black_scholes_call_v0",
        "backend_id": DEFAULT_FD_CAPABILITY_MANIFEST.backend_id,
        "code_version": _distribution_version(),
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
    return CompiledPDESolveResult(
        schema_version="finite-difference-options.compiled-pde-solve-result/v0",
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
]
