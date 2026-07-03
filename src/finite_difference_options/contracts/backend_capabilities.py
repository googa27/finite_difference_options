"""Finite-difference backend capability manifest and route diagnostics.

This module is intentionally domain-neutral. It describes what the FD backend
can support before any numerical grid/operator allocation happens and maps the
shared QuantProblemSpec vocabulary into FD adapter inputs without importing
Haircut Engine or other repository internals.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CapabilityStatus(str, Enum):
    """Maturity of one advertised backend capability."""

    PRODUCTION = "production"
    VALIDATED = "validated"
    EXPERIMENTAL = "experimental"
    UNSUPPORTED = "unsupported"


class UnsupportedReason(str, Enum):
    """Stable diagnostic codes for unsupported solver-route requests."""

    UNSUPPORTED_DIMENSION = "unsupported_dimension"
    UNSUPPORTED_GRID = "unsupported_grid"
    UNSUPPORTED_TERM = "unsupported_pde_term"
    UNSUPPORTED_BOUNDARY = "unsupported_boundary_condition"
    UNSUPPORTED_EXERCISE = "unsupported_exercise_style"
    UNSUPPORTED_OUTPUT = "unsupported_output"
    UNSUPPORTED_STABILITY_CONTROL = "unsupported_stability_control"
    UNSUPPORTED_BACKEND = "unsupported_backend"
    UNSUPPORTED_BENCHMARK = "unsupported_benchmark"
    MISSING_CONVENTION = "missing_convention"


@dataclass(frozen=True)
class UnsupportedRouteDiagnostic:
    """Actionable reason why an FD route request is unsupported."""

    reason: UnsupportedReason
    field: str
    value: str
    supported: tuple[str, ...]
    message: str


@dataclass(frozen=True)
class FDCapabilityManifest:
    """Declarative finite-difference backend support matrix.

    The manifest is intentionally data-only so routers, agents, and docs can
    inspect it before importing any concrete numerical implementation.
    """

    backend_id: str
    contract_version: str
    status: CapabilityStatus
    supported_dimensions: tuple[int, ...]
    grid_types: tuple[str, ...]
    pde_terms: tuple[str, ...]
    boundary_conditions: tuple[str, ...]
    exercise_styles: tuple[str, ...]
    outputs: tuple[str, ...]
    stability_controls: tuple[str, ...]
    required_conventions: tuple[str, ...]
    diagnostics: tuple[str, ...]
    feature_support: Mapping[str, str] = field(default_factory=dict)
    error_budgets: Mapping[str, float | str] = field(default_factory=dict)
    resource_controls: Mapping[str, int | float | str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def supports(self, request: FDRouteRequest) -> bool:
        """Return ``True`` only when no fail-closed diagnostics are produced."""

        return not diagnose_unsupported_route(request, self)


@dataclass(frozen=True)
class FDRouteRequest:
    """FD adapter request distilled from a QuantProblemSpec-like payload.

    This is not the numerical problem itself. It is the capability-screening
    envelope that preserves conventions agents must not drop: measure,
    numeraire, units, valuation/maturity dates, boundary classes, requested
    outputs and stability policy.
    """

    dimension: int
    grid_type: str
    pde_terms: tuple[str, ...]
    boundary_conditions: tuple[str, ...]
    exercise_style: str
    requested_outputs: tuple[str, ...]
    stability_controls: tuple[str, ...]
    measure: str | None
    numeraire: str | None
    units: Mapping[str, str] = field(default_factory=dict)
    boundary_details: Mapping[str, str] = field(default_factory=dict)
    valuation_date: str | None = None
    maturity_date: str | None = None
    time_domain: str | None = None
    source_schema_version: str | None = None
    backend_id: str | None = None

    @classmethod
    def from_quant_problem_spec(cls, payload: Mapping[str, Any]) -> FDRouteRequest:
        """Map a shared QuantProblemSpec-like mapping to an FD route request.

        Accepted keys deliberately mirror the v0 cross-repo problem vocabulary
        while remaining permissive about nesting so public-synthetic fixtures can
        evolve without forcing a hard dependency on Haircut Engine.
        """

        math = _mapping(payload.get("mathematical_problem"))
        solver = _mapping(payload.get("solver_plan"))
        context = _mapping(payload.get("valuation_context"))
        conventions = _mapping(payload.get("conventions"))
        vintage = _mapping(payload.get("vintage"))
        domain = _mapping(math.get("domain"))
        boundary_details = _mapping(math.get("boundary_conditions"))

        dimension = _coerce_dimension(
            _first_present(
                math,
                ("dimension", "dimensions", "state_dimension"),
                default=_state_dimension(math.get("state_variables")),
            )
        )
        grid_type = str(_first_present(solver, ("grid_type", "grid", "grid_family"), default="uniform"))
        exercise_style = str(_first_present(math, ("exercise_style", "exercise"), default="european"))

        return cls(
            dimension=dimension,
            grid_type=grid_type,
            pde_terms=_tuple_of_strings(
                _first_present(
                    math,
                    ("pde_terms", "terms"),
                    default=("drift", "diffusion", "reaction"),
                )
            ),
            boundary_conditions=_boundary_condition_classes(
                _first_present(
                    math,
                    ("boundary_types", "boundaries", "boundary_conditions"),
                    default=("dirichlet",),
                )
            ),
            exercise_style=exercise_style,
            requested_outputs=_tuple_of_strings(
                _first_present(
                    solver,
                    ("requested_outputs", "required_outputs", "outputs"),
                    default=("value",),
                )
            ),
            stability_controls=_tuple_of_strings(
                _first_present(solver, ("stability_controls", "stability"), default=("theta",))
            ),
            measure=_optional_string(
                _first_present(
                    context,
                    ("measure",),
                    default=_first_present(
                        math,
                        ("measure_id", "measure"),
                        default=conventions.get("measure"),
                    ),
                )
            ),
            numeraire=_optional_string(
                _first_present(
                    context,
                    ("numeraire",),
                    default=_first_present(
                        math,
                        ("numeraire_id", "numeraire"),
                        default=conventions.get("numeraire"),
                    ),
                )
            ),
            units=_mapping(
                _first_present(
                    context,
                    ("units",),
                    default=_first_present(math, ("units",), default=conventions.get("units", {})),
                )
            ),
            boundary_details={str(key): str(value) for key, value in boundary_details.items()},
            valuation_date=_optional_string(
                _first_present(
                    context,
                    ("valuation_date", "as_of_date"),
                    default=vintage.get("valuation_date"),
                )
            ),
            maturity_date=_optional_string(
                _first_present(context, ("maturity_date",), default=vintage.get("maturity_date"))
            ),
            time_domain=_optional_string(_first_present(context, ("time_domain",), default=domain.get("t"))),
            source_schema_version=_optional_string(payload.get("schema_version")),
            backend_id=_optional_string(_first_present(solver, ("backend_id", "backend"), default=None)),
        )


DEFAULT_FD_CAPABILITY_MANIFEST = FDCapabilityManifest(
    backend_id="finite_difference_options.fd_backend.v0",
    contract_version="0.1.0",
    status=CapabilityStatus.VALIDATED,
    supported_dimensions=(1, 2, 3),
    grid_types=("uniform", "log_uniform"),
    pde_terms=("drift", "diffusion", "reaction", "source", "mixed_derivative"),
    boundary_conditions=("dirichlet", "neumann", "robin", "second_derivative"),
    exercise_styles=("european", "bermudan", "american"),
    outputs=("value", "delta", "gamma", "exercise_boundary"),
    stability_controls=(
        "theta",
        "crank_nicolson",
        "rannacher",
        "explicit_euler",
        "adi_psor",
        "projected_sor_lcp",
        "policy_iteration_lcp",
    ),
    required_conventions=(
        "measure",
        "numeraire",
        "units",
        "valuation_date",
        "maturity_date",
    ),
    diagnostics=(
        "unsupported dimension",
        "unsupported PDE term",
        "unsupported boundary condition",
        "unsupported exercise style",
        "obstacle/complementarity diagnostics",
        "missing measure/numeraire/units/date convention",
    ),
    feature_support={
        "pinares_fixed_price_proxy": CapabilityStatus.VALIDATED.value,
        "one_dimensional_generator_pde": CapabilityStatus.VALIDATED.value,
        "mixed_derivative": CapabilityStatus.EXPERIMENTAL.value,
        "jump_integral": CapabilityStatus.UNSUPPORTED.value,
        "obstacle_lcp": CapabilityStatus.VALIDATED.value,
        "american_black_scholes_lcp": CapabilityStatus.VALIDATED.value,
        "bermudan_black_scholes_lcp": CapabilityStatus.VALIDATED.value,
        "multidimensional_american_lcp": CapabilityStatus.EXPERIMENTAL.value,
        "hjb_control": CapabilityStatus.UNSUPPORTED.value,
        "rofr_full_family_contract": CapabilityStatus.UNSUPPORTED.value,
    },
    error_budgets={
        "black_scholes_price_abs": 5.0e-4,
        "black_scholes_delta_abs": 5.0e-2,
        "black_scholes_gamma_abs": 2.0e-2,
        "american_lcp_primal_abs": 5.0e-8,
        "american_lcp_dual_abs": 5.0e-5,
        "american_lcp_complementarity_abs": 5.0e-4,
        "pinares_fixed_price_proxy_price_abs_uf": 1.0,
        "pinares_fixed_price_proxy_delta_abs": 1.0e-3,
        "pinares_fixed_price_proxy_gamma_abs": 5.0e-6,
    },
    resource_controls={
        "default_time_scheme": "theta",
        "default_theta": 0.5,
        "pinares_fixed_price_proxy_max_s_steps": 180,
        "pinares_fixed_price_proxy_max_t_steps": 240,
        "deterministic": "true",
    },
    notes=(
        "American/LCP exercise is intentionally unsupported until complementarity diagnostics land.",
        "Jump/PIDE and HJB/control terms must fail closed instead of using placeholder coefficients.",
    ),
)


def diagnose_unsupported_route(
    request: FDRouteRequest,
    manifest: FDCapabilityManifest = DEFAULT_FD_CAPABILITY_MANIFEST,
) -> tuple[UnsupportedRouteDiagnostic, ...]:
    """Return fail-closed diagnostics for unsupported request fields."""

    diagnostics: list[UnsupportedRouteDiagnostic] = []

    if request.backend_id and request.backend_id != manifest.backend_id:
        diagnostics.append(
            _diagnostic(
                UnsupportedReason.UNSUPPORTED_BACKEND,
                "backend_id",
                request.backend_id,
                (manifest.backend_id,),
                f"Unsupported backend_id {request.backend_id!r}; expected {manifest.backend_id!r}.",
            )
        )

    if request.dimension not in manifest.supported_dimensions:
        diagnostics.append(
            _diagnostic(
                UnsupportedReason.UNSUPPORTED_DIMENSION,
                "dimension",
                str(request.dimension),
                tuple(str(item) for item in manifest.supported_dimensions),
                f"FD backend supports dimensions {manifest.supported_dimensions}, got {request.dimension}D.",
            )
        )

    _extend_set_diagnostics(
        diagnostics,
        UnsupportedReason.UNSUPPORTED_GRID,
        "grid_type",
        (request.grid_type,),
        manifest.grid_types,
    )
    _extend_set_diagnostics(
        diagnostics,
        UnsupportedReason.UNSUPPORTED_TERM,
        "pde_terms",
        request.pde_terms,
        manifest.pde_terms,
    )
    _extend_set_diagnostics(
        diagnostics,
        UnsupportedReason.UNSUPPORTED_BOUNDARY,
        "boundary_conditions",
        request.boundary_conditions,
        manifest.boundary_conditions,
    )
    _extend_set_diagnostics(
        diagnostics,
        UnsupportedReason.UNSUPPORTED_EXERCISE,
        "exercise_style",
        (request.exercise_style,),
        manifest.exercise_styles,
    )
    _extend_set_diagnostics(
        diagnostics,
        UnsupportedReason.UNSUPPORTED_OUTPUT,
        "requested_outputs",
        request.requested_outputs,
        manifest.outputs,
    )
    _extend_set_diagnostics(
        diagnostics,
        UnsupportedReason.UNSUPPORTED_STABILITY_CONTROL,
        "stability_controls",
        request.stability_controls,
        manifest.stability_controls,
    )

    conventions = {
        "measure": request.measure,
        "numeraire": request.numeraire,
        "units": request.units,
        "valuation_date": request.valuation_date,
        "maturity_date": request.maturity_date or request.time_domain,
    }
    for field_name in manifest.required_conventions:
        if not conventions.get(field_name):
            diagnostics.append(
                _diagnostic(
                    UnsupportedReason.MISSING_CONVENTION,
                    field_name,
                    "<missing>",
                    ("required",),
                    f"QuantProblemSpec mapping must preserve {field_name}; it was missing or empty.",
                )
            )

    return tuple(diagnostics)


def ensure_route_supported(
    request: FDRouteRequest,
    manifest: FDCapabilityManifest = DEFAULT_FD_CAPABILITY_MANIFEST,
) -> None:
    """Raise :class:`UnsupportedRouteError` if the manifest rejects the request."""

    diagnostics = diagnose_unsupported_route(request, manifest)
    if diagnostics:
        raise UnsupportedRouteError(diagnostics)


class UnsupportedRouteError(ValueError):
    """Raised when a route request is unsupported before numerical work."""

    def __init__(self, diagnostics: Iterable[UnsupportedRouteDiagnostic]) -> None:
        self.diagnostics = tuple(diagnostics)
        reasons = "; ".join(d.message for d in self.diagnostics)
        super().__init__(reasons)


def _diagnostic(
    reason: UnsupportedReason,
    field_name: str,
    value: str,
    supported: tuple[str, ...],
    message: str,
) -> UnsupportedRouteDiagnostic:
    return UnsupportedRouteDiagnostic(
        reason=reason,
        field=field_name,
        value=value,
        supported=supported,
        message=message,
    )


def _extend_set_diagnostics(
    diagnostics: list[UnsupportedRouteDiagnostic],
    reason: UnsupportedReason,
    field_name: str,
    requested: tuple[str, ...],
    supported: tuple[str, ...],
) -> None:
    supported_set = set(supported)
    for value in requested:
        if value not in supported_set:
            diagnostics.append(
                _diagnostic(
                    reason,
                    field_name,
                    value,
                    supported,
                    f"Unsupported {field_name} value {value!r}; supported values are {supported}.",
                )
            )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_present(mapping: Mapping[str, Any], keys: tuple[str, ...], *, default: Any) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _tuple_of_strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable) and not isinstance(value, Mapping):
        return tuple(str(item) for item in value)
    return (str(value),)


def _state_dimension(value: Any) -> int:
    if isinstance(value, str):
        return 1
    if isinstance(value, Iterable) and not isinstance(value, Mapping):
        values = tuple(value)
        return len(values) or 1
    return 1


def _coerce_dimension(value: Any) -> int:
    """Return a positive integer dimension, or -1 for fail-closed diagnostics."""

    if isinstance(value, bool):
        return -1
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        return int(text) if text.isdigit() else -1
    if isinstance(value, Iterable) and not isinstance(value, Mapping):
        return _state_dimension(value)
    return -1


def _boundary_condition_classes(value: Any) -> tuple[str, ...]:
    """Normalize public schema boundary formulas to FD capability classes."""

    raw_items: Iterable[Any]
    if isinstance(value, Mapping):
        raw_items = value.values()
    else:
        raw_items = _tuple_of_strings(value)

    classes: list[str] = []
    for item in raw_items:
        text = str(item).lower().replace("-", "_")
        if "robin" in text:
            boundary_class = "robin"
        elif "second" in text:
            boundary_class = "second_derivative"
        elif "linear" in text or "slope" in text or "growth" in text:
            boundary_class = "neumann"
        elif "neumann" in text:
            boundary_class = "neumann"
        elif "dirichlet" in text or "absorbing" in text or text.strip() in {"0", "zero"}:
            boundary_class = "dirichlet"
        else:
            boundary_class = text
        if boundary_class not in classes:
            classes.append(boundary_class)
    return tuple(classes) or ("dirichlet",)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


__all__ = [
    "CapabilityStatus",
    "DEFAULT_FD_CAPABILITY_MANIFEST",
    "FDCapabilityManifest",
    "FDRouteRequest",
    "UnsupportedReason",
    "UnsupportedRouteDiagnostic",
    "UnsupportedRouteError",
    "diagnose_unsupported_route",
    "ensure_route_supported",
]
