"""Released public-synthetic FD solver contract helpers.

The helpers in this module are the small, wheel-exported contract surface for
external callers that want to verify the finite_difference_options backend
against checked-in public QuantProblemSpec fixtures without importing Haircut
Engine.  They screen the payload with the data-only capability manifest and then
execute only exact public-synthetic fixtures owned by this repository.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

from finite_difference_options.contracts import (
    DEFAULT_FD_CAPABILITY_MANIFEST,
    FDCapabilityManifest,
    FDRouteRequest,
    UnsupportedReason,
    UnsupportedRouteDiagnostic,
    UnsupportedRouteError,
    ensure_route_supported,
)
from finite_difference_options.integrations.public_fixture_identity import matches_exact_public_fixture

PublicContractStatus = Literal["passed", "failed"]


@dataclass(frozen=True)
class ReleasedFDSolverContract:
    """Data-only public solver-contract manifest for released integrations."""

    schema_version: str
    backend_id: str
    contract_version: str
    adapter_schema_version: str
    maturity: str
    supported_problem_ids: tuple[str, ...]
    supported_privacy_classes: tuple[str, ...]
    capability_manifest: dict[str, Any]
    entry_points: tuple[str, ...]
    entry_point_groups: tuple[str, ...]
    issue_refs: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PublicFDSolverResult:
    """Normalized result returned by :func:`solve_public_quant_problem_spec`."""

    schema_version: str
    backend_id: str
    contract_version: str
    status: PublicContractStatus
    problem_id: str
    values: dict[str, float]
    diagnostics: dict[str, Any]
    evidence: dict[str, Any]
    request: dict[str, Any]

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def released_fd_solver_contract(
    manifest: FDCapabilityManifest = DEFAULT_FD_CAPABILITY_MANIFEST,
) -> ReleasedFDSolverContract:
    """Return the released public FD backend contract metadata."""

    manifest_dict = asdict(manifest)
    manifest_dict["status"] = manifest.status.value
    return ReleasedFDSolverContract(
        schema_version="finite-difference-options.public-fd-solver-contract/v0",
        backend_id=manifest.backend_id,
        contract_version=manifest.contract_version,
        adapter_schema_version="public-fd-solver-contract/v0",
        maturity="validated_public_synthetic",
        supported_problem_ids=(
            "public-synthetic.black-scholes-call.v0",
            "pinares.fixed_price_option_proxy.v1",
            "public-synthetic.compiled-pde.black-scholes-call.v0",
        ),
        supported_privacy_classes=("public_synthetic",),
        capability_manifest=manifest_dict,
        entry_points=(
            "finite_difference_options.integrations.public_solver_contract:solve_public_quant_problem_spec",
            "finite_difference_options.integrations.compiled_pde_adapter:solve_compiled_pde_payload",
            "finite_difference_options.integrations.haircut_backend:create_backend",
        ),
        entry_point_groups=("haircut.solver_backends",),
        issue_refs=(
            "googa27/finite_difference_options#55",
            "googa27/finite_difference_options#130",
            "googa27/finite_difference_options#139",
            "googa27/finite_difference_options#141",
        ),
    )


def solve_public_quant_problem_spec(
    payload: Mapping[str, Any],
    *,
    operator_cache: Any | None = None,
    manifest: FDCapabilityManifest = DEFAULT_FD_CAPABILITY_MANIFEST,
) -> PublicFDSolverResult:
    """Screen and execute an exact public-synthetic QuantProblemSpec fixture.

    Only the canonical Black-Scholes and Pinares fixed-price proxy public
    fixtures are executable.  Other mathematically supported-looking payloads
    fail closed with ``UnsupportedRouteError`` until this repository owns an
    executable parity fixture for them.
    """

    request = FDRouteRequest.from_quant_problem_spec(payload)
    ensure_route_supported(request, manifest)
    problem_id = _optional_string(payload.get("problem_id"))
    privacy_class = _optional_string(payload.get("privacy_class"))
    if not _is_public_synthetic_contract_payload(problem_id, privacy_class, payload):
        raise UnsupportedRouteError(
            (
                UnsupportedRouteDiagnostic(
                    reason=UnsupportedReason.UNSUPPORTED_BENCHMARK,
                    field="problem_id",
                    value=problem_id or "<missing>",
                    supported=released_fd_solver_contract(manifest).supported_problem_ids,
                    message=(
                        "Public FD solver contract executes only exact validated public-synthetic "
                        "QuantProblemSpec fixtures."
                    ),
                ),
            )
        )

    report: Any
    if problem_id == "pinares.fixed_price_option_proxy.v1":
        from finite_difference_options.validation.pinares_fixed_price_proxy import (
            run_public_pinares_fixed_price_proxy_fixture,
        )

        report = run_public_pinares_fixed_price_proxy_fixture(operator_cache=operator_cache)
        values = {
            "price": report.price_uf,
            "oracle_price": report.oracle_price_uf,
            "delta": report.delta,
            "reference_delta": report.reference_delta,
            "gamma": report.gamma,
            "reference_gamma": report.reference_gamma,
        }
    else:
        from finite_difference_options.validation.black_scholes_parity import (
            run_public_black_scholes_parity_fixture,
        )

        report = run_public_black_scholes_parity_fixture(operator_cache=operator_cache)
        values = {
            "price": report.price,
            "oracle_price": report.oracle_price,
            "delta": report.delta,
            "reference_delta": report.reference_delta,
            "gamma": report.gamma,
            "reference_gamma": report.reference_gamma,
        }

    assert problem_id is not None
    diagnostics = {
        "errors": report.errors,
        "no_arbitrage": report.no_arbitrage,
        "convergence": report.convergence_table(),
        "resource_controls": report.evidence.resource_controls,
        "operator_cache": (
            operator_cache.info().as_dict() if operator_cache is not None and hasattr(operator_cache, "info") else None
        ),
        "fallbacks": (),
    }
    evidence = {
        **report.evidence.as_dict(),
        "contract_family": "FPF.solver_result_evidence.v1",
        "adapter_schema_version": released_fd_solver_contract(manifest).adapter_schema_version,
        "source_schema_version": request.source_schema_version,
        "problem_id": problem_id,
        "privacy_class": privacy_class,
        "measure": request.measure,
        "numeraire": request.numeraire,
        "units": dict(request.units),
        "status": "passed" if report.converged else "failed",
        "backend_capability_status": manifest.status.value,
    }
    return PublicFDSolverResult(
        schema_version="finite-difference-options.public-fd-solver-result/v0",
        backend_id=manifest.backend_id,
        contract_version=manifest.contract_version,
        status="passed" if report.converged else "failed",
        problem_id=problem_id,
        values=values,
        diagnostics=diagnostics,
        evidence=evidence,
        request=asdict(request),
    )


def _is_public_synthetic_contract_payload(
    problem_id: str | None,
    privacy_class: str | None,
    payload: Mapping[str, Any],
) -> bool:
    if privacy_class != "public_synthetic":
        return False
    if problem_id == "public-synthetic.black-scholes-call.v0":
        from finite_difference_options.validation.black_scholes_parity import public_black_scholes_problem_spec

        expected = public_black_scholes_problem_spec()
    elif problem_id == "pinares.fixed_price_option_proxy.v1":
        from finite_difference_options.validation.pinares_fixed_price_proxy import (
            public_pinares_fixed_price_problem_spec,
        )

        expected = public_pinares_fixed_price_problem_spec()
    else:
        return False
    return matches_exact_public_fixture(payload, expected)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


__all__ = [
    "PublicFDSolverResult",
    "ReleasedFDSolverContract",
    "released_fd_solver_contract",
    "solve_public_quant_problem_spec",
]
