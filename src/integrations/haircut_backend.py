"""Haircut Engine finite-difference backend adapter.

This module is deliberately thin and domain-neutral. It exposes a Haircut-style
backend object without importing Haircut Engine, PDP, API, CLI, UI, plotting, or
frontend packages. The adapter first screens a shared QuantProblemSpec-like
payload through the FD capability manifest. It only executes the public-synthetic
Black-Scholes fixture that this repository owns as validated evidence; all other
supported-looking requests fail closed until a concrete native route is proven.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

from src.contracts import (
    DEFAULT_FD_CAPABILITY_MANIFEST,
    FDCapabilityManifest,
    FDRouteRequest,
    UnsupportedReason,
    UnsupportedRouteDiagnostic,
    UnsupportedRouteError,
    diagnose_unsupported_route,
    ensure_route_supported,
)

AdapterStatus = Literal["supported", "unsupported"]
SolveStatus = Literal["passed", "failed"]

_EXECUTABLE_PUBLIC_BENCHMARKS = {
    "black-scholes-call-001",
    "BS-CALL-PARITY-V0",
    "QPS-VANILLA-CALL-V0",
}
_EXECUTED_REGISTRY_BENCHMARKS = ("BS-CALL-PARITY-V0", "QPS-VANILLA-CALL-V0")
_PUBLIC_SYNTHETIC_PROBLEM_IDS = {
    "public-synthetic-vanilla-call-v0",
    "public-synthetic.black-scholes-call.v0",
}


@dataclass(frozen=True)
class HaircutBackendIdentity:
    """Lightweight backend identity suitable for plugin discovery."""

    backend_id: str
    name: str
    adapter_schema_version: str
    contract_version: str
    maturity: str
    entry_point: str
    issue_refs: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FDBackendScreeningResult:
    """Capability-screening result emitted before numerical work."""

    backend_id: str
    status: AdapterStatus
    request: dict[str, Any]
    diagnostics: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]

    @property
    def supported(self) -> bool:
        return self.status == "supported"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HaircutBackendSolveResult:
    """Normalized public-synthetic solve/evidence bundle for Haircut consumers."""

    backend_id: str
    status: SolveStatus
    problem_id: str | None
    benchmark_ids: tuple[str, ...]
    values: dict[str, float]
    diagnostics: dict[str, Any]
    evidence: dict[str, Any]
    request: dict[str, Any]

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class FiniteDifferenceHaircutBackend:
    """Thin fail-closed adapter implementing the current FD backend contract."""

    def __init__(
        self, manifest: FDCapabilityManifest = DEFAULT_FD_CAPABILITY_MANIFEST
    ) -> None:
        self._manifest = manifest
        self._identity = HaircutBackendIdentity(
            backend_id=manifest.backend_id,
            name="finite_difference_options",
            adapter_schema_version="haircut-fd-backend-adapter/v0",
            contract_version=manifest.contract_version,
            maturity="validated_public_synthetic",
            entry_point="src.integrations.haircut_backend:create_backend",
            issue_refs=(
                "googa27/finite_difference_options#59",
                "googa27/haircut-engine#195",
            ),
        )

    @property
    def identity(self) -> HaircutBackendIdentity:
        return self._identity

    @property
    def manifest(self) -> FDCapabilityManifest:
        return self._manifest

    def capability_manifest(self) -> dict[str, Any]:
        """Return data-only capability metadata without importing numerical routes."""

        manifest = asdict(self._manifest)
        manifest["status"] = self._manifest.status.value
        return manifest

    def screen(self, payload: Mapping[str, Any]) -> FDBackendScreeningResult:
        """Map and validate a QuantProblemSpec-like payload before numerical work."""

        request = FDRouteRequest.from_quant_problem_spec(payload)
        route_diagnostics = diagnose_unsupported_route(request, self._manifest)
        execution_diagnostics = (
            () if route_diagnostics else _execution_diagnostics(payload)
        )
        diagnostics = (*route_diagnostics, *execution_diagnostics)
        return FDBackendScreeningResult(
            backend_id=self._manifest.backend_id,
            status="unsupported" if diagnostics else "supported",
            request=asdict(request),
            diagnostics=tuple(
                _diagnostic_as_dict(diagnostic) for diagnostic in diagnostics
            ),
            manifest=self.capability_manifest(),
        )

    def solve(self, payload: Mapping[str, Any]) -> HaircutBackendSolveResult:
        """Execute the validated public-synthetic Black-Scholes fixture.

        The method intentionally refuses generic supported-looking payloads until
        they carry an executable benchmark/fixture that this repository validates.
        This prevents silent fallback to placeholder coefficients or hard-coded
        route semantics.
        """

        request = FDRouteRequest.from_quant_problem_spec(payload)
        ensure_route_supported(request, self._manifest)
        benchmark_ids = _benchmark_ids(payload)
        problem_id = _optional_string(payload.get("problem_id"))
        if not _is_executable_public_synthetic_payload(
            problem_id, benchmark_ids, payload
        ):
            raise UnsupportedRouteError(
                (
                    UnsupportedRouteDiagnostic(
                        reason=UnsupportedReason.UNSUPPORTED_BENCHMARK,
                        field="benchmark_ids",
                        value=",".join(benchmark_ids) or "<missing>",
                        supported=tuple(sorted(_EXECUTABLE_PUBLIC_BENCHMARKS)),
                        message=(
                            "FD backend solve requires a validated public-synthetic executable benchmark; "
                            "supported mathematical routes are still execution-blocked until a fixture is registered."
                        ),
                    ),
                )
            )

        from src.validation.benchmark_registry import run_registered_benchmark
        from src.validation.black_scholes_parity import (
            run_public_black_scholes_parity_fixture,
        )

        parity_report = run_public_black_scholes_parity_fixture()
        registry_results = {
            benchmark_id: run_registered_benchmark(benchmark_id).as_dict()
            for benchmark_id in _EXECUTED_REGISTRY_BENCHMARKS
        }
        passed = parity_report.converged and all(
            bool(result["passed"]) for result in registry_results.values()
        )
        values = {
            "price": parity_report.price,
            "oracle_price": parity_report.oracle_price,
            "delta": parity_report.delta,
            "reference_delta": parity_report.reference_delta,
            "gamma": parity_report.gamma,
            "reference_gamma": parity_report.reference_gamma,
        }
        diagnostics = {
            "requested_benchmark_ids": benchmark_ids,
            "errors": parity_report.errors,
            "no_arbitrage": parity_report.no_arbitrage,
            "convergence": parity_report.convergence_table(),
            "registry_results": registry_results,
            "resource_controls": parity_report.evidence.resource_controls,
            "requested_resource_controls": _resource_controls(payload),
            "unsupported_route_diagnostics": (),
            "fallbacks": (),
        }
        evidence = {
            **parity_report.evidence.as_dict(),
            "adapter_schema_version": self.identity.adapter_schema_version,
            "source_schema_version": request.source_schema_version,
            "problem_id": problem_id,
            "privacy_class": _optional_string(payload.get("privacy_class")),
        }
        return HaircutBackendSolveResult(
            backend_id=self._manifest.backend_id,
            status="passed" if passed else "failed",
            problem_id=problem_id,
            benchmark_ids=_EXECUTED_REGISTRY_BENCHMARKS,
            values=values,
            diagnostics=diagnostics,
            evidence=evidence,
            request=asdict(request),
        )


def create_backend(
    manifest: FDCapabilityManifest = DEFAULT_FD_CAPABILITY_MANIFEST,
) -> FiniteDifferenceHaircutBackend:
    """Entry-point factory for `haircut.solver_backends` discovery."""

    return FiniteDifferenceHaircutBackend(manifest=manifest)


def _execution_diagnostics(
    payload: Mapping[str, Any]
) -> tuple[UnsupportedRouteDiagnostic, ...]:
    benchmark_ids = _benchmark_ids(payload)
    problem_id = _optional_string(payload.get("problem_id"))
    if _is_executable_public_synthetic_payload(problem_id, benchmark_ids, payload):
        return ()
    return (
        UnsupportedRouteDiagnostic(
            reason=UnsupportedReason.UNSUPPORTED_BENCHMARK,
            field="benchmark_ids",
            value=",".join(benchmark_ids) or "<missing>",
            supported=tuple(sorted(_EXECUTABLE_PUBLIC_BENCHMARKS)),
            message=(
                "FD backend adapter currently executes only registered public-synthetic "
                "benchmarks; register a fixture before claiming solve support."
            ),
        ),
    )


def _resource_controls(payload: Mapping[str, Any]) -> dict[str, Any]:
    solver = payload.get("solver_plan")
    if not isinstance(solver, Mapping):
        return {}
    controls: dict[str, Any] = {}
    if "max_runtime_seconds" in solver:
        controls["max_runtime_seconds"] = solver["max_runtime_seconds"]
    raw_controls = solver.get("resource_controls")
    if isinstance(raw_controls, Mapping):
        controls.update({str(key): value for key, value in raw_controls.items()})
    return controls


def _diagnostic_as_dict(diagnostic: UnsupportedRouteDiagnostic) -> dict[str, Any]:
    payload = asdict(diagnostic)
    payload["reason"] = diagnostic.reason.value
    return payload


def _benchmark_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
    ids: list[str] = []
    for section_name in ("solver_plan", "result_bundle"):
        section = payload.get(section_name)
        if isinstance(section, Mapping):
            ids.extend(str(item) for item in _tuple(section.get("benchmark_ids")))
    financial_graph = payload.get("financial_graph")
    if isinstance(financial_graph, Mapping):
        valuation_graph = financial_graph.get("valuation_graph")
        if isinstance(valuation_graph, Mapping):
            solver_hints = valuation_graph.get("solver_hints")
            if isinstance(solver_hints, Mapping):
                ids.extend(
                    str(item) for item in _tuple(solver_hints.get("benchmark_ids"))
                )
    return tuple(dict.fromkeys(item for item in ids if item))


def _is_executable_public_synthetic_payload(
    problem_id: str | None, benchmark_ids: tuple[str, ...], payload: Mapping[str, Any]
) -> bool:
    privacy_class = _optional_string(payload.get("privacy_class"))
    return bool(
        privacy_class == "public_synthetic"
        and problem_id in _PUBLIC_SYNTHETIC_PROBLEM_IDS
        and (_EXECUTABLE_PUBLIC_BENCHMARKS & set(benchmark_ids))
    )


def _tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(value.values())
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


__all__ = [
    "FDBackendScreeningResult",
    "FiniteDifferenceHaircutBackend",
    "HaircutBackendIdentity",
    "HaircutBackendSolveResult",
    "create_backend",
]
