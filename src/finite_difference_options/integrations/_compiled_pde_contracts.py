"""Private immutable DTOs and identity constants for the exact compiled fixture."""

from __future__ import annotations
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal


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


# Keep the released public class identity for imports and trusted historical pickles.
CompiledPDEDiagnostic.__module__ = "finite_difference_options.integrations.compiled_pde_adapter"
CompiledPDEScreeningResult.__module__ = "finite_difference_options.integrations.compiled_pde_adapter"
CompiledPDESolveResult.__module__ = "finite_difference_options.integrations.compiled_pde_adapter"
CompiledPDEAdapterError.__module__ = "finite_difference_options.integrations.compiled_pde_adapter"
