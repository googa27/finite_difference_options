"""Cross-repository integration adapters for finite_difference_options."""

from .compiled_pde_adapter import (
    CompiledPDEAdapterError,
    CompiledPDEDiagnostic,
    CompiledPDEScreeningResult,
    CompiledPDESolveResult,
    load_compiled_pde_json,
    packaged_compiled_black_scholes_fixture,
    screen_compiled_pde_payload,
    solve_compiled_pde_payload,
)
from .haircut_backend import (
    ContractMajorMismatchError,
    FDBackendScreeningResult,
    FiniteDifferenceHaircutBackend,
    HaircutBackendSolveResult,
    HaircutProtocolUnavailableError,
    create_backend,
)
from .public_solver_contract import (
    PublicFDSolverResult,
    ReleasedFDSolverContract,
    released_fd_solver_contract,
    solve_public_quant_problem_spec,
)

__all__ = [
    "CompiledPDEAdapterError",
    "CompiledPDEDiagnostic",
    "CompiledPDEScreeningResult",
    "CompiledPDESolveResult",
    "ContractMajorMismatchError",
    "FDBackendScreeningResult",
    "FiniteDifferenceHaircutBackend",
    "HaircutBackendSolveResult",
    "HaircutProtocolUnavailableError",
    "PublicFDSolverResult",
    "ReleasedFDSolverContract",
    "create_backend",
    "load_compiled_pde_json",
    "packaged_compiled_black_scholes_fixture",
    "released_fd_solver_contract",
    "screen_compiled_pde_payload",
    "solve_compiled_pde_payload",
    "solve_public_quant_problem_spec",
]
