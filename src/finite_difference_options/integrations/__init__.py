"""Cross-repository integration adapters for finite_difference_options."""

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
    "ContractMajorMismatchError",
    "FDBackendScreeningResult",
    "FiniteDifferenceHaircutBackend",
    "HaircutBackendSolveResult",
    "HaircutProtocolUnavailableError",
    "PublicFDSolverResult",
    "ReleasedFDSolverContract",
    "create_backend",
    "released_fd_solver_contract",
    "solve_public_quant_problem_spec",
]
