"""Cross-repository integration adapters for finite_difference_options."""

from .haircut_backend import (
    FDBackendScreeningResult,
    FiniteDifferenceHaircutBackend,
    HaircutBackendIdentity,
    HaircutBackendSolveResult,
    create_backend,
)
from .public_solver_contract import (
    PublicFDSolverResult,
    ReleasedFDSolverContract,
    released_fd_solver_contract,
    solve_public_quant_problem_spec,
)

__all__ = [
    "FDBackendScreeningResult",
    "FiniteDifferenceHaircutBackend",
    "HaircutBackendIdentity",
    "HaircutBackendSolveResult",
    "PublicFDSolverResult",
    "ReleasedFDSolverContract",
    "create_backend",
    "released_fd_solver_contract",
    "solve_public_quant_problem_spec",
]
