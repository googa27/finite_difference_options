"""Cross-repository integration adapters for finite_difference_options."""

from .haircut_backend import (
    FDBackendScreeningResult,
    FiniteDifferenceHaircutBackend,
    HaircutBackendIdentity,
    HaircutBackendSolveResult,
    create_backend,
)

__all__ = [
    "FDBackendScreeningResult",
    "FiniteDifferenceHaircutBackend",
    "HaircutBackendIdentity",
    "HaircutBackendSolveResult",
    "create_backend",
]
