"""Domain-specific exceptions used across the pricing framework."""

from __future__ import annotations


class FiniteDifferenceError(Exception):
    """Base exception for all finite difference pricing errors."""


class ValidationError(FiniteDifferenceError):
    """Raised when input validation fails."""


class GridError(FiniteDifferenceError):
    """Raised when there are issues with grid generation or parameters."""


class ConvergenceError(FiniteDifferenceError):
    """Raised when numerical methods fail to converge."""


class ModelError(FiniteDifferenceError):
    """Raised when there are issues with financial model parameters."""


class PricingError(FiniteDifferenceError):
    """Raised when pricing computation fails."""


class BoundaryConditionError(FiniteDifferenceError):
    """Raised when boundary conditions are invalid or incompatible."""


class TimeSteppingError(FiniteDifferenceError):
    """Raised when time stepping methods encounter errors."""


class InstrumentError(FiniteDifferenceError):
    """Raised when financial instrument parameters are invalid."""


__all__ = [
    "FiniteDifferenceError",
    "ValidationError",
    "GridError",
    "ModelError",
    "InstrumentError",
    "PricingError",
    "BoundaryConditionError",
    "TimeSteppingError",
    "ConvergenceError",
]
