"""Custom exceptions for the finite difference options pricing library.

This module defines domain-specific exceptions to provide clear error messages
and improve debugging experience for users of the library.
"""
from __future__ import annotations


class FiniteDifferenceError(Exception):
    """Base exception for all finite difference pricing errors."""
    pass


class ValidationError(FiniteDifferenceError):
    """Raised when input validation fails."""
    pass


class GridError(FiniteDifferenceError):
    """Raised when there are issues with grid generation or parameters."""
    pass


class ConvergenceError(FiniteDifferenceError):
    """Raised when numerical methods fail to converge."""
    pass


class ModelError(FiniteDifferenceError):
    """Raised when there are issues with financial model parameters."""
    pass


class PricingError(FiniteDifferenceError):
    """Raised when pricing computation fails."""
    pass


class BoundaryConditionError(FiniteDifferenceError):
    """Raised when boundary conditions are invalid or incompatible."""
    pass


class TimeSteppingError(FiniteDifferenceError):
    """Raised when time stepping methods encounter errors."""
    pass


class InstrumentError(FiniteDifferenceError):
    """Raised when financial instrument parameters are invalid."""
    pass
