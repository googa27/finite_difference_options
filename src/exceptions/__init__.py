"""Unified exceptions package.

This package contains custom exceptions for the unified pricing framework.
"""
# Re-export key exception classes
from ..utils.exceptions import (
    FiniteDifferenceError,
    ValidationError,
    GridError,
    ModelError,
    InstrumentError,
    PricingError,
    BoundaryConditionError,
    TimeSteppingError,
    ConvergenceError,
)

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