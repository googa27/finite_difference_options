"""Facade for domain-specific exceptions used by the pricing framework."""

from __future__ import annotations

from finite_difference_options.exceptions.core import (
    BoundaryConditionError,
    ConvergenceError,
    FiniteDifferenceError,
    GridError,
    InstrumentError,
    ModelError,
    PricingError,
    TimeSteppingError,
    ValidationError,
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
