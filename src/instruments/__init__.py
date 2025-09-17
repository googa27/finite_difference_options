"""Unified instruments package.

This package contains implementations of various financial instruments
for the unified pricing framework.
"""
from .base import Instrument, EuropeanOption, EuropeanCall, EuropeanPut
from .operators import SpatialOperator

__all__ = [
    "Instrument",
    "EuropeanOption",
    "EuropeanCall",
    "EuropeanPut",
    "SpatialOperator",
]