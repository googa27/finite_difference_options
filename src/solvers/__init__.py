"""Unified solvers package.

This package contains implementations of various PDE solvers
for the unified pricing framework.
"""
from .adi import ADISolver, create_adi_solver

__all__ = [
    "ADISolver",
    "create_adi_solver",
]