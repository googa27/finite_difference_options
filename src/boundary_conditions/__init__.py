"""Unified boundary conditions package.

This package contains typed boundary-condition resolvers and findiff adapters
for the unified pricing framework.
"""

from .builder import (
    BlackScholesBoundaryBuilder,
    BoundaryResolution,
    BoundarySpec,
    HestonBoundaryBuilder,
)

__all__ = [
    "BlackScholesBoundaryBuilder",
    "BoundaryResolution",
    "BoundarySpec",
    "HestonBoundaryBuilder",
]
