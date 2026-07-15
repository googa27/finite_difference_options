"""Typed one-dimensional and tensor-product grid contracts.

The FD core needs more than raw coordinate arrays: nonuniform formulas,
coordinate transforms, interpolation domain checks and diagnostics all depend on
stable grid identity. This package facade preserves NumPy interoperability and
legacy imports while implementations live in cohesive modules.
"""

from __future__ import annotations

from finite_difference_options.grids.axis import Array as Array
from finite_difference_options.grids.axis import AxisGrid, as_axis_grid
from finite_difference_options.grids.factories import (
    log_uniform_axis,
    sinh_clustered_axis,
    strike_centered_axis,
    tanh_clustered_axis,
    uniform_axis,
    variance_boundary_axis,
)
from finite_difference_options.grids.tensor import TensorGrid

__all__ = [
    "AxisGrid",
    "TensorGrid",
    "as_axis_grid",
    "log_uniform_axis",
    "sinh_clustered_axis",
    "strike_centered_axis",
    "tanh_clustered_axis",
    "uniform_axis",
    "variance_boundary_axis",
]
