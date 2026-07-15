"""Tensor-product finite-difference grid contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from finite_difference_options.exceptions import ValidationError
from finite_difference_options.grids.axis import Array, AxisGrid, as_axis_grid


@dataclass(frozen=True, slots=True)
class TensorGrid:
    """Tensor product of one-dimensional FD axes."""

    axes: tuple[AxisGrid, ...]

    def __post_init__(self) -> None:
        axes = tuple(as_axis_grid(axis, name=f"axis_{index}") for index, axis in enumerate(self.axes))
        if len(axes) < 1:
            raise ValidationError("TensorGrid requires at least one axis")
        names = [axis.name for axis in axes]
        if len(set(names)) != len(names):
            raise ValidationError("TensorGrid axis names must be unique")
        object.__setattr__(self, "axes", axes)

    def __len__(self) -> int:
        """Return the number of tensor axes."""

        return len(self.axes)

    def __iter__(self) -> Iterator[AxisGrid]:
        """Iterate over axes in tensor-index order."""

        return iter(self.axes)

    def __getitem__(self, index: int) -> AxisGrid:
        """Return one axis by tensor-index position."""

        return self.axes[index]

    @property
    def dimension(self) -> int:
        """Tensor dimension."""

        return len(self.axes)

    @property
    def shape(self) -> tuple[int, ...]:
        """Tensor node shape."""

        return tuple(axis.node_count for axis in self.axes)

    def coordinates(self) -> tuple[Array, ...]:
        """Axis coordinate arrays."""

        return tuple(axis.coordinates_array for axis in self.axes)

    def mesh(self) -> tuple[Array, ...]:
        """Dense tensor mesh with matrix indexing."""

        return tuple(np.meshgrid(*self.coordinates(), indexing="ij"))

    def to_public_dict(self) -> dict[str, Any]:
        """Return serializable tensor-grid diagnostics."""

        return {
            "dimension": self.dimension,
            "shape": list(self.shape),
            "axes": [axis.to_public_dict() for axis in self.axes],
        }


TensorGrid.__module__ = "finite_difference_options.grids"

__all__ = ["TensorGrid"]
