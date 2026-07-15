"""One-dimensional finite-difference axis contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from finite_difference_options.exceptions import ValidationError

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AxisGrid:
    """Immutable one-dimensional finite-difference axis.

    Coordinates are solver coordinates. For a log grid, for example, the
    coordinates are log-spots and :attr:`physical_coordinates` applies the
    declared transform back to spot values for metadata and interpolation
    consumers.
    """

    name: str
    coordinates: Sequence[float]
    family: str = "custom"
    coordinate_system: str = "physical"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        coords = tuple(float(value) for value in self.coordinates)
        if len(coords) < 3:
            raise ValidationError("AxisGrid requires at least three coordinates")
        values = np.asarray(coords, dtype=float)
        if values.ndim != 1:
            raise ValidationError("AxisGrid coordinates must be one-dimensional")
        if not np.all(np.isfinite(values)):
            raise ValidationError("AxisGrid coordinates must be finite")
        if not np.all(np.diff(values) > 0.0):
            raise ValidationError("AxisGrid coordinates must be strictly increasing")
        if not self.name:
            raise ValidationError("AxisGrid name must be non-empty")
        if not self.family:
            raise ValidationError("AxisGrid family must be non-empty")
        if not self.coordinate_system:
            raise ValidationError("AxisGrid coordinate_system must be non-empty")
        object.__setattr__(self, "coordinates", coords)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def __array__(self, dtype: Any | None = None, copy: bool | None = None) -> Array:
        """Return coordinates as a NumPy array for legacy NumPy code."""

        array = np.asarray(self.coordinates, dtype=dtype if dtype is not None else float)
        if copy:
            return array.copy()
        return array

    def __len__(self) -> int:
        """Return the node count."""

        return len(self.coordinates)

    def __iter__(self) -> Iterator[float]:
        """Iterate over solver coordinates."""

        return iter(self.coordinates)

    def __getitem__(self, index: int) -> float:
        """Return one solver coordinate."""

        return self.coordinates[index]

    @property
    def coordinates_array(self) -> Array:
        """Coordinates as a defensive NumPy array."""

        return np.asarray(self.coordinates, dtype=float)

    @property
    def physical_coordinates(self) -> Array:
        """Coordinates mapped back to physical space when a transform exists."""

        values = self.coordinates_array
        if self.coordinate_system == "log":
            return np.exp(values)
        return values.copy()

    @property
    def node_count(self) -> int:
        """Number of coordinates on the axis."""

        return len(self.coordinates)

    @property
    def lower(self) -> float:
        """Lower solver-coordinate bound."""

        return self.coordinates[0]

    @property
    def upper(self) -> float:
        """Upper solver-coordinate bound."""

        return self.coordinates[-1]

    @property
    def boundary_locations(self) -> dict[str, float]:
        """Named boundary locations for boundary algebra and diagnostics."""

        return {"min": self.lower, "max": self.upper}

    @property
    def local_spacings(self) -> Array:
        """Local positive spacings between adjacent nodes."""

        return np.diff(self.coordinates_array)

    @property
    def min_spacing(self) -> float:
        """Minimum adjacent spacing."""

        return float(np.min(self.local_spacings))

    @property
    def max_spacing(self) -> float:
        """Maximum adjacent spacing."""

        return float(np.max(self.local_spacings))

    @property
    def spacing_ratio(self) -> float:
        """Ratio of largest to smallest adjacent spacing."""

        return self.max_spacing / self.min_spacing

    @property
    def uniform(self) -> bool:
        """Whether adjacent spacings are uniform up to roundoff."""

        spacings = self.local_spacings
        return bool(np.allclose(spacings, spacings[0], rtol=1.0e-12, atol=1.0e-15))

    def local_derivative_weights(self, index: int, *, order: int) -> tuple[float, float, float]:
        """Return three-point local finite-difference weights.

        Interior nodes use the centered neighbor triplet. Boundary nodes use the
        nearest three nodes, giving a second-order one-sided closure for the
        first derivative and exactness for quadratic polynomials.
        """

        if order not in {1, 2}:
            raise ValidationError(f"derivative order must be 1 or 2, got {order}")
        if not 0 <= index < self.node_count:
            raise ValidationError(f"grid index {index} outside [0, {self.node_count})")
        if index == 0:
            stencil_indices = (0, 1, 2)
        elif index == self.node_count - 1:
            stencil_indices = (self.node_count - 3, self.node_count - 2, self.node_count - 1)
        else:
            stencil_indices = (index - 1, index, index + 1)
        x0 = self.coordinates[index]
        stencil = np.asarray([self.coordinates[idx] - x0 for idx in stencil_indices], dtype=float)
        matrix = np.vstack([stencil**degree for degree in range(3)])
        rhs = np.zeros(3, dtype=float)
        rhs[order] = 1.0 if order == 1 else 2.0
        weights = np.linalg.solve(matrix, rhs)
        return (float(weights[0]), float(weights[1]), float(weights[2]))

    def derivative(self, values: Sequence[float] | Array, *, order: int) -> Array:
        """Differentiate one-dimensional values with local three-point stencils."""

        array = np.asarray(values, dtype=float)
        if array.shape != (self.node_count,):
            raise ValidationError(f"values must have shape ({self.node_count},), got {array.shape}")
        result = np.empty_like(array, dtype=float)
        for index in range(self.node_count):
            if index == 0:
                stencil_indices = (0, 1, 2)
            elif index == self.node_count - 1:
                stencil_indices = (self.node_count - 3, self.node_count - 2, self.node_count - 1)
            else:
                stencil_indices = (index - 1, index, index + 1)
            weights = self.local_derivative_weights(index, order=order)
            result[index] = sum(weight * array[idx] for weight, idx in zip(weights, stencil_indices, strict=True))
        return result

    def interpolate(
        self,
        values: Sequence[float] | Array,
        points: float | Sequence[float] | Array,
        *,
        extrapolate: bool = False,
    ) -> Any:
        """Interpolate values at solver-coordinate points with domain checks."""

        array = np.asarray(values, dtype=float)
        if array.shape != (self.node_count,):
            raise ValidationError(f"values must have shape ({self.node_count},), got {array.shape}")
        query = np.asarray(points, dtype=float)
        if not np.all(np.isfinite(query)):
            raise ValidationError("interpolation points must be finite")
        if not extrapolate and (np.any(query < self.lower) or np.any(query > self.upper)):
            raise ValidationError("interpolation point outside grid domain")
        clipped = np.clip(query, self.lower, self.upper)
        interpolated = np.interp(clipped, self.coordinates_array, array)
        if np.isscalar(points):
            return float(interpolated)
        return interpolated

    def to_public_dict(self) -> dict[str, Any]:
        """Return a serializable grid diagnostics payload."""

        return {
            "name": self.name,
            "family": self.family,
            "coordinate_system": self.coordinate_system,
            "node_count": self.node_count,
            "lower": self.lower,
            "upper": self.upper,
            "physical_lower": float(self.physical_coordinates[0]),
            "physical_upper": float(self.physical_coordinates[-1]),
            "uniform": self.uniform,
            "min_spacing": self.min_spacing,
            "max_spacing": self.max_spacing,
            "spacing_ratio": self.spacing_ratio,
            "boundary_locations": self.boundary_locations,
            "metadata": dict(self.metadata),
        }


def as_axis_grid(value: AxisGrid | Sequence[float] | Array, *, name: str = "axis") -> AxisGrid:
    """Coerce a raw coordinate sequence into an :class:`AxisGrid`."""

    if isinstance(value, AxisGrid):
        return value
    return AxisGrid(
        name=name,
        coordinates=tuple(float(item) for item in np.asarray(value, dtype=float)),
        family="array",
    )


AxisGrid.__module__ = "finite_difference_options.grids"
as_axis_grid.__module__ = "finite_difference_options.grids"

__all__ = ["Array", "AxisGrid", "as_axis_grid"]
