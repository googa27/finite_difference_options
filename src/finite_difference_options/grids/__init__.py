"""Typed one-dimensional and tensor-product grid contracts.

The FD core needs more than raw coordinate arrays: nonuniform formulas,
coordinate transforms, interpolation domain checks and diagnostics all depend on
stable grid identity.  This module keeps that information in immutable value
objects while preserving NumPy interoperability for legacy solver code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atanh
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from finite_difference_options.exceptions import ValidationError

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AxisGrid:
    """Immutable one-dimensional finite-difference axis.

    Coordinates are solver coordinates.  For a log grid, for example, the
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


def as_axis_grid(value: AxisGrid | Sequence[float] | Array, *, name: str = "axis") -> AxisGrid:
    """Coerce a raw coordinate sequence into an :class:`AxisGrid`."""

    if isinstance(value, AxisGrid):
        return value
    return AxisGrid(
        name=name,
        coordinates=tuple(float(item) for item in np.asarray(value, dtype=float)),
        family="array",
    )


def uniform_axis(*, name: str, lower: float, upper: float, nodes: int, coordinate_system: str = "physical") -> AxisGrid:
    """Create a uniform axis on ``[lower, upper]``."""

    _validate_bounds(lower, upper, nodes)
    return AxisGrid(
        name=name,
        coordinates=tuple(np.linspace(float(lower), float(upper), int(nodes))),
        family="uniform",
        coordinate_system=coordinate_system,
    )


def log_uniform_axis(*, name: str, lower: float, upper: float, nodes: int) -> AxisGrid:
    """Create a uniform log-coordinate axis over positive physical bounds."""

    _validate_bounds(lower, upper, nodes)
    if lower <= 0.0:
        raise ValidationError("log_uniform_axis lower bound must be positive")
    return AxisGrid(
        name=name,
        coordinates=tuple(np.linspace(np.log(float(lower)), np.log(float(upper)), int(nodes))),
        family="log-uniform",
        coordinate_system="log",
        metadata={"physical_lower": float(lower), "physical_upper": float(upper), "transform": "exp"},
    )


def sinh_clustered_axis(
    *,
    name: str,
    lower: float,
    upper: float,
    nodes: int,
    center: float,
    concentration: float = 2.5,
) -> AxisGrid:
    """Create a sinh-stretched axis clustered near ``center``."""

    _validate_centered_factory(lower, upper, nodes, center, concentration)
    q = np.linspace(-1.0, 1.0, int(nodes))
    denom = np.sinh(float(concentration))
    coords = np.empty_like(q)
    for index, value in enumerate(q):
        if value <= 0.0:
            left_scale = (float(center) - float(lower)) * np.sinh(float(concentration) * (-value))
            coords[index] = float(center) - left_scale / denom
        else:
            right_scale = (float(upper) - float(center)) * np.sinh(float(concentration) * value)
            coords[index] = float(center) + right_scale / denom
    coords[0] = float(lower)
    coords[-1] = float(upper)
    return AxisGrid(
        name=name,
        coordinates=tuple(coords),
        family="sinh-clustered",
        metadata={"center": float(center), "concentration": float(concentration)},
    )


def strike_centered_axis(
    *,
    name: str,
    lower: float,
    upper: float,
    nodes: int,
    strike: float,
    concentration: float = 2.5,
) -> AxisGrid:
    """Create a sinh-clustered physical axis centered on an option strike."""

    axis = sinh_clustered_axis(
        name=name,
        lower=lower,
        upper=upper,
        nodes=nodes,
        center=strike,
        concentration=concentration,
    )
    return AxisGrid(
        name=axis.name,
        coordinates=axis.coordinates,
        family="strike-centered-sinh",
        metadata={"strike": float(strike), "concentration": float(concentration)},
    )


def tanh_clustered_axis(
    *,
    name: str,
    lower: float,
    upper: float,
    nodes: int,
    center: float,
    concentration: float = 0.85,
) -> AxisGrid:
    """Create an inverse-tanh clustered axis near ``center``.

    ``concentration`` is an open-unit parameter; values closer to one increase
    clustering near the center while preserving exact endpoints.
    """

    _validate_centered_factory(lower, upper, nodes, center, concentration)
    if not 0.0 < concentration < 1.0:
        raise ValidationError("tanh_clustered_axis concentration must lie in (0, 1)")
    q = np.linspace(-1.0, 1.0, int(nodes))
    denom = atanh(float(concentration))
    coords = np.empty_like(q)
    for index, value in enumerate(q):
        mapped = atanh(float(concentration) * abs(float(value))) / denom
        if value <= 0.0:
            coords[index] = float(center) - (float(center) - float(lower)) * mapped
        else:
            coords[index] = float(center) + (float(upper) - float(center)) * mapped
    coords[0] = float(lower)
    coords[-1] = float(upper)
    return AxisGrid(
        name=name,
        coordinates=tuple(coords),
        family="tanh-clustered",
        metadata={"center": float(center), "concentration": float(concentration)},
    )


def variance_boundary_axis(
    *,
    name: str,
    lower: float,
    upper: float,
    nodes: int,
    boundary: str = "lower",
    concentration: float = 2.5,
) -> AxisGrid:
    """Create a variance axis clustered near an attainable variance boundary."""

    _validate_bounds(lower, upper, nodes)
    if concentration <= 0.0:
        raise ValidationError("variance_boundary_axis concentration must be positive")
    if boundary not in {"lower", "upper"}:
        raise ValidationError("variance_boundary_axis boundary must be 'lower' or 'upper'")
    y = np.linspace(0.0, 1.0, int(nodes))
    denom = np.sinh(float(concentration))
    if boundary == "lower":
        coords = float(lower) + (float(upper) - float(lower)) * np.sinh(float(concentration) * y) / denom
    else:
        coords = float(upper) - (float(upper) - float(lower)) * np.sinh(float(concentration) * (1.0 - y)) / denom
    coords[0] = float(lower)
    coords[-1] = float(upper)
    return AxisGrid(
        name=name,
        coordinates=tuple(coords),
        family="variance-boundary-sinh",
        metadata={"cluster_boundary": boundary, "concentration": float(concentration)},
    )


def _validate_bounds(lower: float, upper: float, nodes: int) -> None:
    if int(nodes) < 3:
        raise ValidationError("grid factories require at least three nodes")
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValidationError("grid bounds must be finite")
    if float(lower) >= float(upper):
        raise ValidationError("grid lower bound must be strictly below upper bound")


def _validate_centered_factory(lower: float, upper: float, nodes: int, center: float, concentration: float) -> None:
    _validate_bounds(lower, upper, nodes)
    if not np.isfinite(center):
        raise ValidationError("grid center must be finite")
    if not float(lower) < float(center) < float(upper):
        raise ValidationError("grid center must lie strictly inside bounds")
    if not np.isfinite(concentration) or concentration <= 0.0:
        raise ValidationError("grid concentration must be positive and finite")


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
