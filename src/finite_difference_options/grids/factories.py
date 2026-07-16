"""Finite-difference grid factory functions."""

from __future__ import annotations

from math import atanh

import numpy as np

from finite_difference_options.exceptions import ValidationError
from finite_difference_options.grids.axis import AxisGrid


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


for _factory in (
    log_uniform_axis,
    sinh_clustered_axis,
    strike_centered_axis,
    tanh_clustered_axis,
    uniform_axis,
    variance_boundary_axis,
):
    _factory.__module__ = "finite_difference_options.grids"


del _factory

__all__ = [
    "log_uniform_axis",
    "sinh_clustered_axis",
    "strike_centered_axis",
    "tanh_clustered_axis",
    "uniform_axis",
    "variance_boundary_axis",
]
