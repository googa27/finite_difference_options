"""Finite difference implementations of Greeks calculators."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from finite_difference_options.exceptions import ValidationError

from .base import GreeksCalculator

Number = float | int
DiagnosticValue = Number | str | bool | None


@dataclass(frozen=True, slots=True)
class GreekEstimate:
    """Requested-coordinate Greek estimate with stencil and error diagnostics."""

    greek: Literal["delta", "gamma"]
    value: float
    requested_coordinate: float
    nearest_node_index: int
    nearest_node_coordinate: float
    nearest_node_value: float
    diagnostics: Mapping[str, DiagnosticValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


class FiniteDifferenceGreeks(GreeksCalculator):
    """Compute option Greeks using finite-difference approximations.

    The price grid is assumed to have shape ``(n_t, n_s[, n_v])`` where axis 0 is
    time and the remaining axes are spatial dimensions.

    Notes
    -----
    These methods are intentionally generic; they do not assume a specific
    diffusion model and therefore operate purely on the provided numeric grids.
    Spatial derivatives use local coordinate values rather than scalar spacing,
    so nonuniform grids use the same local finite-difference contract as the
    typed grid layer.
    """

    def _validate_price_grid(self, grid: NDArray[np.float64]) -> None:
        """Validate that a price grid is provided."""
        if grid.ndim < 1:
            raise ValidationError("Prices array must have at least 1 dimension")
        if not np.all(np.isfinite(grid)):
            raise ValidationError("Prices array must contain only finite values")

    def _validate_asset_grid(self, grid: NDArray[np.float64]) -> None:
        """Validate the asset price grid used for spatial derivatives."""
        if grid.ndim != 1:
            raise ValidationError("Asset price grid must be one-dimensional")
        if grid.shape[0] < 2:
            raise ValidationError("Asset price grid must have at least 2 points")
        if not np.all(np.isfinite(grid)):
            raise ValidationError("Asset price grid must contain only finite values")
        if not np.all(np.diff(grid) > 0.0):
            raise ValidationError("Asset price grid must be strictly increasing")

    def _validate_asset_grid_for_gamma(self, grid: NDArray[np.float64]) -> None:
        """Validate grid for second derivatives with respect to the asset price."""
        self._validate_asset_grid(grid)
        if grid.shape[0] < 3:
            raise ValidationError("Asset price grid must have at least 3 points for second derivative")

    def _validate_time_grid(self, grid: NDArray[np.float64]) -> None:
        """Validate the time grid used for theta calculations."""
        if grid.ndim != 1:
            raise ValidationError("Time grid must be one-dimensional")
        if grid.shape[0] < 2:
            raise ValidationError("Time grid must have at least 2 points")
        if not np.all(np.isfinite(grid)):
            raise ValidationError("Time grid must contain only finite values")
        if not np.all(np.diff(grid) > 0.0):
            raise ValidationError("Time grid must be strictly increasing")

    def _first_derivative(
        self, grid: NDArray[np.float64], spacing: NDArray[np.float64], axis: int
    ) -> NDArray[np.float64]:
        """Return first derivative along ``axis`` using local coordinates."""

        return self._coordinate_derivative(grid, spacing, axis=axis, order=1)

    def _second_derivative(
        self, grid: NDArray[np.float64], spacing: NDArray[np.float64], axis: int
    ) -> NDArray[np.float64]:
        """Return second derivative along ``axis`` using local coordinates."""

        return self._coordinate_derivative(grid, spacing, axis=axis, order=2)

    def _coordinate_derivative(
        self,
        grid: NDArray[np.float64],
        coordinates: NDArray[np.float64],
        *,
        axis: int,
        order: Literal[1, 2],
    ) -> NDArray[np.float64]:
        """Apply a local finite-difference derivative along one axis."""

        if order == 2:
            self._validate_asset_grid_for_gamma(coordinates)
        else:
            self._validate_asset_grid(coordinates)
        if not -grid.ndim <= axis < grid.ndim:
            raise ValidationError(f"Derivative axis {axis} outside price grid rank {grid.ndim}")
        normalized_axis = axis % grid.ndim
        if grid.shape[normalized_axis] != coordinates.shape[0]:
            raise ValidationError(
                "Derivative coordinate length must match selected prices axis: "
                f"axis length {grid.shape[normalized_axis]}, coordinates {coordinates.shape[0]}"
            )

        if _is_uniform(coordinates):
            first = np.gradient(grid, coordinates, axis=normalized_axis, edge_order=2)
            if order == 1:
                return np.asarray(first, dtype=float)
            return np.asarray(np.gradient(first, coordinates, axis=normalized_axis, edge_order=2), dtype=float)

        moved = np.moveaxis(grid, normalized_axis, 0)
        result = np.empty_like(moved, dtype=float)
        for index in range(coordinates.shape[0]):
            stencil_indices, weights = self._local_derivative_stencil(coordinates, index, order=order)
            derivative_slice = np.zeros_like(moved[index], dtype=float)
            for stencil_index, weight in zip(stencil_indices, weights, strict=True):
                derivative_slice += weight * moved[stencil_index]
            result[index] = derivative_slice
        return np.moveaxis(result, 0, normalized_axis)

    def _local_derivative_stencil(
        self,
        coordinates: NDArray[np.float64],
        index: int,
        *,
        order: Literal[1, 2],
    ) -> tuple[tuple[int, ...], tuple[float, ...]]:
        """Return local finite-difference stencil indices and weights."""

        node_count = int(coordinates.shape[0])
        if order == 1 and node_count == 2:
            stencil_indices = (0, 1)
        elif index == 0:
            stencil_indices = (0, 1, 2)
        elif index == node_count - 1:
            stencil_indices = (node_count - 3, node_count - 2, node_count - 1)
        else:
            stencil_indices = (index - 1, index, index + 1)

        x0 = float(coordinates[index])
        stencil = np.asarray([coordinates[stencil_index] - x0 for stencil_index in stencil_indices], dtype=float)
        matrix = np.vstack([stencil**degree for degree in range(len(stencil_indices))])
        rhs = np.zeros(len(stencil_indices), dtype=float)
        rhs[order] = 1.0 if order == 1 else 2.0
        weights = np.linalg.solve(matrix, rhs)
        return stencil_indices, tuple(float(weight) for weight in weights)

    def delta(self, grid: NDArray[np.float64], s: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return Delta of the option across the grid."""

        price_grid = np.asarray(grid, dtype=float)
        asset_grid = np.asarray(s, dtype=float)
        self._validate_price_grid(price_grid)
        self._validate_asset_grid(asset_grid)
        axis = 1 if price_grid.ndim > 1 else 0
        return self._first_derivative(price_grid, asset_grid, axis=axis)

    def gamma(self, grid: NDArray[np.float64], s: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return Gamma of the option across the grid."""
        price_grid = np.asarray(grid, dtype=float)
        asset_grid = np.asarray(s, dtype=float)
        self._validate_price_grid(price_grid)
        self._validate_asset_grid_for_gamma(asset_grid)
        axis = 1 if price_grid.ndim > 1 else 0
        return self._second_derivative(price_grid, asset_grid, axis=axis)

    def theta(self, grid: NDArray[np.float64], t: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return Theta (time decay) across the grid."""
        price_grid = np.asarray(grid, dtype=float)
        time_grid = np.asarray(t, dtype=float)
        self._validate_price_grid(price_grid)
        self._validate_time_grid(time_grid)
        if price_grid.shape[0] != time_grid.shape[0]:
            raise ValidationError("Time grid must match the first dimension of the prices array")
        return -self._first_derivative(price_grid, time_grid, axis=0)

    def sample_delta(
        self,
        values: Sequence[float] | NDArray[np.float64],
        s: Sequence[float] | NDArray[np.float64],
        coordinate: float,
        **kwargs: Any,
    ) -> GreekEstimate:
        """Return a requested-coordinate Delta estimate and diagnostics."""

        return self._sample_spatial_greek(values, s, coordinate, greek="delta", order=1, **kwargs)

    def sample_gamma(
        self,
        values: Sequence[float] | NDArray[np.float64],
        s: Sequence[float] | NDArray[np.float64],
        coordinate: float,
        **kwargs: Any,
    ) -> GreekEstimate:
        """Return a requested-coordinate Gamma estimate and diagnostics."""

        return self._sample_spatial_greek(values, s, coordinate, greek="gamma", order=2, **kwargs)

    def _sample_spatial_greek(
        self,
        values: Sequence[float] | NDArray[np.float64],
        s: Sequence[float] | NDArray[np.float64],
        coordinate: float,
        *,
        greek: Literal["delta", "gamma"],
        order: Literal[1, 2],
        time_index: int = -1,
        reference_value: float | None = None,
        refined_values: Sequence[float] | NDArray[np.float64] | None = None,
        refined_coordinates: Sequence[float] | NDArray[np.float64] | None = None,
        time_to_expiry: float | None = None,
        nonsmooth_coordinates: Sequence[float] = (),
        expiry_tolerance: float = 1.0e-12,
    ) -> GreekEstimate:
        """Sample a spatial Greek at a requested coordinate.

        ``values`` may be a one-dimensional value slice or a ``(time, spot)``
        grid.  Multitime inputs sample the requested ``time_index`` after
        computing the derivative grid.
        """

        value_grid = np.asarray(values, dtype=float)
        coordinates = np.asarray(s, dtype=float)
        coordinate_value = float(coordinate)
        if not np.isfinite(coordinate_value):
            raise ValidationError("Requested coordinate must be finite")
        self._validate_price_grid(value_grid)
        if order == 2:
            self._validate_asset_grid_for_gamma(coordinates)
        else:
            self._validate_asset_grid(coordinates)
        if value_grid.ndim == 1 and value_grid.shape[0] != coordinates.shape[0]:
            raise ValidationError(
                f"One-dimensional values must have shape ({coordinates.shape[0]},), got {value_grid.shape}"
            )
        if value_grid.ndim > 1 and value_grid.shape[1] != coordinates.shape[0]:
            raise ValidationError(
                "Multitime values must use shape (time, coordinate[, ...]) for spatial Greek sampling"
            )
        if coordinate_value < coordinates[0] or coordinate_value > coordinates[-1]:
            raise ValidationError("Requested coordinate lies outside the asset grid domain")
        self._reject_undefined_expiry_kink(
            greek=greek,
            coordinate=coordinate_value,
            time_to_expiry=time_to_expiry,
            nonsmooth_coordinates=nonsmooth_coordinates,
            expiry_tolerance=expiry_tolerance,
        )

        derivative_grid = self.delta(value_grid, coordinates) if order == 1 else self.gamma(value_grid, coordinates)
        derivative_slice = self._select_sampling_slice(derivative_grid, time_index=time_index)
        value, lower_index, upper_index, weight, interpolation_method = self._interpolate_derivative(
            derivative_slice,
            coordinates,
            coordinate_value,
        )
        nearest_node_index = int(np.argmin(np.abs(coordinates - coordinate_value)))
        nearest_node_coordinate = float(coordinates[nearest_node_index])
        nearest_node_value = float(derivative_slice[nearest_node_index])

        refinement_abs_error: float | None = None
        if refined_values is not None or refined_coordinates is not None:
            if refined_values is None or refined_coordinates is None:
                raise ValidationError("Both refined_values and refined_coordinates must be supplied together")
            refined_estimate = self._sample_spatial_greek(
                refined_values,
                refined_coordinates,
                coordinate_value,
                greek=greek,
                order=order,
                time_index=time_index,
                reference_value=None,
                refined_values=None,
                refined_coordinates=None,
                time_to_expiry=time_to_expiry,
                nonsmooth_coordinates=nonsmooth_coordinates,
                expiry_tolerance=expiry_tolerance,
            )
            refinement_abs_error = abs(value - refined_estimate.value)

        reference_abs_error = None if reference_value is None else abs(value - float(reference_value))
        error_terms = [term for term in (refinement_abs_error, reference_abs_error) if term is not None]
        reported_abs_error = max(error_terms) if error_terms else None
        independent_within_reported_error = (
            None
            if reference_abs_error is None or reported_abs_error is None
            else reference_abs_error <= reported_abs_error + 1.0e-15
        )

        diagnostics: dict[str, DiagnosticValue] = {
            "derivative_order": order,
            "stencil_order": 2 if coordinates.shape[0] >= 3 else 1,
            "coordinate_spacing": "uniform" if _is_uniform(coordinates) else "nonuniform",
            "interpolation_method": interpolation_method,
            "lower_index": lower_index,
            "upper_index": upper_index,
            "lower_coordinate": float(coordinates[lower_index]),
            "upper_coordinate": float(coordinates[upper_index]),
            "interpolation_weight": weight,
            "left_spacing": _left_spacing(coordinates, lower_index),
            "right_spacing": _right_spacing(coordinates, upper_index),
            "bracket_spacing": float(coordinates[upper_index] - coordinates[lower_index])
            if upper_index != lower_index
            else 0.0,
            "min_spacing": float(np.min(np.diff(coordinates))),
            "max_spacing": float(np.max(np.diff(coordinates))),
            "spacing_ratio": float(np.max(np.diff(coordinates)) / np.min(np.diff(coordinates))),
            "domain_edge_distance": float(min(coordinate_value - coordinates[0], coordinates[-1] - coordinate_value)),
            "nearest_node_distance": abs(nearest_node_coordinate - coordinate_value),
            "time_index": time_index if value_grid.ndim > 1 else None,
            "time_to_expiry": None if time_to_expiry is None else float(time_to_expiry),
            "expiry_policy": "reject_nonsmooth_expiry_coordinate",
            "reference_abs_error": reference_abs_error,
            "refinement_abs_error": refinement_abs_error,
            "reported_abs_error": reported_abs_error,
            "independent_within_reported_error": independent_within_reported_error,
        }
        return GreekEstimate(
            greek=greek,
            value=float(value),
            requested_coordinate=coordinate_value,
            nearest_node_index=nearest_node_index,
            nearest_node_coordinate=nearest_node_coordinate,
            nearest_node_value=nearest_node_value,
            diagnostics=diagnostics,
        )

    def _select_sampling_slice(self, derivative_grid: NDArray[np.float64], *, time_index: int) -> NDArray[np.float64]:
        if derivative_grid.ndim == 1:
            return derivative_grid
        if derivative_grid.ndim == 2:
            return np.asarray(derivative_grid[time_index], dtype=float)
        raise ValidationError("Requested-coordinate Greek sampling currently supports 1D or (time, coordinate) grids")

    def _interpolate_derivative(
        self,
        derivative_slice: NDArray[np.float64],
        coordinates: NDArray[np.float64],
        coordinate: float,
    ) -> tuple[float, int, int, float, Literal["grid_node", "linear_interpolation"]]:
        exact_matches = np.flatnonzero(np.isclose(coordinates, coordinate, rtol=0.0, atol=1.0e-14))
        if exact_matches.size:
            index = int(exact_matches[0])
            return float(derivative_slice[index]), index, index, 0.0, "grid_node"

        upper_index = int(np.searchsorted(coordinates, coordinate, side="right"))
        lower_index = upper_index - 1
        lower = float(coordinates[lower_index])
        upper = float(coordinates[upper_index])
        weight = (coordinate - lower) / (upper - lower)
        value = (1.0 - weight) * float(derivative_slice[lower_index]) + weight * float(derivative_slice[upper_index])
        return value, lower_index, upper_index, float(weight), "linear_interpolation"

    def _reject_undefined_expiry_kink(
        self,
        *,
        greek: Literal["delta", "gamma"],
        coordinate: float,
        time_to_expiry: float | None,
        nonsmooth_coordinates: Sequence[float],
        expiry_tolerance: float,
    ) -> None:
        if time_to_expiry is None or time_to_expiry > expiry_tolerance:
            return
        for nonsmooth_coordinate in nonsmooth_coordinates:
            if abs(coordinate - float(nonsmooth_coordinate)) <= expiry_tolerance:
                raise ValidationError(
                    f"{greek} is undefined at expiry at nonsmooth coordinate {float(nonsmooth_coordinate)}"
                )


def _is_uniform(coordinates: NDArray[np.float64]) -> bool:
    spacings = np.diff(coordinates)
    return bool(np.allclose(spacings, spacings[0], rtol=1.0e-12, atol=1.0e-15))


def _left_spacing(coordinates: NDArray[np.float64], index: int) -> float | None:
    if index <= 0:
        return None
    return float(coordinates[index] - coordinates[index - 1])


def _right_spacing(coordinates: NDArray[np.float64], index: int) -> float | None:
    if index >= coordinates.shape[0] - 1:
        return None
    return float(coordinates[index + 1] - coordinates[index])
