"""Grid contract tests for issue #47 nonuniform/local-metric grids."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from finite_difference_options.exceptions import ValidationError
from finite_difference_options.grids import (
    AxisGrid,
    TensorGrid,
    log_uniform_axis,
    sinh_clustered_axis,
    strike_centered_axis,
    uniform_axis,
    variance_boundary_axis,
)
from finite_difference_options.solvers.adi import ADISolver


def test_axis_grid_rejects_invalid_coordinates_and_reports_spacing() -> None:
    with pytest.raises(ValidationError, match="strictly increasing"):
        AxisGrid(name="x", coordinates=(0.0, 0.5, 0.5, 1.0))
    with pytest.raises(ValidationError, match="at least three"):
        AxisGrid(name="x", coordinates=(0.0, 1.0))

    grid = AxisGrid(name="x", coordinates=(0.0, 0.1, 0.4, 1.0), family="custom")

    assert grid.node_count == 4
    assert grid.boundary_locations == {"min": 0.0, "max": 1.0}
    assert grid.local_spacings.tolist() == pytest.approx([0.1, 0.3, 0.6])
    assert grid.spacing_ratio == pytest.approx(6.0)
    assert grid.to_public_dict()["family"] == "custom"


def test_nonuniform_axis_derivatives_are_quadratic_exact() -> None:
    grid = AxisGrid(name="x", coordinates=(-1.0, -0.3, 0.2, 0.9, 2.0), family="custom")
    x = grid.coordinates_array
    values = 2.5 * x**2 - 0.7 * x + 3.0

    assert_allclose(grid.derivative(values, order=1), 5.0 * x - 0.7, atol=2e-14)
    assert_allclose(grid.derivative(values, order=2), np.full_like(x, 5.0), atol=2e-14)


def test_log_strike_and_variance_factories_preserve_transform_metadata() -> None:
    log_grid = log_uniform_axis(name="log_spot", lower=50.0, upper=200.0, nodes=9)
    strike_grid = strike_centered_axis(name="spot", lower=20.0, upper=200.0, nodes=21, strike=100.0)
    variance_grid = variance_boundary_axis(name="variance", lower=0.0, upper=0.4, nodes=13)

    assert log_grid.coordinate_system == "log"
    assert_allclose(log_grid.physical_coordinates, np.exp(log_grid.coordinates_array))
    assert log_grid.physical_coordinates[0] == pytest.approx(50.0)
    assert log_grid.physical_coordinates[-1] == pytest.approx(200.0)

    strike_index = int(np.argmin(np.abs(strike_grid.coordinates_array - 100.0)))
    center_spacing = min(
        strike_grid.coordinates_array[strike_index] - strike_grid.coordinates_array[strike_index - 1],
        strike_grid.coordinates_array[strike_index + 1] - strike_grid.coordinates_array[strike_index],
    )
    edge_spacing = max(strike_grid.local_spacings[0], strike_grid.local_spacings[-1])
    assert center_spacing < edge_spacing
    assert strike_grid.to_public_dict()["family"] == "strike-centered-sinh"

    assert variance_grid.lower == pytest.approx(0.0)
    assert variance_grid.to_public_dict()["metadata"]["cluster_boundary"] == "lower"
    assert variance_grid.local_spacings[0] < variance_grid.local_spacings[-1]


def test_axis_grid_interpolation_rejects_extrapolation_by_default() -> None:
    grid = uniform_axis(name="spot", lower=0.0, upper=2.0, nodes=5)
    values = grid.coordinates_array**2

    assert grid.interpolate(values, 1.5) == pytest.approx(2.25)
    with pytest.raises(ValidationError, match="outside grid domain"):
        grid.interpolate(values, -0.1)


def test_tensor_grid_diagnostics_track_axis_local_metrics() -> None:
    tensor = TensorGrid(
        axes=(
            sinh_clustered_axis(name="spot", lower=20.0, upper=200.0, nodes=9, center=100.0),
            uniform_axis(name="variance", lower=0.0, upper=0.5, nodes=5),
        )
    )

    public = tensor.to_public_dict()
    assert public["dimension"] == 2
    assert public["shape"] == [9, 5]
    assert public["axes"][0]["uniform"] is False
    assert public["axes"][1]["uniform"] is True


def test_adi_accepts_axis_grid_objects_and_reports_grid_identity() -> None:
    solver = ADISolver(theta=0.0)
    x_grid = AxisGrid(name="x", coordinates=(-1.0, -0.4, 0.2, 1.1), family="custom-x")
    y_grid = AxisGrid(name="y", coordinates=(-2.0, -0.5, 0.0, 2.0), family="custom-y")
    x_mesh, y_mesh = np.meshgrid(x_grid.coordinates_array, y_grid.coordinates_array, indexing="ij")
    initial = x_mesh * y_mesh
    drift = np.zeros((4, 4, 2), dtype=float)
    covariance = np.zeros((4, 4, 2, 2), dtype=float)
    covariance[..., 0, 0] = 1.0
    covariance[..., 1, 1] = 1.0
    covariance[..., 0, 1] = 0.3
    covariance[..., 1, 0] = 0.3

    result = solver.solve_2d(
        initial_condition=initial,
        drift=drift,
        covariance=covariance,
        time_grid=np.array([0.0, 0.1]),
        spatial_grids=(x_grid, y_grid),
    )

    assert_allclose(result[-1], initial)
    assert_allclose(result[0][1:-1, 1:-1], initial[1:-1, 1:-1] + 0.1 * 0.3, atol=1e-14)
    assert solver.last_diagnostics["grid_diagnostics"][0]["name"] == "x"
    assert solver.last_diagnostics["grid_diagnostics"][0]["family"] == "custom-x"
    assert solver.last_diagnostics["grid_uniformity"] == [False, False]
