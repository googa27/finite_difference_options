"""Regression tests for multidimensional unified-engine coefficient plumbing."""

from __future__ import annotations

import inspect
import pathlib
import sys


import numpy as np
from numpy.testing import assert_allclose
import pytest

from finite_difference_options.exceptions import ValidationError
from finite_difference_options.pricing import UnifiedEuropeanOption
from finite_difference_options.processes import (
    ProcessDimension,
    ProcessType,
    StochasticProcess,
    create_standard_heston,
)
from finite_difference_options.solvers.base import ADISolverWrapper


class RecordingADISolver:
    """ADI test double that records the coefficients it receives."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def solve_2d(
        self,
        initial_condition,
        drift,
        covariance,
        time_grid,
        spatial_grids,
        boundary_conditions=None,
        reaction=None,
    ):
        self.calls.append(
            {
                "initial_condition": np.array(initial_condition, copy=True),
                "drift": np.array(drift, copy=True),
                "covariance": np.array(covariance, copy=True),
                "time_grid": np.array(time_grid, copy=True),
                "spatial_grids": tuple(
                    np.array(grid, copy=True) for grid in spatial_grids
                ),
                "boundary_conditions": boundary_conditions,
                "reaction": None if reaction is None else np.array(reaction, copy=True),
            }
        )
        solution = np.zeros((len(time_grid), *initial_condition.shape), dtype=float)
        solution[-1] = initial_condition
        return solution

    def solve_3d(self, *args, **kwargs):  # pragma: no cover - should not be called here
        raise AssertionError("3D solver should not be reached by these tests")


class FourDimensionalProcess(StochasticProcess):
    @property
    def dimension(self) -> ProcessDimension:
        return ProcessDimension(value=4)

    @property
    def process_type(self) -> ProcessType:
        return ProcessType.NON_AFFINE

    def drift(
        self, time: float, state: np.ndarray
    ) -> np.ndarray:  # pragma: no cover - fail-fast path
        raise AssertionError("drift should not be evaluated for unsupported dimensions")

    def covariance(
        self, time: float, state: np.ndarray
    ) -> np.ndarray:  # pragma: no cover - fail-fast path
        raise AssertionError(
            "covariance should not be evaluated for unsupported dimensions"
        )


class IndefiniteTwoDimensionalProcess(StochasticProcess):
    @property
    def dimension(self) -> ProcessDimension:
        return ProcessDimension(value=2)

    @property
    def process_type(self) -> ProcessType:
        return ProcessType.NON_AFFINE

    def drift(self, time: float, state: np.ndarray) -> np.ndarray:
        return np.zeros((len(state), 2), dtype=float)

    def covariance(self, time: float, state: np.ndarray) -> np.ndarray:
        covariance = np.array([[1.0, 2.0], [2.0, 1.0]])
        return np.broadcast_to(covariance, (len(state), 2, 2)).copy()


def _expected_process_coefficients(process, time_grid, grids):
    mesh = np.meshgrid(*grids, indexing="ij")
    grid_shape = mesh[0].shape
    states = np.stack([axis.reshape(-1) for axis in mesh], axis=-1)
    drift = process.drift(float(time_grid[-1]), states).reshape(
        *grid_shape, process.dimension.value
    )
    covariance = process.covariance(float(time_grid[-1]), states).reshape(
        *grid_shape,
        process.dimension.value,
        process.dimension.value,
    )
    reaction = process.discount(float(time_grid[-1]), states).reshape(*grid_shape)
    return drift, covariance, reaction


def test_multidimensional_adapter_passes_heston_process_coefficients_to_adi() -> None:
    process = create_standard_heston(
        r=0.03, kappa=4.0, theta=0.08, sigma=0.55, rho=-0.35
    )
    recording_solver = RecordingADISolver()
    wrapper = ADISolverWrapper(recording_solver, process=process)

    spot_grid = np.array([80.0, 100.0, 120.0])
    variance_grid = np.array([0.02, 0.06, 0.12])
    grids = (spot_grid, variance_grid)
    time_grid = np.array([0.0, 0.25])
    initial_condition = np.maximum(spot_grid[:, None] - 100.0, 0.0)
    instrument = UnifiedEuropeanOption(strike=100.0, maturity=0.25, option_type="call")

    wrapper.solve(initial_condition, instrument, *grids, time_grid=time_grid)

    assert len(recording_solver.calls) == 1
    call = recording_solver.calls[0]
    expected_drift, expected_covariance, expected_reaction = (
        _expected_process_coefficients(process, time_grid, grids)
    )
    assert_allclose(call["drift"], expected_drift)
    assert_allclose(call["covariance"], expected_covariance)
    assert_allclose(call["reaction"], expected_reaction)

    covariance = call["covariance"]
    assert np.any(
        covariance[..., 0, 1] != 0.0
    ), "Heston mixed covariance must reach ADI assembly"
    assert_allclose(covariance[..., 0, 1], covariance[..., 1, 0])


def test_multidimensional_adapter_coefficients_change_with_selected_process() -> None:
    spot_grid = np.array([90.0, 110.0])
    variance_grid = np.array([0.03, 0.09])
    time_grid = np.array([0.0, 0.5])

    low_vol_process = create_standard_heston(
        r=0.03, kappa=4.0, theta=0.08, sigma=0.2, rho=-0.1
    )
    high_vol_process = create_standard_heston(
        r=0.03, kappa=8.0, theta=0.06, sigma=0.8, rho=-0.7
    )

    _, low_covariance, low_reaction = ADISolverWrapper(
        RecordingADISolver(), process=low_vol_process
    )._build_process_coefficients(
        float(time_grid[-1]),
        (spot_grid, variance_grid),
    )
    _, high_covariance, high_reaction = ADISolverWrapper(
        RecordingADISolver(), process=high_vol_process
    )._build_process_coefficients(
        float(time_grid[-1]),
        (spot_grid, variance_grid),
    )

    assert_allclose(low_reaction, 0.03)
    assert_allclose(high_reaction, 0.03)

    assert not np.allclose(low_covariance, high_covariance)
    expected_shape = low_covariance[..., 1, 1].shape
    expected_low_var = np.broadcast_to(
        low_vol_process.sigma**2 * variance_grid[None, :], expected_shape
    )
    expected_high_var = np.broadcast_to(
        high_vol_process.sigma**2 * variance_grid[None, :], expected_shape
    )
    assert_allclose(low_covariance[..., 1, 1], expected_low_var)
    assert_allclose(high_covariance[..., 1, 1], expected_high_var)


def test_multidimensional_adapter_rejects_unsupported_dimensions_before_allocation() -> (
    None
):
    wrapper = ADISolverWrapper(RecordingADISolver(), process=FourDimensionalProcess())
    instrument = UnifiedEuropeanOption(strike=100.0, maturity=0.25, option_type="call")

    with pytest.raises(ValidationError, match="supports only 2D and 3D"):
        wrapper.solve(
            np.zeros((2, 2, 2, 2)),
            instrument,
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            time_grid=np.array([0.0, 0.25]),
        )


def test_multidimensional_adapter_rejects_indefinite_covariance_before_adi_call() -> (
    None
):
    recording_solver = RecordingADISolver()
    wrapper = ADISolverWrapper(
        recording_solver, process=IndefiniteTwoDimensionalProcess()
    )
    instrument = UnifiedEuropeanOption(strike=100.0, maturity=0.25, option_type="call")

    with pytest.raises(ValidationError, match="positive semi-definite"):
        wrapper.solve(
            np.zeros((2, 2)),
            instrument,
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            time_grid=np.array([0.0, 0.25]),
        )
    assert recording_solver.calls == []


def test_multidimensional_adapter_contains_no_zero_drift_constant_variance_fallback() -> (
    None
):
    source = inspect.getsource(ADISolverWrapper.solve)
    forbidden_fragments = [
        "Create dummy drift",
        "np.zeros(grid_shape + (2,))",
        "np.zeros(grid_shape + (3,))",
        "covariance[..., 0, 0] = 0.04",
        "covariance[..., 1, 1] = 0.01",
        "covariance[..., 2, 2] = 0.005",
    ]
    assert not any(fragment in source for fragment in forbidden_fragments)
