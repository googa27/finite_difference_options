"""Regression tests for issue #46 ADI operator-splitting semantics."""

from __future__ import annotations

import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from numpy.testing import assert_allclose

from src.solvers.adi import ADISolver


def test_adi_mixed_operator_uses_off_diagonal_covariance_once() -> None:
    """For u(x, y)=xy, d_xy u = 1 and the mixed term equals Sigma_12."""

    solver = ADISolver(theta=0.5)
    x_grid = np.array([-1.0, 0.0, 1.0])
    y_grid = np.array([-2.0, 0.0, 2.0])
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid, indexing="ij")
    surface = x_mesh * y_mesh

    covariance = np.zeros((3, 3, 2, 2), dtype=float)
    covariance[..., 0, 0] = 0.1
    covariance[..., 1, 1] = 0.2
    covariance[..., 0, 1] = 0.7
    covariance[..., 1, 0] = 0.7

    mixed = solver._mixed_operator(surface, covariance, (x_grid, y_grid))
    assert_allclose(mixed[1, 1], 0.7)

    covariance[..., 0, 1] = 0.0
    covariance[..., 1, 0] = 0.0
    assert_allclose(
        solver._mixed_operator(surface, covariance, (x_grid, y_grid))[1, 1], 0.0
    )


def test_theta_zero_step_matches_unsplit_explicit_reference() -> None:
    """At theta=0 the ADI stage equals one unsplit explicit generator step."""

    solver = ADISolver(theta=0.0)
    x_grid = np.array([-1.0, 0.0, 1.0])
    y_grid = np.array([-2.0, 0.0, 2.0])
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid, indexing="ij")
    initial = x_mesh * y_mesh + 0.25 * x_mesh**2
    drift = np.zeros((3, 3, 2), dtype=float)
    drift[..., 0] = 0.2
    drift[..., 1] = -0.1
    covariance = np.zeros((3, 3, 2, 2), dtype=float)
    covariance[..., 0, 0] = 0.3
    covariance[..., 1, 1] = 0.4
    covariance[..., 0, 1] = 0.15
    covariance[..., 1, 0] = 0.15
    reaction = np.full((3, 3), 0.07)
    source = np.full((3, 3), 0.11)
    time_grid = np.array([0.0, 0.01])

    directional = solver._directional_operator(
        initial, drift, covariance, (x_grid, y_grid), axis=0
    )
    directional += solver._directional_operator(
        initial, drift, covariance, (x_grid, y_grid), axis=1
    )
    explicit_reference = initial + 0.01 * (
        directional
        + solver._mixed_operator(initial, covariance, (x_grid, y_grid))
        - reaction * initial
        + source
    )

    result = solver.solve_2d(
        initial_condition=initial,
        drift=drift,
        covariance=covariance,
        time_grid=time_grid,
        spatial_grids=(x_grid, y_grid),
        reaction=reaction,
        source=source,
    )

    assert_allclose(result[-1], initial)
    assert_allclose(result[0], explicit_reference)


def test_adi_reaction_is_applied_once_and_calendar_layout_is_terminal_last() -> None:
    """With zero drift/diffusion/source, reaction follows the explicit predictor once per step."""

    solver = ADISolver(theta=0.5)
    x_grid = np.array([0.0, 0.5, 1.0])
    y_grid = np.array([0.0, 1.0, 2.0])
    initial = np.ones((3, 3), dtype=float)
    drift = np.zeros((3, 3, 2), dtype=float)
    covariance = np.zeros((3, 3, 2, 2), dtype=float)
    time_grid = np.array([0.0, 0.1, 0.2])

    result = solver.solve_2d(
        initial_condition=initial,
        drift=drift,
        covariance=covariance,
        time_grid=time_grid,
        spatial_grids=(x_grid, y_grid),
        reaction=0.5,
    )

    assert_allclose(result[-1], initial)
    assert_allclose(result[0], np.full_like(initial, (1.0 - 0.5 * 0.1) ** 2))
    assert solver.last_diagnostics["reaction_treatment"] == "explicit_predictor_once"
    assert (
        solver.last_diagnostics["time_orientation"]
        == "forward_tau_internal_calendar_output"
    )


def test_adi_source_term_is_applied_once_per_step_without_reaction() -> None:
    solver = ADISolver(theta=0.5)
    x_grid = np.array([0.0, 0.5, 1.0])
    y_grid = np.array([0.0, 1.0, 2.0])
    initial = np.zeros((3, 3), dtype=float)
    drift = np.zeros((3, 3, 2), dtype=float)
    covariance = np.zeros((3, 3, 2, 2), dtype=float)
    time_grid = np.array([0.0, 0.1, 0.2, 0.3])

    result = solver.solve_2d(
        initial_condition=initial,
        drift=drift,
        covariance=covariance,
        time_grid=time_grid,
        spatial_grids=(x_grid, y_grid),
        source=2.0,
    )

    assert_allclose(result[-1], initial)
    assert_allclose(result[0], np.full_like(initial, 0.6), atol=1e-14)
    assert solver.last_diagnostics["source_treatment"] == "explicit_predictor_once"


def test_adi_3d_reaction_source_and_calendar_layout() -> None:
    solver = ADISolver(theta=0.5)
    grids = (
        np.array([0.0, 0.5, 1.0]),
        np.array([0.0, 1.0, 2.0]),
        np.array([-1.0, 0.0, 1.0]),
    )
    initial = np.ones((3, 3, 3), dtype=float)
    drift = np.zeros((3, 3, 3, 3), dtype=float)
    covariance = np.zeros((3, 3, 3, 3, 3), dtype=float)
    time_grid = np.array([0.0, 0.2])

    result = solver.solve_3d(
        initial_condition=initial,
        drift=drift,
        covariance=covariance,
        time_grid=time_grid,
        spatial_grids=grids,
        reaction=0.5,
        source=0.25,
    )

    assert_allclose(result[-1], initial)
    assert_allclose(result[0], np.full_like(initial, 0.95))
    assert solver.last_diagnostics["mixed_derivative_pairs"] == [(0, 1), (0, 2), (1, 2)]
    assert (
        solver.last_diagnostics["time_orientation"]
        == "forward_tau_internal_calendar_output"
    )


def test_positivity_floor_is_disclosed_and_only_for_nonnegative_routes() -> None:
    solver = ADISolver(theta=0.5)
    nonnegative_initial = np.zeros((3, 3), dtype=float)
    signed_initial = np.array(
        [[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]], dtype=float
    )
    source = np.zeros((3, 3), dtype=float)

    assert solver._uses_positivity_floor(nonnegative_initial, source)
    assert not solver._uses_positivity_floor(signed_initial, source)
    assert_allclose(
        solver._apply_positivity_floor(np.array([[-1.0, 2.0]]), enabled=True),
        np.array([[0.0, 2.0]]),
    )
    assert_allclose(
        solver._apply_positivity_floor(np.array([[-1.0, 2.0]]), enabled=False),
        np.array([[-1.0, 2.0]]),
    )
