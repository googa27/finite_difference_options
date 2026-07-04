"""Nonuniform-grid Greek estimation diagnostics."""

from __future__ import annotations

from math import erf, exp, log, pi, sqrt

import numpy as np
import pytest

from finite_difference_options.exceptions import ValidationError
from finite_difference_options.greeks import FiniteDifferenceGreeks
from finite_difference_options.grids import strike_centered_axis


def _standard_normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _standard_normal_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def _black_scholes_call_value(s: np.ndarray, *, strike: float, rate: float, sigma: float, tau: float) -> np.ndarray:
    values = np.empty_like(s, dtype=float)
    positive = s > 0.0
    values[~positive] = 0.0
    d1 = (np.log(s[positive] / strike) + (rate + 0.5 * sigma**2) * tau) / (sigma * sqrt(tau))
    d2 = d1 - sigma * sqrt(tau)
    values[positive] = s[positive] * np.vectorize(_standard_normal_cdf)(d1) - strike * exp(
        -rate * tau
    ) * np.vectorize(_standard_normal_cdf)(d2)
    return values


def _black_scholes_delta_gamma(
    spot: float,
    *,
    strike: float,
    rate: float,
    sigma: float,
    tau: float,
) -> tuple[float, float]:
    d1 = (log(spot / strike) + (rate + 0.5 * sigma**2) * tau) / (sigma * sqrt(tau))
    delta = _standard_normal_cdf(d1)
    gamma = _standard_normal_pdf(d1) / (spot * sigma * sqrt(tau))
    return delta, gamma


def test_gamma_uses_direct_local_nonuniform_second_derivative_stencil() -> None:
    s = np.array([0.0, 0.07, 0.23, 0.61, 1.4, 3.0], dtype=float)
    values = s**3 + 2.0 * s**2 - s + 5.0

    gamma = FiniteDifferenceGreeks().gamma(values, s)

    # Interior nonuniform three-point second derivative with actual local
    # spacings.  The old repeated-gradient route gives a different answer.
    index = 3
    h_left = s[index] - s[index - 1]
    h_right = s[index + 1] - s[index]
    expected = (
        2.0 * values[index - 1] / (h_left * (h_left + h_right))
        - 2.0 * values[index] / (h_left * h_right)
        + 2.0 * values[index + 1] / (h_right * (h_left + h_right))
    )

    assert gamma[index] == pytest.approx(expected, abs=1e-12)


def test_sample_delta_distinguishes_requested_coordinate_from_nearest_node() -> None:
    s = np.array([0.0, 0.2, 0.55, 1.1, 2.0], dtype=float)
    values = s**3 + 0.25 * s**2 + 2.0 * s
    spot = 0.8
    calculator = FiniteDifferenceGreeks()

    estimate = calculator.sample_delta(values, s, spot)
    delta_grid = calculator.delta(values, s)
    nearest_index = int(np.argmin(np.abs(s - spot)))

    assert estimate.value == pytest.approx(np.interp(spot, s, delta_grid), abs=1e-12)
    assert estimate.nearest_node_index == nearest_index
    assert estimate.nearest_node_value == pytest.approx(delta_grid[nearest_index], abs=1e-12)
    assert estimate.value != pytest.approx(estimate.nearest_node_value, abs=1e-4)
    assert estimate.diagnostics["interpolation_method"] == "linear_interpolation"
    assert estimate.diagnostics["coordinate_spacing"] == "nonuniform"
    assert estimate.diagnostics["domain_edge_distance"] == pytest.approx(0.8, abs=1e-12)


def test_black_scholes_nonuniform_greeks_converge_and_report_error_diagnostics() -> None:
    strike = 100.0
    spot = 103.0
    rate = 0.03
    sigma = 0.22
    tau = 0.75
    calculator = FiniteDifferenceGreeks()
    exact_delta, exact_gamma = _black_scholes_delta_gamma(
        spot,
        strike=strike,
        rate=rate,
        sigma=sigma,
        tau=tau,
    )

    errors: list[tuple[float, float]] = []
    previous_values: np.ndarray | None = None
    previous_grid: np.ndarray | None = None
    fine_gamma_estimate = None
    for nodes in (31, 61, 121):
        axis = strike_centered_axis(
            name="spot",
            lower=20.0,
            upper=250.0,
            nodes=nodes,
            strike=strike,
            concentration=2.0,
        )
        grid = axis.coordinates_array
        values = _black_scholes_call_value(grid, strike=strike, rate=rate, sigma=sigma, tau=tau)
        delta_estimate = calculator.sample_delta(values, grid, spot, reference_value=exact_delta)
        gamma_estimate = calculator.sample_gamma(
            values,
            grid,
            spot,
            reference_value=exact_gamma,
            refined_values=previous_values,
            refined_coordinates=previous_grid,
        )
        errors.append((abs(delta_estimate.value - exact_delta), abs(gamma_estimate.value - exact_gamma)))
        previous_values = values
        previous_grid = grid
        fine_gamma_estimate = gamma_estimate

    assert errors[2][0] < errors[0][0]
    assert errors[2][1] < errors[0][1]
    assert fine_gamma_estimate is not None
    assert fine_gamma_estimate.diagnostics["reference_abs_error"] == pytest.approx(errors[2][1])
    assert fine_gamma_estimate.diagnostics["refinement_abs_error"] is not None
    assert fine_gamma_estimate.diagnostics["reported_abs_error"] >= fine_gamma_estimate.diagnostics[
        "reference_abs_error"
    ]
    assert fine_gamma_estimate.diagnostics["independent_within_reported_error"] is True


def test_sample_greek_rejects_undefined_expiry_kink() -> None:
    strike = 100.0
    grid = np.array([80.0, 95.0, 100.0, 110.0, 130.0], dtype=float)
    payoff = np.maximum(grid - strike, 0.0)

    with pytest.raises(ValidationError, match="undefined at expiry"):
        FiniteDifferenceGreeks().sample_delta(
            payoff,
            grid,
            strike,
            time_to_expiry=0.0,
            nonsmooth_coordinates=(strike,),
        )
