"""Rannacher startup tests for kinked payoff finite-difference routes."""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.greeks import FiniteDifferenceGreeks  # noqa: E402
from src.instruments.base import EuropeanCall  # noqa: E402
from src.pricing.engines import BlackScholesPDE  # noqa: E402
from src.processes.affine import GeometricBrownianMotion  # noqa: E402
from src.solvers.finite_difference import (  # noqa: E402
    FiniteDifferenceSolver,
    RannacherCrankNicolson,
    ThetaMethod,
)


def _call_setup() -> tuple[EuropeanCall, np.ndarray, np.ndarray]:
    model = GeometricBrownianMotion(mu=0.05, sigma=0.2)
    option = EuropeanCall(strike=1.0, maturity=0.1, model=model)
    # Include the strike exactly so payoff-kink Greek behavior is observable.
    s_grid = np.linspace(0.0, 3.0, 201)
    time_grid = np.linspace(0.0, option.maturity, 4)
    return option, s_grid, time_grid


def _gamma_roughness(values: np.ndarray, s_grid: np.ndarray) -> float:
    greeks = FiniteDifferenceGreeks()
    gamma = greeks.gamma(values, s_grid)
    strike_idx = int(np.argmin(np.abs(s_grid - 1.0)))
    window = gamma[1, strike_idx - 8 : strike_idx + 9]
    # High-frequency oscillations show up as large second differences in Gamma.
    return float(np.sum(np.abs(np.diff(window, n=2))))


def test_rannacher_crank_nicolson_records_startup_schedule() -> None:
    option, s_grid, time_grid = _call_setup()
    stepper = RannacherCrankNicolson(implicit_euler_half_steps=4)
    solver = FiniteDifferenceSolver(time_stepper=stepper)

    solver.solve(
        generator=option.generator(s_grid),
        boundary_conditions=option.boundary_conditions(s_grid),
        initial_conditions=option.payoff(s_grid),
        time_grid=time_grid,
    )

    schedule = solver.last_step_schedule
    assert [entry.label for entry in schedule[:4]] == ["rannacher_be_half_step"] * 4
    assert [entry.theta for entry in schedule[:4]] == [1.0] * 4
    assert [entry.dt_fraction for entry in schedule[:4]] == [0.5] * 4
    assert schedule[0].base_step_index == 0
    assert schedule[2].base_step_index == 1
    assert schedule[4].label == "crank_nicolson"
    assert schedule[4].theta == 0.5
    assert stepper.schedule_summary() == "4 BE half-steps, then theta=0.5"


def test_rannacher_startup_reduces_near_strike_gamma_roughness() -> None:
    option, s_grid, time_grid = _call_setup()

    pure_cn = BlackScholesPDE(instrument=option, time_stepper=ThetaMethod(0.5)).price(
        option=option,
        s=s_grid,
        t=time_grid,
    )
    rannacher_pricer = BlackScholesPDE(
        instrument=option,
        time_stepper=RannacherCrankNicolson(implicit_euler_half_steps=4),
    )
    rannacher = rannacher_pricer.price(option=option, s=s_grid, t=time_grid)

    assert _gamma_roughness(rannacher, s_grid) < 0.85 * _gamma_roughness(pure_cn, s_grid)
    assert [entry.label for entry in rannacher_pricer.last_step_schedule[:4]] == [
        "rannacher_be_half_step"
    ] * 4
    assert np.all(np.isfinite(rannacher))
