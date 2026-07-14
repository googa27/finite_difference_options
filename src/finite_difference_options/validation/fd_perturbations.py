"""Independent perturbation probes for FD verification evidence."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, cast

import numpy as np

from finite_difference_options.integrations.compiled_pde_black_scholes_route import (
    _black_scholes_matrix,
    _run_compiled_black_scholes_route,
    _solve_compiled_black_scholes_grid,
    _upper_call_boundary,
)

PERTURBATION_RESIDUAL_THRESHOLD = 1.0e-3
PERTURBATION_BOUNDARY_THRESHOLD = 1.0e-3


def perturbation_evidence(
    route: Mapping[str, Any],
    finest: Mapping[str, Any],
    manufactured: Mapping[str, Any],
    *,
    residual_tol: float,
    boundary_tol: float,
) -> dict[str, Any]:
    """Return recomputed negative-control residual and boundary evidence."""

    baseline_pde = float(finest["pde_residual_linf"])
    baseline_boundary = float(finest["boundary_linf"])
    rate, q, sigma = _route_rates(route)
    manufactured_rows = cast(list[Mapping[str, Any]], manufactured["rows"])
    return {
        "baseline_passes": baseline_pde <= residual_tol and baseline_boundary <= boundary_tol,
        "baseline": {"pde_residual_linf": baseline_pde, "boundary_linf": baseline_boundary},
        "manufactured_baseline_linf": float(manufactured_rows[-1]["residual_linf"]),
        "cases": {
            "operator_sign_flip": _residual_case(_wrong_operator_residual(route, sign=-1.0)),
            "reaction_sign_flip": _residual_case(_wrong_operator_residual(route, reaction_sign=1.0)),
            "source_shift": _residual_case(
                _source_shift_residual(rate=rate, q=q, sigma=sigma, source_shift=1.0e-2)
            ),
            "static_boundary": _boundary_case(_static_boundary_error(route)),
        },
    }


def _route_rates(route: Mapping[str, Any]) -> tuple[float, float, float]:
    numerics = cast(Mapping[str, Any], route["numerics"])
    return float(numerics["risk_free_rate"]), float(numerics["dividend_yield"]), float(numerics["volatility"])


def _residual_case(value: float) -> dict[str, float | bool | str]:
    threshold = PERTURBATION_RESIDUAL_THRESHOLD
    return {"metric": "residual_linf", "residual_linf": value, "threshold": threshold, "passes": value <= threshold}


def _boundary_case(value: float) -> dict[str, float | bool | str]:
    threshold = PERTURBATION_BOUNDARY_THRESHOLD
    return {"metric": "boundary_linf", "boundary_linf": value, "threshold": threshold, "passes": value <= threshold}


def _source_shift_residual(*, rate: float, q: float, sigma: float, source_shift: float) -> float:
    alpha = 0.17
    tau = 0.37
    grid = np.linspace(0.0, 3.0, 161)
    matrix = _black_scholes_matrix(grid, risk_free_rate=rate, dividend_yield=q, volatility=sigma)
    exact = _manufactured_u(grid, tau, alpha)
    source = _manufactured_source(grid, tau, alpha, rate, q, sigma) + source_shift
    residual = alpha * exact - matrix @ exact - source
    return float(np.max(np.abs(residual[1:-1])))


def _manufactured_u(grid: np.ndarray, tau: float, alpha: float) -> np.ndarray:
    return np.exp(alpha * tau) * (1.0 + 0.2 * grid + 0.05 * grid**3)


def _manufactured_source(grid: np.ndarray, tau: float, alpha: float, rate: float, q: float, sigma: float) -> np.ndarray:
    u = _manufactured_u(grid, tau, alpha)
    e = np.exp(alpha * tau)
    u_s = e * (0.2 + 0.15 * grid**2)
    u_ss = e * (0.3 * grid)
    return alpha * u - (0.5 * sigma * sigma * grid * grid * u_ss + (rate - q) * grid * u_s - rate * u)


def _wrong_operator_residual(route: Mapping[str, Any], *, sign: float = 1.0, reaction_sign: float = -1.0) -> float:
    numerics = cast(Mapping[str, Any], route["numerics"])
    wrong = copy.deepcopy(dict(route))
    wrong["numerics"] = dict(numerics) | {"grid_levels": ((80, 120),)}
    row = cast(Mapping[str, Any], _run_compiled_black_scholes_route(wrong)["convergence"][-1])
    domain = cast(Mapping[str, Any], numerics["domain"])
    s_grid = np.linspace(float(domain["s_min"]), float(domain["s_max"]), int(row["s_steps"]))
    t_grid = np.linspace(float(domain["t_min"]), float(domain["t_max"]), int(row["t_steps"]))
    values, _schedule, _operator = _solve_compiled_black_scholes_grid(
        spot_grid=s_grid,
        time_grid=t_grid,
        strike=float(numerics["strike"]),
        risk_free_rate=float(numerics["risk_free_rate"]),
        dividend_yield=float(numerics["dividend_yield"]),
        volatility=float(numerics["volatility"]),
        theta=float(numerics["theta"]),
    )
    matrix = sign * _black_scholes_matrix(
        s_grid,
        risk_free_rate=float(numerics["risk_free_rate"]),
        dividend_yield=float(numerics["dividend_yield"]),
        volatility=float(numerics["volatility"]),
    )
    if reaction_sign > 0.0:
        matrix += 2.0 * float(numerics["risk_free_rate"]) * np.eye(len(s_grid))
    theta = float(numerics["theta"])
    residual = [
        ((curr - prev) / float(tau_next - tau_prev) - matrix @ (theta * curr + (1.0 - theta) * prev))[1:-1]
        for prev, curr, tau_prev, tau_next in zip(values[:-1], values[1:], t_grid[:-1], t_grid[1:], strict=True)
    ]
    return float(np.max(np.abs(np.concatenate(residual))))


def _static_boundary_error(route: Mapping[str, Any]) -> float:
    numerics = cast(Mapping[str, Any], route["numerics"])
    domain = cast(Mapping[str, Any], numerics["domain"])
    s_max = float(domain["s_max"])
    static_upper = s_max - float(numerics["strike"])
    dynamic_upper = _upper_call_boundary(
        s_max,
        strike=float(numerics["strike"]),
        risk_free_rate=float(numerics["risk_free_rate"]),
        dividend_yield=float(numerics["dividend_yield"]),
        tau=float(numerics["maturity"]),
    )
    return float(abs(dynamic_upper - static_upper))


__all__ = [
    "PERTURBATION_BOUNDARY_THRESHOLD",
    "PERTURBATION_RESIDUAL_THRESHOLD",
    "perturbation_evidence",
]
