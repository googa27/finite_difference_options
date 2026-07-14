"""Numerical executor for the exact public-synthetic compiled Black--Scholes route."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from math import exp
from typing import Any, cast

import numpy as np

from finite_difference_options.validation.black_scholes_parity import (
    black_scholes_call_greeks,
    black_scholes_call_oracle,
)


def _run_compiled_black_scholes_route(route: Mapping[str, Any]) -> dict[str, Any]:
    numerics = cast(Mapping[str, Any], route["numerics"])
    spot = float(numerics["spot"])
    strike = float(numerics["strike"])
    rate = float(numerics["risk_free_rate"])
    dividend_yield = float(numerics["dividend_yield"])
    sigma = float(numerics["volatility"])
    maturity = float(numerics["maturity"])
    theta = float(numerics["theta"])
    tolerance = float(numerics["tolerance"])
    domain = cast(Mapping[str, Any], numerics["domain"])
    s_min = float(domain["s_min"])
    s_max = float(domain["s_max"])
    t_min = float(domain["t_min"])
    t_max = float(domain["t_max"])
    grid_levels = tuple((int(s), int(t)) for s, t in cast(Sequence[Sequence[int]], numerics["grid_levels"]))

    oracle_price = black_scholes_call_oracle(spot, strike, rate, sigma, maturity)
    reference_greeks = black_scholes_call_greeks(spot, strike, rate, sigma, maturity)

    observations: list[dict[str, float | int]] = []
    final_values: np.ndarray | None = None
    final_s_grid: np.ndarray | None = None
    final_t_grid: np.ndarray | None = None
    final_boundary_schedule: tuple[dict[str, float | str], ...] = ()
    final_operator_diagnostics: dict[str, Any] = {}

    for s_steps, t_steps in grid_levels:
        s_grid = np.linspace(s_min, s_max, s_steps, dtype=np.float64)
        t_grid = np.linspace(t_min, t_max, t_steps, dtype=np.float64)
        values, boundary_schedule, operator_diagnostics = _solve_compiled_black_scholes_grid(
            spot_grid=s_grid,
            time_grid=t_grid,
            strike=strike,
            risk_free_rate=rate,
            dividend_yield=dividend_yield,
            volatility=sigma,
            theta=theta,
        )
        price = float(np.interp(spot, s_grid, values[-1]))
        observations.append(
            {
                "s_steps": s_steps,
                "t_steps": t_steps,
                "price": price,
                "oracle_price": oracle_price,
                "abs_error": float(abs(price - oracle_price)),
            }
        )
        if (s_steps, t_steps) == grid_levels[-1]:
            final_values = values.copy()
            final_s_grid = s_grid
            final_t_grid = t_grid
            final_boundary_schedule = boundary_schedule
            final_operator_diagnostics = operator_diagnostics

    assert final_values is not None and final_s_grid is not None and final_t_grid is not None
    delta_slice = np.gradient(final_values[-1], final_s_grid, edge_order=2)
    gamma_slice = np.gradient(delta_slice, final_s_grid, edge_order=2)
    fd_price = float(np.interp(spot, final_s_grid, final_values[-1]))
    fd_delta = float(np.interp(spot, final_s_grid, delta_slice))
    fd_gamma = float(np.interp(spot, final_s_grid, gamma_slice))
    price_abs = float(abs(fd_price - oracle_price))
    delta_abs = float(abs(fd_delta - reference_greeks["delta"]))
    gamma_abs = float(abs(fd_gamma - reference_greeks["gamma"]))
    resource_controls = {
        "max_s_steps": max(level[0] for level in grid_levels),
        "max_t_steps": max(level[1] for level in grid_levels),
        "grid_levels": len(grid_levels),
        "deterministic": "true",
        "boundary_rebuilt_each_time_step": "true",
    }
    return {
        "oracle_price": oracle_price,
        "price": fd_price,
        "delta": fd_delta,
        "gamma": fd_gamma,
        "reference_delta": reference_greeks["delta"],
        "reference_gamma": reference_greeks["gamma"],
        "convergence": tuple(observations),
        "converged": price_abs <= tolerance,
        "errors": {
            "price_abs": price_abs,
            "price_rel": float(price_abs / max(1.0e-12, abs(oracle_price))),
            "delta_abs": delta_abs,
            "delta_rel": float(
                delta_abs / max(1.0e-12, abs(reference_greeks["delta"])),
            ),
            "gamma_abs": gamma_abs,
            "gamma_rel": float(
                gamma_abs / max(1.0e-12, abs(reference_greeks["gamma"])),
            ),
            "max_abs_price_error": float(max(row["abs_error"] for row in observations)),
        },
        "no_arbitrage": _compiled_no_arbitrage(spot, strike, fd_price, fd_delta, fd_gamma),
        "boundary_schedule_applied": {
            "source": "compiled_route_explicit_schedule",
            "lower_expression": "V(0,tau)=0",
            "upper_expression": "V(Smax,tau)=Smax-K*exp(-r*tau)",
            "tau_grid_count": int(len(final_t_grid)),
            "applied": final_boundary_schedule,
        },
        "boundary_assumptions": (
            "lower boundary: explicit compiled PDE Dirichlet V(0,tau)=0",
            "upper boundary: explicit compiled PDE time-dependent far-field V(Smax,tau)=Smax-K*exp(-r*tau)",
            "uniform physical-price grid on the compiled route domain",
            "theta time stepping with per-step boundary-row rebuild",
        ),
        "resource_controls": resource_controls,
        "operator": final_operator_diagnostics,
        "time_schedule": {
            "theta": theta,
            "orientation": "tau_increasing_from_maturity_to_valuation",
        },
        "grid_metadata": _grid_metadata(final_s_grid, final_t_grid),
        "config_hash": _sha256_ref({"route": "compiled_black_scholes_v0", "numerics": numerics}),
    }


def _solve_compiled_black_scholes_grid(
    *,
    spot_grid: np.ndarray,
    time_grid: np.ndarray,
    strike: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    theta: float,
) -> tuple[np.ndarray, tuple[dict[str, float | str], ...], dict[str, Any]]:
    _validate_route_grid(spot_grid, time_grid)
    operator = _black_scholes_matrix(
        spot_grid,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
    )
    identity = np.eye(len(spot_grid), dtype=np.float64)
    values = np.empty((len(time_grid), len(spot_grid)), dtype=np.float64)
    values[0] = np.maximum(spot_grid - strike, 0.0)
    values[0, 0] = 0.0
    values[0, -1] = _upper_call_boundary(
        spot_grid[-1],
        strike=strike,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        tau=0.0,
    )
    schedule = [
        {
            "step_index": 0,
            "tau": float(time_grid[0]),
            "calendar_time": float(time_grid[-1] - time_grid[0]),
            "lower": 0.0,
            "upper": float(values[0, -1]),
            "source": "terminal_payoff_boundary",
        }
    ]
    for step_index, (tau_prev, tau_next) in enumerate(zip(time_grid[:-1], time_grid[1:], strict=True), start=1):
        dt = float(tau_next - tau_prev)
        lhs = identity - theta * dt * operator
        rhs = (identity + (1.0 - theta) * dt * operator) @ values[step_index - 1]
        lower = 0.0
        upper = _upper_call_boundary(
            spot_grid[-1],
            strike=strike,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            tau=float(tau_next),
        )
        _apply_dirichlet_rows(lhs, rhs, lower, upper)
        solution = np.linalg.solve(lhs, rhs)
        solution[0] = lower
        solution[-1] = upper
        values[step_index] = solution
        schedule.append(
            {
                "step_index": int(step_index),
                "tau": float(tau_next),
                "calendar_time": float(time_grid[-1] - tau_next),
                "lower": float(lower),
                "upper": float(upper),
                "source": "compiled_boundary_expression",
            }
        )
    return (
        values,
        tuple(schedule),
        {
            "operator_sign_convention": "dV/dtau = L[V]",
            "drift": "(r-q) S dV/dS",
            "diffusion": "0.5 sigma^2 S^2 d2V/dS2",
            "reaction": "-r V",
            "matrix_shape": tuple(int(item) for item in operator.shape),
            "nonzero_count": int(np.count_nonzero(operator)),
        },
    )


def _black_scholes_matrix(
    grid: np.ndarray,
    *,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
) -> np.ndarray:
    operator = np.zeros((len(grid), len(grid)), dtype=np.float64)
    drift = risk_free_rate - dividend_yield
    variance = volatility * volatility
    for index in range(1, len(grid) - 1):
        s_i = float(grid[index])
        h_minus = float(grid[index] - grid[index - 1])
        h_plus = float(grid[index + 1] - grid[index])
        d1_minus = -h_plus / (h_minus * (h_minus + h_plus))
        d1_center = (h_plus - h_minus) / (h_minus * h_plus)
        d1_plus = h_minus / (h_plus * (h_minus + h_plus))
        d2_minus = 2.0 / (h_minus * (h_minus + h_plus))
        d2_center = -2.0 / (h_minus * h_plus)
        d2_plus = 2.0 / (h_plus * (h_minus + h_plus))
        diffusion_scale = 0.5 * variance * s_i * s_i
        drift_scale = drift * s_i
        operator[index, index - 1] = diffusion_scale * d2_minus + drift_scale * d1_minus
        operator[index, index] = diffusion_scale * d2_center + drift_scale * d1_center - risk_free_rate
        operator[index, index + 1] = diffusion_scale * d2_plus + drift_scale * d1_plus
    return operator


def _apply_dirichlet_rows(matrix: np.ndarray, rhs: np.ndarray, lower: float, upper: float) -> None:
    matrix[0, :] = 0.0
    matrix[0, 0] = 1.0
    rhs[0] = lower
    matrix[-1, :] = 0.0
    matrix[-1, -1] = 1.0
    rhs[-1] = upper


def _upper_call_boundary(
    s_max: float,
    *,
    strike: float,
    risk_free_rate: float,
    dividend_yield: float,
    tau: float,
) -> float:
    return float(
        max(
            s_max * exp(-dividend_yield * tau) - strike * exp(-risk_free_rate * tau),
            0.0,
        )
    )


def _validate_route_grid(spot_grid: np.ndarray, time_grid: np.ndarray) -> None:
    if spot_grid.ndim != 1 or len(spot_grid) < 5:
        raise ValueError("spot grid must be one-dimensional")
    if time_grid.ndim != 1 or len(time_grid) < 2:
        raise ValueError("time grid must be one-dimensional")
    if not np.all(np.isfinite(spot_grid)) or not np.all(np.isfinite(time_grid)):
        raise ValueError("route grids must be finite")
    if np.any(np.diff(spot_grid) <= 0.0) or np.any(np.diff(time_grid) <= 0.0):
        raise ValueError("route grids must be strictly increasing")


def _grid_metadata(s_grid: np.ndarray, t_grid: np.ndarray) -> dict[str, float]:
    s_spacing = np.diff(s_grid)
    t_spacing = np.diff(t_grid)
    return {
        "s_min": float(s_grid.min()),
        "s_max": float(s_grid.max()),
        "t_min": float(t_grid.min()),
        "t_max": float(t_grid.max()),
        "s_spacing_min": float(np.min(s_spacing)),
        "s_spacing_max": float(np.max(s_spacing)),
        "s_spacing_mean": float(np.mean(s_spacing)),
        "t_spacing_min": float(np.min(t_spacing)),
        "t_spacing_max": float(np.max(t_spacing)),
        "t_spacing_mean": float(np.mean(t_spacing)),
    }


def _compiled_no_arbitrage(spot: float, strike: float, value: float, delta: float, gamma: float) -> dict[str, Any]:
    intrinsic = max(spot - strike, 0.0)
    upper_gap = spot - value
    return {
        "value_minus_intrinsic": float(value - intrinsic),
        "upper_gap": float(upper_gap),
        "value_bound_ok": value >= intrinsic - 1.0e-12,
        "upper_bound_ok": upper_gap >= -1.0e-12,
        "delta_lower_bound_ok": delta >= -1.0e-12,
        "delta_upper_bound_ok": delta <= 1.0 + 1.0e-12,
        "gamma_non_negative_ok": gamma >= -1.0e-12,
        "monotone_call_parity_hint": "call value must be monotone increasing in spot (single-point check)",
    }


def _sha256_ref(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return f"sha256:{sha256(encoded.encode('utf-8')).hexdigest()}"
