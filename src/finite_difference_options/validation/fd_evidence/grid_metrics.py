"""Private grid-level refinement and residual measurements for FD evidence."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from math import log
from typing import Any, cast
import numpy as np
from finite_difference_options.integrations.compiled_pde_black_scholes_route import (
    _black_scholes_matrix,
    _solve_compiled_black_scholes_grid,
    _upper_call_boundary,
)


_TEMPORAL_REFERENCE_T_STEPS = 640


def _refinement_table(
    route: Mapping[str, Any],
    levels: tuple[tuple[int, int], ...],
    oracle: float,
    greeks: Mapping[str, float],
) -> dict[str, Any]:
    rows = [_run_grid_level(route, s_steps, t_steps, oracle, greeks) for s_steps, t_steps in levels]
    for index in range(1, len(rows)):
        prev = rows[index - 1]
        curr = rows[index]
        prev_scale, curr_scale = _order_scales(prev, curr)
        curr["observed_price_order"] = _observed_order(
            float(prev["price_abs"]), float(curr["price_abs"]), prev_scale, curr_scale
        )
    return {"levels": levels, "rows": rows, "min_observed_price_order": _min_order(rows)}


def _temporal_refinement_table(
    route: Mapping[str, Any],
    levels: tuple[tuple[int, int], ...],
    oracle: float,
    greeks: Mapping[str, float],
) -> dict[str, Any]:
    if not levels:
        return {"levels": levels, "rows": (), "min_observed_temporal_price_order": None}
    s_steps = levels[-1][0]
    reference = _run_grid_level(route, s_steps, _TEMPORAL_REFERENCE_T_STEPS, oracle, greeks)
    reference_price = float(reference["price"])
    rows = []
    for s_level, t_steps in levels:
        if s_level != s_steps:
            raise ValueError("temporal refinement levels must hold spatial grid fixed")
        row = _run_grid_level(route, s_level, t_steps, oracle, greeks)
        row["temporal_reference_price"] = reference_price
        row["temporal_reference_t_steps"] = _TEMPORAL_REFERENCE_T_STEPS
        row["temporal_price_abs"] = float(abs(float(row["price"]) - reference_price))
        rows.append(row)
    for index in range(1, len(rows)):
        prev = rows[index - 1]
        curr = rows[index]
        curr["observed_temporal_price_order"] = _observed_order(
            float(prev["temporal_price_abs"]),
            float(curr["temporal_price_abs"]),
            float(prev["dt"]),
            float(curr["dt"]),
        )
    return {
        "levels": levels,
        "reference": {
            "s_steps": s_steps,
            "t_steps": _TEMPORAL_REFERENCE_T_STEPS,
            "price": reference_price,
            "price_abs": reference["price_abs"],
            "method": "same-spatial-grid high-time-step reference isolates temporal error",
        },
        "rows": rows,
        "min_observed_temporal_price_order": _min_order(rows, key="observed_temporal_price_order"),
    }


def _run_grid_level(
    route: Mapping[str, Any],
    s_steps: int,
    t_steps: int,
    oracle: float,
    greeks: Mapping[str, float],
) -> dict[str, Any]:
    numerics = cast(Mapping[str, Any], route["numerics"])
    domain = cast(Mapping[str, Any], numerics["domain"])
    s_grid = np.linspace(float(domain["s_min"]), float(domain["s_max"]), s_steps)
    t_grid = np.linspace(float(domain["t_min"]), float(domain["t_max"]), t_steps)
    values, schedule, _operator = _solve_compiled_black_scholes_grid(
        spot_grid=s_grid,
        time_grid=t_grid,
        strike=float(numerics["strike"]),
        risk_free_rate=float(numerics["risk_free_rate"]),
        dividend_yield=float(numerics["dividend_yield"]),
        volatility=float(numerics["volatility"]),
        theta=float(numerics["theta"]),
    )
    delta_slice = np.gradient(values[-1], s_grid, edge_order=2)
    gamma_slice = np.gradient(delta_slice, s_grid, edge_order=2)
    spot = float(numerics["spot"])
    price = float(np.interp(spot, s_grid, values[-1]))
    delta = float(np.interp(spot, s_grid, delta_slice))
    gamma = float(np.interp(spot, s_grid, gamma_slice))
    residuals = _residuals(values, s_grid, t_grid, numerics)
    return {
        "s_steps": s_steps,
        "t_steps": t_steps,
        "h": float(np.max(np.diff(s_grid))),
        "dt": float(np.max(np.diff(t_grid))),
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "oracle_price": oracle,
        "reference_delta": greeks["delta"],
        "reference_gamma": greeks["gamma"],
        "price_abs": float(abs(price - oracle)),
        "delta_abs": float(abs(delta - greeks["delta"])),
        "gamma_abs": float(abs(gamma - greeks["gamma"])),
        "payoff_linf": residuals["payoff_linf"],
        "boundary_linf": residuals["boundary_linf"],
        "algebraic_residual_linf": residuals["algebraic_residual_linf"],
        "algebraic_residual_l2": residuals["algebraic_residual_l2"],
        "boundary_schedule_applied": schedule,
    }


def _residuals(
    values: np.ndarray,
    s_grid: np.ndarray,
    t_grid: np.ndarray,
    numerics: Mapping[str, Any],
) -> dict[str, float]:
    strike = float(numerics["strike"])
    rate = float(numerics["risk_free_rate"])
    q = float(numerics["dividend_yield"])
    sigma = float(numerics["volatility"])
    theta = float(numerics["theta"])
    operator = _black_scholes_matrix(s_grid, risk_free_rate=rate, dividend_yield=q, volatility=sigma)
    residuals = []
    for prev, curr, tau_prev, tau_next in zip(values[:-1], values[1:], t_grid[:-1], t_grid[1:], strict=True):
        dt = float(tau_next - tau_prev)
        blended = theta * curr + (1.0 - theta) * prev
        residuals.append(((curr - prev) / dt - operator @ blended)[1:-1])
    residual = np.concatenate(residuals)
    payoff = np.maximum(s_grid - strike, 0.0)
    upper = np.array(
        [
            _upper_call_boundary(
                s_grid[-1],
                strike=strike,
                risk_free_rate=rate,
                dividend_yield=q,
                tau=float(tau),
            )
            for tau in t_grid
        ]
    )
    boundary_error = max(
        float(np.max(np.abs(values[:, 0]))),
        float(np.max(np.abs(values[:, -1] - upper))),
    )
    return {
        "payoff_linf": float(np.max(np.abs(values[0] - payoff))),
        "boundary_linf": boundary_error,
        "algebraic_residual_linf": float(np.max(np.abs(residual))),
        "algebraic_residual_l2": float(np.linalg.norm(residual) / max(1, residual.size) ** 0.5),
    }


def _order_scales(prev: Mapping[str, Any], curr: Mapping[str, Any]) -> tuple[float, float]:
    prev_h = float(prev["h"])
    curr_h = float(curr["h"])
    if abs(prev_h - curr_h) > 1.0e-15:
        return prev_h, curr_h
    return float(prev["dt"]), float(curr["dt"])


def _observed_order(coarse_error: float, fine_error: float, coarse_h: float, fine_h: float) -> float | None:
    if coarse_error <= 0.0 or fine_error <= 0.0 or coarse_error <= fine_error:
        return None
    return float(log(coarse_error / fine_error) / log(coarse_h / fine_h))


def _min_order(rows: Sequence[Mapping[str, Any]], key: str = "observed_price_order") -> float | None:
    orders = [float(row[key]) for row in rows if row.get(key) is not None]
    return min(orders) if orders else None
