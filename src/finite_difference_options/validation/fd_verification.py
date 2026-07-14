"""Deterministic FD numerical verification/evidence bundle for issue #142."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from math import log
from pathlib import Path
from typing import Any, cast

import numpy as np

from finite_difference_options.contracts import DEFAULT_FD_CAPABILITY_MANIFEST
from finite_difference_options.integrations.compiled_pde_adapter import (
    EXPECTED_COMPILED_HASH,
    EXPECTED_PROBLEM_ID,
    EXPECTED_SOURCE_IR_HASH,
    packaged_compiled_black_scholes_fixture,
    screen_compiled_pde_payload,
)
from finite_difference_options.integrations.compiled_pde_black_scholes_route import (
    _black_scholes_matrix,
    _solve_compiled_black_scholes_grid,
    _upper_call_boundary,
)
from finite_difference_options.integrations.haircut_protocol import installed_distribution_version
from finite_difference_options.validation.black_scholes_parity import (
    black_scholes_call_greeks,
    black_scholes_call_oracle,
)
from finite_difference_options.validation.fd_evidence_integrity import (
    HASH_KEYS as _HASH_KEYS,
    canonicalize as _canonicalize,
    hashes_for_bundle as _hashes_for_bundle,
)
from finite_difference_options.validation.fd_perturbations import perturbation_evidence

FD_BS_VERIFICATION_BENCHMARK_ID = "fd-bs-001"
FD_BS_VERIFICATION_VERSIONED_ID = "FD-BS-001-V0"
_SPATIAL_LEVELS = ((40, 120), (80, 120), (120, 120))
_TEMPORAL_LEVELS = ((160, 40), (160, 80), (160, 160))
_TEMPORAL_REFERENCE_T_STEPS = 640
_FULL_LEVELS = ((40, 40), (80, 120), (120, 200))
_PRICE_TOL = 5.0e-4
_DELTA_TOL = 1.0e-3
_GAMMA_TOL = 8.0e-3
_RESIDUAL_TOL = 1.0e-10
_PAYOFF_TOL = 1.0e-12
_BOUNDARY_TOL = 1.0e-12
_TEMPORAL_ORDER_TOL = 1.8
_MANUFACTURED_ORDER_TOL = 1.8



class FDVerificationError(ValueError):
    """Raised when a verification artifact fails content/hash/numerical checks."""

    def __init__(self, failures: tuple[str, ...]) -> None:
        self.failures = failures
        super().__init__("; ".join(failures))


def run_fd_bs_verification_benchmark() -> dict[str, Any]:
    """Return the public-synthetic Black-Scholes FD evidence bundle."""

    route = screen_compiled_pde_payload(packaged_compiled_black_scholes_fixture()).route
    numerics = cast(Mapping[str, Any], route["numerics"])
    spot = float(numerics["spot"])
    strike = float(numerics["strike"])
    rate = float(numerics["risk_free_rate"])
    q = float(numerics["dividend_yield"])
    sigma = float(numerics["volatility"])
    maturity = float(numerics["maturity"])
    oracle = black_scholes_call_oracle(spot, strike, rate, sigma, maturity, dividend_yield=q)
    greeks = black_scholes_call_greeks(spot, strike, rate, sigma, maturity, dividend_yield=q)
    spatial = _refinement_table(route, _SPATIAL_LEVELS, oracle, greeks)
    temporal = _temporal_refinement_table(route, _TEMPORAL_LEVELS, oracle, greeks)
    full = _refinement_table(route, _FULL_LEVELS, oracle, greeks)
    finest = full["rows"][-1]
    manufactured = _manufactured_residual_table(rate=rate, q=q, sigma=sigma)
    perturbations = perturbation_evidence(
        route,
        finest,
        manufactured,
        residual_tol=_RESIDUAL_TOL,
        boundary_tol=_BOUNDARY_TOL,
    )
    results = {
        "black_scholes_oracle": {"price": oracle, **greeks, "dividend_yield": q},
        "spatial_refinement": spatial,
        "temporal_refinement": temporal,
        "full_refinement": full,
        "manufactured_solution": manufactured,
        "perturbations": perturbations,
        "tolerances": {
            "price_abs": _PRICE_TOL,
            "delta_abs": _DELTA_TOL,
            "gamma_abs": _GAMMA_TOL,
            "pde_residual_linf": _RESIDUAL_TOL,
            "payoff_linf": _PAYOFF_TOL,
            "boundary_linf": _BOUNDARY_TOL,
        },
    }
    request = {
        "benchmark_id": FD_BS_VERIFICATION_BENCHMARK_ID,
        "versioned_benchmark_id": FD_BS_VERIFICATION_VERSIONED_ID,
        "problem_id": EXPECTED_PROBLEM_ID,
        "route_id": "fd.compiled_pde.black_scholes_call_v0",
        "requested_outputs": ("value", "delta", "gamma"),
        "privacy_class": "public_synthetic",
    }
    config = {
        "backend_id": DEFAULT_FD_CAPABILITY_MANIFEST.backend_id,
        "code_version": installed_distribution_version(),
        "source_ir_canonical_hash": EXPECTED_SOURCE_IR_HASH,
        "compiled_hash": EXPECTED_COMPILED_HASH,
        "spatial_levels": _SPATIAL_LEVELS,
        "temporal_levels": _TEMPORAL_LEVELS,
        "temporal_reference_t_steps": _TEMPORAL_REFERENCE_T_STEPS,
        "full_levels": _FULL_LEVELS,
        "theta": float(numerics["theta"]),
        "operator": "dV/dtau = 0.5*sigma^2*S^2*V_SS + (r-q)*S*V_S - r*V",
    }
    convention = {
        "measure": route["measure"],
        "numeraire": route["numeraire"],
        "units": route["units"],
        "time_orientation": route["time_orientation"],
        "state_coordinate": "spot S",
        "boundary_conditions": route["boundary_conditions"],
        "boundary_schedule_source": "compiled_route_explicit_schedule",
    }
    bundle = {
        "schema_version": "finite-difference-options.fd-verification-evidence/v0",
        "benchmark_id": FD_BS_VERIFICATION_BENCHMARK_ID,
        "request": request,
        "config": config,
        "convention": convention,
        "results": results,
    }
    status = "passed" if _evaluate_gates(results) else "failed"
    bundle["evidence"] = {"status": status}
    bundle["evidence"] = {
        "status": status,
        "hashes": _hashes_for_bundle(bundle),
        "validation_rule": "status, hashes, and numerical truth are recomputed by validate_fd_bs_verification_bundle",
    }
    return bundle


def write_fd_bs_verification_json(path: str | Path) -> dict[str, Any]:
    """Run the verification benchmark, validate it, and persist deterministic JSON."""

    bundle = run_fd_bs_verification_benchmark()
    validate_fd_bs_verification_bundle(bundle)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bundle


def validate_fd_bs_verification_bundle(bundle: Mapping[str, Any]) -> None:
    """Recompute hashes and numerical evidence; never trust stored booleans."""

    failures: list[str] = []
    supplied = copy.deepcopy(dict(bundle))
    hashes = cast(Mapping[str, Any], cast(Mapping[str, Any], supplied.get("evidence", {})).get("hashes", {}))
    recomputed_hashes = _hashes_for_bundle(supplied)
    for key in _HASH_KEYS:
        if hashes.get(key) != recomputed_hashes[key]:
            failures.append(f"hash mismatch: {key}")

    fresh = run_fd_bs_verification_benchmark()
    if _canonicalize(supplied.get("request")) != _canonicalize(fresh["request"]):
        failures.append("request does not match executable benchmark")
    if _canonicalize(supplied.get("config")) != _canonicalize(fresh["config"]):
        failures.append("config does not match executable benchmark")
    if _canonicalize(supplied.get("convention")) != _canonicalize(fresh["convention"]):
        failures.append("convention does not match executable benchmark")
    if _canonicalize(supplied.get("results")) != _canonicalize(fresh["results"]):
        failures.append("results do not match recomputed numerical truth")
    supplied_results = cast(Mapping[str, Any], supplied.get("results", {}))
    supplied_status = cast(Mapping[str, Any], supplied.get("evidence", {})).get("status")
    expected_status = "passed" if _evaluate_gates(supplied_results) else "failed"
    if supplied_status != expected_status:
        failures.append("evidence status does not match recomputed gates")
    if not _evaluate_gates(supplied_results):
        failures.append("numerical gates failed from recomputed metrics")
    if failures:
        raise FDVerificationError(tuple(failures))


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
        "pde_residual_linf": residuals["pde_residual_linf"],
        "pde_residual_l2": residuals["pde_residual_l2"],
        "boundary_linf": residuals["boundary_linf"],
        "algebraic_residual_linf": residuals["algebraic_residual_linf"],
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
        "pde_residual_linf": float(np.max(np.abs(residual))),
        "pde_residual_l2": float(np.linalg.norm(residual) / max(1, residual.size) ** 0.5),
        "boundary_linf": boundary_error,
        "algebraic_residual_linf": float(np.max(np.abs(residual))),
    }


def _manufactured_residual_table(*, rate: float, q: float, sigma: float) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    alpha = 0.17
    tau = 0.37
    for s_steps in (41, 81, 161):
        grid = np.linspace(0.0, 3.0, s_steps)
        matrix = _black_scholes_matrix(grid, risk_free_rate=rate, dividend_yield=q, volatility=sigma)
        exact = _manufactured_u(grid, tau, alpha)
        residual = alpha * exact - matrix @ exact - _manufactured_source(grid, tau, alpha, rate, q, sigma)
        interior = residual[1:-1]
        rows.append(
            {
                "s_steps": s_steps,
                "h": float(np.max(np.diff(grid))),
                "residual_linf": float(np.max(np.abs(interior))),
                "residual_l2": float(np.linalg.norm(interior) / max(1, interior.size) ** 0.5),
            }
        )
    for index in range(1, len(rows)):
        rows[index]["observed_residual_order"] = _observed_order(
            float(rows[index - 1]["residual_linf"]),
            float(rows[index]["residual_linf"]),
            float(rows[index - 1]["h"]),
            float(rows[index]["h"]),
        )
    return {
        "exact_solution": "u(S,tau)=exp(alpha*tau)*(1 + 0.2*S + 0.05*S^3)",
        "source": "f=u_tau-L[u] evaluated analytically",
        "rows": rows,
        "min_observed_residual_order": _min_order(rows, key="observed_residual_order"),
    }


def _manufactured_u(grid: np.ndarray, tau: float, alpha: float) -> np.ndarray:
    return np.exp(alpha * tau) * (1.0 + 0.2 * grid + 0.05 * grid**3)


def _manufactured_source(grid: np.ndarray, tau: float, alpha: float, rate: float, q: float, sigma: float) -> np.ndarray:
    u = _manufactured_u(grid, tau, alpha)
    e = np.exp(alpha * tau)
    u_s = e * (0.2 + 0.15 * grid**2)
    u_ss = e * (0.3 * grid)
    l_cont = 0.5 * sigma * sigma * grid * grid * u_ss + (rate - q) * grid * u_s - rate * u
    return alpha * u - l_cont


def _evaluate_gates(results: Mapping[str, Any]) -> bool:
    full = cast(Mapping[str, Any], results.get("full_refinement", {}))
    rows = cast(list[Mapping[str, Any]], full.get("rows", ()))
    if len(rows) < 3:
        return False
    finest = rows[-1]
    no_arb = finest["price"] >= 0.0 and 0.0 <= finest["delta"] <= 1.0 and finest["gamma"] >= 0.0
    residuals = (
        float(finest["payoff_linf"]) <= _PAYOFF_TOL
        and float(finest["pde_residual_linf"]) <= _RESIDUAL_TOL
        and float(finest["boundary_linf"]) <= _BOUNDARY_TOL
    )
    oracle = _oracle_bounded(finest)
    temporal = cast(Mapping[str, Any], results.get("temporal_refinement", {}))
    temporal_order = temporal.get("min_observed_temporal_price_order")
    temporal_rows = cast(list[Mapping[str, Any]], temporal.get("rows", ()))
    temporal_ok = (
        temporal_order is not None
        and float(temporal_order) >= _TEMPORAL_ORDER_TOL
        and all(_oracle_bounded(row) for row in temporal_rows)
    )
    manufactured = cast(Mapping[str, Any], results.get("manufactured_solution", {}))
    manufactured_order = manufactured.get("min_observed_residual_order")
    manufactured_ok = manufactured_order is not None and float(manufactured_order) >= _MANUFACTURED_ORDER_TOL
    return oracle and residuals and no_arb and temporal_ok and manufactured_ok and _perturbations_fail(results)


def _oracle_bounded(row: Mapping[str, Any]) -> bool:
    return (
        float(row["price_abs"]) <= _PRICE_TOL
        and float(row["delta_abs"]) <= _DELTA_TOL
        and float(row["gamma_abs"]) <= _GAMMA_TOL
    )


def _perturbations_fail(results: Mapping[str, Any]) -> bool:
    perturb = cast(Mapping[str, Any], results.get("perturbations", {}))
    cases = cast(Mapping[str, Mapping[str, Any]], perturb.get("cases", {}))
    required = {"operator_sign_flip", "reaction_sign_flip", "source_shift", "static_boundary"}
    baseline_ok = bool(perturb.get("baseline_passes")) and set(cases) == required
    return baseline_ok and all(_case_recomputes_fail(c) for c in cases.values())


def _case_recomputes_fail(case: Mapping[str, Any]) -> bool:
    metric = str(case.get("metric"))
    if metric not in {"residual_linf", "boundary_linf"} or metric not in case:
        return False
    recomputed_pass = float(case[metric]) <= float(case.get("threshold", 0.0))
    return bool(case.get("passes")) == recomputed_pass and not recomputed_pass


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


__all__ = [
    "FD_BS_VERIFICATION_BENCHMARK_ID",
    "FD_BS_VERIFICATION_VERSIONED_ID",
    "FDVerificationError",
    "run_fd_bs_verification_benchmark",
    "validate_fd_bs_verification_bundle",
    "write_fd_bs_verification_json",
]
