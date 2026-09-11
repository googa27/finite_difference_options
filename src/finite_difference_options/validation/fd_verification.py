"""Deterministic FD numerical verification/evidence bundle for issue #142."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
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
from finite_difference_options.validation.fd_evidence.grid_metrics import (
    _TEMPORAL_REFERENCE_T_STEPS as _TEMPORAL_REFERENCE_T_STEPS,
    _refinement_table as _refinement_table,
    _temporal_refinement_table as _temporal_refinement_table,
    _run_grid_level as _run_grid_level,
    _residuals as _residuals,
    _order_scales as _order_scales,
    _observed_order as _observed_order,
    _min_order as _min_order,
)
from finite_difference_options.validation.fd_evidence.integrity import (
    HASH_KEYS as _HASH_KEYS,
    canonicalize as _canonicalize,
    hashes_for_bundle as _hashes_for_bundle,
)
from finite_difference_options.validation.fd_evidence.manufactured import (
    manufactured_source,
    manufactured_u,
)
from finite_difference_options.validation.fd_evidence.perturbations import perturbation_evidence

FD_BS_VERIFICATION_BENCHMARK_ID = "fd-bs-001"
FD_BS_VERIFICATION_VERSIONED_ID = "FD-BS-001-V0"
FD_BS_VERIFICATION_SCHEMA_VERSION = "finite-difference-options.fd-verification-evidence/v0"
_SPATIAL_LEVELS = ((40, 120), (80, 120), (120, 120))
_TEMPORAL_LEVELS = ((160, 40), (160, 80), (160, 160))
_FULL_LEVELS = ((40, 40), (80, 120), (120, 200))
_PRICE_TOL = 5.0e-4
_DELTA_TOL = 1.0e-3
_GAMMA_TOL = 8.0e-3
_ALGEBRAIC_RESIDUAL_TOL = 1.0e-8
_PDE_CONSISTENCY_H2_COEFFICIENT_TOL = 3.0e-2
_PAYOFF_TOL = 1.0e-12
_BOUNDARY_TOL = 1.0e-10
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
        algebraic_tol=_ALGEBRAIC_RESIDUAL_TOL,
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
            "algebraic_residual_linf": _ALGEBRAIC_RESIDUAL_TOL,
            "pde_consistency_h2_coefficient": _PDE_CONSISTENCY_H2_COEFFICIENT_TOL,
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
        "source_ir_canonical_hash": EXPECTED_SOURCE_IR_HASH,
        "compiled_hash": EXPECTED_COMPILED_HASH,
        "spatial_levels": _SPATIAL_LEVELS,
        "temporal_levels": _TEMPORAL_LEVELS,
        "temporal_reference_t_steps": _TEMPORAL_REFERENCE_T_STEPS,
        "full_levels": _FULL_LEVELS,
        "theta": float(numerics["theta"]),
        "operator": "dV/dtau = 0.5*sigma^2*S^2*V_SS + (r-q)*S*V_S - r*V",
    }
    provenance = {
        "distribution": "finite-difference-options",
        "code_version": installed_distribution_version(),
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
        "schema_version": FD_BS_VERIFICATION_SCHEMA_VERSION,
        "benchmark_id": FD_BS_VERIFICATION_BENCHMARK_ID,
        "request": request,
        "config": config,
        "provenance": provenance,
        "convention": convention,
        "results": results,
    }
    status = "passed" if _evaluate_gates(results) else "failed"
    evidence: dict[str, Any] = {"status": status}
    bundle["evidence"] = evidence
    evidence["hashes"] = _hashes_for_bundle(bundle)
    evidence["validation_rule"] = (
        "status, hashes, and numerical truth are recomputed by validate_fd_bs_verification_bundle"
    )
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
    if supplied.get("schema_version") != FD_BS_VERIFICATION_SCHEMA_VERSION:
        failures.append("schema_version does not match FD verification contract")
    if supplied.get("benchmark_id") != FD_BS_VERIFICATION_BENCHMARK_ID:
        failures.append("benchmark_id does not match FD verification contract")
    provenance = supplied.get("provenance")
    if not isinstance(provenance, Mapping):
        failures.append("provenance must be an object")
    elif (
        provenance.get("distribution") != "finite-difference-options"
        or not isinstance(provenance.get("code_version"), str)
        or not provenance.get("code_version")
    ):
        failures.append("provenance distribution/code_version is invalid")
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


def _manufactured_residual_table(*, rate: float, q: float, sigma: float) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    alpha = 0.17
    tau = 0.37
    for s_steps in (41, 81, 161):
        grid = np.linspace(0.0, 3.0, s_steps)
        matrix = _black_scholes_matrix(grid, risk_free_rate=rate, dividend_yield=q, volatility=sigma)
        exact = manufactured_u(grid, tau, alpha)
        residual = alpha * exact - matrix @ exact - manufactured_source(grid, tau, alpha, rate, q, sigma)
        interior = residual[1:-1]
        rows.append(
            {
                "s_steps": s_steps,
                "h": float(np.max(np.diff(grid))),
                "pde_consistency_linf": float(np.max(np.abs(interior))),
                "pde_consistency_l2": float(np.linalg.norm(interior) / max(1, interior.size) ** 0.5),
            }
        )
    for index in range(1, len(rows)):
        rows[index]["observed_pde_consistency_order"] = _observed_order(
            float(rows[index - 1]["pde_consistency_linf"]),
            float(rows[index]["pde_consistency_linf"]),
            float(rows[index - 1]["h"]),
            float(rows[index]["h"]),
        )
    return {
        "exact_solution": "u(S,tau)=exp(alpha*tau)*(1 + 0.2*S + 0.05*S^3)",
        "source": "f=u_tau-L[u] evaluated analytically",
        "rows": rows,
        "min_observed_pde_consistency_order": _min_order(rows, key="observed_pde_consistency_order"),
    }


def _evaluate_gates(results: Mapping[str, Any]) -> bool:
    full = cast(Mapping[str, Any], results.get("full_refinement", {}))
    rows = cast(list[Mapping[str, Any]], full.get("rows", ()))
    if len(rows) < 3:
        return False
    finest = rows[-1]
    no_arb = finest["price"] >= 0.0 and 0.0 <= finest["delta"] <= 1.0 and finest["gamma"] >= 0.0
    residuals = (
        float(finest["payoff_linf"]) <= _PAYOFF_TOL
        and float(finest["algebraic_residual_linf"]) <= _ALGEBRAIC_RESIDUAL_TOL
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
    manufactured_order = manufactured.get("min_observed_pde_consistency_order")
    manufactured_rows = cast(list[Mapping[str, Any]], manufactured.get("rows", ()))
    manufactured_ok = (
        manufactured_order is not None
        and float(manufactured_order) >= _MANUFACTURED_ORDER_TOL
        and len(manufactured_rows) >= 3
        and float(manufactured_rows[-1]["pde_consistency_linf"])
        <= _PDE_CONSISTENCY_H2_COEFFICIENT_TOL * float(manufactured_rows[-1]["h"]) ** 2
    )
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


__all__ = [
    "FD_BS_VERIFICATION_BENCHMARK_ID",
    "FD_BS_VERIFICATION_SCHEMA_VERSION",
    "FD_BS_VERIFICATION_VERSIONED_ID",
    "FDVerificationError",
    "run_fd_bs_verification_benchmark",
    "validate_fd_bs_verification_bundle",
    "write_fd_bs_verification_json",
]
