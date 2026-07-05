"""Greek derivative convergence and stability gates.

The gates in this module are intentionally small enough for pull-request CI but
structured like production benchmark artifacts.  They validate that requested
Delta/Gamma estimates on nonuniform grids converge against independent
Black--Scholes analytical derivatives, that strike-node alignment sensitivity is
bounded, that expiry kinks fail closed, and that Rannacher startup reduces the
near-strike Gamma oscillation seen with pure Crank--Nicolson.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from math import erf, exp, log, pi, sqrt
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from finite_difference_options.exceptions import ValidationError
from finite_difference_options.greeks import FiniteDifferenceGreeks, GreekEstimate
from finite_difference_options.grids import strike_centered_axis
from finite_difference_options.instruments.base import EuropeanCall
from finite_difference_options.pricing.engines import BlackScholesPDE
from finite_difference_options.processes.affine import GeometricBrownianMotion
from finite_difference_options.solvers.finite_difference import RannacherCrankNicolson, ThetaMethod

GREEK_DERIVATIVE_VALIDATION_BENCHMARK_ID = "FD-GREEKS-VALIDATION-V0"

ValidationMode = Literal["pr", "broad"]
MetricValue = float | bool | int | str


@dataclass(frozen=True)
class GreekDerivativeValidationThresholds:
    """Acceptance bands for Issue #58 derivative gates."""

    max_delta_abs_error: float = 1.0e-3
    max_gamma_abs_error: float = 3.0e-4
    max_finest_to_middle_error_ratio: float = 0.75
    refinement_noise_floor_abs: float = 2.0e-6
    strike_alignment_delta_diff_abs: float = 1.0e-4
    strike_alignment_gamma_diff_abs: float = 2.0e-5
    rannacher_gamma_roughness_ratio: float = 0.85
    max_runtime_seconds: float = 30.0


@dataclass(frozen=True)
class GreekDerivativeValidationReport:
    """Serializable result of the derivative validation gate."""

    benchmark_id: str
    mode: ValidationMode
    passed: bool
    thresholds: GreekDerivativeValidationThresholds
    metrics: dict[str, MetricValue]
    invariants: dict[str, bool]
    matrix: tuple[dict[str, Any], ...]
    strike_alignment: dict[str, Any]
    rannacher: dict[str, Any]
    expiry_policy: dict[str, Any]
    runtime_seconds: float

    def as_dict(self) -> dict[str, Any]:
        """Return the stable artifact payload."""

        return {
            "schema_version": "finite-difference-greek-validation/v0",
            "artifact_kind": "pr-fast-matrix" if self.mode == "pr" else "scheduled-broad-matrix",
            "benchmark_id": self.benchmark_id,
            "mode": self.mode,
            "passed": self.passed,
            "thresholds": asdict(self.thresholds),
            "metrics": self.metrics,
            "invariants": self.invariants,
            "matrix": list(self.matrix),
            "strike_alignment": self.strike_alignment,
            "rannacher": self.rannacher,
            "expiry_policy": self.expiry_policy,
            "runtime_seconds": self.runtime_seconds,
        }


@dataclass(frozen=True)
class _GreekValidationCase:
    moneyness: float
    maturity: float
    sigma: float

    @property
    def spot(self) -> float:
        return 100.0 * self.moneyness


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _normal_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def _black_scholes_call_values(
    spots: NDArray[np.float64],
    *,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
) -> NDArray[np.float64]:
    values = np.zeros_like(spots, dtype=float)
    positive = spots > 0.0
    if not np.any(positive):
        return values
    d1 = (np.log(spots[positive] / strike) + (rate + 0.5 * sigma**2) * maturity) / (
        sigma * sqrt(maturity)
    )
    d2 = d1 - sigma * sqrt(maturity)
    cdf = np.vectorize(_normal_cdf)
    values[positive] = spots[positive] * cdf(d1) - strike * exp(-rate * maturity) * cdf(d2)
    return values


def _black_scholes_delta_gamma(
    spot: float,
    *,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
) -> tuple[float, float]:
    d1 = (log(spot / strike) + (rate + 0.5 * sigma**2) * maturity) / (sigma * sqrt(maturity))
    return _normal_cdf(d1), _normal_pdf(d1) / (spot * sigma * sqrt(maturity))


def _validation_cases(mode: ValidationMode) -> tuple[_GreekValidationCase, ...]:
    maturities = (0.25, 1.0)
    sigmas = (0.15, 0.35)
    moneyness = (0.9, 1.0, 1.1)
    cases = tuple(
        _GreekValidationCase(m, maturity, sigma)
        for m in moneyness
        for maturity in maturities
        for sigma in sigmas
    )
    if mode == "pr":
        return cases
    return cases + tuple(
        _GreekValidationCase(m, maturity, sigma)
        for m in (0.8, 1.2)
        for maturity in (0.1, 2.0)
        for sigma in (0.2, 0.5)
    )


def _grid_values(
    *,
    nodes: int,
    case: _GreekValidationCase,
    strike: float,
    rate: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    axis = strike_centered_axis(
        name="spot",
        lower=20.0,
        upper=250.0,
        nodes=nodes,
        strike=strike,
        concentration=2.5,
    )
    coordinates = np.asarray(axis.coordinates_array, dtype=float)
    values = _black_scholes_call_values(
        coordinates,
        strike=strike,
        rate=rate,
        sigma=case.sigma,
        maturity=case.maturity,
    )
    return coordinates, values


def _material_error_ratio(finest_error: float, middle_error: float, *, noise_floor: float) -> float:
    """Return a refinement ratio, ignoring roundoff-level nonmonotonicity."""

    if max(abs(finest_error), abs(middle_error)) <= noise_floor:
        return 0.0
    return float(finest_error / max(middle_error, 1.0e-18))


def _estimate_payload(estimate: GreekEstimate) -> dict[str, Any]:
    return {
        "value": estimate.value,
        "requested_coordinate": estimate.requested_coordinate,
        "nearest_node_index": estimate.nearest_node_index,
        "nearest_node_coordinate": estimate.nearest_node_coordinate,
        "nearest_node_value": estimate.nearest_node_value,
        "reference_abs_error": estimate.diagnostics["reference_abs_error"],
        "refinement_abs_error": estimate.diagnostics["refinement_abs_error"],
        "reported_abs_error": estimate.diagnostics["reported_abs_error"],
        "diagnostics": dict(estimate.diagnostics),
    }


def _evaluate_validation_case(
    case: _GreekValidationCase,
    *,
    thresholds: GreekDerivativeValidationThresholds,
) -> dict[str, Any]:
    strike = 100.0
    rate = 0.03
    levels = (("coarse", 41), ("middle", 81), ("finest", 161))
    coordinates_by_level: dict[str, NDArray[np.float64]] = {}
    values_by_level: dict[str, NDArray[np.float64]] = {}
    for level_name, nodes in levels:
        coordinates, values = _grid_values(nodes=nodes, case=case, strike=strike, rate=rate)
        coordinates_by_level[level_name] = coordinates
        values_by_level[level_name] = values

    reference_delta, reference_gamma = _black_scholes_delta_gamma(
        case.spot,
        strike=strike,
        rate=rate,
        sigma=case.sigma,
        maturity=case.maturity,
    )
    greeks = FiniteDifferenceGreeks()
    estimates: dict[str, dict[str, Any]] = {}
    for level_name, _nodes in levels:
        refined_name = {"coarse": "middle", "middle": "finest"}.get(level_name)
        refined_values = None if refined_name is None else values_by_level[refined_name]
        refined_coordinates = None if refined_name is None else coordinates_by_level[refined_name]
        delta = greeks.sample_delta(
            values_by_level[level_name],
            coordinates_by_level[level_name],
            case.spot,
            reference_value=reference_delta,
            refined_values=refined_values,
            refined_coordinates=refined_coordinates,
            time_to_expiry=case.maturity,
            nonsmooth_coordinates=(strike,),
        )
        gamma = greeks.sample_gamma(
            values_by_level[level_name],
            coordinates_by_level[level_name],
            case.spot,
            reference_value=reference_gamma,
            refined_values=refined_values,
            refined_coordinates=refined_coordinates,
            time_to_expiry=case.maturity,
            nonsmooth_coordinates=(strike,),
        )
        estimates[level_name] = {
            "nodes": int(coordinates_by_level[level_name].shape[0]),
            "min_spacing": float(np.min(np.diff(coordinates_by_level[level_name]))),
            "max_spacing": float(np.max(np.diff(coordinates_by_level[level_name]))),
            "delta": _estimate_payload(delta),
            "gamma": _estimate_payload(gamma),
        }

    middle_delta_error = float(estimates["middle"]["delta"]["reference_abs_error"])
    middle_gamma_error = float(estimates["middle"]["gamma"]["reference_abs_error"])
    finest_delta_error = float(estimates["finest"]["delta"]["reference_abs_error"])
    finest_gamma_error = float(estimates["finest"]["gamma"]["reference_abs_error"])
    delta_ratio = _material_error_ratio(
        finest_delta_error,
        middle_delta_error,
        noise_floor=thresholds.refinement_noise_floor_abs,
    )
    gamma_ratio = _material_error_ratio(
        finest_gamma_error,
        middle_gamma_error,
        noise_floor=thresholds.refinement_noise_floor_abs,
    )
    return {
        "moneyness": case.moneyness,
        "spot": case.spot,
        "strike": strike,
        "rate": rate,
        "sigma": case.sigma,
        "maturity": case.maturity,
        "grid_family": "strike_centered_nonuniform",
        "reference": {"delta": reference_delta, "gamma": reference_gamma},
        "coarse": estimates["coarse"],
        "middle": estimates["middle"],
        "finest": estimates["finest"],
        "finest_to_middle_error_ratio": {"delta": delta_ratio, "gamma": gamma_ratio},
        "passed": bool(
            finest_delta_error <= thresholds.max_delta_abs_error
            and finest_gamma_error <= thresholds.max_gamma_abs_error
            and delta_ratio <= thresholds.max_finest_to_middle_error_ratio
            and gamma_ratio <= thresholds.max_finest_to_middle_error_ratio
        ),
    }


def _strike_alignment_check() -> dict[str, Any]:
    strike = 100.0
    rate = 0.03
    sigma = 0.25
    maturity = 0.5
    requested_spot = strike
    step = 0.25
    greeks = FiniteDifferenceGreeks()
    grids = {
        "strike_on_node": np.arange(20.0, 180.0 + step / 2.0, step, dtype=float),
        "strike_half_cell_shifted": np.arange(20.0 + step / 2.0, 180.0 + step / 2.0, step, dtype=float),
    }
    estimates: dict[str, dict[str, Any]] = {}
    for key, grid in grids.items():
        values = _black_scholes_call_values(
            grid,
            strike=strike,
            rate=rate,
            sigma=sigma,
            maturity=maturity,
        )
        delta_reference, gamma_reference = _black_scholes_delta_gamma(
            requested_spot,
            strike=strike,
            rate=rate,
            sigma=sigma,
            maturity=maturity,
        )
        estimates[key] = {
            "delta": _estimate_payload(
                greeks.sample_delta(values, grid, requested_spot, reference_value=delta_reference)
            ),
            "gamma": _estimate_payload(
                greeks.sample_gamma(values, grid, requested_spot, reference_value=gamma_reference)
            ),
        }
    delta_on_node = float(estimates["strike_on_node"]["delta"]["value"])
    delta_shifted = float(estimates["strike_half_cell_shifted"]["delta"]["value"])
    gamma_on_node = float(estimates["strike_on_node"]["gamma"]["value"])
    gamma_shifted = float(estimates["strike_half_cell_shifted"]["gamma"]["value"])
    delta_diff = abs(delta_on_node - delta_shifted)
    gamma_diff = abs(gamma_on_node - gamma_shifted)
    return {
        "grid_step": step,
        "requested_spot": requested_spot,
        "estimates": estimates,
        "delta_diff_abs": float(delta_diff),
        "gamma_diff_abs": float(gamma_diff),
    }


def _gamma_roughness(values: NDArray[np.float64], s_grid: NDArray[np.float64]) -> float:
    greeks = FiniteDifferenceGreeks()
    gamma = greeks.gamma(values, s_grid)
    strike_index = int(np.argmin(np.abs(s_grid - 1.0)))
    window = gamma[1, strike_index - 8 : strike_index + 9]
    return float(np.sum(np.abs(np.diff(window, n=2))))


def _rannacher_stability_check() -> dict[str, Any]:
    process = GeometricBrownianMotion(mu=0.05, sigma=0.2)
    option = EuropeanCall(strike=1.0, maturity=0.1, model=process)
    s_grid = np.linspace(0.0, 3.0, 201)
    time_grid = np.linspace(0.0, option.maturity, 4)

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
    pure_roughness = _gamma_roughness(pure_cn, s_grid)
    rannacher_roughness = _gamma_roughness(rannacher, s_grid)
    return {
        "pure_crank_nicolson_gamma_roughness": pure_roughness,
        "rannacher_gamma_roughness": rannacher_roughness,
        "roughness_ratio": float(rannacher_roughness / pure_roughness),
        "schedule_labels": [entry.label for entry in rannacher_pricer.last_step_schedule[:4]],
        "finite": bool(np.all(np.isfinite(rannacher))),
    }


def _expiry_policy_check() -> dict[str, Any]:
    strike = 100.0
    grid = np.linspace(80.0, 120.0, 41)
    payoff = np.maximum(grid - strike, 0.0)
    try:
        FiniteDifferenceGreeks().sample_delta(
            payoff,
            grid,
            strike,
            time_to_expiry=0.0,
            nonsmooth_coordinates=(strike,),
        )
    except ValidationError as exc:
        return {"expiry_kink_rejected": True, "message": str(exc)}
    return {"expiry_kink_rejected": False, "message": "expiry kink unexpectedly accepted"}


def run_greek_derivative_validation(
    *,
    mode: ValidationMode = "pr",
    thresholds: GreekDerivativeValidationThresholds | None = None,
) -> GreekDerivativeValidationReport:
    """Run the deterministic derivative validation gate."""

    start = time.perf_counter()
    active_thresholds = thresholds or GreekDerivativeValidationThresholds()
    matrix = tuple(_evaluate_validation_case(case, thresholds=active_thresholds) for case in _validation_cases(mode))
    strike_alignment = _strike_alignment_check()
    rannacher = _rannacher_stability_check()
    expiry_policy = _expiry_policy_check()
    runtime_seconds = time.perf_counter() - start

    finest_delta_errors = [float(row["finest"]["delta"]["reference_abs_error"]) for row in matrix]
    finest_gamma_errors = [float(row["finest"]["gamma"]["reference_abs_error"]) for row in matrix]
    ratios = [
        float(row["finest_to_middle_error_ratio"][greek])
        for row in matrix
        for greek in ("delta", "gamma")
    ]
    max_delta_abs_error = max(finest_delta_errors)
    max_gamma_abs_error = max(finest_gamma_errors)
    max_ratio = max(ratios)
    strike_alignment_delta_diff_abs = float(strike_alignment["delta_diff_abs"])
    strike_alignment_gamma_diff_abs = float(strike_alignment["gamma_diff_abs"])
    rannacher_gamma_roughness_ratio = float(rannacher["roughness_ratio"])
    metrics: dict[str, MetricValue] = {
        "benchmark_cases": len(matrix),
        "max_delta_abs_error": max_delta_abs_error,
        "max_gamma_abs_error": max_gamma_abs_error,
        "max_finest_to_middle_error_ratio": max_ratio,
        "strike_alignment_delta_diff_abs": strike_alignment_delta_diff_abs,
        "strike_alignment_gamma_diff_abs": strike_alignment_gamma_diff_abs,
        "rannacher_gamma_roughness_ratio": rannacher_gamma_roughness_ratio,
        "runtime_seconds": runtime_seconds,
    }
    invariants = {
        "nonuniform_delta_converged": max_delta_abs_error <= active_thresholds.max_delta_abs_error,
        "nonuniform_gamma_converged": max_gamma_abs_error <= active_thresholds.max_gamma_abs_error,
        "refinement_improves_all_cases": max_ratio <= active_thresholds.max_finest_to_middle_error_ratio,
        "strike_alignment_bounded": (
            strike_alignment_delta_diff_abs <= active_thresholds.strike_alignment_delta_diff_abs
            and strike_alignment_gamma_diff_abs <= active_thresholds.strike_alignment_gamma_diff_abs
        ),
        "rannacher_smooths_kinked_gamma": (
            rannacher_gamma_roughness_ratio <= active_thresholds.rannacher_gamma_roughness_ratio
            and bool(rannacher["finite"])
        ),
        "expiry_kink_rejected": bool(expiry_policy["expiry_kink_rejected"]),
        "runtime_recorded": runtime_seconds >= 0.0,
        "runtime_within_budget": runtime_seconds <= active_thresholds.max_runtime_seconds,
    }
    return GreekDerivativeValidationReport(
        benchmark_id=GREEK_DERIVATIVE_VALIDATION_BENCHMARK_ID,
        mode=mode,
        passed=all(invariants.values()) and all(bool(row["passed"]) for row in matrix),
        thresholds=active_thresholds,
        metrics=metrics,
        invariants=invariants,
        matrix=matrix,
        strike_alignment=strike_alignment,
        rannacher=rannacher,
        expiry_policy=expiry_policy,
        runtime_seconds=runtime_seconds,
    )


def write_greek_derivative_validation_artifact(
    path: str | Path,
    report: GreekDerivativeValidationReport | None = None,
    *,
    mode: ValidationMode = "pr",
) -> GreekDerivativeValidationReport:
    """Write a Greek derivative validation artifact and return the report."""

    active_report = report or run_greek_derivative_validation(mode=mode)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(active_report.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return active_report


__all__ = [
    "GREEK_DERIVATIVE_VALIDATION_BENCHMARK_ID",
    "GreekDerivativeValidationReport",
    "GreekDerivativeValidationThresholds",
    "run_greek_derivative_validation",
    "write_greek_derivative_validation_artifact",
]
