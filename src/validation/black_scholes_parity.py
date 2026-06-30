"""Public-synthetic Black--Scholes parity fixtures for FD routing.

This module publishes a deterministic, serializable fixture for a 1D European call
that external benchmark consumers (e.g., arXiv-Lab adapter tests) can consume
without importing internal project modules.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import exp, log, sqrt
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.stats import norm

from src.contracts import DEFAULT_FD_CAPABILITY_MANIFEST, SolverEvidence
from src.greeks import FiniteDifferenceGreeks
from src.instruments.base import EuropeanCall
from src.pricing.engines import BlackScholesPDE
from src.processes.affine import GeometricBrownianMotion


BoundaryType = Literal["dirichlet", "neumann", "robin", "second_derivative", "asymptotic"]
TimeDirection = Literal["increasing", "decreasing"]


@dataclass(frozen=True)
class BoundarySpec:
    """Typed boundary specification for fixture serialization."""

    coordinate: str
    location: str
    boundary_type: BoundaryType
    value: float | str
    expression: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TimeAxisSpec:
    """Explicit time-axis orientation metadata."""

    axis: str
    direction: TimeDirection
    valuation_index: int
    maturity_index: int
    valuation_time: float
    maturity_time: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BlackScholesParityCase:
    """Public-synthetic Black--Scholes call-option fixture definition."""

    fixture_id: str = "public-synthetic.black-scholes-call.v0"
    route_id: str = "fd.black_scholes_1d.crank_nicolson"
    backend_id: str = DEFAULT_FD_CAPABILITY_MANIFEST.backend_id
    code_version: str = "local-checkout"
    spot: float = 1.0
    strike: float = 1.0
    rate: float = 0.05
    sigma: float = 0.2
    maturity: float = 1.0
    s_max: float = 3.0
    tolerance: float = 5.0e-4
    valuation_date: str = "2026-01-02"
    maturity_date: str = "2027-01-02"
    measure: str = "risk_neutral"
    numeraire: str = "money_market_account"
    units: dict[str, str] | None = None
    seed: int | None = None

    def normalized_units(self) -> dict[str, str]:
        """Return explicit synthetic units for evidence serialization."""

        return self.units or {"underlying": "synthetic_currency", "time": "ACT/365F"}


@dataclass(frozen=True)
class ConvergenceObservation:
    """One row in the parity fixture convergence table."""

    s_steps: int
    t_steps: int
    price: float
    oracle_price: float
    abs_error: float


@dataclass(frozen=True)
class BlackScholesParityReport:
    """Black--Scholes fixture result with solver evidence and benchmark export."""

    case: BlackScholesParityCase
    evidence: SolverEvidence
    oracle_price: float
    observations: tuple[ConvergenceObservation, ...]
    price: float
    delta: float
    gamma: float
    reference_delta: float
    reference_gamma: float
    boundary_specs: tuple[BoundarySpec, ...]
    time_axis: TimeAxisSpec
    grid_metadata: dict[str, Any]
    no_arbitrage: dict[str, Any]
    errors: dict[str, Any]

    @property
    def max_abs_error(self) -> float:
        """Maximum absolute pricing error across convergence rows."""

        return max(observation.abs_error for observation in self.observations)

    @property
    def final_abs_error(self) -> float:
        """Absolute pricing error on the finest configured grid."""

        return self.observations[-1].abs_error

    @property
    def converged(self) -> bool:
        """Whether the finest grid satisfies the fixture tolerance."""

        return self.final_abs_error <= self.case.tolerance

    def convergence_table(self) -> tuple[dict[str, float | int], ...]:
        """Return a JSON-friendly convergence table."""

        return tuple(
            {
                "s_steps": row.s_steps,
                "t_steps": row.t_steps,
                "price": row.price,
                "oracle_price": row.oracle_price,
                "abs_error": row.abs_error,
            }
            for row in self.observations
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the arXiv-Lab-friendly serialized export payload."""

        final_s_steps = self.observations[-1].s_steps
        final_t_steps = self.observations[-1].t_steps
        grid = self.grid_metadata

        return {
            "schema_version": "arxiv-lab/fd-oracle-fixture/v0",
            "fixture_id": self.case.fixture_id,
            "problem_spec": _build_public_problem_spec(self.case),
            "result_export": {
                "time_axis": self.time_axis.as_dict(),
                "domain": {
                    "s_min": grid["s_min"],
                    "s_max": grid["s_max"],
                    "t_min": grid["t_min"],
                    "t_max": grid["t_max"],
                },
                "grid": {
                    "s_steps": final_s_steps,
                    "t_steps": final_t_steps,
                    "s_spacing": {
                        "min": grid["s_spacing_min"],
                        "max": grid["s_spacing_max"],
                        "mean": grid["s_spacing_mean"],
                    },
                    "t_spacing": {
                        "min": grid["t_spacing_min"],
                        "max": grid["t_spacing_max"],
                        "mean": grid["t_spacing_mean"],
                    },
                    "s_grid_family": "uniform",
                    "t_grid_family": "uniform",
                    "monotone": True,
                    "valuation_time_index": self.time_axis.valuation_index,
                    "maturity_time_index": self.time_axis.maturity_index,
                },
                "boundary": {
                    "typed": tuple(spec.as_dict() for spec in self.boundary_specs),
                    "notes": [
                        "left boundary is hard value 0",
                        "right boundary is first derivative asymptote",
                    ],
                },
                "solution": {
                    "spot": self.case.spot,
                    "price": self.price,
                    "delta": self.delta,
                    "gamma": self.gamma,
                },
                "reference": {
                    "price": self.oracle_price,
                    "delta": self.reference_delta,
                    "gamma": self.reference_gamma,
                },
                "errors": self.errors,
                "no_arbitrage": self.no_arbitrage,
            },
            "convergence": self.convergence_table(),
            "evidence": self.evidence.as_dict(),
        }



def black_scholes_call_oracle(
    spot: float, strike: float, rate: float, sigma: float, maturity: float
) -> float:
    """Analytical Black--Scholes call price used as the public oracle."""

    d1 = (
        log(spot / strike) + (rate + 0.5 * sigma**2) * maturity
    ) / (sigma * sqrt(maturity))
    d2 = d1 - sigma * sqrt(maturity)
    return float(spot * norm.cdf(d1) - strike * exp(-rate * maturity) * norm.cdf(d2))


def black_scholes_call_greeks(
    spot: float, strike: float, rate: float, sigma: float, maturity: float
) -> dict[str, float]:
    """Analytical Black--Scholes call delta and gamma used for benchmark deltas."""

    d1 = (
        log(spot / strike) + (rate + 0.5 * sigma**2) * maturity
    ) / (sigma * sqrt(maturity))
    return {
        "delta": float(norm.cdf(d1)),
        "gamma": float(
            norm.pdf(d1) / (spot * sigma * sqrt(maturity))
        ),
    }


def _build_public_problem_spec(case: BlackScholesParityCase) -> dict[str, Any]:
    """Build a tiny public QuantProblemSpec payload for the deterministic fixture."""

    return {
        "schema_version": "quant-problem-spec/v0",
        "artifact_manifest": {
            "schema_version": "artifact-manifest/v0",
            "manifest_id": "public-synthetic-bs-call-oracle-fixture-v0",
            "benchmark_ids": ["BS-FD-ORACLE-V0", "QPS-BS-CALL-PUBLIC-V0"],
        },
        "problem_id": "public-synthetic.black-scholes-call.v0",
        "problem_hash": "publicsyntheticbscall001",
        "valuation_context": {
            "measure": case.measure,
            "numeraire": case.numeraire,
            "valuation_date": case.valuation_date,
            "maturity_date": case.maturity_date,
            "time_domain": "[0, 1]",
            "units": case.normalized_units(),
            "privacy_tier": "public_synthetic",
        },
        "mathematical_problem": {
            "dimension": 1,
            "state_variables": [
                {
                    "name": "S",
                    "role": "underlying",
                    "unit": case.normalized_units().get("underlying", "synthetic_currency"),
                    "coordinate": "spot",
                }
            ],
            "pde_terms": ["drift", "diffusion", "reaction"],
            "boundary_conditions": {
                "S=0": "dirichlet",
                "S=S_max": "neumann",
            },
            "exercise_style": "european",
            "requested_outputs": ["value", "delta", "gamma"],
        },
        "solver_plan": {
            "backend_id": case.backend_id,
            "grid_type": "uniform",
            "method_id": "arxiv-lab-bs-oracle-fixture",
            "stability_controls": ["theta"],
            "requested_outputs": ["value", "delta", "gamma"],
            "time_controls": {"theta": 0.6},
            "resource_controls": {
                "grid_levels": 3,
                "max_s_steps": 120,
                "max_t_steps": 200,
            },
        },
        "financial_graph": {
            "instrument": {
                "kind": "european_call",
                "currency": "synthetic_currency",
                "strike": case.strike,
                "maturity": case.maturity,
            }
        },
        "result_bundle": {
            "benchmark_ids": ["BS-FD-ORACLE-V0"],
            "references": ["BS-CALL-PARITY-V0"],
        },
        "benchmark_ids": ["BS-FD-ORACLE-V0", "QPS-BS-CALL-PUBLIC-V0"],
    }


def run_public_black_scholes_parity_fixture(
    *,
    case: BlackScholesParityCase | None = None,
    grid_levels: tuple[tuple[int, int], ...] = ((40, 40), (80, 120), (120, 200)),
) -> BlackScholesParityReport:
    """Run the public-synthetic Black--Scholes fixture and emit evidence.

    The fixture uses only synthetic parameters and deterministic finite-difference
    grids. It records boundary assumptions and resource controls so a router can
    compare evidence without importing private market data or hidden defaults.
    """

    case = case or BlackScholesParityCase()
    oracle_price = black_scholes_call_oracle(
        case.spot, case.strike, case.rate, case.sigma, case.maturity
    )
    reference_greeks = black_scholes_call_greeks(
        case.spot, case.strike, case.rate, case.sigma, case.maturity
    )

    model = GeometricBrownianMotion(mu=case.rate, sigma=case.sigma)
    instrument = EuropeanCall(strike=case.strike, maturity=case.maturity, model=model)
    pricer = BlackScholesPDE(instrument=instrument)
    greek_calculator = FiniteDifferenceGreeks()

    observations: list[ConvergenceObservation] = []
    final_values: np.ndarray | None = None
    final_s_grid: np.ndarray | None = None
    final_t_grid: np.ndarray | None = None
    valuation_index = -1
    maturity_index = 0

    for s_steps, t_steps in grid_levels:
        s_grid = np.linspace(0.0, case.s_max, s_steps)
        t_grid = np.linspace(0.0, case.maturity, t_steps)
        values = pricer.price(option=instrument, s=s_grid, t=t_grid)
        price = float(np.interp(case.spot, s_grid, values[valuation_index]))

        observations.append(
            ConvergenceObservation(
                s_steps=s_steps,
                t_steps=t_steps,
                price=price,
                oracle_price=oracle_price,
                abs_error=abs(price - oracle_price),
            )
        )

        if (s_steps, t_steps) == grid_levels[-1]:
            final_values = values.copy()
            final_s_grid = s_grid
            final_t_grid = t_grid

    assert final_values is not None and final_s_grid is not None and final_t_grid is not None

    delta_slice = greek_calculator.delta(final_values, final_s_grid)[valuation_index]
    gamma_slice = greek_calculator.gamma(final_values, final_s_grid)[valuation_index]
    fd_price = float(np.interp(case.spot, final_s_grid, final_values[valuation_index]))
    fd_delta = float(np.interp(case.spot, final_s_grid, delta_slice))
    fd_gamma = float(np.interp(case.spot, final_s_grid, gamma_slice))

    no_arbitrage = _compute_no_arbitrage_assertions(case, fd_price, fd_delta, fd_gamma)

    s_spacing = np.diff(final_s_grid)
    t_spacing = np.diff(final_t_grid)
    grid_metadata = {
        "s_min": float(final_s_grid.min()),
        "s_max": float(final_s_grid.max()),
        "t_min": float(final_t_grid.min()),
        "t_max": float(final_t_grid.max()),
        "s_spacing_min": float(np.min(s_spacing)),
        "s_spacing_max": float(np.max(s_spacing)),
        "s_spacing_mean": float(np.mean(s_spacing)),
        "t_spacing_min": float(np.min(t_spacing)),
        "t_spacing_max": float(np.max(t_spacing)),
        "t_spacing_mean": float(np.mean(t_spacing)),
    }

    delta_abs_error = abs(fd_delta - reference_greeks["delta"])
    gamma_abs_error = abs(fd_gamma - reference_greeks["gamma"])
    errors = {
        "price_abs": float(abs(fd_price - oracle_price)),
        "price_rel": float(abs(fd_price - oracle_price) / max(1.0, abs(oracle_price))),
        "delta_abs": float(delta_abs_error),
        "delta_rel": float(delta_abs_error / max(1e-12, abs(reference_greeks["delta"]))),
        "gamma_abs": float(gamma_abs_error),
        "gamma_rel": float(
            gamma_abs_error / max(1e-12, abs(reference_greeks["gamma"]))
        ),
        "max_abs_price_error": float(max(row.abs_error for row in observations)),
    }

    boundary_specs = (
        BoundarySpec(
            coordinate="S",
            location="S=0",
            boundary_type="dirichlet",
            value=0.0,
            expression="v(0,t)=0",
        ),
        BoundarySpec(
            coordinate="S",
            location="S=s_max",
            boundary_type="neumann",
            value=1.0,
            expression="dV/dS=1",
        ),
    )
    time_axis = TimeAxisSpec(
        axis="time",
        direction="decreasing",
        valuation_index=valuation_index,
        maturity_index=maturity_index,
        valuation_time=0.0,
        maturity_time=case.maturity,
    )

    evidence = SolverEvidence(
        route_id=case.route_id,
        backend_id=case.backend_id,
        code_version=case.code_version,
        config_hash=_config_hash(case, grid_levels),
        fixture_id=case.fixture_id,
        seed=case.seed,
        valuation_date=case.valuation_date,
        maturity_date=case.maturity_date,
        measure=case.measure,
        numeraire=case.numeraire,
        units=case.normalized_units(),
        boundary_assumptions=(
            "left boundary: zero call value at S=0",
            "right boundary: derivative approaches one for call far field",
            "uniform physical-price grid on [0, s_max]",
            "theta time stepping",
        ),
        resource_controls={
            "max_s_steps": max(level[0] for level in grid_levels),
            "max_t_steps": max(level[1] for level in grid_levels),
            "grid_levels": len(grid_levels),
            "deterministic": "true",
        },
    )

    return BlackScholesParityReport(
        case=case,
        evidence=evidence,
        oracle_price=oracle_price,
        observations=tuple(observations),
        price=fd_price,
        delta=fd_delta,
        gamma=fd_gamma,
        reference_delta=reference_greeks["delta"],
        reference_gamma=reference_greeks["gamma"],
        boundary_specs=boundary_specs,
        time_axis=time_axis,
        grid_metadata=grid_metadata,
        no_arbitrage=no_arbitrage,
        errors=errors,
    )



def export_public_black_scholes_fixture_json(
    *,
    path: Path | str,
    case: BlackScholesParityCase | None = None,
    grid_levels: tuple[tuple[int, int], ...] = ((40, 40), (80, 120), (120, 200)),
) -> Path:
    """Write the arXiv-Lab fixture payload to a JSON path.

    The generated payload is intentionally plain JSON and can be consumed without
    importing project internals.
    """

    report = run_public_black_scholes_parity_fixture(case=case, grid_levels=grid_levels)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.as_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return output_path



def _compute_no_arbitrage_assertions(
    case: BlackScholesParityCase, value: float, delta: float, gamma: float
) -> dict[str, Any]:
    intrinsic = max(case.spot - case.strike, 0.0)
    no_upper_gap = case.spot - value

    return {
        "value_minus_intrinsic": float(value - intrinsic),
        "upper_gap": float(no_upper_gap),
        "value_bound_ok": value >= intrinsic - 1e-12,
        "upper_bound_ok": no_upper_gap >= -1e-12,
        "delta_lower_bound_ok": delta >= -1e-12,
        "delta_upper_bound_ok": delta <= 1.0 + 1e-12,
        "gamma_non_negative_ok": gamma >= -1e-12,
        "monotone_call_parity_hint": "call value must be monotone increasing in spot (single-point check)",
    }



def _config_hash(case: BlackScholesParityCase, grid_levels: tuple[tuple[int, int], ...]) -> str:
    payload: dict[str, Any] = {
        "fixture_id": case.fixture_id,
        "spot": case.spot,
        "strike": case.strike,
        "rate": case.rate,
        "sigma": case.sigma,
        "maturity": case.maturity,
        "s_max": case.s_max,
        "tolerance": case.tolerance,
        "grid_levels": grid_levels,
        "measure": case.measure,
        "numeraire": case.numeraire,
        "units": case.normalized_units(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


__all__ = [
    "BlackScholesParityCase",
    "BlackScholesParityReport",
    "ConvergenceObservation",
    "BoundarySpec",
    "TimeAxisSpec",
    "black_scholes_call_greeks",
    "black_scholes_call_oracle",
    "run_public_black_scholes_parity_fixture",
    "export_public_black_scholes_fixture_json",
]
