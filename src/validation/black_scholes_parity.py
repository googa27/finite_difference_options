"""Public-synthetic Black--Scholes parity fixtures for FD routing."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import exp, log, sqrt
import json
from typing import Any

import numpy as np
from scipy.stats import norm

from src.contracts import DEFAULT_FD_CAPABILITY_MANIFEST, SolverEvidence
from src.instruments.base import EuropeanCall
from src.pricing.engines import BlackScholesPDE
from src.processes.affine import GeometricBrownianMotion


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
    """Black--Scholes fixture result with solver evidence."""

    case: BlackScholesParityCase
    evidence: SolverEvidence
    oracle_price: float
    observations: tuple[ConvergenceObservation, ...]

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


def black_scholes_call_oracle(spot: float, strike: float, rate: float, sigma: float, maturity: float) -> float:
    """Analytical Black--Scholes call price used as the public oracle."""

    d1 = (log(spot / strike) + (rate + 0.5 * sigma**2) * maturity) / (sigma * sqrt(maturity))
    d2 = d1 - sigma * sqrt(maturity)
    return float(spot * norm.cdf(d1) - strike * exp(-rate * maturity) * norm.cdf(d2))


def run_public_black_scholes_parity_fixture(
    *,
    case: BlackScholesParityCase | None = None,
    grid_levels: tuple[tuple[int, int], ...] = ((40, 40), (80, 120), (120, 200)),
) -> BlackScholesParityReport:
    """Run the public-synthetic Black--Scholes fixture and emit evidence.

    The fixture uses only synthetic parameters and deterministic finite-difference
    grids. It records boundary assumptions and resource controls so a router can
    compare evidence without relying on private market data or hidden defaults.
    """

    case = case or BlackScholesParityCase()
    oracle = black_scholes_call_oracle(case.spot, case.strike, case.rate, case.sigma, case.maturity)

    model = GeometricBrownianMotion(mu=case.rate, sigma=case.sigma)
    instrument = EuropeanCall(strike=case.strike, maturity=case.maturity, model=model)
    pricer = BlackScholesPDE(instrument=instrument)

    observations: list[ConvergenceObservation] = []
    for s_steps, t_steps in grid_levels:
        s_grid = np.linspace(0.0, case.s_max, s_steps)
        t_grid = np.linspace(0.0, case.maturity, t_steps)
        values = pricer.price(option=instrument, s=s_grid, t=t_grid)
        price = float(np.interp(case.spot, s_grid, values[-1]))
        observations.append(
            ConvergenceObservation(
                s_steps=s_steps,
                t_steps=t_steps,
                price=price,
                oracle_price=oracle,
                abs_error=abs(price - oracle),
            )
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
            "right boundary: first derivative approaches one for call far field",
            "uniform physical-price grid on [0, s_max]",
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
        oracle_price=oracle,
        observations=tuple(observations),
    )


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
    "black_scholes_call_oracle",
    "run_public_black_scholes_parity_fixture",
]
