"""Public-synthetic Pinares fixed-price proxy finite-difference evidence.

Pinares owns the family real-estate deal semantics. This module only validates a
small, public-synthetic fixed-price purchase-option proxy that can be represented
as a one-dimensional Black--Scholes-style European call under the project-wide
``Q*`` proxy measure. Full family-contract, ROFR, legal coordination, mortality
schedule, liquidity/default, tax and market-rent routes remain fail-closed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from finite_difference_options.contracts import DEFAULT_FD_CAPABILITY_MANIFEST, SolverEvidence

PINARES_FIXED_PRICE_PROXY_BENCHMARK_ID = "PINARES-FD-FIXED-PRICE-PROXY-V0"
PINARES_QPS_CONTRACT_BENCHMARK_ID = "PINARES-QPS-FIXED-PRICE-PROXY-V0"
PINARES_FAIL_CLOSED_BENCHMARK_ID = "PINARES-FD-FAIL-CLOSED-V0"
PINARES_FIXED_PRICE_PROXY_BENCHMARK_IDS = (
    PINARES_FIXED_PRICE_PROXY_BENCHMARK_ID,
    PINARES_QPS_CONTRACT_BENCHMARK_ID,
)
PINARES_FIXED_PRICE_PROXY_PROBLEM_ID = "pinares.fixed_price_option_proxy.v1"
PINARES_FIXED_PRICE_PROXY_PROBLEM_HASH = "publicsyntheticpinares001"
PINARES_FIXED_PRICE_PROXY_FIXTURE_ID = "public-synthetic.pinares-fixed-price-proxy.v1"
PINARES_FIXED_PRICE_PROXY_ROUTE_ID = "fd.pinares_fixed_price_proxy.crank_nicolson"
PINARES_FIXED_PRICE_PROXY_SCHEMA_VERSION = "finite-difference-pinares-fixed-price-proxy/v0"
PINARES_PROXY_GRID_LEVELS: tuple[tuple[int, int], ...] = ((80, 80), (120, 160), (180, 240))


@dataclass(frozen=True)
class PinaresFixedPriceProxyCase:
    """Public-synthetic Pinares fixed-price proxy benchmark inputs.

    The ``survival_probability`` scales a terminal fixed-price call payoff. It is
    not a full mortality table, full deal valuation, ROFR valuation, legal/tax
    conclusion, or production Pinares scenario.
    """

    fixture_id: str = PINARES_FIXED_PRICE_PROXY_FIXTURE_ID
    problem_id: str = PINARES_FIXED_PRICE_PROXY_PROBLEM_ID
    problem_hash: str = PINARES_FIXED_PRICE_PROXY_PROBLEM_HASH
    route_id: str = PINARES_FIXED_PRICE_PROXY_ROUTE_ID
    backend_id: str = DEFAULT_FD_CAPABILITY_MANIFEST.backend_id
    code_version: str = "local-checkout"
    spot_uf: float = 6000.0
    strike_uf: float = 6200.0
    risk_free_rate: float = 0.015
    volatility: float = 0.12
    maturity_years: float = 1.0
    survival_probability: float = 0.97
    s_max_uf: float = 12000.0
    valuation_date: str = "2026-06-30"
    maturity_date: str = "2027-06-30"
    price_abs_tolerance_uf: float = 1.0
    delta_abs_tolerance: float = 1.0e-3
    gamma_abs_tolerance: float = 5.0e-6
    seed: int | None = None

    def normalized_units(self) -> dict[str, str]:
        """Return explicit Pinares proxy units."""

        return {"S": "UF", "underlying": "UF", "value": "UF", "rate": "1/year", "time": "year"}

    def as_black_scholes_case(self) -> Any:
        """Return the unscaled Black--Scholes case used by the linear FD solve."""

        from finite_difference_options.validation.black_scholes_parity import BlackScholesParityCase

        return BlackScholesParityCase(
            fixture_id=self.fixture_id,
            route_id=self.route_id,
            backend_id=self.backend_id,
            code_version=self.code_version,
            spot=self.spot_uf,
            strike=self.strike_uf,
            rate=self.risk_free_rate,
            sigma=self.volatility,
            maturity=self.maturity_years,
            s_max=self.s_max_uf,
            tolerance=self.price_abs_tolerance_uf / self.survival_probability,
            valuation_date=self.valuation_date,
            maturity_date=self.maturity_date,
            measure="Q*",
            numeraire="UF_money_market_account_proxy",
            units={"underlying": "UF", "time": "year"},
            seed=self.seed,
        )


@dataclass(frozen=True)
class PinaresFixedPriceProxyReport:
    """Scaled Pinares fixed-price proxy result with evidence payloads."""

    case: PinaresFixedPriceProxyCase
    evidence: SolverEvidence
    base_report: Any
    oracle_price_uf: float
    price_uf: float
    delta: float
    gamma: float
    reference_delta: float
    reference_gamma: float
    errors: dict[str, float]
    no_arbitrage: dict[str, Any]

    @property
    def final_abs_error_uf(self) -> float:
        """Absolute price error on the finest configured grid, in UF."""

        return self.errors["price_abs"]

    def _unscaled_analytical_call(self) -> float:
        """Return the unscaled Black--Scholes oracle lazily to avoid import cycles."""

        from finite_difference_options.validation.black_scholes_parity import black_scholes_call_oracle

        return black_scholes_call_oracle(
            self.case.spot_uf,
            self.case.strike_uf,
            self.case.risk_free_rate,
            self.case.volatility,
            self.case.maturity_years,
        )

    @property
    def converged(self) -> bool:
        """Whether all declared Pinares proxy tolerances pass."""

        return (
            self.errors["price_abs"] <= self.case.price_abs_tolerance_uf
            and self.errors["delta_abs"] <= self.case.delta_abs_tolerance
            and self.errors["gamma_abs"] <= self.case.gamma_abs_tolerance
            and all(bool(value) for key, value in self.no_arbitrage.items() if key.endswith("_ok"))
        )

    def convergence_table(self) -> tuple[dict[str, float | int], ...]:
        """Return scaled convergence rows for JSON serialization."""

        scale = self.case.survival_probability
        return tuple(
            {
                "s_steps": row.s_steps,
                "t_steps": row.t_steps,
                "price_uf": scale * row.price,
                "oracle_price_uf": scale * row.oracle_price,
                "abs_error_uf": scale * row.abs_error,
            }
            for row in self.base_report.observations
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly export for downstream contract tests."""

        return {
            "schema_version": PINARES_FIXED_PRICE_PROXY_SCHEMA_VERSION,
            "fixture_id": self.case.fixture_id,
            "problem_spec": public_pinares_fixed_price_problem_spec(case=self.case),
            "result_export": {
                "route_id": self.case.route_id,
                "backend_id": self.case.backend_id,
                "solution": {
                    "spot_uf": self.case.spot_uf,
                    "price_uf": self.price_uf,
                    "delta": self.delta,
                    "gamma": self.gamma,
                },
                "reference": {
                    "oracle_price_uf": self.oracle_price_uf,
                    "analytical_unscaled_call_uf": self._unscaled_analytical_call(),
                    "survival_probability": self.case.survival_probability,
                    "delta": self.reference_delta,
                    "gamma": self.reference_gamma,
                },
                "errors": self.errors,
                "no_arbitrage": self.no_arbitrage,
                "convergence": self.convergence_table(),
                "time_axis": self.base_report.as_dict()["result_export"]["time_axis"],
                "grid": self.base_report.as_dict()["result_export"]["grid"],
                "boundary": self.base_report.as_dict()["result_export"]["boundary"],
            },
            "unsupported_scope": {
                "rofr": "unsupported; right of first refusal is not a vanilla call",
                "full_family_contract": "unsupported; requires Pinares contract valuation semantics and PDP inputs",
                "legal_tax_conclusion": "unsupported; FD backend does not provide legal/tax advice",
            },
            "evidence": self.evidence.as_dict(),
        }


def public_pinares_fixed_price_problem_spec(*, case: PinaresFixedPriceProxyCase | None = None) -> dict[str, Any]:
    """Return the canonical public-synthetic Pinares fixed-price QuantProblemSpec."""

    case = case or PinaresFixedPriceProxyCase()
    coefficient_terms = [
        {
            "name": "drift",
            "operator": "S * dV/dS",
            "coefficient": case.risk_free_rate,
            "expression": "r S ∂V/∂S",
        },
        {
            "name": "diffusion",
            "operator": "S^2 * d²V/dS²",
            "coefficient": 0.5 * case.volatility**2,
            "variance": case.volatility**2,
            "expression": "0.5 σ² S² ∂²V/∂S²",
        },
        {
            "name": "reaction",
            "operator": "V",
            "coefficient": -case.risk_free_rate,
            "expression": "-r V",
        },
    ]
    resource_controls = {
        "grid_levels": len(PINARES_PROXY_GRID_LEVELS),
        "max_s_steps": max(level[0] for level in PINARES_PROXY_GRID_LEVELS),
        "max_t_steps": max(level[1] for level in PINARES_PROXY_GRID_LEVELS),
        "deterministic": "true",
    }
    error_budgets = {
        "price_abs_uf": case.price_abs_tolerance_uf,
        "delta_abs": case.delta_abs_tolerance,
        "gamma_abs": case.gamma_abs_tolerance,
    }
    return {
        "schema_version": "quant-problem-spec/v0",
        "privacy_class": "public_synthetic",
        "artifact_manifest": {
            "schema_version": "artifact-manifest/v0",
            "manifest_id": "pinares-fd-fixed-price-proxy-public-synthetic-v1",
            "fixture_id": case.fixture_id,
            "benchmark_ids": list(PINARES_FIXED_PRICE_PROXY_BENCHMARK_IDS),
            "issue_refs": ["googa27/finite_difference_options#119"],
        },
        "problem_id": case.problem_id,
        "problem_hash": case.problem_hash,
        "valuation_context": {
            "measure": "Q*",
            "numeraire": "UF_money_market_account_proxy",
            "valuation_date": case.valuation_date,
            "maturity_date": case.maturity_date,
            "time_domain": f"[0, {case.maturity_years:g}]",
            "units": case.normalized_units(),
            "privacy_tier": "public_synthetic",
        },
        "mathematical_problem": {
            "dimension": 1,
            "state_variables": [
                {
                    "name": "S",
                    "role": "underlying",
                    "unit": "UF",
                    "description": "public synthetic property-value proxy",
                }
            ],
            "measure_id": "Q*",
            "numeraire_id": "UF_money_market_account_proxy",
            "pde_terms": [term["name"] for term in coefficient_terms],
            "pde_operator_terms": coefficient_terms,
            "pde_coefficients": {
                "risk_free_rate": case.risk_free_rate,
                "volatility": case.volatility,
                "terms": coefficient_terms,
            },
            "boundary_conditions": {
                "S=0": "dirichlet",
                "S=S_max": "linear_growth",
            },
            "exercise_style": "european",
            "terminal_payoff": {
                "payoff_id": "mortality_scaled_fixed_price_call_payoff",
                "expression": "p_survival * max(S - K, 0)",
                "timing": "terminal",
                "units": "UF",
                "parameters": {
                    "K_uf": case.strike_uf,
                    "p_survival": case.survival_probability,
                },
            },
            "unsupported_full_deal_terms": [
                "rofr",
                "legal_coordination",
                "tax_transfer_analysis",
                "liquidity_default",
                "market_rent_alternative",
            ],
            "requested_outputs": ["value", "delta", "gamma"],
        },
        "solver_plan": {
            "backend_id": case.backend_id,
            "grid_type": "uniform",
            "method_id": "pinares-fd-fixed-price-proxy-crank-nicolson",
            "method_kind": "finite_difference",
            "stability_controls": ["theta"],
            "requested_outputs": ["value", "delta", "gamma"],
            "time_controls": {"theta": 0.5},
            "resource_controls": resource_controls,
            "error_budgets": error_budgets,
        },
        "financial_graph": {
            "instrument": {
                "kind": "mortality_scaled_fixed_price_purchase_option_proxy",
                "unit": "UF",
                "strike_uf": case.strike_uf,
                "survival_probability": case.survival_probability,
            },
            "valuation_graph": {"solver_hints": {"benchmark_ids": list(PINARES_FIXED_PRICE_PROXY_BENCHMARK_IDS)}},
        },
        "result_bundle": {
            "benchmark_ids": list(PINARES_FIXED_PRICE_PROXY_BENCHMARK_IDS),
            "references": ["Pinares THEORY.md: fixed-price option is separate from ROFR/full contract"],
        },
        "benchmark_ids": list(PINARES_FIXED_PRICE_PROXY_BENCHMARK_IDS),
    }


def public_pinares_full_deal_unsupported_problem_spec() -> dict[str, Any]:
    """Return a public-synthetic full-deal request that FD must reject."""

    payload = public_pinares_fixed_price_problem_spec()
    payload["problem_id"] = "pinares.full_family_contract.unsupported.v1"
    payload["problem_hash"] = "publicsyntheticpinaresfullunsupported001"
    payload["benchmark_ids"] = [PINARES_FAIL_CLOSED_BENCHMARK_ID]
    payload["artifact_manifest"] = {
        **payload["artifact_manifest"],
        "benchmark_ids": [PINARES_FAIL_CLOSED_BENCHMARK_ID],
    }
    payload["result_bundle"] = {"benchmark_ids": [PINARES_FAIL_CLOSED_BENCHMARK_ID]}
    payload["solver_plan"] = {
        **payload["solver_plan"],
        "requested_outputs": ["value", "delta", "gamma", "legal_tax_conclusion"],
    }
    payload["financial_graph"] = {
        "instrument": {
            "kind": "family_real_estate_use_right_full_contract",
            "unit": "UF",
            "contains_rofr": True,
            "legal_coordination": "proposal_assumption",
        }
    }
    payload["mathematical_problem"] = {
        "dimension": 4,
        "state_variables": ["property_value", "father_alive", "liquidity_state", "legal_coordination_state"],
        "measure_id": "P/Q*_mixed_unsupported",
        "numeraire_id": "UF_money_market_account_proxy",
        "pde_terms": [
            "drift",
            "diffusion",
            "reaction",
            "hazard_killing",
            "liquidity_jump",
            "hjb_control",
        ],
        "boundary_conditions": {
            "sale_or_stress": "absorbing",
            "legal_coordination": "legal_coordination_constraint",
        },
        "exercise_style": "rofr_full_family_contract",
        "requested_outputs": ["value", "delta", "gamma", "legal_tax_conclusion"],
    }
    return payload


def run_public_pinares_fixed_price_proxy_fixture(
    *,
    case: PinaresFixedPriceProxyCase | None = None,
    grid_levels: tuple[tuple[int, int], ...] = PINARES_PROXY_GRID_LEVELS,
) -> PinaresFixedPriceProxyReport:
    """Run the Pinares public-synthetic fixed-price proxy parity fixture."""

    case = case or PinaresFixedPriceProxyCase()
    from finite_difference_options.validation.black_scholes_parity import run_public_black_scholes_parity_fixture

    base_report = run_public_black_scholes_parity_fixture(case=case.as_black_scholes_case(), grid_levels=grid_levels)
    scale = case.survival_probability
    oracle_price = scale * base_report.oracle_price
    price = scale * base_report.price
    delta = scale * base_report.delta
    gamma = scale * base_report.gamma
    reference_delta = scale * base_report.reference_delta
    reference_gamma = scale * base_report.reference_gamma
    errors = {
        "price_abs": scale * base_report.final_abs_error,
        "price_rel": scale * base_report.final_abs_error / max(1e-12, abs(oracle_price)),
        "delta_abs": abs(delta - reference_delta),
        "delta_rel": abs(delta - reference_delta) / max(1e-12, abs(reference_delta)),
        "gamma_abs": abs(gamma - reference_gamma),
        "gamma_rel": abs(gamma - reference_gamma) / max(1e-12, abs(reference_gamma)),
        "max_abs_price_error": scale * base_report.max_abs_error,
    }
    intrinsic = scale * max(case.spot_uf - case.strike_uf, 0.0)
    upper_bound = scale * case.spot_uf
    no_arbitrage = {
        "value_minus_intrinsic_uf": price - intrinsic,
        "upper_gap_uf": upper_bound - price,
        "value_bound_ok": price >= intrinsic - 1e-12,
        "upper_bound_ok": price <= upper_bound + 1e-12,
        "delta_lower_bound_ok": delta >= -1e-12,
        "delta_upper_bound_ok": delta <= scale + 1e-12,
        "gamma_non_negative_ok": gamma >= -1e-12,
        "survival_scale_ok": 0.0 <= scale <= 1.0,
    }
    evidence = SolverEvidence(
        route_id=case.route_id,
        backend_id=case.backend_id,
        code_version=case.code_version,
        config_hash=_config_hash(case, grid_levels),
        fixture_id=case.fixture_id,
        seed=case.seed,
        valuation_date=case.valuation_date,
        maturity_date=case.maturity_date,
        measure="Q*",
        numeraire="UF_money_market_account_proxy",
        units=case.normalized_units(),
        boundary_assumptions=(
            "left boundary: Dirichlet fixed-price proxy value V(0,tau)=0",
            "right boundary: linear-growth call proxy scaled by survival_probability",
            "uniform UF spot grid on [0, s_max_uf]",
            "theta time stepping with Crank-Nicolson theta=0.5",
            "not a ROFR or full family-contract valuation",
        ),
        resource_controls={
            "max_s_steps": max(level[0] for level in grid_levels),
            "max_t_steps": max(level[1] for level in grid_levels),
            "grid_levels": len(grid_levels),
            "deterministic": "true",
        },
    )
    return PinaresFixedPriceProxyReport(
        case=case,
        evidence=evidence,
        base_report=base_report,
        oracle_price_uf=oracle_price,
        price_uf=price,
        delta=delta,
        gamma=gamma,
        reference_delta=reference_delta,
        reference_gamma=reference_gamma,
        errors=errors,
        no_arbitrage=no_arbitrage,
    )


def export_public_pinares_fixed_price_proxy_fixture_json(
    *,
    path: Path | str,
    case: PinaresFixedPriceProxyCase | None = None,
    grid_levels: tuple[tuple[int, int], ...] = PINARES_PROXY_GRID_LEVELS,
) -> Path:
    """Write the public-synthetic Pinares fixed-price fixture JSON."""

    report = run_public_pinares_fixed_price_proxy_fixture(case=case, grid_levels=grid_levels)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def _config_hash(case: PinaresFixedPriceProxyCase, grid_levels: tuple[tuple[int, int], ...]) -> str:
    payload = {
        "fixture_id": case.fixture_id,
        "problem_id": case.problem_id,
        "problem_hash": case.problem_hash,
        "spot_uf": case.spot_uf,
        "strike_uf": case.strike_uf,
        "risk_free_rate": case.risk_free_rate,
        "volatility": case.volatility,
        "maturity_years": case.maturity_years,
        "survival_probability": case.survival_probability,
        "s_max_uf": case.s_max_uf,
        "grid_levels": grid_levels,
        "valuation_date": case.valuation_date,
        "maturity_date": case.maturity_date,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


__all__ = [
    "PINARES_FAIL_CLOSED_BENCHMARK_ID",
    "PINARES_FIXED_PRICE_PROXY_BENCHMARK_ID",
    "PINARES_FIXED_PRICE_PROXY_BENCHMARK_IDS",
    "PINARES_FIXED_PRICE_PROXY_FIXTURE_ID",
    "PINARES_FIXED_PRICE_PROXY_PROBLEM_HASH",
    "PINARES_FIXED_PRICE_PROXY_PROBLEM_ID",
    "PINARES_FIXED_PRICE_PROXY_ROUTE_ID",
    "PINARES_QPS_CONTRACT_BENCHMARK_ID",
    "PinaresFixedPriceProxyCase",
    "PinaresFixedPriceProxyReport",
    "export_public_pinares_fixed_price_proxy_fixture_json",
    "public_pinares_fixed_price_problem_spec",
    "public_pinares_full_deal_unsupported_problem_spec",
    "run_public_pinares_fixed_price_proxy_fixture",
]
