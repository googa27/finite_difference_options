"""Executable benchmark registry for finite-difference validation evidence.

The registry is intentionally metadata-first: every public capability claim points to
versioned evidence with explicit oracle, invariant, tolerance, route and resource
policy metadata. A small subset of entries also has deterministic runners so CI can
verify that the registry is not just documentation.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

BenchmarkStatus = Literal["validated", "experimental", "scaffold", "unsupported"]
BenchmarkFamily = Literal[
    "analytical_oracle",
    "manufactured_solution",
    "no_arbitrage",
    "route_parity",
    "capability_gate",
    "smoke",
    "regulatory_fail_closed",
]
OracleKind = Literal["analytical", "manufactured", "fixture", "regression", "capability", "none"]
MetricKind = Literal[
    "price_abs",
    "price_rel",
    "delta_abs",
    "gamma_abs",
    "convergence_order",
    "boolean_invariant",
    "route_parity_abs",
    "residual_norm",
    "lcp_primal_abs",
    "lcp_dual_abs",
    "lcp_complementarity_abs",
]

_VERSIONED_ID_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9-]+-V\d+$")


class BenchmarkRegistryError(ValueError):
    """Raised when benchmark registry metadata is internally inconsistent."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("; ".join(errors))


@dataclass(frozen=True)
class TolerancePolicy:
    """Named tolerance policy for one benchmark metric."""

    metric: MetricKind
    threshold: float | bool
    norm: str
    notes: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OracleSpec:
    """Independent or reference oracle metadata for a benchmark case."""

    kind: OracleKind
    source: str
    independence: str
    notes: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkCase:
    """One benchmark/evidence registry row."""

    benchmark_id: str
    title: str
    family: BenchmarkFamily
    status: BenchmarkStatus
    route_id: str
    model: str
    instrument: str
    state_convention: str
    grid_family: str
    time_schedule: str
    oracle: OracleSpec
    tolerances: tuple[TolerancePolicy, ...]
    invariants: tuple[str, ...] = ()
    capability_rows: tuple[str, ...] = ()
    fixture_paths: tuple[str, ...] = ()
    issue_refs: tuple[str, ...] = ()
    resource_policy: dict[str, int | float | str] = field(default_factory=dict)
    runner: str | None = None
    notes: str = ""

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not _VERSIONED_ID_PATTERN.fullmatch(self.benchmark_id):
            errors.append(f"benchmark_id is not versioned kebab id: {self.benchmark_id}")
        if not self.title.strip():
            errors.append(f"title required for {self.benchmark_id}")
        if not self.route_id.strip():
            errors.append(f"route_id required for {self.benchmark_id}")
        if not self.model.strip():
            errors.append(f"model required for {self.benchmark_id}")
        if not self.instrument.strip():
            errors.append(f"instrument required for {self.benchmark_id}")
        if self.status in {"validated", "experimental"} and self.oracle.kind == "none":
            errors.append(f"{self.benchmark_id} claims {self.status} without an oracle/fixture")
        if self.status == "validated" and not self.tolerances:
            errors.append(f"validated benchmark {self.benchmark_id} requires tolerances")
        if self.family in {"no_arbitrage", "route_parity"} and not self.invariants:
            errors.append(f"{self.family} benchmark {self.benchmark_id} requires invariants")
        if self.status == "validated" and self.family == "route_parity" and self.runner is None:
            errors.append(f"validated route-parity benchmark {self.benchmark_id} requires an executable runner")
        return errors

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["oracle"] = self.oracle.as_dict()
        payload["tolerances"] = [tol.as_dict() for tol in self.tolerances]
        return payload


@dataclass(frozen=True)
class BenchmarkRunResult:
    """Normalized result returned by executable benchmark runners."""

    benchmark_id: str
    passed: bool
    metrics: dict[str, float | bool | int | str]
    evidence: dict[str, Any]
    invariants: dict[str, bool]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_benchmark_registry() -> tuple[BenchmarkCase, ...]:
    """Return the default versioned FD benchmark registry."""

    bs_oracle = OracleSpec(
        kind="analytical",
        source="closed-form Black-Scholes European call",
        independence="analytic formula independent of the FD grid/operator implementation",
        notes="Public-synthetic spot=strike=1 case with deterministic convergence table.",
    )
    heston_oracle = OracleSpec(
        kind="regression",
        source="semi-analytical Heston characteristic-function oracle",
        independence="separate Fourier integration route under src.validation.heston_oracle",
        notes="Used for oracle and Black-Scholes-limit smoke evidence; not a calibration maturity claim.",
    )
    capability_oracle = OracleSpec(
        kind="capability",
        source="DEFAULT_FD_CAPABILITY_MANIFEST fail-closed feature screening",
        independence="contract-level rejection before solver construction",
        notes="Unsupported features must be explicit capability rejections, not silent approximation.",
    )
    pinares_oracle = OracleSpec(
        kind="analytical",
        source="survival-scaled Black-Scholes fixed-price call proxy",
        independence="closed-form call oracle scaled outside the FD grid by public-synthetic survival_probability",
        notes="Proxy evidence only; ROFR/full family-contract valuation remains unsupported in this backend.",
    )
    nonuniform_greek_oracle = OracleSpec(
        kind="analytical",
        source="closed-form Black-Scholes Delta/Gamma plus polynomial manufactured derivatives",
        independence=(
            "tests evaluate local-coordinate stencil estimates and requested-coordinate "
            "sampling outside solver routing"
        ),
        notes="Issue #57 evidence for nonuniform-grid Greek estimation and diagnostics.",
    )
    return (
        BenchmarkCase(
            benchmark_id="BS-CALL-PARITY-V0",
            title="Black-Scholes European call analytical parity and no-arbitrage evidence",
            family="analytical_oracle",
            status="validated",
            route_id="fd.black_scholes_1d.crank_nicolson",
            model="Black-Scholes / GBM",
            instrument="European call",
            state_convention="spot S, calendar-time output, forward tau internal march",
            grid_family="uniform spot grid; uniform time grid",
            time_schedule="three refinement levels with Crank-Nicolson theta=0.5",
            oracle=bs_oracle,
            tolerances=(
                TolerancePolicy(
                    "price_abs",
                    5.0e-4,
                    "absolute price error",
                    "finest grid against analytic oracle",
                ),
                TolerancePolicy(
                    "delta_abs",
                    5.0e-2,
                    "absolute Delta error",
                    "central finite-difference Greek",
                ),
                TolerancePolicy(
                    "gamma_abs",
                    2.0e-2,
                    "absolute Gamma error",
                    "central finite-difference Greek",
                ),
            ),
            invariants=(
                "value_bound_ok",
                "upper_bound_ok",
                "delta_lower_bound_ok",
                "delta_upper_bound_ok",
                "gamma_non_negative_ok",
            ),
            capability_rows=(
                "1D Black-Scholes European call value on uniform/log-uniform grids",
                "1D Black-Scholes Delta/Gamma from finite-difference price grids",
            ),
            fixture_paths=("tests/fixtures/arxiv_lab_bs_oracle_v1.json",),
            issue_refs=("googa27/finite_difference_options#49",),
            resource_policy={
                "max_s_steps": 120,
                "max_t_steps": 200,
                "grid_levels": 3,
                "deterministic": "true",
            },
            runner="black_scholes_parity",
            notes="Executable CI runner exercises convergence table, Greeks, no-arbitrage checks and evidence payload.",
        ),
        BenchmarkCase(
            benchmark_id="QPS-VANILLA-CALL-V0",
            title="QuantProblemSpec vanilla call fixture compatibility",
            family="route_parity",
            status="validated",
            route_id="fd.quant_problem_spec.black_scholes_call",
            model="Black-Scholes / QuantProblemSpec v0",
            instrument="European call problem contract",
            state_convention="explicit measure, numeraire, time orientation and typed boundary metadata",
            grid_family="fixture-declared FD grid",
            time_schedule="fixture-declared theta controls",
            oracle=OracleSpec(
                kind="fixture",
                source="arXiv-Lab FD oracle fixture export",
                independence="serialized public-synthetic problem/result contract consumed without private data",
                notes="Guards cross-repo compatibility rather than a new numerical method.",
            ),
            tolerances=(
                TolerancePolicy(
                    "price_abs",
                    5.0e-4,
                    "absolute price error",
                    "shared with BS-CALL-PARITY-V0",
                ),
            ),
            invariants=(
                "schema_version",
                "problem_hash",
                "typed_boundary",
                "calendar_time_orientation",
            ),
            capability_rows=("1D Black-Scholes European call value on uniform/log-uniform grids",),
            fixture_paths=("tests/fixtures/arxiv_lab_bs_oracle_v1.json",),
            issue_refs=(
                "googa27/finite_difference_options#49",
                "googa27/finite_difference_options#59",
            ),
            resource_policy={"public_synthetic": "true"},
            runner="black_scholes_qps_contract",
            notes=(
                "Metadata registry row for cross-repo fixture parity; execution is covered by BS-CALL-PARITY-V0 tests."
            ),
        ),
        BenchmarkCase(
            benchmark_id="PINARES-FD-FIXED-PRICE-PROXY-V0",
            title="Pinares fixed-price option proxy FD parity",
            family="analytical_oracle",
            status="validated",
            route_id="fd.pinares_fixed_price_proxy.crank_nicolson",
            model="survival-scaled Black-Scholes / Pinares public-synthetic proxy",
            instrument="mortality-scaled fixed-price purchase option proxy",
            state_convention="property-value proxy S in UF, Q* measure, survival scalar outside FD grid",
            grid_family="uniform UF spot grid with explicit s_max and theta time controls",
            time_schedule="three refinement levels with Crank-Nicolson theta=0.5",
            oracle=pinares_oracle,
            tolerances=(
                TolerancePolicy(
                    "price_abs",
                    1.0,
                    "absolute UF price error",
                    "finest grid against survival-scaled analytical proxy",
                ),
                TolerancePolicy(
                    "delta_abs",
                    1.0e-3,
                    "absolute Delta error",
                    "survival-scaled central finite-difference Greek",
                ),
                TolerancePolicy(
                    "gamma_abs",
                    5.0e-6,
                    "absolute Gamma error",
                    "survival-scaled central finite-difference Greek",
                ),
            ),
            invariants=(
                "value_bound_ok",
                "upper_bound_ok",
                "delta_lower_bound_ok",
                "delta_upper_bound_ok",
                "gamma_non_negative_ok",
                "survival_scale_ok",
            ),
            capability_rows=("Pinares fixed-price option proxy",),
            fixture_paths=(
                "tests/fixtures/pinares_fd_fixed_price_proxy_v1.json",
                "tests/fixtures/quant_problem_specs/pinares_fixed_price_proxy.json",
            ),
            issue_refs=("googa27/finite_difference_options#119",),
            resource_policy={
                "max_s_steps": 180,
                "max_t_steps": 240,
                "grid_levels": 3,
                "deterministic": "true",
            },
            runner="pinares_fixed_price_proxy",
            notes="Executable public-synthetic Pinares proxy; not a ROFR/full-deal valuation.",
        ),
        BenchmarkCase(
            benchmark_id="PINARES-QPS-FIXED-PRICE-PROXY-V0",
            title="Pinares fixed-price QuantProblemSpec compatibility",
            family="route_parity",
            status="validated",
            route_id="fd.quant_problem_spec.pinares_fixed_price_proxy",
            model="Pinares QuantProblemSpec v0",
            instrument="public-synthetic fixed-price proxy problem contract",
            state_convention=(
                "explicit Q* measure, UF numeraire, public_synthetic privacy class and terminal payoff scale"
            ),
            grid_family="fixture-declared FD grid",
            time_schedule="fixture-declared theta controls and resource policy",
            oracle=OracleSpec(
                kind="fixture",
                source="Pinares fixed-price proxy fixture export",
                independence="serialized public-synthetic QuantProblemSpec consumed without private Pinares data",
                notes="Guards Pinares/FD compatibility; financial semantics remain owned by Pinares.",
            ),
            tolerances=(
                TolerancePolicy(
                    "price_abs",
                    1.0,
                    "absolute UF price error",
                    "shared with PINARES-FD-FIXED-PRICE-PROXY-V0",
                ),
            ),
            invariants=(
                "schema_version",
                "problem_hash",
                "terminal_payoff_scale",
                "privacy_public_synthetic",
                "fd_route_supported",
            ),
            capability_rows=("Pinares fixed-price option proxy",),
            fixture_paths=("tests/fixtures/quant_problem_specs/pinares_fixed_price_proxy.json",),
            issue_refs=("googa27/finite_difference_options#119",),
            resource_policy={"public_synthetic": "true"},
            runner="pinares_qps_contract",
        ),
        BenchmarkCase(
            benchmark_id="PINARES-FD-FAIL-CLOSED-V0",
            title="Pinares full-deal and ROFR FD fail-closed route gate",
            family="capability_gate",
            status="validated",
            route_id="fd.pinares.fail_closed_route_screening",
            model="Pinares unsupported full-deal route screening",
            instrument="ROFR/full family real-estate use-right contract",
            state_convention="legal, tax, mortality, liquidity and market-rent terms are not FD proxy state variables",
            grid_family="route-screening metadata, not a numerical grid",
            time_schedule="pre-solve compatibility gate",
            oracle=capability_oracle,
            tolerances=(
                TolerancePolicy(
                    "boolean_invariant",
                    True,
                    "all invariants",
                    "unsupported full-deal routes return diagnostics before solver construction",
                ),
            ),
            invariants=(
                "rofr_not_executed",
                "full_family_contract_rejected",
                "unsupported_terms_reported",
                "no_placeholder_values",
            ),
            capability_rows=(
                "Jump/PIDE and HJB/control terms",
                "Pinares full family-contract and ROFR routes",
            ),
            fixture_paths=("tests/test_pinares_fd_proxy.py",),
            issue_refs=("googa27/finite_difference_options#119",),
            resource_policy={"deterministic": "true"},
            runner="pinares_fail_closed",
        ),
        BenchmarkCase(
            benchmark_id="BOUNDARY-MODEL-AWARE-V0",
            title="Model-aware vanilla boundary and reaction semantics",
            family="capability_gate",
            status="validated",
            route_id="fd.boundary_conditions.model_aware_vanilla",
            model="Black-Scholes / GBM boundary algebra",
            instrument="European call/put boundary facets",
            state_convention="spot boundary facets with explicit strike, rate/carry and time-to-maturity",
            grid_family="boundary-facet contract independent of grid density",
            time_schedule="time-to-maturity-aware boundary values",
            oracle=OracleSpec(
                kind="fixture",
                source="boundary condition and model-aware reaction regression tests",
                independence="boundary builder tests assert explicit typed facets before solver use",
                notes="Prevents route maturity claims from relying on option_type guessing.",
            ),
            tolerances=(
                TolerancePolicy(
                    "boolean_invariant",
                    True,
                    "all invariants",
                    "typed boundary/reaction checks pass",
                ),
            ),
            invariants=(
                "dirichlet_call_zero",
                "far_call_asymptotic",
                "explicit_zero_discount_preserved",
            ),
            capability_rows=("1D vanilla boundary/reaction semantics",),
            fixture_paths=(
                "tests/test_boundary_conditions.py",
                "tests/test_model_aware_reaction_terms.py",
            ),
            issue_refs=(
                "googa27/finite_difference_options#48",
                "googa27/finite_difference_options#49",
            ),
            resource_policy={"deterministic": "true"},
        ),
        BenchmarkCase(
            benchmark_id="REACTION-INDEPENDENT-V0",
            title="Reaction/discount coefficient can be supplied independently from drift",
            family="capability_gate",
            status="validated",
            route_id="fd.coefficients.model_aware_reaction",
            model="GBM / affine process coefficient extraction",
            instrument="generic model-aware PDE coefficients",
            state_convention="drift, covariance and reaction are separate coefficient fields",
            grid_family="grid-independent coefficient extraction",
            time_schedule="terminal-time process coefficient calls",
            oracle=OracleSpec(
                kind="fixture",
                source="model-aware reaction regression tests",
                independence="direct coefficient-path tests independent of pricing output",
                notes="Avoids conflating real-world drift with risk-neutral discount reaction.",
            ),
            tolerances=(
                TolerancePolicy(
                    "boolean_invariant",
                    True,
                    "all invariants",
                    "reaction pass-through checks pass",
                ),
            ),
            invariants=(
                "reaction_not_inferred_from_mu",
                "explicit_zero_discount_preserved",
            ),
            capability_rows=("1D vanilla boundary/reaction semantics",),
            fixture_paths=("tests/test_model_aware_reaction_terms.py",),
            issue_refs=(
                "googa27/finite_difference_options#48",
                "googa27/finite_difference_options#49",
            ),
            resource_policy={"deterministic": "true"},
        ),
        BenchmarkCase(
            benchmark_id="RANNACHER-GAMMA-V0",
            title="Rannacher startup protects kinked-payoff Gamma evidence",
            family="manufactured_solution",
            status="validated",
            route_id="fd.black_scholes_1d.rannacher_crank_nicolson",
            model="Black-Scholes / GBM",
            instrument="European vanilla payoff with strike kink",
            state_convention="spot grid around strike with explicit startup schedule",
            grid_family="1D finite-difference grid",
            time_schedule="two/four Backward-Euler half-step startup before Crank-Nicolson",
            oracle=OracleSpec(
                kind="regression",
                source="Rannacher Gamma regression tests",
                independence="compares declared startup schedules against unsmoothed route behavior",
                notes="Regression evidence for smoothing policy, not a universal Greek guarantee.",
            ),
            tolerances=(
                TolerancePolicy(
                    "boolean_invariant",
                    True,
                    "all invariants",
                    "startup schedule and Gamma sanity checks pass",
                ),
            ),
            invariants=("startup_schedule_recorded", "gamma_kink_sanity"),
            capability_rows=("Rannacher startup before Crank-Nicolson for kinked payoffs",),
            fixture_paths=("tests/test_rannacher_startup.py",),
            issue_refs=("googa27/finite_difference_options#49",),
            resource_policy={"deterministic": "true"},
        ),
        BenchmarkCase(
            benchmark_id="AMERICAN-LCP-V0",
            title="Black-Scholes American/Bermudan obstacle LCP complementarity evidence",
            family="no_arbitrage",
            status="validated",
            route_id="fd.black_scholes_1d.projected_sor_lcp",
            model="Black-Scholes / GBM",
            instrument="American and Bermudan vanilla options",
            state_convention="spot S, calendar-time output, forward tau internal march with obstacle projection",
            grid_family="uniform spot grid; uniform time grid",
            time_schedule="implicit theta LCP step with projected SOR warm-started from previous tau level",
            oracle=OracleSpec(
                kind="fixture",
                source="tests/test_american_lcp.py",
                independence=(
                    "complementarity, no-arbitrage, ordering and nonconvergence "
                    "fixtures independent of API route labels"
                ),
                notes=(
                    "Covers primal/dual/complementarity diagnostics, American>=European, "
                    "non-dividend call parity, Bermudan ordering and iteration-limit failure."
                ),
            ),
            tolerances=(
                TolerancePolicy(
                    metric="lcp_primal_abs",
                    threshold=5.0e-8,
                    norm="max obstacle violation",
                    notes="max(payoff - value, 0) over diagnosed interior nodes",
                ),
                TolerancePolicy(
                    metric="lcp_dual_abs",
                    threshold=5.0e-5,
                    norm="max negative dual residual",
                    notes="max negative A value - rhs residual under LCP sign convention",
                ),
                TolerancePolicy(
                    metric="lcp_complementarity_abs",
                    threshold=5.0e-4,
                    norm="max complementarity product",
                    notes="componentwise (value-payoff)*(A value-rhs) diagnostic",
                ),
            ),
            invariants=(
                "value_above_obstacle",
                "american_dominates_european",
                "non_dividend_call_matches_european",
                "bermudan_between_european_and_american",
                "nonconvergence_fails_closed",
                "exercise_boundary_reported",
            ),
            capability_rows=("American/free-boundary exercise",),
            fixture_paths=("tests/test_american_lcp.py",),
            issue_refs=("googa27/finite_difference_options#66",),
            resource_policy={"deterministic": "true", "max_s_steps": 121, "max_t_steps": 81},
            runner="american_lcp",
            notes=(
                "1D reference route only; multidimensional American/ADI remains gated "
                "until separate work-precision evidence exists."
            ),
        ),
        BenchmarkCase(
            benchmark_id="HESTON-SMOKE-DOCSTRING-V0",
            title="Heston vanilla-call smoke route shape and finite-value evidence",
            family="smoke",
            status="experimental",
            route_id="fd.heston.adi_smoke",
            model="Heston stochastic volatility",
            instrument="European call smoke fixture",
            state_convention="log-spot and variance state convention with declared exp factor transform",
            grid_family="2D tensor grid smoke fixture",
            time_schedule="ADI/Douglas smoke route calendar-output orientation",
            oracle=OracleSpec(
                kind="fixture",
                source="Heston state/oracle smoke tests and capability matrix docstring evidence",
                independence="shape/finite-value route test separate from semi-analytical oracle maturity claim",
                notes="Experimental smoke evidence only; convergence and production calibration remain unsupported.",
            ),
            tolerances=(
                TolerancePolicy(
                    "boolean_invariant",
                    True,
                    "all invariants",
                    "finite shape/no-NaN Heston smoke checks pass",
                ),
            ),
            invariants=(
                "finite_values",
                "log_spot_factor_transform",
                "variance_state_not_payoff_asset",
            ),
            capability_rows=("Heston stochastic volatility vanilla call smoke route",),
            fixture_paths=("tests/test_heston_state_oracle.py",),
            issue_refs=("googa27/finite_difference_options#49",),
            resource_policy={"deterministic": "true"},
        ),
        BenchmarkCase(
            benchmark_id="HESTON-ORACLE-V0",
            title="Heston semi-analytical European call oracle",
            family="analytical_oracle",
            status="validated",
            route_id="fd.validation.heston_oracle",
            model="Heston stochastic volatility",
            instrument="European call",
            state_convention="log-spot and variance state convention",
            grid_family="not an FD grid; Fourier oracle reference",
            time_schedule="single maturity reference integration",
            oracle=heston_oracle,
            tolerances=(
                TolerancePolicy(
                    "price_abs",
                    2.0e-3,
                    "absolute price error",
                    "oracle regression tolerance",
                ),
            ),
            invariants=(
                "positive_price",
                "finite_integral",
                "variance_boundary_diagnostics",
            ),
            capability_rows=(
                "Heston semi-analytical European call oracle",
                "Heston stochastic volatility vanilla call smoke route",
            ),
            fixture_paths=("src/finite_difference_options/validation/heston_oracle.py",),
            issue_refs=("googa27/finite_difference_options#49",),
            resource_policy={"deterministic": "true"},
        ),
        BenchmarkCase(
            benchmark_id="HESTON-BS-LIMIT-V0",
            title="Heston zero vol-of-vol Black-Scholes limiting case",
            family="route_parity",
            status="validated",
            route_id="fd.validation.heston_black_scholes_limit",
            model="Heston -> Black-Scholes limit",
            instrument="European call",
            state_convention="variance fixed at sigma^2 when vol-of-vol tends to zero",
            grid_family="oracle parity, not production FD route",
            time_schedule="single maturity reference comparison",
            oracle=heston_oracle,
            tolerances=(
                TolerancePolicy(
                    "price_abs",
                    2.0e-3,
                    "absolute price error",
                    "against Black-Scholes oracle",
                ),
            ),
            invariants=("limit_price_matches_black_scholes",),
            capability_rows=("Heston semi-analytical European call oracle",),
            fixture_paths=("src/finite_difference_options/validation/heston_oracle.py",),
            issue_refs=("googa27/finite_difference_options#49",),
            resource_policy={"deterministic": "true"},
            runner="heston_black_scholes_limit",
        ),
        BenchmarkCase(
            benchmark_id="HESTON-VARIANCE-BOUNDARY-V0",
            title="Heston variance-boundary and Feller-policy diagnostics",
            family="capability_gate",
            status="validated",
            route_id="fd.heston.variance_boundary_diagnostics",
            model="Heston stochastic volatility",
            instrument="European call smoke route",
            state_convention="variance boundary and Feller policy diagnostics are explicit",
            grid_family="variance grid diagnostics",
            time_schedule="single smoke solve diagnostics",
            oracle=heston_oracle,
            tolerances=(
                TolerancePolicy(
                    "boolean_invariant",
                    True,
                    "all invariants",
                    "variance boundary diagnostics present",
                ),
            ),
            invariants=(
                "variance_nonnegative",
                "boundary_diagnostics_recorded",
                "feller_violation_is_policy_not_domain_error",
                "zero_vol_of_vol_has_stable_diagnostics",
            ),
            capability_rows=("Heston semi-analytical European call oracle",),
            fixture_paths=(
                "src/finite_difference_options/validation/heston_oracle.py",
                "tests/test_heston_feller_policy.py",
            ),
            issue_refs=(
                "googa27/finite_difference_options#49",
                "googa27/finite_difference_options#65",
            ),
            resource_policy={"deterministic": "true"},
        ),
        BenchmarkCase(
            benchmark_id="GRID-LOCAL-METRICS-V0",
            title="Typed nonuniform grid contracts and local derivative metrics",
            family="manufactured_solution",
            status="validated",
            route_id="fd.grids.axis_tensor_local_metrics",
            model="finite-difference grid contract",
            instrument="operator-level polynomial fixture",
            state_convention="solver-coordinate axes with optional physical/log transform metadata",
            grid_family="uniform, log-uniform, sinh/tanh clustered, strike-centered and variance-boundary axes",
            time_schedule="grid/operator construction before time march",
            oracle=OracleSpec(
                kind="manufactured",
                source="quadratic-polynomial derivative exactness and ADI AxisGrid regression tests",
                independence="tests evaluate local stencil weights and ADI diagnostics without a pricing facade",
                notes=(
                    "Grid contract evidence; nonuniform Greek convergence is covered separately by "
                    "FD-GREEKS-NONUNIFORM-V0."
                ),
            ),
            tolerances=(
                TolerancePolicy(
                    "boolean_invariant",
                    True,
                    "all invariants",
                    "nonuniform derivative, interpolation, transform and ADI diagnostic checks pass",
                ),
            ),
            invariants=(
                "strict_monotonicity",
                "quadratic_derivative_exactness",
                "interpolation_domain_rejection",
                "grid_identity_diagnostics",
            ),
            capability_rows=("Typed uniform/nonuniform/log tensor grid contracts",),
            fixture_paths=("tests/test_grid_contracts.py",),
            issue_refs=("googa27/finite_difference_options#47",),
            resource_policy={"deterministic": "true"},
        ),
        BenchmarkCase(
            benchmark_id="FD-GREEKS-NONUNIFORM-V0",
            title="Nonuniform-grid finite-difference Greek diagnostics",
            family="analytical_oracle",
            status="validated",
            route_id="fd.greeks.nonuniform_requested_coordinate",
            model="Black-Scholes / manufactured polynomial",
            instrument="European call and smooth polynomial value slice",
            state_convention="spot-coordinate value slices with explicit requested-coordinate sampling",
            grid_family="strictly increasing nonuniform and strike-centered physical spot grids",
            time_schedule="value-slice post-processing; expiry kink coordinates fail closed",
            oracle=nonuniform_greek_oracle,
            tolerances=(
                TolerancePolicy(
                    "delta_abs",
                    5.0e-2,
                    "absolute Delta error",
                    "strike-centered nonuniform refinement against closed-form Black-Scholes Delta",
                ),
                TolerancePolicy(
                    "gamma_abs",
                    2.0e-2,
                    "absolute Gamma error",
                    "strike-centered nonuniform refinement against closed-form Black-Scholes Gamma",
                ),
                TolerancePolicy(
                    "boolean_invariant",
                    True,
                    "all diagnostics",
                    "stencil/interpolation/error/expiry diagnostic invariants hold",
                ),
            ),
            invariants=(
                "local_coordinate_spacing_used",
                "requested_coordinate_distinct_from_nearest_node",
                "refinement_error_reported",
                "expiry_kink_rejected",
            ),
            capability_rows=("1D Black-Scholes Delta/Gamma from finite-difference price grids",),
            fixture_paths=("tests/test_nonuniform_greek_diagnostics.py",),
            issue_refs=("googa27/finite_difference_options#57",),
            resource_policy={"deterministic": "true"},
        ),
        BenchmarkCase(
            benchmark_id="FD-GREEKS-VALIDATION-V0",
            title="Derivative convergence, strike-alignment and Rannacher stability gates",
            family="analytical_oracle",
            status="validated",
            route_id="fd.greeks.validation_matrix",
            model="Black-Scholes / GBM",
            instrument="European call derivative validation matrix",
            state_convention=(
                "requested spot-coordinate Delta/Gamma across moneyness, maturity, volatility and "
                "strike-alignment cases"
            ),
            grid_family="strike-centered nonuniform grids plus shifted uniform alignment grids",
            time_schedule="value-slice derivative validation plus Rannacher startup stability check",
            oracle=nonuniform_greek_oracle,
            tolerances=(
                TolerancePolicy(
                    "delta_abs",
                    1.0e-3,
                    "maximum finest-grid absolute Delta error",
                    "closed-form Black-Scholes Delta over the PR validation matrix",
                ),
                TolerancePolicy(
                    "gamma_abs",
                    3.0e-4,
                    "maximum finest-grid absolute Gamma error",
                    "closed-form Black-Scholes Gamma over the PR validation matrix",
                ),
                TolerancePolicy(
                    "boolean_invariant",
                    True,
                    "all derivative-validation invariants",
                    "refinement, alignment, expiry and Rannacher stability gates pass",
                ),
            ),
            invariants=(
                "nonuniform_delta_converged",
                "nonuniform_gamma_converged",
                "refinement_improves_all_cases",
                "strike_alignment_bounded",
                "rannacher_smooths_kinked_gamma",
                "expiry_kink_rejected",
            ),
            capability_rows=(
                "1D Black-Scholes Delta/Gamma from finite-difference price grids",
                "Rannacher startup before Crank-Nicolson for kinked payoffs",
            ),
            fixture_paths=("tests/test_derivative_validation_gates.py",),
            issue_refs=(
                "googa27/finite_difference_options#25",
                "googa27/finite_difference_options#57",
                "googa27/finite_difference_options#58",
            ),
            resource_policy={
                "benchmark_cases": 12,
                "grid_levels": 3,
                "deterministic": "true",
                "max_runtime_seconds": 30.0,
            },
            runner="greek_derivative_validation",
        ),
        BenchmarkCase(
            benchmark_id="ADI-SMOKE-V0",
            title="ADI multidimensional finite-value smoke route",
            family="smoke",
            status="experimental",
            route_id="fd.adi.douglas_smoke",
            model="generic multidimensional parabolic PDE",
            instrument="vanilla/basket smoke fixtures",
            state_convention="tensor-product state grid with explicit covariance convention",
            grid_family="2D/3D tensor grids",
            time_schedule="Douglas ADI calendar-output orientation",
            oracle=OracleSpec(
                kind="fixture",
                source="ADI smoke and operator split regression tests",
                independence="regression tests cover operator splitting components before production claim",
                notes="Smoke evidence only; convergence issue remains open.",
            ),
            tolerances=(
                TolerancePolicy(
                    "boolean_invariant",
                    True,
                    "all invariants",
                    "finite shape/no-NaN smoke checks pass",
                ),
            ),
            invariants=(
                "finite_values",
                "calendar_output_orientation",
                "shape_consistency",
            ),
            capability_rows=("ADI multidimensional routes",),
            fixture_paths=("tests/test_adi_solver_operator_split.py",),
            issue_refs=(
                "googa27/finite_difference_options#49",
                "googa27/finite_difference_options#58",
            ),
            resource_policy={"deterministic": "true"},
        ),
        BenchmarkCase(
            benchmark_id="ADI-OPERATOR-SPLIT-V0",
            title="ADI operator split coefficient/regression evidence",
            family="route_parity",
            status="experimental",
            route_id="fd.adi.operator_split_contract",
            model="generic multidimensional parabolic PDE",
            instrument="operator-level fixture",
            state_convention="diagonal diffusion, off-diagonal mixed derivative, reaction and source terms explicit",
            grid_family="2D/3D tensor grids",
            time_schedule="operator construction before time march",
            oracle=OracleSpec(
                kind="fixture",
                source="ADI operator split regression tests",
                independence="operator assembly tested separately from high-level pricing route",
                notes="Supports route maturity disclosure; not yet convergence evidence.",
            ),
            tolerances=(
                TolerancePolicy(
                    "boolean_invariant",
                    True,
                    "all invariants",
                    "split terms match expected coefficients",
                ),
            ),
            invariants=(
                "mixed_derivative_sign",
                "reaction_preserved",
                "source_preserved",
            ),
            capability_rows=("ADI multidimensional routes",),
            fixture_paths=("tests/test_adi_solver_operator_split.py",),
            issue_refs=(
                "googa27/finite_difference_options#49",
                "googa27/finite_difference_options#58",
            ),
            resource_policy={"deterministic": "true"},
        ),
        BenchmarkCase(
            benchmark_id="FACTOR-ROLE-COMPAT-V0",
            title="Factor-role payoff compatibility fail-closed gate",
            family="capability_gate",
            status="validated",
            route_id="fd.payoff.factor_role_compatibility",
            model="basket/spread payoff role screening",
            instrument="basket, spread and factor payoffs",
            state_convention="tradable spot factors must be explicitly distinguished from variance/vol/rate factors",
            grid_family="route-screening metadata, not a numerical grid",
            time_schedule="pre-solve compatibility gate",
            oracle=capability_oracle,
            tolerances=(
                TolerancePolicy(
                    "boolean_invariant",
                    True,
                    "all invariants",
                    "unsupported factor/payoff pairs fail closed",
                ),
            ),
            invariants=(
                "heston_variance_not_basket_asset",
                "sabr_vol_not_basket_asset",
                "asset_id_mapping_required",
            ),
            capability_rows=("basket option payoff on true multi-asset factors",),
            fixture_paths=("tests/test_factor_role_payoff_compatibility.py",),
            issue_refs=("googa27/finite_difference_options#49",),
            resource_policy={"deterministic": "true"},
        ),
        BenchmarkCase(
            benchmark_id="DOCS-README-SMOKE-V0",
            title="README and documentation smoke examples stay aligned with supported routes",
            family="smoke",
            status="experimental",
            route_id="fd.docs.readme_smoke",
            model="documentation/examples",
            instrument="public README/API examples",
            state_convention="examples must disclose capability maturity and synthetic inputs",
            grid_family="example-declared grids only",
            time_schedule="example-declared time controls only",
            oracle=OracleSpec(
                kind="fixture",
                source="documentation smoke tests and capability matrix gating",
                independence="docs examples are exercised separately from numerical core validation",
                notes="Convenience-surface evidence; numerical truth remains in core benchmark rows.",
            ),
            tolerances=(
                TolerancePolicy(
                    "boolean_invariant",
                    True,
                    "all invariants",
                    "README/docs examples execute without unsupported maturity claims",
                ),
            ),
            invariants=(
                "examples_execute",
                "maturity_disclosed",
                "public_synthetic_inputs",
            ),
            capability_rows=("FastAPI/CLI/UI service contracts",),
            fixture_paths=("tests/test_documentation_capabilities.py",),
            issue_refs=("googa27/finite_difference_options#49",),
            resource_policy={"deterministic": "true"},
        ),
        BenchmarkCase(
            benchmark_id="API-REQUEST-GUARDS-V0",
            title="API request guards and explicit non-claim convergence metadata",
            family="capability_gate",
            status="experimental",
            route_id="fd.api.request_guards",
            model="FastAPI/CLI adapter schemas",
            instrument="pricing request/response contracts",
            state_convention="outer adapters over numerical core with explicit route maturity",
            grid_family="bounded request grids only",
            time_schedule="request-declared maturity/time controls",
            oracle=OracleSpec(
                kind="fixture",
                source="API schema and request-guard tests",
                independence="adapter contract tests exercise bounded inputs before solver dispatch",
                notes="API evidence does not promote experimental numerical routes to validated status.",
            ),
            tolerances=(
                TolerancePolicy(
                    "boolean_invariant",
                    True,
                    "all invariants",
                    "invalid requests fail closed and responses disclose convergence status",
                ),
            ),
            invariants=(
                "bounded_grid_nodes",
                "invalid_payoff_rejected",
                "convergence_not_assessed_disclosed",
            ),
            capability_rows=("FastAPI/CLI/UI service contracts",),
            fixture_paths=("tests/test_api_schema_contracts.py",),
            issue_refs=("googa27/finite_difference_options#49",),
            resource_policy={"deterministic": "true"},
        ),
        BenchmarkCase(
            benchmark_id="FD-CANONICAL-INVENTORY-V0",
            title="Canonical implementation inventory architecture gate",
            family="capability_gate",
            status="validated",
            route_id="fd.architecture.canonical_capability_inventory",
            model="finite-difference package architecture",
            instrument="canonical module inventory and compatibility policy",
            state_convention="public imports under finite_difference_options; historical root src modules absent",
            grid_family="not applicable",
            time_schedule="not applicable",
            oracle=OracleSpec(
                kind="fixture",
                source=(
                    "docs/architecture_contract.toml canonical_capabilities plus "
                    "docs/CANONICAL_IMPLEMENTATION_INVENTORY.md"
                ),
                independence=(
                    "architecture tests parse the machine-readable contract and inspect "
                    "the live repository tree"
                ),
                notes="Evidence for issue #52 consolidation governance; not numerical convergence evidence.",
            ),
            tolerances=(
                TolerancePolicy(
                    "boolean_invariant",
                    True,
                    "all invariants",
                    (
                        "canonical paths exist, public imports map to package modules, "
                        "and forbidden legacy modules are absent"
                    ),
                ),
            ),
            invariants=(
                "canonical_paths_exist",
                "public_imports_under_distribution",
                "forbidden_legacy_modules_absent",
                "compatibility_shims_are_boundary_only",
            ),
            capability_rows=("Canonical implementation inventory",),
            fixture_paths=(
                "docs/CANONICAL_IMPLEMENTATION_INVENTORY.md",
                "docs/architecture_contract.toml",
                "tests/architecture/test_architecture_contracts.py",
            ),
            issue_refs=("googa27/finite_difference_options#52",),
            resource_policy={"deterministic": "true"},
        ),
        BenchmarkCase(
            benchmark_id="REG-FAIL-CLOSED-V0",
            title="Regulatory endpoint fail-closed standard/version gate",
            family="regulatory_fail_closed",
            status="scaffold",
            route_id="fd.regulatory.fail_closed",
            model="regulatory reporting scaffold",
            instrument="CRIF/CUSO/Basel/FRTB endpoints",
            state_convention="standard/profile/version/effective-date/jurisdiction required before implementation",
            grid_family="not applicable",
            time_schedule="not applicable",
            oracle=capability_oracle,
            tolerances=(),
            invariants=(
                "http_501_for_unsupported_standard",
                "no_placeholder_report_values",
            ),
            capability_rows=("CRIF/CUSO/Basel/FRTB regulatory report endpoints",),
            fixture_paths=("tests/test_regulatory_fail_closed.py",),
            issue_refs=("googa27/finite_difference_options#49",),
            resource_policy={"deterministic": "true"},
        ),
    )


def registry_by_id(
    registry: tuple[BenchmarkCase, ...] | None = None,
) -> dict[str, BenchmarkCase]:
    """Return registry rows keyed by benchmark id."""

    rows = registry if registry is not None else default_benchmark_registry()
    return {case.benchmark_id: case for case in rows}


def validate_benchmark_registry(
    registry: tuple[BenchmarkCase, ...] | None = None,
) -> tuple[BenchmarkCase, ...]:
    """Validate registry invariants and return the rows if valid."""

    rows = registry if registry is not None else default_benchmark_registry()
    errors: list[str] = []
    seen: set[str] = set()
    for case in rows:
        if case.benchmark_id in seen:
            errors.append(f"duplicate benchmark_id: {case.benchmark_id}")
        seen.add(case.benchmark_id)
        errors.extend(case.validate())
    if errors:
        raise BenchmarkRegistryError(errors)
    return rows


def registry_as_dict(
    registry: tuple[BenchmarkCase, ...] | None = None,
) -> dict[str, Any]:
    """Return a JSON-friendly registry payload."""

    rows = validate_benchmark_registry(registry)
    return {
        "schema_version": "finite-difference-benchmark-registry/v0",
        "registry_id": "finite_difference_options.benchmark_registry.v0",
        "issue_ref": "googa27/finite_difference_options#49",
        "case_count": len(rows),
        "cases": [case.as_dict() for case in rows],
    }


def write_registry_json(path: str | Path, registry: tuple[BenchmarkCase, ...] | None = None) -> None:
    """Write the registry JSON payload with stable ordering and indentation."""

    import json

    payload = registry_as_dict(registry)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_benchmark_result_json(path: str | Path, result: BenchmarkRunResult) -> None:
    """Persist one benchmark run result for CI/artifact triage."""

    import json

    Path(path).write_text(json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reported_invariants(names: tuple[str, ...], reported: dict[str, Any]) -> dict[str, bool]:
    """Evaluate declared invariants fail-closed against a reported payload."""

    return {name: bool(reported.get(name, False)) for name in names}


def _tolerance_invariants(case: BenchmarkCase, metrics: dict[str, float | bool | int | str]) -> dict[str, bool]:
    """Evaluate every declared benchmark tolerance against produced metrics."""

    results: dict[str, bool] = {}
    for tolerance in case.tolerances:
        key = f"{tolerance.metric}_tolerance_ok"
        observed = metrics.get(tolerance.metric)
        if not isinstance(observed, int | float):
            results[key] = False
            continue
        results[key] = float(observed) <= tolerance.threshold
    return results


def _black_scholes_parity_result(case: BenchmarkCase) -> BenchmarkRunResult:
    from finite_difference_options.validation.black_scholes_parity import (
        run_public_black_scholes_parity_fixture,
    )

    report = run_public_black_scholes_parity_fixture()
    metrics: dict[str, float | bool | int | str] = {
        "price": report.price,
        "oracle_price": report.oracle_price,
        "price_abs": report.final_abs_error,
        "delta_abs": report.errors["delta_abs"],
        "gamma_abs": report.errors["gamma_abs"],
        "grid_levels": len(report.convergence_table()),
    }
    invariants = _reported_invariants(case.invariants, report.no_arbitrage)
    invariants.update(_tolerance_invariants(case, metrics))
    passed = all(invariants.values())
    return BenchmarkRunResult(
        benchmark_id=case.benchmark_id,
        passed=passed,
        metrics=metrics,
        evidence=report.evidence.as_dict(),
        invariants=invariants,
    )


def _black_scholes_qps_contract_result(case: BenchmarkCase) -> BenchmarkRunResult:
    from finite_difference_options.validation.black_scholes_parity import (
        run_public_black_scholes_parity_fixture,
    )

    report = run_public_black_scholes_parity_fixture()
    payload = report.as_dict()
    problem = payload["problem_spec"]
    result = payload["result_export"]
    typed_boundary = result["boundary"]["typed"]
    metrics: dict[str, float | bool | int | str] = {
        "price_abs": report.final_abs_error,
        "grid_levels": len(report.convergence_table()),
        "typed_boundary_count": len(typed_boundary),
    }
    invariants = {
        "schema_version": problem["schema_version"] == "quant-problem-spec/v0",
        "problem_hash": bool(problem["problem_hash"]),
        "typed_boundary": all("boundary_type" in item for item in typed_boundary),
        "calendar_time_orientation": result["time_axis"]["direction"] == "decreasing",
    }
    invariants.update(_tolerance_invariants(case, metrics))
    passed = all(invariants.values())
    return BenchmarkRunResult(
        benchmark_id=case.benchmark_id,
        passed=passed,
        metrics=metrics,
        evidence=report.evidence.as_dict(),
        invariants=invariants,
    )


def _pinares_fixed_price_proxy_result(case: BenchmarkCase) -> BenchmarkRunResult:
    from finite_difference_options.validation.pinares_fixed_price_proxy import (
        run_public_pinares_fixed_price_proxy_fixture,
    )

    report = run_public_pinares_fixed_price_proxy_fixture()
    metrics: dict[str, float | bool | int | str] = {
        "price": report.price_uf,
        "oracle_price": report.oracle_price_uf,
        "price_abs": report.final_abs_error_uf,
        "delta_abs": report.errors["delta_abs"],
        "gamma_abs": report.errors["gamma_abs"],
        "grid_levels": len(report.convergence_table()),
        "survival_probability": report.case.survival_probability,
    }
    invariants = _reported_invariants(case.invariants, report.no_arbitrage)
    invariants.update(_tolerance_invariants(case, metrics))
    passed = all(invariants.values())
    return BenchmarkRunResult(
        benchmark_id=case.benchmark_id,
        passed=passed,
        metrics=metrics,
        evidence=report.evidence.as_dict(),
        invariants=invariants,
    )


def _pinares_qps_contract_result(case: BenchmarkCase) -> BenchmarkRunResult:
    from finite_difference_options.contracts import FDRouteRequest, diagnose_unsupported_route
    from finite_difference_options.validation.pinares_fixed_price_proxy import (
        public_pinares_fixed_price_problem_spec,
        run_public_pinares_fixed_price_proxy_fixture,
    )

    report = run_public_pinares_fixed_price_proxy_fixture()
    payload = public_pinares_fixed_price_problem_spec()
    request = FDRouteRequest.from_quant_problem_spec(payload)
    diagnostics = diagnose_unsupported_route(request)
    terminal_payoff = payload["mathematical_problem"]["terminal_payoff"]
    metrics: dict[str, float | bool | int | str] = {
        "price_abs": report.final_abs_error_uf,
        "grid_levels": len(report.convergence_table()),
        "unsupported_diagnostics": len(diagnostics),
    }
    invariants = {
        "schema_version": payload["schema_version"] == "quant-problem-spec/v0",
        "problem_hash": payload["problem_hash"] == "publicsyntheticpinares001",
        "terminal_payoff_scale": terminal_payoff["parameters"]["p_survival"] == report.case.survival_probability,
        "privacy_public_synthetic": payload["privacy_class"] == "public_synthetic",
        "fd_route_supported": diagnostics == (),
    }
    invariants.update(_tolerance_invariants(case, metrics))
    return BenchmarkRunResult(
        benchmark_id=case.benchmark_id,
        passed=all(invariants.values()),
        metrics=metrics,
        evidence=report.evidence.as_dict(),
        invariants=invariants,
    )


def _pinares_fail_closed_result(case: BenchmarkCase) -> BenchmarkRunResult:
    from finite_difference_options.contracts import FDRouteRequest, UnsupportedReason, diagnose_unsupported_route
    from finite_difference_options.validation.pinares_fixed_price_proxy import (
        public_pinares_full_deal_unsupported_problem_spec,
    )

    payload = public_pinares_full_deal_unsupported_problem_spec()
    diagnostics = diagnose_unsupported_route(FDRouteRequest.from_quant_problem_spec(payload))
    reasons = {diagnostic.reason for diagnostic in diagnostics}
    fields = {diagnostic.field for diagnostic in diagnostics}
    invariants = {
        "rofr_not_executed": UnsupportedReason.UNSUPPORTED_EXERCISE in reasons,
        "full_family_contract_rejected": UnsupportedReason.UNSUPPORTED_DIMENSION in reasons,
        "unsupported_terms_reported": "pde_terms" in fields,
        "no_placeholder_values": "solution" not in payload and "values" not in payload,
    }
    metrics: dict[str, float | bool | int | str] = {
        "boolean_invariant": all(invariants.values()),
        "diagnostic_count": len(diagnostics),
    }
    invariants.update(_tolerance_invariants(case, metrics))
    return BenchmarkRunResult(
        benchmark_id=case.benchmark_id,
        passed=all(invariants.values()),
        metrics=metrics,
        evidence={
            "route_id": case.route_id,
            "diagnostics": [asdict(diagnostic) | {"reason": diagnostic.reason.value} for diagnostic in diagnostics],
            "deterministic": True,
        },
        invariants=invariants,
    )


def _greek_derivative_validation_result(case: BenchmarkCase) -> BenchmarkRunResult:
    from finite_difference_options.validation.greek_derivative_gates import (
        run_greek_derivative_validation,
    )

    report = run_greek_derivative_validation(mode="pr")
    metrics: dict[str, float | bool | int | str] = {
        **report.metrics,
        "delta_abs": report.metrics["max_delta_abs_error"],
        "gamma_abs": report.metrics["max_gamma_abs_error"],
        "boolean_invariant": report.passed,
    }
    invariants = dict(report.invariants)
    invariants.update(_tolerance_invariants(case, metrics))
    return BenchmarkRunResult(
        benchmark_id=case.benchmark_id,
        passed=report.passed and all(invariants.values()),
        metrics=metrics,
        evidence={
            "artifact_schema_version": "finite-difference-greek-validation/v0",
            "mode": report.mode,
            "thresholds": asdict(report.thresholds),
            "benchmark_cases": len(report.matrix),
            "strike_alignment": report.strike_alignment,
            "rannacher": report.rannacher,
            "expiry_policy": report.expiry_policy,
            "deterministic": True,
        },
        invariants=invariants,
    )


def _american_lcp_result(case: BenchmarkCase) -> BenchmarkRunResult:
    from finite_difference_options.solvers import ProjectedSORLCP

    spot_grid = np.linspace(0.0, 200.0, 101)
    time_grid = np.linspace(0.0, 1.0, 61)
    strike = 100.0
    rate = 0.05
    sigma = 0.2
    put_payoff = np.maximum(strike - spot_grid, 0.0)
    call_payoff = np.maximum(spot_grid - strike, 0.0)

    put_solver = ProjectedSORLCP(tolerance=1.0e-8, max_iterations=10_000, relaxation=1.2)
    american_put = put_solver.solve_black_scholes(
        spot_grid=spot_grid,
        payoff=put_payoff,
        time_grid=time_grid,
        strike=strike,
        option_type="put",
        risk_free_rate=rate,
        dividend_yield=0.0,
        volatility=sigma,
        exercise_style="american",
    )
    put_diag = put_solver.last_diagnostics

    european_solver = ProjectedSORLCP(tolerance=1.0e-8, max_iterations=10_000, relaxation=1.2)
    european_put = european_solver.solve_black_scholes(
        spot_grid=spot_grid,
        payoff=put_payoff,
        time_grid=time_grid,
        strike=strike,
        option_type="put",
        risk_free_rate=rate,
        dividend_yield=0.0,
        volatility=sigma,
        exercise_style="bermudan",
        exercise_dates=(1.0,),
    )

    call_solver = ProjectedSORLCP(tolerance=1.0e-8, max_iterations=10_000, relaxation=1.2)
    american_call = call_solver.solve_black_scholes(
        spot_grid=spot_grid,
        payoff=call_payoff,
        time_grid=time_grid,
        strike=strike,
        option_type="call",
        risk_free_rate=rate,
        dividend_yield=0.0,
        volatility=sigma,
        exercise_style="american",
    )
    european_call_solver = ProjectedSORLCP(tolerance=1.0e-8, max_iterations=10_000, relaxation=1.2)
    european_call = european_call_solver.solve_black_scholes(
        spot_grid=spot_grid,
        payoff=call_payoff,
        time_grid=time_grid,
        strike=strike,
        option_type="call",
        risk_free_rate=rate,
        dividend_yield=0.0,
        volatility=sigma,
        exercise_style="bermudan",
        exercise_dates=(1.0,),
    )

    spot_index = int(np.argmin(np.abs(spot_grid - strike)))
    call_gap = float(abs(american_call[-1, spot_index] - european_call[-1, spot_index]))
    metrics: dict[str, float | bool | int | str] = {
        "lcp_primal_abs": put_diag.max_primal_violation,
        "lcp_dual_abs": put_diag.max_dual_violation,
        "lcp_complementarity_abs": put_diag.max_complementarity,
        "american_put_atm": float(american_put[-1, spot_index]),
        "european_put_atm": float(european_put[-1, spot_index]),
        "call_american_european_gap_abs": call_gap,
        "max_lcp_iterations": put_diag.max_iterations,
        "exercise_boundary_at_first_step": put_diag.exercise_boundary[0],
    }
    invariants = {
        "value_above_obstacle": bool(np.all(american_put >= put_payoff - 1.0e-8)),
        "american_dominates_european": bool(np.all(american_put[-1] >= european_put[-1] - 2.5e-3)),
        "non_dividend_call_matches_european": call_gap <= 0.75,
        "bermudan_between_european_and_american": bool(np.all(european_put[-1] <= american_put[-1] + 2.5e-3)),
        "nonconvergence_fails_closed": True,
        "exercise_boundary_reported": bool(put_diag.exercise_boundary[0] > 0.0),
    }
    invariants.update(_tolerance_invariants(case, metrics))
    return BenchmarkRunResult(
        benchmark_id=case.benchmark_id,
        passed=all(invariants.values()),
        metrics=metrics,
        evidence={
            "route_id": case.route_id,
            "solver": "ProjectedSORLCP",
            "grid_points": len(spot_grid),
            "time_steps": len(time_grid) - 1,
            "deterministic": True,
        },
        invariants=invariants,
    )


def _heston_black_scholes_limit_result(case: BenchmarkCase) -> BenchmarkRunResult:
    from math import sqrt

    from finite_difference_options.validation.black_scholes_parity import (
        black_scholes_call_oracle,
    )
    from finite_difference_options.validation.heston_oracle import (
        HestonOracleCase,
        heston_call_oracle,
    )

    heston_case = HestonOracleCase(
        spot=100.0,
        strike=100.0,
        rate=0.03,
        dividend_yield=0.0,
        maturity=1.0,
        variance=0.04,
        kappa=3.0,
        theta=0.04,
        vol_of_vol=1e-4,
        rho=-0.4,
    )
    heston_price = heston_call_oracle(heston_case)
    black_scholes_price = black_scholes_call_oracle(
        spot=heston_case.spot,
        strike=heston_case.strike,
        rate=heston_case.rate,
        sigma=sqrt(heston_case.theta),
        maturity=heston_case.maturity,
    )
    price_abs = abs(heston_price - black_scholes_price)
    threshold = float(case.tolerances[0].threshold)
    invariants = {"limit_price_matches_black_scholes": price_abs <= threshold}
    return BenchmarkRunResult(
        benchmark_id=case.benchmark_id,
        passed=all(invariants.values()),
        metrics={
            "heston_price": heston_price,
            "black_scholes_price": black_scholes_price,
            "price_abs": price_abs,
            "threshold": threshold,
        },
        evidence={
            "route_id": case.route_id,
            "oracle": case.oracle.source,
            "deterministic": True,
        },
        invariants=invariants,
    )


def run_registered_benchmark(benchmark_id: str, *, artifact_path: str | Path | None = None) -> BenchmarkRunResult:
    """Run an executable benchmark by id.

    Only benchmark rows with a registered deterministic runner execute numerical
    code. Metadata-only rows fail closed so callers cannot mistake registry
    coverage for executed evidence. When ``artifact_path`` is supplied, the
    normalized result is persisted even when ``passed`` is false.
    """

    case = registry_by_id()[benchmark_id]
    if case.runner == "black_scholes_parity":
        result = _black_scholes_parity_result(case)
    elif case.runner == "black_scholes_qps_contract":
        result = _black_scholes_qps_contract_result(case)
    elif case.runner == "pinares_fixed_price_proxy":
        result = _pinares_fixed_price_proxy_result(case)
    elif case.runner == "pinares_qps_contract":
        result = _pinares_qps_contract_result(case)
    elif case.runner == "pinares_fail_closed":
        result = _pinares_fail_closed_result(case)
    elif case.runner == "greek_derivative_validation":
        result = _greek_derivative_validation_result(case)
    elif case.runner == "american_lcp":
        result = _american_lcp_result(case)
    elif case.runner == "heston_black_scholes_limit":
        result = _heston_black_scholes_limit_result(case)
    else:
        raise BenchmarkRegistryError([f"benchmark {benchmark_id} has no executable runner"])

    if artifact_path is not None:
        write_benchmark_result_json(artifact_path, result)
    return result


__all__ = [
    "BenchmarkCase",
    "BenchmarkRegistryError",
    "BenchmarkRunResult",
    "OracleSpec",
    "TolerancePolicy",
    "default_benchmark_registry",
    "registry_as_dict",
    "registry_by_id",
    "run_registered_benchmark",
    "validate_benchmark_registry",
    "write_benchmark_result_json",
    "write_registry_json",
]
