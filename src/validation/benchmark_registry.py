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
OracleKind = Literal[
    "analytical", "manufactured", "fixture", "regression", "capability", "none"
]
MetricKind = Literal[
    "price_abs",
    "price_rel",
    "delta_abs",
    "gamma_abs",
    "convergence_order",
    "boolean_invariant",
    "route_parity_abs",
    "residual_norm",
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
            errors.append(
                f"benchmark_id is not versioned kebab id: {self.benchmark_id}"
            )
        if not self.title.strip():
            errors.append(f"title required for {self.benchmark_id}")
        if not self.route_id.strip():
            errors.append(f"route_id required for {self.benchmark_id}")
        if not self.model.strip():
            errors.append(f"model required for {self.benchmark_id}")
        if not self.instrument.strip():
            errors.append(f"instrument required for {self.benchmark_id}")
        if self.status in {"validated", "experimental"} and self.oracle.kind == "none":
            errors.append(
                f"{self.benchmark_id} claims {self.status} without an oracle/fixture"
            )
        if self.status == "validated" and not self.tolerances:
            errors.append(
                f"validated benchmark {self.benchmark_id} requires tolerances"
            )
        if self.family in {"no_arbitrage", "route_parity"} and not self.invariants:
            errors.append(
                f"{self.family} benchmark {self.benchmark_id} requires invariants"
            )
        if (
            self.status == "validated"
            and self.family == "route_parity"
            and self.runner is None
        ):
            errors.append(
                f"validated route-parity benchmark {self.benchmark_id} requires an executable runner"
            )
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
            capability_rows=(
                "1D Black-Scholes European call value on uniform/log-uniform grids",
            ),
            fixture_paths=("tests/fixtures/arxiv_lab_bs_oracle_v1.json",),
            issue_refs=(
                "googa27/finite_difference_options#49",
                "googa27/finite_difference_options#59",
            ),
            resource_policy={"public_synthetic": "true"},
            runner="black_scholes_qps_contract",
            notes=(
                "Metadata registry row for cross-repo fixture parity; execution is covered by "
                "BS-CALL-PARITY-V0 tests."
            ),
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
            capability_rows=(
                "Rannacher startup before Crank-Nicolson for kinked payoffs",
            ),
            fixture_paths=("tests/test_rannacher_startup.py",),
            issue_refs=("googa27/finite_difference_options#49",),
            resource_policy={"deterministic": "true"},
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
            fixture_paths=("src/validation/heston_oracle.py",),
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
            fixture_paths=("src/validation/heston_oracle.py",),
            issue_refs=("googa27/finite_difference_options#49",),
            resource_policy={"deterministic": "true"},
            runner="heston_black_scholes_limit",
        ),
        BenchmarkCase(
            benchmark_id="HESTON-VARIANCE-BOUNDARY-V0",
            title="Heston variance-boundary diagnostics",
            family="capability_gate",
            status="validated",
            route_id="fd.heston.variance_boundary_diagnostics",
            model="Heston stochastic volatility",
            instrument="European call smoke route",
            state_convention="variance boundary has explicit lower/upper diagnostics",
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
            invariants=("variance_nonnegative", "boundary_diagnostics_recorded"),
            capability_rows=("Heston semi-analytical European call oracle",),
            fixture_paths=("src/validation/heston_oracle.py",),
            issue_refs=("googa27/finite_difference_options#49",),
            resource_policy={"deterministic": "true"},
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
    registry: tuple[BenchmarkCase, ...] | None = None
) -> dict[str, BenchmarkCase]:
    """Return registry rows keyed by benchmark id."""

    rows = registry if registry is not None else default_benchmark_registry()
    return {case.benchmark_id: case for case in rows}


def validate_benchmark_registry(
    registry: tuple[BenchmarkCase, ...] | None = None
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
    registry: tuple[BenchmarkCase, ...] | None = None
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


def write_registry_json(
    path: str | Path, registry: tuple[BenchmarkCase, ...] | None = None
) -> None:
    """Write the registry JSON payload with stable ordering and indentation."""

    import json

    payload = registry_as_dict(registry)
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_benchmark_result_json(path: str | Path, result: BenchmarkRunResult) -> None:
    """Persist one benchmark run result for CI/artifact triage."""

    import json

    Path(path).write_text(
        json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _black_scholes_parity_result(case: BenchmarkCase) -> BenchmarkRunResult:
    from src.validation.black_scholes_parity import (
        run_public_black_scholes_parity_fixture,
    )

    report = run_public_black_scholes_parity_fixture()
    invariants = {
        name: bool(report.no_arbitrage[name])
        for name in case.invariants
        if name in report.no_arbitrage
    }
    passed = report.converged and all(invariants.values())
    return BenchmarkRunResult(
        benchmark_id=case.benchmark_id,
        passed=passed,
        metrics={
            "price": report.price,
            "oracle_price": report.oracle_price,
            "price_abs": report.final_abs_error,
            "delta_abs": report.errors["delta_abs"],
            "gamma_abs": report.errors["gamma_abs"],
            "grid_levels": len(report.convergence_table()),
        },
        evidence=report.evidence.as_dict(),
        invariants=invariants,
    )


def _black_scholes_qps_contract_result(case: BenchmarkCase) -> BenchmarkRunResult:
    from src.validation.black_scholes_parity import (
        run_public_black_scholes_parity_fixture,
    )

    report = run_public_black_scholes_parity_fixture()
    payload = report.as_dict()
    problem = payload["problem_spec"]
    result = payload["result_export"]
    typed_boundary = result["boundary"]["typed"]
    invariants = {
        "schema_version": problem["schema_version"] == "quant-problem-spec/v0",
        "problem_hash": bool(problem["problem_hash"]),
        "typed_boundary": all("boundary_type" in item for item in typed_boundary),
        "calendar_time_orientation": result["time_axis"]["direction"] == "decreasing",
    }
    passed = report.converged and all(invariants.values())
    return BenchmarkRunResult(
        benchmark_id=case.benchmark_id,
        passed=passed,
        metrics={
            "price_abs": report.final_abs_error,
            "grid_levels": len(report.convergence_table()),
            "typed_boundary_count": len(typed_boundary),
        },
        evidence=report.evidence.as_dict(),
        invariants=invariants,
    )


def _heston_black_scholes_limit_result(case: BenchmarkCase) -> BenchmarkRunResult:
    from math import sqrt

    from src.validation.black_scholes_parity import black_scholes_call_oracle
    from src.validation.heston_oracle import HestonOracleCase, heston_call_oracle

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


def run_registered_benchmark(
    benchmark_id: str, *, artifact_path: str | Path | None = None
) -> BenchmarkRunResult:
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
    elif case.runner == "heston_black_scholes_limit":
        result = _heston_black_scholes_limit_result(case)
    else:
        raise BenchmarkRegistryError(
            [f"benchmark {benchmark_id} has no executable runner"]
        )

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
