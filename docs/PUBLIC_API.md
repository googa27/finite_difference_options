# Public API — Finite Difference Options

Status: v0.1 audit snapshot. This document is a documentation contract for the import surface that this checkout intentionally exposes today; it does not promote experimental routes to production maturity.

The package is a typed finite-difference mathematics library. The package root intentionally exports no convenience symbols so optional adapters and legacy facades do not load accidentally. Public users should import from the stable subpackages below and consult `docs/CAPABILITY_MATRIX.md` before treating a route as validated.

## Capability boundaries

- Numerical core: grids, stochastic-process coefficient contracts, boundary algebra, theta-family stepping, ADI/LCP solvers, Greeks, validation oracles and diagnostics.
- Integration seams: exact public-synthetic QuantProblemSpec and compiled-PDE adapters, plus the canonical Haircut backend entry point `haircut.solver_backends`.
- Optional outer surfaces: CLI, API, plotting and UI profiles are extras; the core package must not require them.
- Unsupported routes fail closed before grid/operator construction. Labels such as dimension, option type, or model name are not enough to infer coefficients, boundaries or exercise policy.

## Import policy

Use explicit subpackage imports:

```python
from finite_difference_options.pricing import create_log_grid, create_unified_european_call
from finite_difference_options.processes import create_black_scholes_process
from finite_difference_options.solvers import BandedOperatorCache
from finite_difference_options.integrations import solve_public_quant_problem_spec
```

Do not import through `src.*`, repository-relative paths, or private modules. API stability is strongest for dataclasses/functions exported through package `__all__` values below and weakest for modules that are only retained as legacy compatibility shims.

Import order is part of this contract. A fresh interpreter may import `finite_difference_options.solvers.base` before `finite_difference_options.pricing`; type-only references from the solver layer must not eagerly initialize the pricing facade. `tests/architecture/test_isolated_public_imports.py` verifies both the isolated solver import and the documented pricing-facade exports in one subprocess.

## Runtime export manifest

The following lists are synchronized by `tests/architecture/test_public_docs_contract.py` against every package facade with `__all__` discovered from the machine-readable setuptools package contract in `pyproject.toml`.

### `finite_difference_options`

Package-root convenience exports are intentionally empty.

<!-- public-api:finite_difference_options:start -->
<!-- public-api:finite_difference_options:end -->

### `finite_difference_options.boundary_conditions`

<!-- public-api:finite_difference_options.boundary_conditions:start -->
- `BlackScholesBoundaryBuilder`
- `BoundaryResolution`
- `BoundarySpec`
- `HestonBoundaryBuilder`
<!-- public-api:finite_difference_options.boundary_conditions:end -->

### `finite_difference_options.contracts`

<!-- public-api:finite_difference_options.contracts:start -->
- `CapabilityStatus`
- `DEFAULT_FD_CAPABILITY_MANIFEST`
- `FDCapabilityManifest`
- `FDRouteRequest`
- `UnsupportedReason`
- `UnsupportedRouteDiagnostic`
- `UnsupportedRouteError`
- `SolverEvidence`
- `diagnose_unsupported_route`
- `ensure_route_supported`
- `finite_difference_formula_bundle`
- `formula_bundle_json`
- `validate_formula_bundle`
<!-- public-api:finite_difference_options.contracts:end -->

### `finite_difference_options.exceptions`

<!-- public-api:finite_difference_options.exceptions:start -->
- `FiniteDifferenceError`
- `ValidationError`
- `GridError`
- `ModelError`
- `InstrumentError`
- `PricingError`
- `BoundaryConditionError`
- `TimeSteppingError`
- `ConvergenceError`
<!-- public-api:finite_difference_options.exceptions:end -->

### `finite_difference_options.greeks`

<!-- public-api:finite_difference_options.greeks:start -->
- `FiniteDifferenceGreeks`
- `GreekEstimate`
- `GreeksCalculator`
- `FDCalculator1D`
- `FDCalculator2D`
- `GreeksCalculatorFactory`
<!-- public-api:finite_difference_options.greeks:end -->

### `finite_difference_options.grids`

<!-- public-api:finite_difference_options.grids:start -->
- `AxisGrid`
- `TensorGrid`
- `as_axis_grid`
- `log_uniform_axis`
- `sinh_clustered_axis`
- `strike_centered_axis`
- `tanh_clustered_axis`
- `uniform_axis`
- `variance_boundary_axis`
<!-- public-api:finite_difference_options.grids:end -->

### `finite_difference_options.instruments`

<!-- public-api:finite_difference_options.instruments:start -->
- `Instrument`
- `EuropeanOption`
- `EuropeanCall`
- `EuropeanPut`
- `SpatialOperator`
<!-- public-api:finite_difference_options.instruments:end -->

### `finite_difference_options.integrations`

<!-- public-api:finite_difference_options.integrations:start -->
- `CompiledPDEAdapterError`
- `CompiledPDEDiagnostic`
- `CompiledPDEScreeningResult`
- `CompiledPDESolveResult`
- `ContractMajorMismatchError`
- `FDBackendScreeningResult`
- `FiniteDifferenceHaircutBackend`
- `HaircutBackendSolveResult`
- `HaircutProtocolUnavailableError`
- `PublicFDSolverResult`
- `ReleasedFDSolverContract`
- `create_backend`
- `load_compiled_pde_json`
- `packaged_compiled_black_scholes_fixture`
- `released_fd_solver_contract`
- `screen_compiled_pde_payload`
- `solve_compiled_pde_payload`
- `solve_public_quant_problem_spec`
<!-- public-api:finite_difference_options.integrations:end -->

The `compiled_pde_adapter` public module retains its existing imports and signatures. Its private DTO and validation modules are implementation details. The extraction preserves exact v0 fixture screening and numerical behavior; no new route or schema is enabled.

### `finite_difference_options.models`

<!-- public-api:finite_difference_options.models:start -->
- `Market`
- `StochasticProcess`
- `AffineProcess`
- `NonAffineProcess`
- `GeometricBrownianMotion`
- `OrnsteinUhlenbeck`
- `CoxIngersollRoss`
- `HestonModel`
- `ConstantElasticityVariance`
- `SABRModel`
- `create_gbm`
- `create_ou`
- `create_cir`
- `create_heston`
- `create_cev`
- `create_sabr`
<!-- public-api:finite_difference_options.models:end -->

### `finite_difference_options.plotting`

<!-- public-api:finite_difference_options.plotting:start -->
- `PlotOptions`
- `Plotter`
- `BasePlotter`
- `MatplotlibSeabornPlotter`
- `get_plotter`
- `PlotlyPlotter`
- `map_matplotlib_to_plotly`
- `DEFAULT_SEQUENTIAL`
- `DEFAULT_DIVERGING`
- `symmetric_bounds`
<!-- public-api:finite_difference_options.plotting:end -->

### `finite_difference_options.pricing`

<!-- public-api:finite_difference_options.pricing:start -->
- `GridParameters`
- `PDEModel`
- `PricingEngine`
- `PricingResult`
- `UnifiedPricingEngine`
- `create_default_pricing_engine`
- `create_linear_grid`
- `create_log_grid`
- `create_unified_pricing_engine`
- `GridResult`
- `OptionPricer`
- `UnifiedInstrument`
- `SpreadOption`
- `StandardBasketOption`
- `UnifiedEuropeanOption`
- `UnifiedAmericanOption`
- `UnifiedBermudanOption`
- `UnifiedBasketOption`
- `create_spread_call`
- `create_spread_put`
- `create_standard_basket_call`
- `create_standard_basket_put`
- `create_unified_european_call`
- `create_unified_european_put`
- `create_unified_american_call`
- `create_unified_american_put`
- `create_unified_bermudan_call`
- `create_unified_bermudan_put`
- `create_unified_basket_call`
- `BlackScholesPDE`
- `BondCashFlow`
- `CallScheduleEntry`
- `CallableBondExerciseRecord`
- `CallableBondPDEModel`
<!-- public-api:finite_difference_options.pricing:end -->

### `finite_difference_options.pricing.engines`

<!-- public-api:finite_difference_options.pricing.engines:start -->
- `GridParameters`
- `PDEModel`
- `PricingEngine`
- `PricingResult`
- `UnifiedPricingEngine`
- `create_default_pricing_engine`
- `create_linear_grid`
- `create_log_grid`
- `create_unified_pricing_engine`
- `BlackScholesPDE`
- `BondCashFlow`
- `CallScheduleEntry`
- `CallableBondExerciseRecord`
- `CallableBondPDEModel`
<!-- public-api:finite_difference_options.pricing.engines:end -->

### `finite_difference_options.pricing.instruments`

<!-- public-api:finite_difference_options.pricing.instruments:start -->
- `UnifiedInstrument`
- `SpreadOption`
- `StandardBasketOption`
- `UnifiedEuropeanOption`
- `UnifiedBasketOption`
- `create_spread_call`
- `create_spread_put`
- `create_standard_basket_call`
- `create_standard_basket_put`
- `create_unified_european_call`
- `create_unified_european_put`
- `create_unified_basket_call`
- `create_unified_basket_put`
- `PayoffCalculator`
- `EuropeanPayoffCalculator`
- `BasketPayoffCalculator`
- `PayoffCalculatorFactory`
<!-- public-api:finite_difference_options.pricing.instruments:end -->

### `finite_difference_options.pricing.workflows`

<!-- public-api:finite_difference_options.pricing.workflows:start -->
- `GridResult`
- `OptionPricer`
<!-- public-api:finite_difference_options.pricing.workflows:end -->

### `finite_difference_options.processes`

<!-- public-api:finite_difference_options.processes:start -->
- `StochasticProcess`
- `AffineProcess`
- `NonAffineProcess`
- `AffineCovarianceForm`
- `CovarianceValidationResult`
- `FactorRole`
- `ProcessCoefficientEvaluation`
- `ProcessDimension`
- `ProcessFactor`
- `ProcessFactorMetadata`
- `ProcessType`
- `GeometricBrownianMotion`
- `OrnsteinUhlenbeck`
- `CoxIngersollRoss`
- `HestonModel`
- `create_black_scholes_process`
- `create_vasicek_process`
- `create_cir_process`
- `create_standard_heston`
- `FellerDiagnostics`
- `FellerPolicy`
- `ZeroBoundaryClassification`
- `diagnose_feller_condition`
- `ConstantElasticityVariance`
- `SABRModel`
- `create_cev_process`
- `create_sabr_model`
<!-- public-api:finite_difference_options.processes:end -->

### `finite_difference_options.risk`

<!-- public-api:finite_difference_options.risk:start -->
- `Trade`
- `RiskFactor`
- `Exposure`
- `NotImplementedForStandard`
- `RegulatoryStandard`
- `exposures_to_crif`
- `calculate_cuso`
- `calculate_basel`
- `calculate_frtb`
<!-- public-api:finite_difference_options.risk:end -->

### `finite_difference_options.solvers`

<!-- public-api:finite_difference_options.solvers:start -->
- `ADISolver`
- `BandedOperatorCache`
- `CachedBlackScholesFiniteDifferenceSolver`
- `CrankNicolson`
- `ExplicitEuler`
- `FiniteDifferenceSolver`
- `LCPDiagnostics`
- `LCPLevelDiagnostics`
- `OperatorCacheInfo`
- `PDESolver`
- `ProjectedSORLCP`
- `RannacherCrankNicolson`
- `ThetaMethod`
- `ThetaSubstepRecord`
- `TimeStepper`
- `create_adi_solver`
- `create_default_solver`
<!-- public-api:finite_difference_options.solvers:end -->

### `finite_difference_options.validation`

<!-- public-api:finite_difference_options.validation:start -->
- `BenchmarkCase`
- `BenchmarkRegistryError`
- `BenchmarkRunResult`
- `OracleSpec`
- `TolerancePolicy`
- `PINARES_FAIL_CLOSED_BENCHMARK_ID`
- `PINARES_FIXED_PRICE_PROXY_BENCHMARK_ID`
- `PINARES_FIXED_PRICE_PROXY_BENCHMARK_IDS`
- `PINARES_FIXED_PRICE_PROXY_FIXTURE_ID`
- `PINARES_FIXED_PRICE_PROXY_PROBLEM_HASH`
- `PINARES_FIXED_PRICE_PROXY_PROBLEM_ID`
- `PINARES_FIXED_PRICE_PROXY_ROUTE_ID`
- `PINARES_QPS_CONTRACT_BENCHMARK_ID`
- `PinaresFixedPriceProxyCase`
- `PinaresFixedPriceProxyReport`
- `default_benchmark_registry`
- `export_public_pinares_fixed_price_proxy_fixture_json`
- `public_pinares_fixed_price_problem_spec`
- `public_pinares_full_deal_unsupported_problem_spec`
- `registry_as_dict`
- `registry_by_id`
- `run_public_pinares_fixed_price_proxy_fixture`
- `run_registered_benchmark`
- `validate_benchmark_registry`
- `write_benchmark_result_json`
- `write_registry_json`
- `validate_positive`
- `validate_non_negative`
- `validate_probability`
- `validate_grid_parameters`
- `validate_option_parameters`
- `validate_model_parameters`
- `validate_array`
- `validate_spot_price`
<!-- public-api:finite_difference_options.validation:end -->

## Entry points and extras

- CLI script:

  ```text
  fd-options -> finite_difference_options.cli.main:app
  ```

- Haircut backend entry point:

  ```text
  group: haircut.solver_backends
  name: finite_difference_options
  target: finite_difference_options.integrations.haircut_backend:create_backend
  ```

- Optional extras: `api`, `cli`, `viz`, `ui`, `validation`, `build`, `audit`, and `dev`.

## Compatibility notes

The `pricing`, `instruments`, `exceptions`, `models`, and `risk` surfaces include legacy compatibility names. They remain public while tests cover them, but new integrations should prefer typed contracts, explicit problem payloads, and the capability manifest. A public symbol does not imply every model/product combination is mature; maturity is governed by benchmark and capability docs.
