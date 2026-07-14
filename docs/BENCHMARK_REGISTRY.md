# Finite-difference benchmark registry

Issue: [googa27/finite_difference_options#49](https://github.com/googa27/finite_difference_options/issues/49)
Schema: `finite-difference-benchmark-registry/v0`
Executable source: `finite_difference_options.validation.benchmark_registry`
Static fixture: `tests/fixtures/fd_benchmark_registry_v1.json`

The registry is the evidence index behind `docs/CAPABILITY_MATRIX.md`. A capability row is not a production claim just because a route can run; it needs a versioned benchmark/evidence ID with explicit model, route, oracle, tolerance, invariants, fixture paths and resource policy.

## Evidence layers

| Layer | Purpose | Example |
|---|---|---|
| Analytical oracle | Compare the FD route to a closed-form or independent semi-analytical result. | `BS-CALL-PARITY-V0`, `HESTON-ORACLE-V0` |
| Manufactured/fixture evidence | Pin expected behavior for a route-specific contract before broader production claims. | `RANNACHER-GAMMA-V0`, `BOUNDARY-MODEL-AWARE-V0`, `AMERICAN-LCP-V0` |
| No-arbitrage invariants | Guard positivity, bounds, Delta/Gamma signs and parity-style properties. | `BS-CALL-PARITY-V0` |
| Route parity | Assert that a generic/cross-repo contract and a specialized local route agree on semantics. | `QPS-VANILLA-CALL-V0`, `HESTON-BS-LIMIT-V0` |
| Capability gate | Fail closed for unsupported products, regulatory reports, factors or solver features. | `FACTOR-ROLE-COMPAT-V0`, `REG-FAIL-CLOSED-V0` |
| Smoke evidence | Prove shape/orientation/finite values only; this is not convergence maturity. | `ADI-SMOKE-V0`, `HESTON-SMOKE-DOCSTRING-V0` |

## Validation policy

`validate_benchmark_registry()` enforces:

- unique versioned benchmark IDs ending in `-Vn`;
- route/model/instrument/state/grid/time metadata on every case;
- validated and experimental rows must cite an oracle or fixture;
- validated rows require tolerance policies;
- no-arbitrage and route-parity rows must name invariants;
- all capability-matrix evidence IDs must be represented by registry rows in tests.

Only rows with an explicit runner can execute numerical code through `run_registered_benchmark(...)`. Metadata-only rows fail closed if called directly so no consumer mistakes registry coverage for executed evidence.

`BS-CALL-PARITY-V0`, `QPS-VANILLA-CALL-V0`, `VQPW-FD-COMPILED-PDE-BS-CALL-V0`, `PINARES-FD-FIXED-PRICE-PROXY-V0`, `PINARES-QPS-FIXED-PRICE-PROXY-V0`, `PINARES-FD-FAIL-CLOSED-V0`, `FD-GREEKS-VALIDATION-V0`, `AMERICAN-LCP-V0` and `HESTON-BS-LIMIT-V0` execute deterministic runners through `run_registered_benchmark(...)`.

`FD-GREEKS-VALIDATION-V0` writes the Issue #58 Greek derivative validation artifact: PR-fast CI covers 12 Black-Scholes cases across moneyness, maturity and volatility; scheduled broad CI extends the matrix and uploads `fd-greek-derivative-validation-broad.json`.

The issue #142 CLI runner `fd-options validation run-benchmark fd-bs-001 --out <path>` is intentionally artifact-oriented rather than a registry-row alias: it recomputes the exact compiled Black-Scholes native FD route, analytical price/Delta/Gamma, spatial and temporal three-level tables, manufactured-solution residual evidence, no-arbitrage checks, explicit boundary schedule, independent sign/source/reaction/boundary perturbations, and request/config/convention/result/evidence hashes. `validate_fd_bs_verification_bundle(...)` recomputes hashes and numerical truth after hash recomputation; stored `passed` booleans are not trusted.

## Registry rows

The static registry fixture currently contains these versioned evidence IDs:

| Benchmark ID | Evidence role |
|---|---|
| `BS-CALL-PARITY-V0` | executable Black-Scholes analytical parity, Greeks, convergence and no-arbitrage runner |
| `QPS-VANILLA-CALL-V0` | executable QuantProblemSpec/static-fixture route-parity runner |
| `VQPW-FD-COMPILED-PDE-BS-CALL-V0` | executable compiled `pde_ir.v0` fixture adapter screening/solve runner for issue #141 |
| `BOUNDARY-MODEL-AWARE-V0` | typed model-aware boundary/reaction gate |
| `REACTION-INDEPENDENT-V0` | reaction/discount coefficient independence gate |
| `RANNACHER-GAMMA-V0` | Rannacher startup and kinked-payoff Gamma evidence |
| `AMERICAN-LCP-V0` | American/Bermudan obstacle LCP complementarity, ordering, exercise-boundary and nonconvergence diagnostics |
| `HESTON-SMOKE-DOCSTRING-V0` | Heston ADI smoke/shape/finite-value evidence |
| `HESTON-ORACLE-V0` | semi-analytical Heston oracle evidence |
| `HESTON-BS-LIMIT-V0` | executable Heston-to-Black-Scholes limit runner |
| `HESTON-VARIANCE-BOUNDARY-V0` | Heston variance-boundary and Feller-policy diagnostic evidence |
| `GRID-LOCAL-METRICS-V0` | typed AxisGrid/TensorGrid validation, local nonuniform derivative metrics, log/clustered factories and ADI grid diagnostics |
| `FD-GREEKS-NONUNIFORM-V0` | nonuniform requested-coordinate Delta/Gamma stencil, interpolation, refinement/reference-error and expiry-kink diagnostics |
| `FD-GREEKS-VALIDATION-V0` | executable derivative convergence, strike-alignment sensitivity, expiry-kink and Rannacher stability artifact gate for #58 |
| `ADI-SMOKE-V0` | ADI finite-value/orientation smoke evidence |
| `ADI-OPERATOR-SPLIT-V0` | ADI operator split coefficient evidence |
| `FACTOR-ROLE-COMPAT-V0` | factor-role payoff compatibility fail-closed gate |
| `DOCS-README-SMOKE-V0` | README/docs smoke and maturity-disclosure evidence |
| `API-REQUEST-GUARDS-V0` | API request guards and convergence non-claim metadata |
| `FD-CANONICAL-INVENTORY-V0` | canonical implementation inventory and duplicate-stack architecture gate |
| `REG-FAIL-CLOSED-V0` | regulatory/reporting endpoint fail-closed gate |

## Successor work

The registry deliberately records several experimental/smoke rows. Closing issue #49 does not graduate ADI/Heston/basket/regulatory routes to production; it makes their evidence status explicit and machine-checkable. Subsequent issues should add executable runners for convergence, stability, multidimensional American/LCP, Heston ADI parity, and cross-backend route parity before changing maturity in `docs/CAPABILITY_MATRIX.md`.
