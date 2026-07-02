# FD capability and maturity matrix

This is the authoritative public status matrix for `finite_difference_options` until the package has a generated API-reference pipeline. A README example or application endpoint is not a production claim unless the route below is marked `production` or `validated` and cites evidence.

Status vocabulary:

- `production`: validated, documented, release-gated, and intended for external use.
- `validated`: tested against a public oracle/fixture or regression gate, but not yet release-hardened.
- `experimental`: implemented smoke path with known limitations and no production claim.
- `scaffold`: API or placeholder exists but must not be selected for real valuation.
- `unsupported`: fail closed; do not route or silently approximate.

| Capability | Status | Evidence / benchmark ID | Notes |
|---|---:|---|---|
| 1D Black-Scholes European call value on uniform/log-uniform grids | `validated` | `BS-CALL-PARITY-V0`, `QPS-VANILLA-CALL-V0`, `BOUNDARY-MODEL-AWARE-V0` | Public-synthetic analytical oracle and QuantProblemSpec fixture; #48 boundary resolver carries strike, rate, carry/dividend and time-to-maturity explicitly. |
| 1D Black-Scholes Delta/Gamma from finite-difference price grids | `validated` | `BS-CALL-PARITY-V0` | Greeks share grid metadata; use only with validated value route. |
| 1D vanilla boundary/reaction semantics | `validated` | `BOUNDARY-MODEL-AWARE-V0`, `REACTION-INDEPENDENT-V0` | Dirichlet/asymptotic call/put facets are model-aware and fail closed for unsupported products; spatial reaction can be supplied independently from drift while legacy GBM `mu` fallback remains disclosed for old APIs. |
| Rannacher startup before Crank-Nicolson for kinked payoffs | `validated` | `RANNACHER-GAMMA-V0` | Two/four Backward-Euler half-step schedules are explicit and recorded. |
| Pinares fixed-price option proxy | `validated` | `PINARES-FD-FIXED-PRICE-PROXY-V0`, `PINARES-QPS-FIXED-PRICE-PROXY-V0` | Public-synthetic one-dimensional UF fixed-price purchase-option proxy under `Q*`, with survival scaling, explicit boundary conditions, theta controls and error budgets. This row is not a ROFR/full-family-contract valuation. |
| Callable fixed-rate bond cash-flow and Bermudan call-date reference route | `experimental` | none | `CallableBondPDEModel` now requires explicit cash-flow/coupon and call schedules, applies issuer exercise only on contractual dates, keeps same-date coupons outside the call cap, supports clean/dirty call settlement with coupon-date clean calls treated ex-coupon, spans remaining maturity from settlement, and exposes exercise diagnostics. It is a reference backward-induction route, not a production Hull-White/QuantLib-validated callable-bond engine. |
| Heston stochastic volatility vanilla call smoke route | `experimental` | `HESTON-SMOKE-DOCSTRING-V0`, `HESTON-ORACLE-V0` | Uses explicit `(log_spot, variance)` state; vanilla payoffs receive spot through the declared `exp` factor transform. Shape/finite-value ADI smoke only; Fourier-oracle and Black-Scholes-limit evidence are regression tests, not a production calibration claim. |
| Heston semi-analytical European call oracle | `validated` | `HESTON-ORACLE-V0`, `HESTON-BS-LIMIT-V0`, `HESTON-VARIANCE-BOUNDARY-V0` | Characteristic-function inversion, tighter-integration stability check, benchmark regression value, variance-boundary diagnostics, and zero-vol-of-vol Black-Scholes limit. |
| ADI multidimensional routes | `experimental` | `ADI-SMOKE-V0`, `ADI-OPERATOR-SPLIT-V0` | Douglas split uses one calendar-output orientation in 2D/3D; drift, diagonal diffusion, off-diagonal mixed derivatives, reaction, and source terms are covered by regression evidence. Positivity flooring is disclosed for nonnegative pricing routes; production calibration remains unsupported until convergence/benchmark gates graduate this row. |
| basket option payoff on true multi-asset factors | `unsupported` | `FACTOR-ROLE-COMPAT-V0` | Standard one-strike baskets, explicit leg-strike baskets, and two-leg spreads have separate product identities; pricing routes require tradable-spot factor metadata plus optional asset-ID matching. Heston variance, SABR volatility, and rate factors fail closed and are not basket assets; tracked by #45/#62. |
| American/free-boundary exercise | `unsupported` | none | Requires diagnosed LCP/complementarity implementation; tracked by #66. |
| Jump/PIDE and HJB/control terms | `unsupported` | `PINARES-FD-FAIL-CLOSED-V0` | Capability manifest rejects these terms fail-closed; Pinares full-family/ROFR requests return diagnostics before any solver allocation. |
| Haircut backend adapter / solver contract screening | `validated` | `QPS-VANILLA-CALL-V0`, `BS-CALL-PARITY-V0`, `PINARES-QPS-FIXED-PRICE-PROXY-V0`, `PINARES-FD-FIXED-PRICE-PROXY-V0` | Transitional `finite_difference_options.integrations.haircut_backend:create_backend` factory exposes identity, capability manifest, fail-closed screening, and public-synthetic solve evidence without importing Haircut/PDP/API/UI stacks. Numerical execution requires an exact checked-in public-synthetic fixture (`public_black_scholes_problem_spec()` or `public_pinares_fixed_price_problem_spec()`) plus `privacy_class=public_synthetic`; label-compatible private fixtures are screened but fail closed until a matching executable oracle fixture exists. Clean-wheel entry-point publication remains tied to the package-namespace work in #51/#52. |
| CRIF/CUSO/Basel/FRTB regulatory report endpoints | `scaffold` | `REG-FAIL-CLOSED-V0` | Endpoints and Python strategy/converter entry points return typed HTTP 501 / `NotImplementedForStandard` metadata until exact standard/profile/version, effective date, jurisdiction, licensing status, and conformance fixtures exist. |
| FastAPI/CLI/UI service contracts | `experimental` | `DOCS-README-SMOKE-V0`, `API-REQUEST-GUARDS-V0` | FastAPI pricing endpoints use versioned schemas, enum payoff validation, explicit spot semantics, finite/range request checks, pre-solve node budgets, and explicit opt-in for full grids. Convenience interfaces only; numerical truth remains in the Python core. |

Maintained documentation set:

- Product requirements: `docs/PRD.md`
- Architecture and migration map: `docs/ARCHITECTURE.md`
- Capability matrix: this file
- Benchmark evidence registry: `docs/BENCHMARK_REGISTRY.md`
- CI policy: `docs/CI_POLICY.md`

Archived planning material under `docs/planning/` and `docs/archive/` is not a current capability source of truth.
