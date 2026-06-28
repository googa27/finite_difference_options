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
| 1D Black-Scholes European call value on uniform/log-uniform grids | `validated` | `BS-CALL-PARITY-V0`, `QPS-VANILLA-CALL-V0` | Public-synthetic analytical oracle and QuantProblemSpec fixture. |
| 1D Black-Scholes Delta/Gamma from finite-difference price grids | `validated` | `BS-CALL-PARITY-V0` | Greeks share grid metadata; use only with validated value route. |
| Rannacher startup before Crank-Nicolson for kinked payoffs | `validated` | `RANNACHER-GAMMA-V0` | Two/four Backward-Euler half-step schedules are explicit and recorded. |
| Heston stochastic volatility vanilla call smoke route | `experimental` | `HESTON-SMOKE-DOCSTRING-V0` | Shape/finite-value smoke only; not a production benchmark or calibration claim. |
| ADI multidimensional routes | `experimental` | `ADI-SMOKE-V0` | Mixed-derivative/reaction completeness remains tracked by #46. |
| basket option payoff on true multi-asset factors | `unsupported` | `FACTOR-ROLE-COMPAT-V0` | Standard one-strike baskets, explicit leg-strike baskets, and two-leg spreads have separate product identities; pricing routes require tradable-spot factor metadata plus optional asset-ID matching. Heston variance, SABR volatility, and rate factors fail closed and are not basket assets; tracked by #45/#62. |
| American/free-boundary exercise | `unsupported` | none | Requires diagnosed LCP/complementarity implementation; tracked by #66. |
| Jump/PIDE and HJB/control terms | `unsupported` | none | Capability manifest rejects these terms fail-closed. |
| CRIF/CUSO/Basel/FRTB regulatory report endpoints | `scaffold` | `REG-FAIL-CLOSED-V0` | Endpoints and Python strategy/converter entry points return typed HTTP 501 / `NotImplementedForStandard` metadata until exact standard/profile/version, effective date, jurisdiction, licensing status, and conformance fixtures exist. |
| FastAPI/CLI/UI service contracts | `experimental` | `DOCS-README-SMOKE-V0`, `API-REQUEST-GUARDS-V0` | FastAPI pricing endpoints use versioned schemas, enum payoff validation, explicit spot semantics, finite/range request checks, pre-solve node budgets, and explicit opt-in for full grids. Convenience interfaces only; numerical truth remains in the Python core. |

Maintained documentation set:

- Product requirements: `docs/PRD.md`
- Architecture and migration map: `docs/ARCHITECTURE.md`
- Capability matrix: this file
- CI policy: `docs/CI_POLICY.md`

Archived planning material under `docs/planning/` and `docs/archive/` is not a current capability source of truth.
