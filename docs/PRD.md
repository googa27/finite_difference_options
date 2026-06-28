# Finite Difference Options — Product Requirements Document

**Status:** Canonical target-state PRD; current repository is an experimental library/application hybrid  
**Audit baseline:** 2026-06-26  
**Repository:** `googa27/finite_difference_options`  
**Default branch:** `main`  
**Portfolio epic:** [`haircut-engine` #62](https://github.com/googa27/haircut-engine/issues/62)  
**Local modernization epic:** [#50](https://github.com/googa27/finite_difference_options/issues/50)

---

## 1. Product statement

Finite Difference Options is a reusable, library-first finite-difference toolkit for parabolic, obstacle and related pricing PDEs. Its stable responsibility is the numerical mechanism: grids, derivative operators, boundary conditions, time integrators, ADI and LCP policies, numerical sensitivities, convergence diagnostics and accuracy-adjusted performance evidence.

Option products, stochastic-process adapters, API, CLI, Streamlit and the Next.js client are validation or delivery clients of the numerical library. They must not define hidden coefficients, boundaries or fallback behavior in the stable core.

The repository may implement the Haircut Engine solver contract through a thin plugin, but it must not depend on Haircut domain entities or PDP internals.

## 2. Portfolio role and boundaries

| Concern | This repository owns | This repository does not own |
|---|---|---|
| FD core | Grids, stencils/operators, boundaries and time stepping | PDP ingestion or Haircut business logic |
| Multidimensional methods | ADI splitting, mixed derivatives and validated covariance/operator construction | Placeholder drift/covariance or dimension-only model selection |
| Obstacles | LCP/PSOR/policy iteration when validated | Hidden product-specific exercise policy |
| Sensitivities | Grid-consistent values, Greeks and error diagnostics | Portfolio aggregation or regulatory reporting policy |
| Validation | Manufactured, analytical and convergence evidence | Production finite-element ownership |
| Integration | Thin versioned backend adapter and capability manifest | Imports from Haircut domain/application modules |
| Delivery | Optional CLI/API/UI/client examples | Mandatory web/frontend dependencies in numerical core |

```text
Haircut generic solver contract
            │
            ▼
FD integration adapter
            │
            ▼
public FD contracts → grids/operators/BCs/time/solvers/Greeks
```

Cross-repository integration uses independently versioned wheels and fixtures, not Git submodules or checkout-relative imports.

## 3. Users and maturity

| User | Required outcome |
|---|---|
| Numerical developer | Explicit PDE semantics, consistency, stability and convergence evidence |
| Quant/model developer | Stable problem/config/result API and complete diagnostics |
| Model validator | Analytical/manufactured references and failure semantics |
| Performance engineer | Accuracy-adjusted stage, memory and reuse benchmarks |
| Backend consumer | Clean wheel, capability manifest and no private-module coupling |
| Application developer | Thin clients over one public numerical API |

Capabilities are `production`, `validated`, `experimental`, `scaffold`, `deprecated` or `unsupported`. `docs/CAPABILITY_MATRIX.md` is the authoritative current-status matrix; a unified facade, example or endpoint is not evidence of a validated numerical capability.

## 4. Functional requirements

### FR-FD-001 — Installable package

Define complete PEP 621 project/build metadata and a real `finite_difference_options` package under `src/`. Built wheels must work outside the repository. The public API cannot use the literal package `src`, mixed `src.*` imports, path injection or requirements files as its only runtime metadata. Owners: #51 and #52.

### FR-FD-002 — Explicit PDE contracts

Native immutable contracts describe domain, coordinates, time orientation, drift, diffusion/covariance, discount/reaction, source, initial/terminal condition, boundaries, optional mixed terms/jumps/systems/obstacle, requested outputs, discretization/tolerances/resources, dtype/device and result diagnostics.

No selectable route may invent coefficients, boundaries or dynamics. Owner: #59.

### FR-FD-003 — Grids

Uniform, nonuniform, transformed and multidimensional tensor grids have typed representations with monotonicity, spacing, coordinate, truncation and boundary checks. Grid identity and transforms are preserved in result metadata.

### FR-FD-004 — Differential operators

First, second and mixed derivative operators declare stencil order, local-spacing assumptions, upwind/central bias, boundary closure and coordinate transform. Variable coefficients are evaluated at the correct locations. Consistency, dimensions, sparsity and signs are tested independently of products. Process-level generator contracts provide canonical batched drift/covariance/discount shapes, state-factor roles, validated exact affine covariance tensor accessors, PSD diagnostics and explicit generator application to polynomial fixtures. Inexact affine-covariance claims fail closed for quadratic/bilinear native covariances, including direct raw coefficient access. Discount/reaction is an independent field and is never inferred from drift.

### FR-FD-005 — Boundary conditions

Dirichlet, Neumann, Robin, periodic and asymptotic conditions use typed representations and correct algebraic treatment. A generic solver cannot infer boundaries from an `option_type` string. Unsupported classes fail before operator construction. Tests include boundary residuals and corner policy.

### FR-FD-006 — One-dimensional time integration

Validated theta-family routes include backward Euler and Crank–Nicolson with explicit time orientation, coefficient evaluation and stability policy. Rannacher smoothing is a separate capability with payoff-kink and start-up evidence; issue #56 adds configurable two/four BE half-step startup, realised schedule records and a near-strike Gamma roughness regression. Owner: #56.

### FR-FD-007 — Multidimensional ADI

ADI receives actual drift, covariance and mixed-derivative coefficients from the problem contract. Covariance is checked for shape, symmetry and positive semidefiniteness over required states/times. Mixed derivative signs and split operators are independently tested.

Dummy arrays, hard-coded variances and dimension-only routing are prohibited. Each ADI variant advertises only validated dimensions and operator classes.

### FR-FD-008 — Obstacles and free boundaries

American or optimal-stopping routes use explicit obstacle/LCP contracts and validated methods such as PSOR or policy iteration. Results include complementarity residual, iteration count, stopping criterion and exercise-boundary diagnostics. Clipping a European solution is not a validated LCP route.

### FR-FD-009 — Greeks and sensitivities

Every Greek identifies differentiated coordinate/parameter, grid and transform, stencil/order, boundary behavior, payoff-kink policy, interpolation point and error/reference evidence. Nonuniform-grid derivatives require dedicated coefficients and convergence tests. Owner: #57.

### FR-FD-010 — Diagnostics and failures

Results include convergence/residual status, time steps, operator/matrix/factorization reuse, ADI or LCP iterations, boundary and covariance checks, dtype/device, memory estimate, stage timings, warnings/regularization/fallback trace and version provenance.

A failed solve or invalid operator cannot be returned as a successful price array.

### FR-FD-011 — Numerical validation

Tests cover heat and advection–diffusion–reaction manufactured solutions, Black–Scholes values and boundaries, spatial/temporal convergence, nonuniform grids and payoff kinks, time/state-dependent coefficients, correlated mixed derivatives, ADI splitting error, obstacle complementarity when supported, and invalid covariance/grid/boundary/solver requests.

A single endpoint price comparison is insufficient.

### FR-FD-012 — Haircut backend plugin

After package, consolidation and correctness blockers pass, publish an optional `haircut.solver_backends` entry point. The adapter exposes identity/version/maturity/capabilities, preserves conventions, rejects unsupported requests before operator work, returns complete diagnostics, uses canonical public FD APIs and imports no Haircut domain/application, PDP or delivery modules. Owner: #59.

### FR-FD-013 — Canonical implementation

Inventory legacy and newer pricers, processes, boundaries, Greeks and solvers. Each capability has one canonical implementation plus a documented migration/deprecation path. Duplicate modules cannot remain indefinitely as equal APIs. Owner: #52.

### FR-FD-014 — Optional applications

CLI, FastAPI, Streamlit, plotting and Next.js are separately tested applications over the installed package. They are not core dependencies or alternate numerical implementations. Frontend and service contracts have independent locks and CI. Regulatory/reporting service routes must return HTTP 501 problem details unless the route declares an exact standard/profile/version, effective date, jurisdiction, licensing status and conformance fixture set.

## 5. Non-functional requirements

### Correctness and reproducibility

- PDE/operator, time, grid and boundary conventions are explicit.
- Observed consistency and convergence match expected order or deviations are explained.
- Covariance and mixed derivatives are validated mathematically.
- Unsupported requests fail before numerical work.
- Problem, grid and solver configurations are snapshot-able.
- Results include versions, grid identity, dtype/device, solver policy and reuse state.

### Performance and scalability

- Preserve sparse or banded structure where applicable.
- Reuse operators, matrices and factorizations only with complete invalidation keys.
- Avoid unnecessary full solution history and repeated reshaping/materialization.
- Benchmark operator construction, boundaries, factorization, solve/substeps, Greeks and serialization separately.
- Compare routes at equal error and equal mathematical problem.

### Maintainability and release quality

- One import package and one canonical implementation per capability.
- Core has no API, CLI, UI, frontend, plotting or Haircut/PDP dependency.
- Dependency direction is architecture-tested.
- Compatibility shims contain no new numerical logic and have removal dates.
- GitHub issues and versioned docs are authoritative; local agent databases are not.
- Solver work and service work are bounded by explicit resource policies.
- Release profiles produce dependency, license and vulnerability evidence or an owned exception.

## 6. Dependency profiles

The current runtime requirements combine numerical, API, CLI, UI and plotting packages; SciPy appears only in development requirements. The target uses project metadata, published extras and development groups.

| Profile | Purpose |
|---|---|
| `core` | NumPy and SciPy plus `findiff` only where a canonical validated route uses it |
| `api` | FastAPI/Uvicorn service |
| `cli` | Typer command line |
| `ui` | Streamlit application |
| `viz` | Matplotlib/Plotly rendering |
| `validation` | Analytical/reference and benchmark tooling |
| `docs` | Documentation build |
| development groups | pytest, Ruff, mypy, pre-commit, build, twine and audit tools |

Visualization packages such as Seaborn do not belong in core. A development lock may pin an environment; wheel metadata publishes compatible ranges. Minimum-supported and latest-compatible profiles are tested separately.

## 7. Release, compatibility and production gates

- Distribution and solver-contract versions are independent.
- Wheels and sdists install and test outside the repository root.
- Haircut compatibility is recorded in the matrix owned by Haircut #65.
- Unknown combinations are unsupported.
- Deprecations include replacement, warning version, removal version/date and migration example.
- Breaking operator, boundary, time or Greek semantics require versioning, migration notes and parity evidence.

Required production gates:

| Dimension | Gate |
|---|---|
| Packaging | PEP 621 project, real namespace, wheel import smoke |
| Architecture | Canonical modules and optional-application isolation |
| Correctness | Manufactured/analytical, convergence, boundary, covariance and negative tests |
| Multidimensional | Real coefficients, mixed-term and ADI splitting evidence |
| Greeks/obstacles | Nonuniform/kink or complementarity evidence when advertised |
| Diagnostics | Residual, convergence, timing and provenance metadata |
| Plugin | Clean-wheel discovery, capability rejection and shared parity |
| Performance | Accuracy-adjusted stage and memory benchmarks |
| Release | Compatibility, changelog and supply-chain evidence |

## 8. Roadmap and issue ownership

| Workstream | Owner issues |
|---|---|
| Numerical correctness, API semantics and validation | #42–#49 |
| Repository modernization epic | #50 |
| Package foundation and import cleanup | #51 |
| Canonical implementation consolidation | #52 |
| API, capability and documentation contracts | #53–#55 |
| Rannacher, nonuniform Greeks and numerical refinements | #56–#58 |
| Haircut backend adapter | #59 |
| Portfolio protocol, parity and release governance | `haircut-engine` #62–#65 |

## 9. Non-goals

- A full collateral, reporting or data platform.
- A second production finite-element library.
- A mandatory environment containing API, UI, plotting and frontend stacks.
- Product-specific boundary guessing in generic solver code.
- Placeholder coefficients or proxy dynamics in a validated route.
- Git submodules or checkout-relative imports as integration contracts.
- Production claims based on a facade, endpoint or speed alone.

## 10. Change policy

Changes to PDE/operator convention, grid/stencil semantics, boundaries, time integration, ADI/LCP, Greeks, public APIs, dependency profiles, backend capabilities or compatibility must update this PRD, `docs/ARCHITECTURE.md`, `AGENTS.md`, relevant fixtures/benchmarks and issue/release metadata together.

---

## References

- PyPA project metadata: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
- PyPA `src` layout: https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/
- PyPA plugin discovery: https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/
- Python entry points: https://docs.python.org/3/library/importlib.metadata.html
- uv dependency profiles: https://docs.astral.sh/uv/concepts/projects/dependencies/
- SciPy sparse linear algebra: https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html

*End of canonical Finite Difference Options PRD.*
