# AGENTS.md — Finite Difference Options Operating Contract

**Canonical for:** human contributors and coding/review agents  
**Audit baseline:** 2026-06-26  
**Default branch:** `main`  
**Portfolio epic:** [`haircut-engine` #62](https://github.com/googa27/haircut-engine/issues/62)  
**Local modernization epic:** [#50](https://github.com/googa27/finite_difference_options/issues/50)

> The prior guide is preserved at
> [`docs/archive/AGENTS.pre-federated-audit-2026-06-26.md`](docs/archive/AGENTS.pre-federated-audit-2026-06-26.md).

## 1. Mission

Build an installable, modular and numerically trustworthy finite-difference library. Explicit PDE coefficients, boundary conditions, stability/convergence, diagnostics, reproducibility and dependency boundaries take priority over feature count, facade uniformity or speed.

A successful-looking array is not a valid result when coefficients were invented, a solver failed or the requested capability was unsupported.

## 2. Required reading

Before a non-trivial change, read:

1. `docs/PRD.md`.
2. `docs/ARCHITECTURE.md`.
3. The owning GitHub issue and its parent.
4. Relevant numerical tests and implementation.
5. README/API/client documentation when public behavior changes.

GitHub issues, canonical docs, tests, benchmark registries and release manifests are authoritative. Local agent databases, vector stores, caches and untracked notebooks are not.

## 3. Ownership

This repository owns grids, derivative operators, boundary algebra, time integrators, ADI/LCP policies, numerical Greeks and FD diagnostics.

It does not own Haircut Engine domain or CASCADE policy, PDP internals, a second production FEM implementation, or a mandatory combined API/UI/frontend environment.

Cross-repository integration uses released wheels, a versioned solver contract, entry points and parity fixtures. Do not add production Git submodules, local-path release dependencies, branch dependencies or repository-relative imports.

## 4. Workflow

1. Start from current `main`.
2. Link the change to the correct parent and child issue.
3. Add a characterization or failing numerical test before changing semantics.
4. Separate package/consolidation moves from mathematical changes.
5. Update contracts, tests, benchmarks, docs and deprecation notes together.
6. Run targeted numerical tests, then architecture, packaging and profile gates.
7. Describe mathematical risk, compatibility, performance and rollback in the PR.
8. Do not increase a capability's maturity without the required evidence.

## 5. Dependency direction

```text
contracts and diagnostic schemas
             ↑
grids / operators / boundary conditions
             ↑
time integration / one-dimensional / ADI / LCP / Greeks
             ↑
validation and integration adapters
             ↑
API / CLI / UI / frontend / examples
```

Rules:

- The stable import package is `finite_difference_options`; no new public `src.*` imports.
- Contracts do not import `findiff`, FastAPI, Typer, Streamlit, plotting, frontend code or Haircut.
- Numerical core does not import products, API, CLI, UI, reporting or integration adapters.
- Solvers consume explicit coefficient, grid and boundary records; they do not inspect product strings to invent semantics.
- Optional imports are lazy and name the required extra.
- API, CLI, UI and examples import the installed public package.
- The frontend communicates through a versioned service schema.
- Compatibility shims live only at package boundaries and have removal milestones.
- Add architecture tests for every new package boundary.

## 6. Mathematical preflight

For every PDE or FD change, document:

- PDE/generator sign and forward/backward-time transform;
- domain, coordinates and transforms;
- drift, diffusion/covariance, reaction/discount and source terms;
- initial or terminal condition;
- boundary partition, type and formula;
- grid family and truncation/far-field policy;
- stencil/order and upwind/central policy;
- time, ADI or obstacle method and expected order;
- linear/LCP solver and stopping criterion;
- requested value, Greek or sensitivity convention.

Dimension, process class or option type alone does not define this information.

## 7. Coefficient and covariance rules

- Coefficients are explicit problem inputs or produced by a typed model adapter.
- Selectable routes must not create dummy drift, covariance or discount arrays.
- Validate coefficient shapes, finiteness and coordinate convention before operator construction.
- Validate covariance symmetry and positive semidefiniteness over required states and times.
- Regularization is explicit, quantified and recorded in diagnostics.
- Mixed-derivative coefficients and signs are independently tested.
- State/time dependence is declared as a capability and drives reassembly/reuse policy.

## 8. Grid and operator rules

- Validate grid monotonicity, minimum size, spacing ratios, transforms and requested evaluation points.
- Use `AxisGrid`/`TensorGrid` for new grid semantics so local spacings, transforms, boundary locations and diagnostics are preserved instead of passing anonymous arrays.
- Distinguish uniform, nonuniform and transformed-grid formulas.
- Every derivative operator declares order, local stencil, boundary closure and bias.
- Upwinding is selected by typed policy, not hidden branching.
- Preserve sparse or banded structure where appropriate.
- Boundary rows are owned by boundary algebra, not accidental stencil truncation.
- Use `findiff` only where its behavior is validated and compatible with required diagnostics.
- Operator cache keys include grid, coefficients, boundary policy, time step, scheme and dtype/device.

## 9. Boundary-condition rules

Every boundary condition is typed and identifies its boundary set.

- Generic solvers must not infer boundaries from `option_type`.
- Product adapters translate financial far-field assumptions into explicit BC records.
- Test Dirichlet values, Neumann/Robin residuals, corners, time dependence and transformed coordinates.
- Unsupported boundary classes fail before operator construction.
- ADI substeps document and test boundary application order.

## 10. Time, ADI and obstacle rules

- State the theta-scheme operator sign and formula used by implementation.
- Test spatial and temporal convergence separately.
- Time-dependent coefficients and boundaries have explicit rebuild/reuse behavior.
- Rannacher smoothing is a separate start-up policy with payoff-kink evidence.
- ADI policies declare dimension, split, mixed operator, variant, boundary treatment and stability assumptions.
- ADI is never routed solely from process dimension.
- Obstacle/LCP routes declare obstacle, method, tolerance, iteration limit and warm-start policy.
- LCP results include primal, dual and complementarity residuals.
- Failed convergence is a typed failure, not a successful field.

## 11. Greeks and sensitivities

Every Greek states:

- differentiated coordinate or parameter;
- units and coordinate transform;
- stencil/order and one-sided boundary treatment;
- evaluation or interpolation point;
- smoothing and payoff-kink policy;
- reference or error estimate.

Nonuniform-grid derivatives use dedicated coefficients and convergence tests. Parameter bumps disclose bump size and interaction with solver tolerance.

## 12. Backend plugin

The Haircut adapter must:

- expose lightweight identity, solver-contract range, maturity and capabilities;
- validate the request before grid or operator work;
- map generic records into canonical native FD contracts without reinterpretation;
- reject absent coefficients, unsupported BCs and invalid covariance;
- use only validated canonical numerical policies;
- return solution, Greeks and complete diagnostics;
- work from a clean installed wheel;
- import no Haircut domain/application, PDP or delivery modules.

Advertise only capabilities backed by repository-local tests and shared parity evidence.

## 13. Canonical implementation and applications

Classify every duplicate process, pricer, solver, boundary and Greek module as canonical, compatibility, reference-test-only or remove.

- Keep one implementation per stable capability.
- Do not create a larger conditional facade to preserve every duplicate path.
- API, CLI, Streamlit, plotting and Next.js are outer applications, not numerical core.
- Frontend and service code have separate locks and contract tests.
- Regulatory report examples are not FD-core ownership unless a separate validated domain contract establishes them.
- Preserve behavior with tests before moves; change numerics separately.

## 14. Packaging and dependencies

Packaging changes must:

- use complete PEP 621 project metadata and an explicit build backend;
- place the package under `src/finite_difference_options`;
- declare `requires-python`, bounded numerical dependencies, license and project URLs;
- put actual runtime numerical dependencies in core metadata;
- retain `findiff` only where canonical validated routes need it;
- isolate `api`, `cli`, `ui`, `viz`, `validation` and `docs` extras;
- keep tests, lint, typing, build and audit tools in dependency groups;
- maintain a reproducible lock but publish compatible runtime ranges;
- test minimum-supported and latest-compatible profiles separately;
- build and install sdist/wheel outside the repository;
- test missing-extra messages and wheel contents;
- produce release dependency/license/vulnerability evidence or an owned exception.

Evaluate dependencies by API stability, maintenance, platform support, transitive size, license, security history and measured value. Visualization libraries do not belong in core.

## 15. Performance

Optimization order:

1. Prove analytical or manufactured correctness and convergence.
2. Profile grids, operators, boundaries, factorization, solve/ADI substeps, Greeks and serialization.
3. Remove dense conversion, repeated construction and unnecessary histories.
4. Reuse invariant operators and factorizations with complete invalidation keys.
5. Compare alternatives at equal numerical error and the same mathematical problem.
6. Establish noise-aware regression budgets.

A benchmark records problem/grid/config hash, dimensions, nodes, nonzeros, time steps, versions, hardware/BLAS/threads, dtype/device, cold/warm/cache state, stage timings, memory, residual/iterations, achieved error and reuse count.

## 16. Tests and verification

Required layers are architecture, contract, unit, numerical, integration, shared parity, performance and packaging. Default numerical tests are deterministic and offline; API/UI/frontend tests are separate profiles.

Unified-engine regression policy:

- `tests/test_unified_pricing_engine.py` is part of the blocking stable regression suite.
- Public unified-engine solutions are in calendar-time order: `prices[0]` is valuation time and `prices[-1]` is maturity/terminal payoff.
- European options on multidimensional process grids use the first state coordinate as the underlying; terminal payoff broadcasting across non-underlying dimensions must be explicit in tests.
- Any remaining unified-engine quarantine must use `pytest.mark.xfail(strict=True, reason="#<issue>: ...")` with a removal condition. Silent file-level exclusion is not allowed.

Current repository commands include:

```bash
python -m pip install -e '.[dev]'
python -m pip install -r requirements-dev.lock.txt
python -m pip check
pre-commit run --all-files
ruff check . --select E9,F63,F7,F82
mypy --ignore-missing-imports --follow-imports=silent src/finite_difference_options/contracts src/finite_difference_options/validation scripts/check_architecture_contract.py
python scripts/check_architecture_contract.py
pytest -q tests/architecture tests/test_packaging_contract.py --no-cov
pytest -q
python -m build --sdist --wheel
python -m twine check dist/*
python -m pip_audit --progress-spinner=off --skip-editable
cyclonedx-py environment --of JSON -o sbom.json
```

The blocking pull-request/push baseline is documented in [`docs/CI_POLICY.md`](docs/CI_POLICY.md). That policy separates the fast stable Python regression suite, optional Node application profile, and advisory Gemini automation so documentation-only or narrowly-scoped numerical PRs are not forced to repay unrelated repository-wide debt.

Package modernization requires lock validation, `python -m build`, `twine check`, clean-wheel import tests, optional-profile wheel smoke tests, SBOM/pip-audit evidence, and the Haircut backend conformance suite.

Do not report an unconfigured or unrun gate as passing. Record the gap and owner issue.

## 17. Evidence by change type

| Change | Minimum evidence |
|---|---|
| PDE coefficients/operator | Mathematical statement, consistency/reference and residual tests |
| Boundary algebra | Boundary residual, corner, time-dependent and rejection cases |
| Grid/stencil | Coefficient derivation, polynomial consistency and convergence |
| Time/Rannacher | Temporal convergence and payoff-kink/start-up evidence |
| ADI/mixed terms | Real coefficients, split definition, mixed-sign and convergence evidence |
| LCP/obstacle | Complementarity residual and iteration/failure behavior |
| Greeks | Convention, stencil, smoothing and reference error |
| Package/consolidation | Characterization, clean wheel and legacy mapping |
| Optional application | Missing-extra and isolated profile/contract test |
| Haircut adapter | Entry-point discovery, capability negatives, parity and compatibility update |
| Breaking semantics | Version bump, migration/deprecation note and downstream evidence |

## 18. Compatibility, docs and review

Distribution API and Haircut solver-contract versions are separate. Unknown combinations are unsupported. Public deprecations name replacement, warning version, removal version/date and migration example; shims contain no new numerical behavior.

Update `docs/PRD.md`, `docs/ARCHITECTURE.md`, `AGENTS.md`, tests, benchmarks and issues together when responsibilities or semantics change. Keep README and client examples aligned with the installed namespace and versioned API.

Before review, confirm:

- [ ] The change belongs in the FD repository.
- [ ] PDE coefficients, time, grid and boundary conventions are explicit.
- [ ] No placeholder coefficient or product-boundary inference remains on the route.
- [ ] Numerical evidence includes convergence, residuals and negative behavior.
- [ ] No application dependency leaked into core.
- [ ] No new checkout-only import assumption was added.
- [ ] Solver failure and fallback behavior are explicit.
- [ ] Performance compares equal error.
- [ ] API, compatibility and deprecation impact are handled.
- [ ] Docs, issues and release metadata are synchronized.

---

*This file is the canonical Finite Difference Options contributor and agent contract.*

- Package topology changes must update `docs/architecture_contract.toml`, `docs/ARCHITECTURE.md`, architecture tests, and `scripts/check_architecture_contract.py` architecture contract gate in the same PR.
