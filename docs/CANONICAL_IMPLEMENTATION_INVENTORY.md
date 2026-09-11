# Canonical Implementation Inventory

**Owner issue:** finite_difference_options #52  
**Status:** executable architecture inventory  
**Rule:** every public numerical capability has exactly one canonical implementation path. Legacy names may exist only as thin compatibility shims with no numerical logic and an explicit removal milestone.

## Capability map

Machine-readable capability keys: `stochastic-processes-and-coefficients`, `instruments-payoffs-product-adapters`, `one-dimensional-pricing-workflow`, `multidimensional-adi-solving`, `grids-and-local-metrics`, `boundary-conditions`, `greeks-and-sensitivities`, `validation-benchmarks-capability-evidence`.

| Capability group | Canonical implementation | Public import surface | Legacy/duplicate paths that must not reappear | Evidence |
|---|---|---|---|---|
| Stochastic processes and coefficient extraction | `src/finite_difference_options/processes/` plus `src/finite_difference_options/instruments/operators.py` | `finite_difference_options.processes`, `finite_difference_options.instruments.operators` | `src/processes.py`, `src/multidimensional_processes.py`, `src/stochastic_processes.py` | `tests/test_process_generator_contracts.py`, `tests/test_unified_processes.py`, `tests/test_factor_role_payoff_compatibility.py` |
| Instruments, payoffs, and product adapters | `src/finite_difference_options/pricing/instruments/` with package-boundary adapters under `src/finite_difference_options/instruments/` | `finite_difference_options.pricing.instruments`, `finite_difference_options.instruments` | `src/instruments.py`, `src/options.py`, `src/payoffs.py` | `tests/test_options.py`, `tests/test_unified_pricing_engine.py`, `tests/test_callable_bond.py` |
| One-dimensional pricing workflow | `src/finite_difference_options/pricing/workflows/option_pricer.py` and `src/finite_difference_options/pricing/engines/finite_difference.py` | `finite_difference_options.pricing.workflows`, `finite_difference_options.pricing.engines` | `src/option_pricer.py`, `src/pde_pricer.py`, `src/pricer.py` | `tests/test_option_pricer.py`, `tests/test_pde_pricer.py`, `tests/test_pricing_engine_imports.py` |
| Multidimensional / ADI solving | `src/finite_difference_options/solvers/adi.py` and `src/finite_difference_options/solvers/base.py` | `finite_difference_options.solvers` | `src/multidimensional_solver.py`, `src/adi_solver.py`, `src/solver_factory.py` | `tests/test_adi_solver_operator_split.py`, `tests/test_multidimensional_adapter_coefficients.py` |
| Grid contracts and local metrics | `src/finite_difference_options/grids/` | `finite_difference_options.grids` | `src/grids.py`, `src/grid.py`, `src/nonuniform_grids.py` | `tests/test_grid_contracts.py`, `tests/test_adi_solver_operator_split.py` |
| Boundary conditions | `src/finite_difference_options/boundary_conditions/` | `finite_difference_options.boundary_conditions` | `src/boundary_conditions.py`, `src/multidimensional_boundary_conditions.py` | `tests/test_boundary_conditions.py`, `tests/test_model_aware_reaction_terms.py` |
| Greeks and sensitivities | `src/finite_difference_options/greeks/` | `finite_difference_options.greeks` | `src/greeks.py`, `src/risk_greeks.py` | `tests/test_greeks.py`, `tests/test_finite_difference_greeks.py` |
| Validation, benchmarks, and capability evidence | `src/finite_difference_options/validation/` and `src/finite_difference_options/contracts/` | `finite_difference_options.validation`, `finite_difference_options.contracts` | `src/validation.py`, `src/benchmarks.py`, `src/capabilities.py` | `tests/test_benchmark_registry.py`, `tests/test_fd_backend_capabilities.py`, `tests/test_documentation_capabilities.py` |

## Compatibility policy

- There are currently **no** repository-root compatibility shim files. The package boundary gate rejects new root Python files and historical `src.*` imports.
- Any future shim must be listed in `docs/architecture_contract.toml`, import or delegate to the canonical module, emit a targeted `DeprecationWarning`, and carry a removal version/date.
- Shims must not define finite-difference formulas, payoff algebra, stochastic-process coefficients, boundary conditions, solver loops, or Greek stencils.
- Compatibility tests must prove old import names delegate to the canonical implementation during the deprecation window.

## Migration table

| Historical request | Replacement |
|---|---|
| `src.option_pricer`, `src.pde_pricer` | `finite_difference_options.pricing.workflows` / `finite_difference_options.pricing.engines` |
| `src.multidimensional_processes` | `finite_difference_options.processes` |
| `src.multidimensional_boundary_conditions` | `finite_difference_options.boundary_conditions` |
| `src.pde_solver`, `src.multidimensional_solver`, `src.adi_solver` | `finite_difference_options.solvers` |
| `src.greeks` | `finite_difference_options.greeks` |
| `src.validation`, `src.benchmarks` | `finite_difference_options.validation` |

## Executable gates

`docs/architecture_contract.toml` carries the same inventory in machine-readable form. This architecture contract is enforced by `scripts/check_architecture_contract.py` and `tests/architecture/test_architecture_contracts.py`, which verify that:

1. every canonical module path exists;
2. each advertised public import maps under `finite_difference_options`;
3. historical duplicate modules are absent;
4. any declared compatibility shim stays inside the package, has a removal milestone, and does not reintroduce standalone numerical code.


## Cached and generic time stepping

- `cached-black-scholes-tridiagonal-solving`: public `finite_difference_options.solvers` owns `solvers/black_scholes.py` and its private `solvers/_tridiagonal.py` LAPACK adapter. Private implementation is not a new public export.
- `generic-theta-time-stepping`: `solvers/finite_difference.py` owns generic theta/Rannacher stepping through its existing public import.

`docs/architecture_contract.toml` inventories the complete `solvers` package. The checker rejects any non-initializer Python implementation without a canonical owner, including an unregistered new private kernel. Removing the registered tridiagonal owner and adding an unknown solver both fail negative architecture tests. Other packages retain their explicit existing topology/import gates; this scoped ratchet does not claim all leaf modules elsewhere were individually inventoried.

### Compiled fixture contract and verification ownership

- `compiled-pde-contract-validation`: `integrations/compiled_pde_adapter.py` owns public screening and solving orchestration; `_compiled_pde_contracts.py` owns immutable DTOs/identities; `_compiled_pde_validation.py` owns exact serialized validation. Contracts cannot import integration, solver or verification modules; private validation cannot import its facade or execution routes.
- `compiled-fd-grid-verification`: `validation/fd_verification.py` owns evidence orchestration; `validation/fd_evidence/grid_metrics.py` owns refinement and residual measurements and cannot import its orchestrator.

`docs/architecture_contract.toml` declares each required implementation file and dependency rule. The architecture contract checker resolves both absolute and relative static imports, including `from package import module`; negative mutation tests enforce these boundaries. Historical public names remain re-exported at their original facades.
