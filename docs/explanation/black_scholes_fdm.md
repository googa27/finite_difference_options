# Black-Scholes finite-difference method

This page is the durable theory companion for the repository's Black-Scholes finite-difference examples. The implementation target is not just "produce a price array"; it is to solve a precisely stated parabolic pricing PDE with explicit coordinates, boundary conditions, time orientation, and diagnostics.

## Pricing equation

For a non-dividend European claim with payoff `g(S)` at maturity `T`, the Black-Scholes value `V(S,t)` solves

```text
∂_t V + 0.5 σ² S² ∂²_SS V + r S ∂_S V - r V = 0,
V(S,T) = g(S).
```

Finite-difference code usually evolves forward in time-to-maturity `τ = T - t`:

```text
∂_τ u = 0.5 σ² S² ∂²_SS u + r S ∂_S u - r u,
u(S,0) = g(S).
```

The sign convention matters. In this repository, any route that advertises a value at valuation time must make the calendar-time/time-to-maturity orientation explicit in tests and result metadata.

## Discretization contract

A validated Black-Scholes FD route needs all of the following as inputs or declared policy:

- spatial coordinate: spot `S`, log-spot `x = log(S)`, or another declared transform;
- domain truncation: lower and upper spot/log-spot bounds;
- grid family: uniform, nonuniform, or transformed, including node count and spacing policy;
- boundary conditions: type, boundary side, formula, and time dependence;
- time scheme: backward Euler, Crank-Nicolson, Rannacher-smoothed CN, or another documented θ-route;
- linear solver and tolerances;
- output convention: valuation-time point value, full grid, Greeks, or diagnostics.

A route is not mathematically defined by option type alone. For example, a call's far-field boundary and a put's far-field boundary are product-adapter responsibilities; the FD core should receive them as explicit boundary records or fail closed.

## Boundary examples

For vanilla European options in spot coordinates on `[S_min, S_max]`, common benchmark boundaries are:

```text
call: V(0,t) = 0,
      V(S_max,t) ≈ S_max - K exp(-r(T-t));

put:  V(0,t) ≈ K exp(-r(T-t)),
      V(S_max,t) = 0.
```

These are truncation approximations, not universal truths. Local volatility, dividends, stochastic rates, transformed coordinates, American exercise, or model-specific asymptotics require different contracts.

## Stability and convergence expectations

Blocking validation should separate:

1. temporal convergence: vary time steps while holding spatial error small;
2. spatial convergence: vary grid size while holding time error small;
3. boundary sensitivity: move the truncated domain and verify stable target values;
4. payoff-kink behavior: use Rannacher smoothing or document the observed degradation;
5. Greek convergence: use grid-aware derivative formulas, especially on nonuniform grids.

The repository's Project #5 architecture requires capability claims to be benchmark-backed. A single endpoint match to a closed-form price is useful smoke evidence, but it is not a complete numerical validation gate.

## Relation to the architecture contract

Black-Scholes is the simplest shared fixture for package, API, and solver-contract health:

```text
explicit Black-Scholes problem
        ↓
grid + boundary records
        ↓
θ time stepper
        ↓
price / Greeks / residual diagnostics
```

The same ownership rules apply here as for more complex models:

- finite_difference_options owns FD grids, operators, boundary algebra, stepping, and diagnostics;
- product adapters own financial payoff/boundary assumptions;
- haircut-engine owns CASCADE/domain orchestration, not hidden FD semantics;
- PDP owns data products, not pricing kernels.
