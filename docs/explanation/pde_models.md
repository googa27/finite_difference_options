# PDE model contracts

This page explains the model-theory boundary behind the repository architecture. A finite-difference solver route is valid only when the mathematical PDE, coordinate convention, boundary conditions, and numerical policy are explicit. Model names and process dimensions are not enough.

## Generic parabolic pricing form

The canonical backward pricing problem can be written as

```text
∂_t V + L_t V - c(x,t) V + f(x,t) = 0,
V(x,T) = g(x),
```

where the generator is

```text
L_t V = 0.5 Σ_ij a_ij(x,t) ∂²_ij V + Σ_i b_i(x,t) ∂_i V.
```

In time-to-maturity `τ = T - t`, the same problem is evolved with a consistent sign convention and an initial condition at `τ = 0`. Every implementation must state which orientation it uses.

## Required problem fields

A model-correct FD problem contract includes:

- state variables and coordinate transforms;
- domain and truncation policy;
- drift vector `b(x,t)`;
- covariance/diffusion matrix `A(x,t)` with symmetry and positive-semidefinite checks where required;
- reaction/discount `c(x,t)`;
- source term `f(x,t)`;
- initial or terminal payoff/condition;
- typed boundary conditions for every boundary set;
- optional obstacle/free-boundary, jump, or coupled-system terms;
- requested outputs: value, grid, Greeks, residuals, exercise boundary, or diagnostics;
- discretization controls and resource limits.

No route should replace missing data with dummy drift, hard-coded covariance, generic zero discounting, or product-string boundary guesses.

## Model families

### Black-Scholes / local volatility

Black-Scholes has one tradable state and an analytic reference. Local-volatility variants keep the parabolic structure but make diffusion state/time dependent. They are good validation fixtures for boundary handling, time stepping, and Greek convergence.

### Short-rate models

Vasicek/Ornstein-Uhlenbeck and CIR-style models may price claims in a rate coordinate. Boundary behavior and discounting are model-specific. Positivity constraints, transformed coordinates, and degeneracy near zero must be explicit.

### Stochastic volatility

Heston-style contracts require at least spot/log-spot and variance state variables. The executable convention in this repository's target architecture is log-spot plus variance when used by native problem contracts. The covariance tensor, mixed derivative, variance lower boundary, Feller/numerical policy, and payoff transform must be explicit.

The Feller condition is not a universal validity switch for every numerical route; it is one input into a declared boundary/numerical policy.

### Correlated multidimensional models and ADI

Alternating-direction implicit methods require a real operator split. Correctness depends on drift, covariance, reaction, mixed derivative signs, boundary application, and time dependence. Selecting ADI because a process has dimension greater than one is not sufficient.

### American and callable claims

Early exercise is an obstacle or complementarity problem:

```text
V ≥ payoff,
∂_t V + L_t V - cV + f ≤ 0,
(V - payoff)(∂_t V + L_t V - cV + f) = 0.
```

Post-step payoff clipping is not, by itself, a diagnosed LCP solver. Valid routes report complementarity residuals, iteration counts, stopping criteria, and exercise-boundary diagnostics.

## Validation expectations

A model enters the supported capability matrix only after evidence exists for the advertised route:

- analytical or manufactured reference where available;
- convergence/stability tests at fixed mathematical problem;
- boundary residuals and truncation sensitivity;
- covariance/mixed-derivative validation for multidimensional routes;
- explicit failure tests for unsupported inputs;
- complete result diagnostics and reproducibility metadata.

## Cross-repository meaning

Within the Quant PDE Platform:

- finite_difference_options owns finite-difference model mechanics and diagnostics;
- finite_element_options owns finite-element mechanics and diagnostics;
- haircut-engine owns graph/CASCADE orchestration and solver routing;
- PDP owns data products and point-in-time data contracts.

Integration happens through explicit problem specs, capability manifests, shared fixtures, and compatibility matrices, not through hidden source-tree imports or model-name conventions.
