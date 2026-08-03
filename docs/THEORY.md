# Theory — Finite Difference Options

Status: canonical v0.1 theory summary. This file consolidates the mathematical conventions scattered through architecture, capability and validation docs; it does not add new capability claims.

## Pricing PDE convention

The reusable finite-difference problem is a linear parabolic equation in solver time to expiry `tau`:

```text
partial_tau u(x,tau)
  = 0.5 sum_ij a_ij(x,tau) partial_ij u
    + sum_i b_i(x,tau) partial_i u
    - c(x,tau) u
    + f(x,tau)
```

Here `x` is the declared computational coordinate, `a` is the covariance or diffusion matrix in that coordinate, `b` is drift, `c` is the reaction/discount term, and `f` is a source term. The repository normalizes generator application as `0.5 tr(a Hessian) + b dot grad - c value + source`. Callers must supply time orientation, domain, terminal or initial condition, boundary data, units, measure and numeraire metadata.

For a risk-neutral Black-Scholes stock with continuous dividend yield `q` in physical spot coordinate `S`, the European value satisfies:

```text
partial_tau V
  = 0.5 sigma^2 S^2 partial_SS V
    + (r - q) S partial_S V
    - r V
```

In log coordinate x, the transformed drift is `(r-q-0.5 sigma^2)` and the covariance term is `sigma^2`. Tests that compare spot and log routes must account for this coordinate change.

## Grids and truncation

A grid is part of the numerical problem, not a display detail. Supported public grid records preserve monotone coordinates, spacing family, local spacing ratios, physical-coordinate transforms and boundary locations. Current validated families include uniform, log-uniform, strike-centered and smooth clustered axes plus tensor-product grids.

Truncation and far-field behavior must be explicit. For example, a vanilla equity call on a bounded spot domain uses lower-boundary and upper-boundary assumptions that approximate the infinite-domain financial payoff; those assumptions are product-adapter boundary records rather than hidden stencil behavior.

## Operators and finite differences

Spatial derivative operators must declare coordinate, order, stencil, local spacing and boundary closure. Uniform-grid formulas and nonuniform-grid formulas are different policies. Mixed derivatives use the declared covariance convention and sign; covariance is validated for shape, finiteness, symmetry and positive semidefiniteness before operator assembly.

The core must not invent dummy drifts, covariances, discounts or sources. A selectable route that cannot obtain an explicit coefficient field fails closed before numerical work.

## Time stepping

One-dimensional routes use theta-family stepping for the semidiscrete operator `L`:

```text
(I - theta dt L_next) u_next
  = (I + (1 - theta) dt L_now) u_now
    + dt (theta f_next + (1 - theta) f_now)
```

Important named cases are explicit Euler (theta equal to zero), backward Euler (theta equal to one) and Crank-Nicolson (theta equal to one half). Time-dependent coefficients or boundaries require a declared reassembly/reuse policy. Rannacher smoothing is represented as a startup sequence of backward-Euler half-steps before returning to the requested theta scheme; it is not an undocumented Boolean tweak.

## Boundary algebra

Boundary conditions are typed records over identified boundary sets. The public native boundary builder currently supports Dirichlet facets, selected Neumann/second-derivative/degenerate/extrapolated typed records where exposed by route-specific resolvers, and route-specific asymptotic schedules. Robin conditions are part of the general mathematical boundary form, but they are not a native typed public boundary kind or builder capability in this release; a route requiring Robin data must fail closed or provide its own explicitly validated adapter rather than relying on an advertised core Robin implementation. Boundary rows are owned by boundary algebra, not by accidental stencil truncation. Corners, transformed coordinates and time-dependent boundary values require explicit validation.

## ADI and multidimensional routes

An ADI route declares dimension, directional/mixed operator split, variant, theta parameters, coefficient dependence, boundary order, linear solvers, stability assumptions and diagnostics. A process dimension by itself is not sufficient to select ADI. Heston routes use executable state `(log_spot, variance)`; the variance state is not a second tradable basket leg.

## Obstacles and exercise

American or Bermudan exercise introduces an obstacle `psi` and complementarity conditions:

```text
u >= psi
residual(u) >= 0
(u - psi) residual(u) = 0
```

The LCP route records primal, dual and complementarity residuals, active set evidence, iteration count, tolerance and nonconvergence failure. European PDE output must not be relabeled as American output.

## Greeks and sensitivities

A Greek states the differentiated coordinate or parameter, units, transform, stencil order, interpolation point, one-sided boundary treatment and payoff-kink policy. Nonuniform-grid Delta and Gamma use local finite-difference weights and return diagnostics that distinguish requested-coordinate interpolation from nearest-node values. At nonsmooth payoff coordinates, a caller must name the kink policy; undefined derivatives fail closed.

## Evidence obligations

A route can be documented as validated only when tests or fixtures cover the relevant mathematical claim. Current evidence includes Black-Scholes parity, Pinares public-synthetic proxy convergence, nonuniform Greek validation, Rannacher kink smoothing, American LCP diagnostics, benchmark-registry checks and architecture/package gates. The capability matrix remains the maturity source of truth.
