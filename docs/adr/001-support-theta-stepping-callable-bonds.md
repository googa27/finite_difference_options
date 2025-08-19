# 001: Support θ-Stepping and Callable Bonds

## Status
Accepted

## Context
The initial solver handled only fixed schemes for European options. To broaden the framework we needed a unified time-stepper and support for instruments with early redemption features.

## Decision
Adopt a configurable θ-method for time-stepping and introduce a callable bond PDE model. The model caps grid values at the call price and uses boundary conditions that enforce the same limit.

## Consequences
- Users can choose between explicit, implicit and Crank–Nicolson stepping.
- Callable bonds can be priced alongside vanilla options.
- Additional documentation explains the method and modelling approach.
