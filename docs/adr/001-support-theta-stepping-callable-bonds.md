# 001: Support θ-Stepping and Callable Bonds

## Status
Superseded by the issue #63 schedule-aware callable-bond route.

## Context
The initial solver handled only fixed schemes for European options. To broaden the framework we needed a unified time-stepper and support for instruments with early redemption features.

The original callable-bond implementation was not an exercise model: it set a constant terminal payoff and clipped every grid value by one call price, even on dates where the issuer had no exercise right.

## Decision
Adopt a configurable θ-method for time-stepping. Callable fixed-rate bonds now use an explicit cash-flow and callability schedule:

- coupon and redemption cash flows are contractual events;
- call schedule entries carry exercise time, settlement price and clean/dirty convention;
- issuer exercise applies only when backward induction crosses a contractual call date;
- same-date coupon cash flows remain outside the issuer exercise cap;
- clean calls include accrued interest in the settlement value;
- the time grid spans remaining maturity from settlement to final redemption;
- the old global post-solve cap path fails closed rather than masquerading as a PDE boundary condition.

The current callable-bond path is an experimental reference route. Production maturity still requires a governed short-rate model, calibration, boundary derivation, convergence evidence and independent QuantLib/tree comparisons.

## Consequences
- Users can choose between explicit, implicit and Crank–Nicolson stepping for supported PDE routes.
- Callable-bond smoke tests now check schedule economics instead of validating a global cap.
- Continuous exercise remains unsupported until a diagnosed obstacle/LCP route exists.
