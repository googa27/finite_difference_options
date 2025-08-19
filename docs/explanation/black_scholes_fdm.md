# Black--Scholes Equation and Finite Difference Method

This note outlines how the Black--Scholes partial differential equation (PDE) arises and how it can be solved numerically with finite differences.

## PDE derivation

Assume the underlying asset price $S_t$ follows a geometric Brownian motion under the risk-neutral measure,

$$
\mathrm{d}S_t = r S_t\,\mathrm{d}t + \sigma S_t\,\mathrm{d}W_t,
$$

where $r$ is the continuously compounded risk-free rate, $\sigma$ the volatility and $W_t$ a Wiener process. For a derivative price $V(S,t)$, apply Itô's lemma and construct a self-financing, delta-hedged portfolio. Eliminating the stochastic term and requiring the portfolio to earn the risk-free rate yields the Black--Scholes PDE,

$$
\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0.
$$

The terminal condition at maturity $T$ is given by the option payoff $V(S,T)=\Phi(S)$.

## Finite difference discretization

Let $S_i = i\,\Delta S$ for $i=0,\dots,I$ and $t_n = n\,\Delta t$ for $n=0,\dots,N$. Approximating spatial derivatives with central differences and time with an implicit step gives the scheme

$$
\frac{V_i^{n+1} - V_i^n}{\Delta t} = \frac{1}{2}\sigma^2 S_i^2 \frac{V_{i+1}^{n+1} - 2 V_i^{n+1} + V_{i-1}^{n+1}}{(\Delta S)^2} + r S_i \frac{V_{i+1}^{n+1} - V_{i-1}^{n+1}}{2\Delta S} - r V_i^{n+1}.
$$

Collecting terms yields a tridiagonal linear system $A V^{n+1} = V^n$ that can be solved efficiently at each step. Using a weighted average of the explicit and implicit schemes leads to the Crank--Nicolson method, which is second-order accurate in both time and space.

## Boundary conditions

Appropriate boundary conditions ensure stability and accuracy:

- **Lower boundary ($S=0$):** the value of a call is $0$; a put equals $K e^{-r(T-t)}$.
- **Upper boundary ($S \to S_{\max}$):** the value of a call approaches $S_{\max} - K e^{-r(T-t)}$; a put tends to $0$.
- **Terminal condition ($t=T$):** option payoff $\Phi(S)$.

Choosing $S_{\max}$ several standard deviations above the strike keeps truncation error low while maintaining a manageable grid.

## Further reading

- Fischer Black and Myron Scholes. "The Pricing of Options and Corporate Liabilities." *Journal of Political Economy* (1973).
- John C. Hull. *Options, Futures, and Other Derivatives*. Pearson, 2022.
- Paul Wilmott, Sam Howison, and Jeff Dewynne. *The Mathematics of Financial Derivatives*. Cambridge University Press, 1995.
- Damiano Brigo and Fabio Mercurio. *Option Pricing Models and Volatility*. Springer, 2001.
