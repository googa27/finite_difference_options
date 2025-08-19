# θ-Method and Callable Bond PDE Models

This note derives the general θ-method for time-stepping parabolic PDEs and explains how callable bonds are handled in the finite difference framework.

## θ-method derivation

Consider a linear parabolic PDE of the form

$$
\frac{\partial V}{\partial t} = \mathcal{L}V,
$$

where $\mathcal{L}$ is the spatial differential operator. On a grid with time step $\Delta t$, the θ-method approximates the time derivative by a weighted average of the operator evaluated at the current and next time levels:

$$
\frac{V^{n+1}-V^{n}}{\Delta t} = (1-\theta)\,\mathcal{L}V^{n} + \theta\,\mathcal{L}V^{n+1}.
$$

Rearranging gives a linear system for the unknown $V^{n+1}$,

$$
\bigl(I - \theta\,\Delta t\,\mathcal{L}\bigr) V^{n+1} = \bigl(I + (1-\theta)\,\Delta t\,\mathcal{L}\bigr) V^{n}.
$$

Different choices of $\theta$ recover common schemes:

### Explicit Euler ($\theta=0$)

$$
V^{n+1} = \bigl(I + \Delta t\,\mathcal{L}\bigr) V^{n}.
$$

This scheme is first-order accurate in time and conditionally stable; $\Delta t$ must satisfy a Courant condition.

### Implicit Euler ($\theta=1$)

$$
\bigl(I - \Delta t\,\mathcal{L}\bigr) V^{n+1} = V^{n}.
$$

It remains first-order accurate but is unconditionally stable because the operator is evaluated fully at the new time level.

### Crank–Nicolson ($\theta=1/2$)

$$
\bigl(I - \tfrac{1}{2}\Delta t\,\mathcal{L}\bigr) V^{n+1} = \bigl(I + \tfrac{1}{2}\Delta t\,\mathcal{L}\bigr) V^{n}.
$$

Averaging the explicit and implicit forms yields second-order accuracy in both time and space while retaining unconditional stability.

## Callable bond modelling

A callable bond allows the issuer to redeem the bond early at a fixed call price $C$. The PDE for the bond value is solved as usual, but the early redemption feature is enforced numerically.

### Grid capping

After each time step the grid values are capped at the call price:

$$
V^{n+1}_i \leftarrow \min\bigl(V^{n+1}_i, C\bigr),
$$

which emulates the optimal calling strategy and prevents the numerical solution from exceeding $C$.

### Boundary conditions

Dirichlet boundaries keep the solution within admissible bounds:

- **Lower boundary:** as the short rate approaches zero, the bond value tends to zero.
- **Upper boundary:** the bond price cannot exceed the call price $C$.

Together, boundary conditions and grid capping ensure the finite difference solution respects the callable feature at all times.

## Further reading

- John C. Hull. *Options, Futures, and Other Derivatives*. Pearson, 2022.
- Paul Wilmott, Sam Howison, and Jeff Dewynne. *The Mathematics of Financial Derivatives*. Cambridge University Press, 1995.
