"""High-level option pricing interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import numpy as np
from numpy.typing import NDArray

from .pde_pricer import BlackScholesPDE
from .instruments.base import Instrument


class GridResult(NamedTuple):
    """Named result for pricing grid and Greeks.

    Orientation convention: ``values`` and Greeks are shaped ``(t, s)`` where the
    first axis is time ascending from 0 to maturity, and the second axis is the
    spatial asset-price grid from 0 to ``s_max``.
    """

    s: NDArray[np.float64]
    t: NDArray[np.float64]
    values: NDArray[np.float64]
    delta: Optional[NDArray[np.float64]]
    gamma: Optional[NDArray[np.float64]]
    theta: Optional[NDArray[np.float64]]


@dataclass
class OptionPricer:
    """Compute value grids using finite difference methods.

    Parameters
    ----------
    instrument:
        The financial instrument to be priced.
    """

    instrument: Instrument
    _pde_model: BlackScholesPDE | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        """Initialise PDE solver from instrument."""
        # For now, we assume BlackScholesPDE can handle any Instrument.
        # In a more complex scenario, a factory might be needed here.
        self._pde_model = BlackScholesPDE(instrument=self.instrument)

    def compute_grid(
        self,
        *,
        s_max: float,
        s_steps: int,
        t_steps: int,
        return_greeks: bool = False,
    ) -> GridResult:
        """Return grids and values with optional Greeks as a NamedTuple."""

        s = np.linspace(0, s_max, s_steps)
        t = np.linspace(0, self.instrument.maturity, t_steps)

        values = self._pde_model.price(option=self.instrument, s=s, t=t)

        if not return_greeks:
            return GridResult(
                s=s, t=t, values=values, delta=None, gamma=None, theta=None
            )

        calculator = FiniteDifferenceGreeks()
        delta = calculator.delta(values, s)
        gamma = calculator.gamma(values, s)
        theta = calculator.theta(values, t)
        return GridResult(
            s=s, t=t, values=values, delta=delta, gamma=gamma, theta=theta
        )