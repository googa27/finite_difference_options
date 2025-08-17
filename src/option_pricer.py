"""High-level option pricing interface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from .models import GeometricBrownianMotion, Market
from .options import EuropeanCall, EuropeanOption, EuropeanPut
from .pde_pricer import BlackScholesPDE


@dataclass
class OptionPricer:
    """Compute option value grids using finite difference methods."""

    rate: float
    sigma: float

    def __post_init__(self) -> None:
        """Initialize market, model and PDE pricer from inputs."""
        self.market = Market(rate=self.rate)
        self.model = GeometricBrownianMotion(rate=self.rate, sigma=self.sigma)
        self._pricer = BlackScholesPDE(model=self.model, market=self.market)

    def compute_grid(
        self,
        strike: float,
        maturity: float,
        option_type: str,
        s_max: float,
        s_steps: int,
        t_steps: int,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return asset and time grids with option values.

        Parameters
        ----------
        strike:
            Strike price of the option.
        maturity:
            Time to maturity in years.
        option_type:
            "Call" for call option, "Put" for put option.
        s_max:
            Maximum underlying asset price to consider.
        s_steps:
            Number of discrete asset price steps.
        t_steps:
            Number of discrete time steps.
        """
        option_cls: type[EuropeanOption]
        option_cls = EuropeanCall if option_type == "Call" else EuropeanPut
        option = option_cls(strike=strike)

        s = np.linspace(0, s_max, s_steps)
        t = np.linspace(0, maturity, t_steps)
        values = self._pricer.price(option=option, s=s, t=t)
        return s, t, values
