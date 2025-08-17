"""High-level option pricing interface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .models import GeometricBrownianMotion, Market
from .options import EuropeanCall, EuropeanOption, EuropeanPut
from .pde_pricer import BlackScholesPDE, PDEModel
from .greeks import FiniteDifferenceGreeks


@dataclass
class OptionPricer:
    """Compute value grids using finite difference methods.

    Parameters
    ----------
    rate, sigma:
        Parameters of the default Black--Scholes model.  They are ignored when
        ``pde_model`` is provided.
    pde_model:
        Custom :class:`~src.pde_pricer.PDEModel` implementation.  When supplied
        the pricer delegates all computations to this model.
    """

    rate: float | None = None
    sigma: float | None = None
    pde_model: PDEModel | None = None

    def __post_init__(self) -> None:
        """Initialise market, model and PDE solver from inputs."""
        if self.pde_model is None:
            if self.rate is None or self.sigma is None:
                raise ValueError("rate and sigma must be set when no model supplied")
            self.market = Market(rate=self.rate)
            self.model = GeometricBrownianMotion(rate=self.rate, sigma=self.sigma)
            self._default_pricer = BlackScholesPDE(model=self.model, market=self.market)
        else:
            self._default_pricer = None

    def compute_grid(
        self,
        *,
        maturity: float,
        s_max: float,
        s_steps: int,
        t_steps: int,
        strike: float | None = None,
        option_type: str | None = None,
        return_greeks: bool = False,
    ) -> Union[
        Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ]:
        """Return asset and time grids with instrument values and Greeks."""

        s = np.linspace(0, s_max, s_steps)
        t = np.linspace(0, maturity, t_steps)

        if self.pde_model is None:
            if strike is None or option_type is None:
                msg = "strike and option_type must be provided for default model"
                raise ValueError(msg)
            option_cls: type[EuropeanOption]
            option_cls = EuropeanCall if option_type == "Call" else EuropeanPut
            option = option_cls(strike=strike)
            values = self._default_pricer.price(option=option, s=s, t=t)
        else:
            values = self.pde_model.price(option=None, s=s, t=t)

        if not return_greeks:
            return s, t, values

        calculator = FiniteDifferenceGreeks()
        delta = calculator.delta(values, s)
        gamma = calculator.gamma(values, s)
        theta = calculator.theta(values, t)
        return s, t, values, delta, gamma, theta
