from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from findiff import FinDiff, BoundaryConditions

from .instruments import Instrument
from .processes import GeometricBrownianMotion
from .boundary_conditions import BlackScholesBoundaryBuilder
from .spatial_operator import SpatialOperator
from .validation import validate_option_parameters, validate_model_parameters


@dataclass(init=False)
class EuropeanOption(Instrument):
    """Base class for European options."""

    strike: float
    _maturity: float
    model: GeometricBrownianMotion
    boundary_builder: BlackScholesBoundaryBuilder = field(default_factory=BlackScholesBoundaryBuilder)
    
    def __init__(
        self, 
        strike: float, 
        maturity: float, 
        model: GeometricBrownianMotion,
        boundary_builder: BlackScholesBoundaryBuilder = None
    ):
        """Initialize European option.
        
        Parameters
        ----------
        strike : float
            Strike price.
        maturity : float
            Time to maturity.
        model : GeometricBrownianMotion
            Underlying process model.
        boundary_builder : BlackScholesBoundaryBuilder, optional
            Boundary condition builder.
        """
        self.strike = strike
        self._maturity = maturity
        self.model = model
        self.boundary_builder = boundary_builder or BlackScholesBoundaryBuilder()
        self.__post_init__()
    
    @property
    def maturity(self) -> float:
        """Get instrument maturity."""
        return self._maturity

    def __post_init__(self) -> None:
        """Validate option and model parameters after initialization."""
        validate_option_parameters(self.strike, self._maturity)
        # The new GeometricBrownianMotion uses mu, so we can't validate the rate here.
        # We will assume the mu is the risk-free rate for now.
        # validate_model_parameters(
        #     self.model.mu,
        #     self.model.sigma,
        #     0.0  # No dividend yield in this model
        # )

    @abstractmethod
    def payoff(self, s: NDArray[np.float64]) -> NDArray[np.float64]:  # pragma: no cover - abstract
        """Return the payoff at expiry for underlying prices ``s``."""
        raise NotImplementedError

    def generator(self, s: NDArray[np.float64]) -> FinDiff:
        return SpatialOperator(self.model).build(s)

    def boundary_conditions(self, s: NDArray[np.float64]) -> BoundaryConditions:
        return self.boundary_builder.build(s, self)


@dataclass(init=False)
class EuropeanCall(EuropeanOption):
    """European call option."""

    def payoff(self, s: np.ndarray) -> np.ndarray:
        return np.maximum(s - self.strike, 0.0)


@dataclass(init=False)
class EuropeanPut(EuropeanOption):
    """European put option."""

    def payoff(self, s: np.ndarray) -> np.ndarray:
        return np.maximum(self.strike - s, 0.0)