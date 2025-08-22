from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from findiff import FinDiff, BoundaryConditions

from .instruments import Instrument
from .models import GeometricBrownianMotion
from .boundary_conditions import BlackScholesBoundaryBuilder
from .spatial_operator import SpatialOperator
from .validation import validate_option_parameters, validate_model_parameters


@dataclass
class EuropeanOption(Instrument):
    """Base class for European options."""

    strike: float
    maturity: float
    model: GeometricBrownianMotion
    boundary_builder: BlackScholesBoundaryBuilder = field(default_factory=BlackScholesBoundaryBuilder)

    def __post_init__(self) -> None:
        """Validate option and model parameters after initialization."""
        validate_option_parameters(self.strike, self.maturity)
        validate_model_parameters(
            self.model.rate,
            self.model.sigma,
            0.0  # No dividend yield in this model
        )

    @abstractmethod
    def payoff(self, s: NDArray[np.float64]) -> NDArray[np.float64]:  # pragma: no cover - abstract
        """Return the payoff at expiry for underlying prices ``s``."""
        raise NotImplementedError

    def generator(self, s: NDArray[np.float64]) -> FinDiff:
        return SpatialOperator(self.model).build(s)

    def boundary_conditions(self, s: NDArray[np.float64]) -> BoundaryConditions:
        return self.boundary_builder.build(s, self)


@dataclass
class EuropeanCall(EuropeanOption):
    """European call option."""

    def payoff(self, s: np.ndarray) -> np.ndarray:
        return np.maximum(s - self.strike, 0.0)


@dataclass
class EuropeanPut(EuropeanOption):
    """European put option."""

    def payoff(self, s: np.ndarray) -> np.ndarray:
        return np.maximum(self.strike - s, 0.0)