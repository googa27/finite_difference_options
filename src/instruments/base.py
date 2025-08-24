"""Base classes for financial instruments.

This module contains the abstract interfaces and base implementations
for all financial instruments in the unified framework.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np
from numpy.typing import NDArray

from ..utils.exceptions import ValidationError


@dataclass
class Instrument(ABC):
    """Abstract base class for all financial instruments."""
    
    @property
    @abstractmethod
    def maturity(self) -> float:
        """Get instrument maturity."""
        ...
    
    @abstractmethod
    def payoff(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute payoff at maturity.
        
        Parameters
        ----------
        state : NDArray[np.float64]
            Underlying asset state(s) at maturity.
            
        Returns
        -------
        NDArray[np.float64]
            Payoff value(s).
        """
        ...


@dataclass
class EuropeanOption(Instrument):
    """Base class for European options."""
    
    strike: float
    _maturity: float
    
    def __post_init__(self) -> None:
        """Validate option parameters."""
        if self.strike <= 0:
            raise ValidationError(f"Strike price must be positive, got {self.strike}")
        if self._maturity <= 0:
            raise ValidationError(f"Maturity must be positive, got {self._maturity}")
    
    @property
    def maturity(self) -> float:
        """Get instrument maturity."""
        return self._maturity


@dataclass
class EuropeanCall(EuropeanOption):
    """European call option."""
    
    def payoff(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute call option payoff."""
        return np.maximum(state - self.strike, 0.0)


@dataclass
class EuropeanPut(EuropeanOption):
    """European put option."""
    
    def payoff(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute put option payoff."""
        return np.maximum(self.strike - state, 0.0)