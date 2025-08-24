"""Base classes for financial instruments.

This module contains the abstract interfaces and base implementations
for all financial instruments in the unified framework.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator, ConfigDict

from ..utils.exceptions import ValidationError


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


class EuropeanOption(Instrument, BaseModel):
    """Base class for European options."""
    
    strike: float
    maturity: float
    
    model_config = ConfigDict(frozen=True, extra='forbid')
    
    @field_validator('strike')
    @classmethod
    def validate_strike(cls, v):
        """Validate strike price."""
        if v <= 0:
            raise ValidationError(f"Strike price must be positive, got {v}")
        return v
    
    @field_validator('maturity')
    @classmethod
    def validate_maturity(cls, v):
        """Validate maturity."""
        if v <= 0:
            raise ValidationError(f"Maturity must be positive, got {v}")
        return v


class EuropeanCall(EuropeanOption):
    """European call option."""
    
    def payoff(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute call option payoff."""
        return np.maximum(state - self.strike, 0.0)


class EuropeanPut(EuropeanOption):
    """European put option."""
    
    def payoff(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute put option payoff."""
        return np.maximum(self.strike - state, 0.0)