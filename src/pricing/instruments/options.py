"""Option instruments implementation.

This module contains implementations of various option types
for the unified pricing framework.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator, ConfigDict

from .base import UnifiedInstrument
from .payoff_calculators import PayoffCalculatorFactory
from ...validation import validate_positive
from ...utils.process_validators import validate_weights_sum_to_one
from ...utils.exceptions import ValidationError


class UnifiedEuropeanOption(UnifiedInstrument, BaseModel):
    """European option for unified pricing framework."""
    
    strike: float
    maturity: float
    option_type: str  # 'call' or 'put'
    
    model_config = ConfigDict(frozen=True, extra='forbid')
    
    @field_validator('strike')
    @classmethod
    def validate_strike(cls, v):
        """Validate strike price."""
        validate_positive(v, "strike")
        return v
    
    @field_validator('maturity')
    @classmethod
    def validate_maturity(cls, v):
        """Validate maturity."""
        validate_positive(v, "maturity")
        return v
    
    @field_validator('option_type')
    @classmethod
    def validate_option_type(cls, v):
        """Validate option type."""
        if v not in ['call', 'put']:
            raise ValidationError(f"option_type must be 'call' or 'put', got {v}")
        return v
    
    def payoff(self, *grids: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute European option payoff."""
        calculator = PayoffCalculatorFactory.create_calculator(self)
        return calculator.calculate_payoff(self, *grids)


class UnifiedBasketOption(UnifiedInstrument, BaseModel):
    """Basket option for unified pricing framework."""
    
    strikes: NDArray[np.float64]
    weights: NDArray[np.float64]
    maturity: float
    option_type: str  # 'call' or 'put'
    
    model_config = ConfigDict(frozen=True, extra='forbid')
    
    @field_validator('maturity')
    @classmethod
    def validate_maturity(cls, v):
        """Validate maturity."""
        validate_positive(v, "maturity")
        return v
    
    @field_validator('strikes')
    @classmethod
    def validate_strikes(cls, v):
        """Validate strikes."""
        if not np.all(v > 0):
            raise ValidationError("All strikes must be positive")
        return v
    
    @field_validator('option_type')
    @classmethod
    def validate_option_type(cls, v):
        """Validate option type."""
        if v not in ['call', 'put']:
            raise ValidationError(f"option_type must be 'call' or 'put', got {v}")
        return v
    
    def __init__(self, **data):
        """Initialize and validate basket option parameters."""
        super().__init__(**data)
        
        # Additional validation that requires multiple fields
        if len(self.strikes) != len(self.weights):
            raise ValidationError("strikes and weights must have same length")
        
        validate_weights_sum_to_one(self.weights)
    
    def payoff(self, *grids: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute basket option payoff."""
        calculator = PayoffCalculatorFactory.create_calculator(self)
        return calculator.calculate_payoff(self, *grids)


# Convenience functions
def create_unified_european_call(strike: float, maturity: float) -> UnifiedEuropeanOption:
    """Create European call option."""
    return UnifiedEuropeanOption(strike=strike, maturity=maturity, option_type='call')


def create_unified_european_put(strike: float, maturity: float) -> UnifiedEuropeanOption:
    """Create European put option."""
    return UnifiedEuropeanOption(strike=strike, maturity=maturity, option_type='put')


def create_unified_basket_call(
    strikes: NDArray[np.float64], 
    weights: NDArray[np.float64], 
    maturity: float
) -> UnifiedBasketOption:
    """Create basket call option."""
    return UnifiedBasketOption(strikes=strikes, weights=weights, maturity=maturity, option_type='call')


def create_unified_basket_put(
    strikes: NDArray[np.float64], 
    weights: NDArray[np.float64], 
    maturity: float
) -> UnifiedBasketOption:
    """Create basket put option."""
    return UnifiedBasketOption(strikes=strikes, weights=weights, maturity=maturity, option_type='put')