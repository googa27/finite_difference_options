"""Option instruments implementation.

This module contains implementations of various option types
for the unified pricing framework.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from .base import UnifiedInstrument
from .payoff_calculators import PayoffCalculatorFactory
from ...utils.validation import validate_positive
from ...utils.process_validators import validate_weights_sum_to_one
from ...utils.exceptions import ValidationError


@dataclass
class UnifiedEuropeanOption(UnifiedInstrument):
    """European option for unified pricing framework."""
    
    strike: float
    _maturity: float
    option_type: str  # 'call' or 'put'
    
    def __init__(self, strike: float, maturity: float, option_type: str = 'call'):
        """Initialize European option.
        
        Parameters
        ----------
        strike : float
            Strike price.
        maturity : float
            Time to maturity.
        option_type : str
            Option type ('call' or 'put').
        """
        self.strike = strike
        self._maturity = maturity
        self.option_type = option_type
        self.__post_init__()
    
    @property
    def maturity(self) -> float:
        """Get instrument maturity."""
        return self._maturity
    
    def __post_init__(self) -> None:
        """Validate option parameters."""
        validate_positive(self.strike, "strike")
        validate_positive(self.maturity, "maturity")
        
        if self.option_type not in ['call', 'put']:
            raise ValidationError(f"option_type must be 'call' or 'put', got {self.option_type}")
    
    def payoff(self, *grids: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute European option payoff."""
        calculator = PayoffCalculatorFactory.create_calculator(self)
        return calculator.calculate_payoff(self, *grids)


@dataclass
class UnifiedBasketOption(UnifiedInstrument):
    """Basket option for unified pricing framework."""
    
    strikes: NDArray[np.float64]
    weights: NDArray[np.float64]
    _maturity: float
    option_type: str  # 'call' or 'put'
    
    def __init__(
        self, 
        strikes: NDArray[np.float64], 
        weights: NDArray[np.float64], 
        maturity: float, 
        option_type: str = 'call'
    ):
        """Initialize basket option.
        
        Parameters
        ----------
        strikes : NDArray[np.float64]
            Strike prices for each asset.
        weights : NDArray[np.float64]
            Weights for each asset in the basket.
        maturity : float
            Time to maturity.
        option_type : str
            Option type ('call' or 'put').
        """
        self.strikes = strikes
        self.weights = weights
        self._maturity = maturity
        self.option_type = option_type
        self.__post_init__()
    
    @property
    def maturity(self) -> float:
        """Get instrument maturity."""
        return self._maturity
    
    def __post_init__(self) -> None:
        """Validate basket option parameters."""
        validate_positive(self.maturity, "maturity")
        
        if len(self.strikes) != len(self.weights):
            raise ValidationError("strikes and weights must have same length")
        
        if self.option_type not in ['call', 'put']:
            raise ValidationError(f"option_type must be 'call' or 'put', got {self.option_type}")
        
        if not np.all(self.strikes > 0):
            raise ValidationError("All strikes must be positive")
        
        validate_weights_sum_to_one(self.weights)
    
    def payoff(self, *grids: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute basket option payoff."""
        calculator = PayoffCalculatorFactory.create_calculator(self)
        return calculator.calculate_payoff(self, *grids)


# Convenience functions
def create_unified_european_call(strike: float, maturity: float) -> UnifiedEuropeanOption:
    """Create European call option."""
    return UnifiedEuropeanOption(strike, maturity, 'call')


def create_unified_european_put(strike: float, maturity: float) -> UnifiedEuropeanOption:
    """Create European put option."""
    return UnifiedEuropeanOption(strike, maturity, 'put')


def create_unified_basket_call(
    strikes: NDArray[np.float64], 
    weights: NDArray[np.float64], 
    maturity: float
) -> UnifiedBasketOption:
    """Create basket call option."""
    return UnifiedBasketOption(strikes, weights, maturity, 'call')


def create_unified_basket_put(
    strikes: NDArray[np.float64], 
    weights: NDArray[np.float64], 
    maturity: float
) -> UnifiedBasketOption:
    """Create basket put option."""
    return UnifiedBasketOption(strikes, weights, maturity, 'put')