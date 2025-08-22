"""Option instruments implementation.

This module contains implementations of various option types
for the unified pricing framework.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from .base import UnifiedInstrument
from ...utils.validation import validate_positive
from ...utils.process_validators import validate_weights_sum_to_one
from ...utils.exceptions import ValidationError


@dataclass
class UnifiedEuropeanOption(UnifiedInstrument):
    """European option for unified pricing framework."""
    
    strike: float
    _maturity: float
    option_type: str  # 'call' or 'put'
    
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
        if len(grids) == 0:
            raise ValidationError("At least one grid required")
        
        # First grid is always the underlying asset price
        price_grid = grids[0]
        
        if self.option_type == 'call':
            return np.maximum(price_grid - self.strike, 0.0)
        else:  # put
            return np.maximum(self.strike - price_grid, 0.0)


@dataclass
class UnifiedBasketOption(UnifiedInstrument):
    """Basket option for unified pricing framework."""
    
    strikes: NDArray[np.float64]
    weights: NDArray[np.float64]
    _maturity: float
    option_type: str  # 'call' or 'put'
    
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
        if len(grids) != len(self.weights):
            raise ValidationError(
                f"Number of grids ({len(grids)}) must match number of assets ({len(self.weights)})"
            )
        
        # Create meshgrid for all dimensions
        mesh_grids = np.meshgrid(*grids, indexing='ij')
        
        # Compute basket value
        basket_value = np.zeros_like(mesh_grids[0])
        for weight, grid in zip(self.weights, mesh_grids, strict=True):
            basket_value += weight * grid
        
        # Compute weighted strike
        weighted_strike = np.sum(self.weights * self.strikes)
        
        if self.option_type == 'call':
            return np.maximum(basket_value - weighted_strike, 0.0)
        else:  # put
            return np.maximum(weighted_strike - basket_value, 0.0)


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
