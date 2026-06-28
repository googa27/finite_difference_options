"""Payoff calculators for financial instruments.

This module contains implementations of payoff calculation strategies
for various financial instruments in the unified pricing framework.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast
import numpy as np
from numpy.typing import NDArray

from .base import UnifiedInstrument
from src.exceptions import ValidationError


class PayoffCalculator(ABC):
    """Abstract base class for payoff calculation strategies."""
    
    @abstractmethod
    def calculate_payoff(self, instrument: UnifiedInstrument, *grids: NDArray[np.float64]) -> NDArray[np.float64]:
        """Calculate instrument payoff at maturity.
        
        Parameters
        ----------
        instrument : UnifiedInstrument
            The financial instrument.
        *grids : NDArray[np.float64]
            Spatial grids for each dimension.
            
        Returns
        -------
        NDArray[np.float64]
            Payoff values on the grid.
        """
        ...


class EuropeanPayoffCalculator(PayoffCalculator):
    """Payoff calculator for European options."""
    
    def calculate_payoff(self, instrument: UnifiedInstrument, *grids: NDArray[np.float64]) -> NDArray[np.float64]:
        """Calculate European option payoff."""
        if len(grids) == 0:
            raise ValidationError("At least one grid required")
        
        # First grid is always the underlying asset price
        price_grid = grids[0]
        
        # Validate that the instrument has the expected attributes
        if not hasattr(instrument, 'strike') or not hasattr(instrument, 'option_type'):
            raise ValidationError("Instrument must have 'strike' and 'option_type' attributes")
        
        # For European option, only the first grid (price) matters
        # Return payoff based on the first grid only, regardless of other grids
        if instrument.option_type == 'call':
            return np.maximum(price_grid - instrument.strike, 0.0)
        else:  # put
            return np.maximum(instrument.strike - price_grid, 0.0)


class BasketPayoffCalculator(PayoffCalculator):
    """Payoff calculator for basket options."""

    def calculate_payoff(self, instrument: UnifiedInstrument, *grids: NDArray[np.float64]) -> NDArray[np.float64]:
        """Calculate basket option payoff."""
        # Validate that the instrument has the expected attributes
        if not hasattr(instrument, 'weights') or not hasattr(instrument, 'option_type'):
            raise ValidationError("Instrument must have 'weights' and 'option_type' attributes")

        basket_instrument = cast(Any, instrument)
        weights = np.asarray(basket_instrument.weights, dtype=np.float64)
        option_type = basket_instrument.option_type

        if len(grids) != len(weights):
            raise ValidationError(
                f"Expected {len(weights)} grids, got {len(grids)}"
            )

        basket_value = self._basket_value(weights, *grids)
        basket_strike = self._basket_strike(instrument)

        if option_type == 'call':
            return np.maximum(basket_value - basket_strike, 0.0)
        else:  # put
            return np.maximum(basket_strike - basket_value, 0.0)

    @staticmethod
    def _basket_value(weights: NDArray[np.float64], *grids: NDArray[np.float64]) -> NDArray[np.float64]:
        arrays = [np.asarray(grid, dtype=np.float64) for grid in grids]
        if len(arrays) == 1:
            return weights[0] * arrays[0]
        if arrays[0].ndim > 1 and all(array.shape == arrays[0].shape for array in arrays):
            basket_value = np.zeros_like(arrays[0], dtype=np.float64)
            for weight, grid in zip(weights, arrays, strict=True):
                basket_value += weight * grid
            return basket_value

        if not all(array.ndim == 1 for array in arrays):
            raise ValidationError("Basket grids must be one-dimensional coordinate arrays or matching pointwise arrays")
        output_shape = tuple(array.shape[0] for array in arrays)
        basket_value = np.zeros(output_shape, dtype=np.float64)
        for axis, (weight, grid) in enumerate(zip(weights, arrays, strict=True)):
            view_shape = [1] * len(arrays)
            view_shape[axis] = grid.shape[0]
            basket_value += weight * grid.reshape(view_shape)
        return basket_value

    @staticmethod
    def _basket_strike(instrument: UnifiedInstrument) -> float:
        basket_instrument = cast(Any, instrument)
        if hasattr(instrument, 'strike'):
            return float(basket_instrument.strike)
        if hasattr(instrument, 'strikes'):
            weights = np.asarray(basket_instrument.weights, dtype=np.float64)
            strikes = np.asarray(basket_instrument.strikes, dtype=np.float64)
            return float(np.sum(weights * strikes))
        raise ValidationError("Instrument must have either 'strike' or 'strikes' attribute")


# Factory for creating appropriate payoff calculators
class PayoffCalculatorFactory:
    """Factory for creating payoff calculators."""
    
    @staticmethod
    def create_calculator(instrument: UnifiedInstrument) -> PayoffCalculator:
        """Create appropriate payoff calculator for instrument.
        
        Parameters
        ----------
        instrument : UnifiedInstrument
            The financial instrument.
            
        Returns
        -------
        PayoffCalculator
            Appropriate payoff calculator for the instrument.
        """
        # Import here to avoid circular imports
        from .options import SpreadOption, StandardBasketOption, UnifiedEuropeanOption, UnifiedBasketOption
        
        if isinstance(instrument, UnifiedEuropeanOption):
            return EuropeanPayoffCalculator()
        elif isinstance(instrument, (UnifiedBasketOption, StandardBasketOption, SpreadOption)):
            return BasketPayoffCalculator()
        else:
            raise ValidationError(f"Unsupported instrument type: {type(instrument)}")