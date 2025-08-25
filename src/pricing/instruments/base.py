"""Base classes for financial instruments.

This module contains the abstract interfaces for financial instruments
in the unified pricing framework.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


class UnifiedInstrument(ABC):
    """Abstract base class for financial instruments in unified framework."""
    
    @abstractmethod
    def payoff(self, *grids: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute instrument payoff at maturity.
        
        Parameters
        ----------
        *grids : NDArray[np.float64]
            Spatial grids for each dimension.
            
        Returns
        -------
        NDArray[np.float64]
            Payoff values on the grid.
        """
        ...
    
    # Define maturity as a regular field that subclasses must implement
    # This will be implemented as a Pydantic field in subclasses
    maturity: float
