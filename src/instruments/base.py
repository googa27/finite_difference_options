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
from findiff import FinDiff, BoundaryConditions

from ..utils.exceptions import ValidationError
from ..processes.base import StochasticProcess
from ..spatial_operator import SpatialOperator


class Instrument(BaseModel):
    """Base class for all financial instruments."""
    
    model_config = ConfigDict(frozen=True, extra='forbid')
    
    maturity: float
    
    @field_validator('maturity')
    @classmethod
    def validate_maturity(cls, v):
        """Validate maturity."""
        if v <= 0:
            raise ValidationError(f"Maturity must be positive, got {v}")
        return v
    
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
        raise NotImplementedError("Subclasses must implement payoff method")
    
    def generator(self, s: NDArray[np.float64]) -> FinDiff:
        """Return the discretised infinitesimal generator on the spatial grid.
        
        This method is used by the PDE solver to construct the spatial operator.
        
        Parameters
        ----------
        s : NDArray[np.float64]
            Spatial grid for the underlying asset price.
            
        Returns
        -------
        FinDiff
            Discretised infinitesimal generator.
        """
        raise NotImplementedError("Subclasses must implement generator method")
    
    def boundary_conditions(self, s: NDArray[np.float64]) -> BoundaryConditions:
        """Return boundary conditions for the spatial grid.
        
        This method is used by the PDE solver to construct the boundary conditions.
        
        Parameters
        ----------
        s : NDArray[np.float64]
            Spatial grid for the underlying asset price.
            
        Returns
        -------
        BoundaryConditions
            Boundary conditions for the spatial grid.
        """
        raise NotImplementedError("Subclasses must implement boundary_conditions method")


class EuropeanOption(Instrument):
    """Base class for European options."""
    
    strike: float
    model: StochasticProcess
    
    model_config = ConfigDict(
        frozen=True, 
        extra='forbid',
        arbitrary_types_allowed=True  # Allow arbitrary types like StochasticProcess
    )
    
    @field_validator('strike')
    @classmethod
    def validate_strike(cls, v):
        """Validate strike price."""
        if v <= 0:
            raise ValidationError(f"Strike price must be positive, got {v}")
        return v
    
    def generator(self, s: NDArray[np.float64]) -> FinDiff:
        """Return the discretised infinitesimal generator on the spatial grid."""
        operator = SpatialOperator(self.model)
        return operator.build(s)
    
    def boundary_conditions(self, s: NDArray[np.float64]) -> BoundaryConditions:
        """Return boundary conditions for the spatial grid."""
        # This will be implemented by subclasses
        raise NotImplementedError("Subclasses must implement boundary_conditions method")


class EuropeanCall(EuropeanOption):
    """European call option."""
    
    def payoff(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute call option payoff."""
        return np.maximum(state - self.strike, 0.0)
    
    def boundary_conditions(self, s: NDArray[np.float64]) -> BoundaryConditions:
        """Return boundary conditions for the spatial grid."""
        from ..boundary_conditions.builder import BlackScholesBoundaryBuilder
        builder = BlackScholesBoundaryBuilder()
        return builder.build(s, self)


class EuropeanPut(EuropeanOption):
    """European put option."""
    
    def payoff(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute put option payoff."""
        return np.maximum(self.strike - state, 0.0)
    
    def boundary_conditions(self, s: NDArray[np.float64]) -> BoundaryConditions:
        """Return boundary conditions for the spatial grid."""
        from ..boundary_conditions.builder import BlackScholesBoundaryBuilder
        builder = BlackScholesBoundaryBuilder()
        return builder.build(s, self)