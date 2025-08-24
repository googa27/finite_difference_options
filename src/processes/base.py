"""Base classes for stochastic processes.

This module contains the abstract interfaces and base implementations
for all stochastic processes in the unified framework.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, field_validator, ConfigDict

from ..utils.exceptions import ValidationError
from ..utils.state_handling import ensure_state_array, validate_state_dimensions


class ProcessDimension(BaseModel):
    """Represents the dimension of a stochastic process."""
    
    value: int
    
    model_config = ConfigDict(frozen=True, extra='forbid')
    
    @field_validator('value')
    @classmethod
    def validate_value(cls, v):
        """Validate that dimension is positive."""
        if v < 1:
            raise ValidationError(f"Process dimension must be positive, got {v}")
        return v
    
    @property
    def is_univariate(self) -> bool:
        """Check if process is 1-dimensional."""
        return self.value == 1
    
    @property
    def is_multivariate(self) -> bool:
        """Check if process is multi-dimensional."""
        return self.value > 1
    
    def __eq__(self, other) -> bool:
        if isinstance(other, ProcessDimension):
            return self.value == other.value
        return self.value == other
    
    def __repr__(self) -> str:
        return f"ProcessDimension(value={self.value})"


class ProcessType(Enum):
    """Enumeration of process types for optimization purposes."""
    AFFINE = "affine"
    NON_AFFINE = "non_affine"


class StochasticProcess(ABC):
    """Abstract base class for all stochastic processes."""
    
    @property
    @abstractmethod
    def dimension(self) -> ProcessDimension:
        """Get process dimension."""
        ...
    
    @property
    @abstractmethod
    def process_type(self) -> ProcessType:
        """Get process type for optimization."""
        ...
    
    @abstractmethod
    def drift(self, time: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute drift vector μ(x,t).
        
        Parameters
        ----------
        time : float
            Current time.
        state : NDArray[np.float64]
            Current state vector(s).
            
        Returns
        -------
        NDArray[np.float64]
            Drift vector(s).
        """
        ...
    
    @abstractmethod
    def covariance(self, time: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute covariance matrix Σ(x,t).
        
        Parameters
        ----------
        time : float
            Current time.
        state : NDArray[np.float64]
            Current state vector(s).
            
        Returns
        -------
        NDArray[np.float64]
            Covariance matrix/matrices.
        """
        ...
    
    def validate_state(self, state: NDArray[np.float64]) -> None:
        """Validate state vector dimensions and values."""
        state = ensure_state_array(state)
        validate_state_dimensions(state, self.dimension.value, self.__class__.__name__)
    
    def diffusion(self, time: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute diffusion matrix σ(x,t) from covariance.
        
        This is a convenience method that computes the matrix square root
        of the covariance matrix.
        
        Parameters
        ----------
        time : float
            Current time.
        state : NDArray[np.float64]
            Current state vector(s).
            
        Returns
        -------
        NDArray[np.float64]
            Diffusion matrix/matrices.
        """
        from ..utils.covariance_utils import matrix_sqrt
        
        cov = self.covariance(time, state)
        
        if cov.ndim == 2:
            # Single covariance matrix
            return matrix_sqrt(cov)
        else:
            # Batch of covariance matrices
            batch_size = cov.shape[0]
            dim = cov.shape[1]
            result = np.zeros((batch_size, dim, dim))
            
            for i in range(batch_size):
                result[i] = matrix_sqrt(cov[i])
            
            return result


class AffineProcess(StochasticProcess):
    """Base class for affine stochastic processes.
    
    Affine processes have linear drift and affine covariance:
    - μ(x,t) = α(t) + β(t)x
    - Σ(x,t) = γ(t) + δ(t)⊙x (⊙ is element-wise or appropriate operation)
    
    This structure enables computational optimizations and analytical solutions.
    """
    
    @property
    def process_type(self) -> ProcessType:
        return ProcessType.AFFINE
    
    @abstractmethod
    def affine_drift_coefficients(
        self, 
        time: float = 0.0
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get affine drift coefficients α(t), β(t).
        
        Returns
        -------
        Tuple[NDArray[np.float64], NDArray[np.float64]]
            (α, β) where μ(x,t) = α + βx
        """
        ...
    
    @abstractmethod
    def affine_covariance_coefficients(
        self, 
        time: float = 0.0
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get affine covariance coefficients γ(t), δ(t).
        
        Returns
        -------
        Tuple[NDArray[np.float64], NDArray[np.float64]]
            (γ, δ) where Σ(x,t) = γ + δ⊙x (⊙ is element-wise or appropriate operation)
        """
        ...
    
    def drift(
        self, 
        time: float, 
        state: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute affine drift μ(x,t) = α(t) + β(t)x."""
        # Ensure state is array
        state = ensure_state_array(state)
        self.validate_state(state)
        
        alpha, beta = self.affine_drift_coefficients(time)
        
        if state.ndim == 1:
            # For single state, alpha is (d,) and beta is (d, d)
            # Return alpha + beta @ state
            return alpha + beta @ state
        else:
            # Vectorized computation for multiple states
            # state is (n, d), alpha is (d,), beta is (d, d)
            # For each row of state, compute alpha + beta @ state[i]
            n_states = state.shape[0]
            result = np.zeros((n_states, self.dimension.value))
            for i in range(n_states):
                result[i] = alpha + beta @ state[i]
            return result
    
    def covariance(
        self,
        time: float,
        state: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute affine covariance Σ(x,t) = γ(t) + δ(t)⊙x."""
        # Ensure state is array
        state = ensure_state_array(state)
        self.validate_state(state)
        
        gamma, delta = self.affine_covariance_coefficients(time)
        
        if state.ndim == 1:
            # For 1D case, delta operation is element-wise multiplication
            return gamma + delta * state
        else:
            # Vectorized computation for multiple states
            return gamma[None, :, :] + delta[None, :, :] * state[:, :, None]


class NonAffineProcess(StochasticProcess):
    """Base class for non-affine stochastic processes."""
    
    @property
    def process_type(self) -> ProcessType:
        return ProcessType.NON_AFFINE