"""Multi-dimensional stochastic processes for advanced option pricing models.

This module implements stochastic processes in 2D and 3D using drift and volatility
parameterization, which is more natural for multi-dimensional models than the
generator-based approach used for 1D processes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .validation import validate_positive
from .exceptions import ValidationError


class ProcessDimension(Enum):
    """Enumeration of supported process dimensions."""
    TWO_D = 2
    THREE_D = 3


class MultiDimensionalProcess(ABC):
    """Abstract base class for multi-dimensional stochastic processes.
    
    Multi-dimensional processes are parameterized by drift and volatility functions
    rather than generators, which provides more flexibility for complex models.
    """
    
    @property
    @abstractmethod
    def dimension(self) -> ProcessDimension:
        """Return the dimension of the stochastic process."""
        ...
    
    @abstractmethod
    def drift(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute drift vector at given state(s).
        
        Parameters
        ----------
        state : NDArray[np.float64]
            State vector(s) with shape (..., d) where d is dimension.
            
        Returns
        -------
        NDArray[np.float64]
            Drift vector(s) with same shape as input.
        """
        ...
    
    @abstractmethod
    def diffusion_matrix(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute diffusion matrix at given state(s).
        
        Parameters
        ----------
        state : NDArray[np.float64]
            State vector(s) with shape (..., d) where d is dimension.
            
        Returns
        -------
        NDArray[np.float64]
            Diffusion matrix with shape (..., d, d).
        """
        ...
    
    @abstractmethod
    def validate_parameters(self) -> None:
        """Validate model parameters."""
        ...


@dataclass
class HestonModel(MultiDimensionalProcess):
    """Heston stochastic volatility model.
    
    The Heston model describes the evolution of an asset price S and its variance V:
    dS = r*S*dt + sqrt(V)*S*dW1
    dV = κ(θ-V)*dt + σ_v*sqrt(V)*dW2
    
    where dW1 and dW2 are correlated Brownian motions with correlation ρ.
    
    Parameters
    ----------
    r : float
        Risk-free interest rate.
    kappa : float
        Mean reversion speed for variance.
    theta : float
        Long-term variance level.
    sigma_v : float
        Volatility of variance (vol of vol).
    rho : float
        Correlation between asset and variance Brownian motions.
    """
    
    r: float
    kappa: float
    theta: float
    sigma_v: float
    rho: float

    def __post_init__(self) -> None:
        """Validate Heston parameters."""
        self.validate_parameters()

    @property
    def dimension(self) -> ProcessDimension:
        """Return 2D dimension for Heston model."""
        return ProcessDimension.TWO_D

    def validate_parameters(self) -> None:
        """Validate Heston model parameters."""
        validate_positive(self.kappa, "kappa")
        validate_positive(self.theta, "theta")
        validate_positive(self.sigma_v, "sigma_v")
        
        if not -1.0 <= self.rho <= 1.0:
            raise ValidationError(f"Correlation rho must be between -1 and 1, got {self.rho}")
        
        # Check Feller condition: 2κθ >= σ_v²
        feller_lhs = 2 * self.kappa * self.theta
        feller_rhs = self.sigma_v ** 2
        if feller_lhs < feller_rhs:
            raise ValidationError(
                f"Feller condition violated: 2κθ = {feller_lhs:.6f} < σ_v² = {feller_rhs:.6f}. "
                "This may lead to negative variance."
            )

    def drift(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute Heston drift vector.
        
        Parameters
        ----------
        state : NDArray[np.float64]
            State vector [S, V] or array of state vectors.
            
        Returns
        -------
        NDArray[np.float64]
            Drift vector [r*S, κ(θ-V)].
        """
        state = np.asarray(state)
        
        if state.ndim == 1:
            # Single state vector
            s, v = state[0], state[1]
            drift_s = self.r * s
            drift_v = self.kappa * (self.theta - v)
            return np.array([drift_s, drift_v])
        else:
            # Array of state vectors
            s = state[..., 0]
            v = state[..., 1]
            drift_s = self.r * s
            drift_v = self.kappa * (self.theta - v)
            
            result = np.empty_like(state)
            result[..., 0] = drift_s
            result[..., 1] = drift_v
            return result

    def diffusion_matrix(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute Heston diffusion matrix.
        
        The diffusion matrix G satisfies: dX = μ dt + G dW
        where dW is a vector of independent Brownian motions.
        
        Parameters
        ----------
        state : NDArray[np.float64]
            State vector [S, V] or array of state vectors.
            
        Returns
        -------
        NDArray[np.float64]
            Diffusion matrix with shape (..., 2, 2).
        """
        state = np.asarray(state)
        
        if state.ndim == 1:
            # Single state vector
            s, v = state[0], state[1]
            sqrt_v = np.sqrt(np.maximum(v, 0.0))  # Ensure non-negative
            
            # Diffusion matrix for correlated Brownian motions
            g11 = s * sqrt_v
            g12 = 0.0
            g21 = self.rho * self.sigma_v * sqrt_v
            g22 = self.sigma_v * sqrt_v * np.sqrt(1 - self.rho**2)
            
            return np.array([[g11, g12], [g21, g22]])
        else:
            # Array of state vectors
            s = state[..., 0]
            v = state[..., 1]
            sqrt_v = np.sqrt(np.maximum(v, 0.0))
            
            # Create diffusion matrices
            shape = state.shape[:-1] + (2, 2)
            diffusion = np.zeros(shape)
            
            diffusion[..., 0, 0] = s * sqrt_v
            diffusion[..., 0, 1] = 0.0
            diffusion[..., 1, 0] = self.rho * self.sigma_v * sqrt_v
            diffusion[..., 1, 1] = self.sigma_v * sqrt_v * np.sqrt(1 - self.rho**2)
            
            return diffusion


@dataclass
class ThreeFactorModel(MultiDimensionalProcess):
    """Three-factor stochastic model example.
    
    This is a generic three-dimensional model where each factor follows:
    dX_i = μ_i*X_i*dt + σ_i*X_i*dW_i
    
    with correlated Brownian motions.
    
    Parameters
    ----------
    r : float
        Risk-free rate.
    mu : NDArray[np.float64]
        Drift parameters for each factor (length 3).
    sigma : NDArray[np.float64]
        Volatility parameters for each factor (length 3).
    correlation_matrix : NDArray[np.float64]
        3x3 correlation matrix for Brownian motions.
    """
    
    r: float
    mu: NDArray[np.float64]
    sigma: NDArray[np.float64]
    correlation_matrix: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate three-factor model parameters."""
        self.mu = np.asarray(self.mu)
        self.sigma = np.asarray(self.sigma)
        self.correlation_matrix = np.asarray(self.correlation_matrix)
        self.validate_parameters()

    @property
    def dimension(self) -> ProcessDimension:
        """Return 3D dimension for three-factor model."""
        return ProcessDimension.THREE_D

    def validate_parameters(self) -> None:
        """Validate three-factor model parameters."""
        if len(self.mu) != 3:
            raise ValidationError(f"mu must have length 3, got {len(self.mu)}")
        
        if len(self.sigma) != 3:
            raise ValidationError(f"sigma must have length 3, got {len(self.sigma)}")
        
        if self.correlation_matrix.shape != (3, 3):
            raise ValidationError(f"correlation_matrix must be 3x3, got {self.correlation_matrix.shape}")
        
        # Check if correlation matrix is symmetric
        if not np.allclose(self.correlation_matrix, self.correlation_matrix.T):
            raise ValidationError("Correlation matrix must be symmetric")
        
        # Check if correlation matrix is positive definite
        eigenvals = np.linalg.eigvals(self.correlation_matrix)
        if np.any(eigenvals <= 0):
            raise ValidationError("Correlation matrix must be positive definite")
        
        # Check diagonal elements are 1
        if not np.allclose(np.diag(self.correlation_matrix), 1.0):
            raise ValidationError("Correlation matrix diagonal elements must be 1")
        
        # Check volatilities are positive
        for i, sig in enumerate(self.sigma):
            validate_positive(sig, f"sigma[{i}]")

    def drift(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute three-factor drift vector.
        
        Parameters
        ----------
        state : NDArray[np.float64]
            State vector [X1, X2, X3] or array of state vectors.
            
        Returns
        -------
        NDArray[np.float64]
            Drift vector [μ1*X1, μ2*X2, μ3*X3].
        """
        state = np.asarray(state)
        
        if state.ndim == 1:
            # Single state vector
            return self.mu * state
        else:
            # Array of state vectors
            return self.mu[np.newaxis, :] * state

    def diffusion_matrix(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute three-factor diffusion matrix.
        
        Parameters
        ----------
        state : NDArray[np.float64]
            State vector [X1, X2, X3] or array of state vectors.
            
        Returns
        -------
        NDArray[np.float64]
            Diffusion matrix with shape (..., 3, 3).
        """
        state = np.asarray(state)
        
        # Compute Cholesky decomposition of correlation matrix
        chol = np.linalg.cholesky(self.correlation_matrix)
        
        if state.ndim == 1:
            # Single state vector
            diagonal_vol = np.diag(self.sigma * state)
            return diagonal_vol @ chol.T
        else:
            # Array of state vectors
            shape = state.shape[:-1] + (3, 3)
            diffusion = np.zeros(shape)
            
            for i in range(3):
                for j in range(3):
                    diffusion[..., i, j] = self.sigma[i] * state[..., i] * chol[i, j]
            
            return diffusion


# Convenience functions for creating standard models

def create_standard_heston(
    r: float = 0.05,
    kappa: float = 2.0,
    theta: float = 0.04,
    sigma_v: float = 0.3,
    rho: float = -0.7
) -> HestonModel:
    """Create Heston model with standard parameters.
    
    Parameters
    ----------
    r : float, optional
        Risk-free rate (default: 0.05).
    kappa : float, optional
        Mean reversion speed (default: 2.0).
    theta : float, optional
        Long-term variance (default: 0.04).
    sigma_v : float, optional
        Vol of vol (default: 0.3).
    rho : float, optional
        Correlation (default: -0.7).
        
    Returns
    -------
    HestonModel
        Configured Heston model.
    """
    return HestonModel(r=r, kappa=kappa, theta=theta, sigma_v=sigma_v, rho=rho)


def create_uncorrelated_three_factor(
    r: float = 0.05,
    mu: Optional[NDArray[np.float64]] = None,
    sigma: Optional[NDArray[np.float64]] = None
) -> ThreeFactorModel:
    """Create uncorrelated three-factor model.
    
    Parameters
    ----------
    r : float, optional
        Risk-free rate (default: 0.05).
    mu : NDArray[np.float64], optional
        Drift parameters (default: [0.1, 0.08, 0.06]).
    sigma : NDArray[np.float64], optional
        Volatility parameters (default: [0.2, 0.15, 0.1]).
        
    Returns
    -------
    ThreeFactorModel
        Configured three-factor model.
    """
    if mu is None:
        mu = np.array([0.1, 0.08, 0.06])
    if sigma is None:
        sigma = np.array([0.2, 0.15, 0.1])
    
    correlation_matrix = np.eye(3)
    
    return ThreeFactorModel(r=r, mu=mu, sigma=sigma, correlation_matrix=correlation_matrix)
