"""Non-affine stochastic processes implementation.

This module contains implementations of non-affine stochastic processes
that don't fit the linear drift/affine covariance structure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from .base import NonAffineProcess, ProcessDimension
from ..utils.validation import validate_positive
from ..utils.process_validators import validate_cev_beta, validate_sabr_parameters
from ..utils.state_handling import validate_positive_state_components


@dataclass
class ConstantElasticityVariance(NonAffineProcess):
    """Constant Elasticity of Variance (CEV) process.
    
    dS = μS dt + σS^β dW
    
    Parameters
    ----------
    mu : float
        Drift parameter.
    sigma : float
        Volatility parameter.
    beta : float
        Elasticity parameter (0 ≤ β ≤ 1).
    """
    
    mu: float
    sigma: float
    beta: float
    
    def __post_init__(self) -> None:
        validate_positive(self.sigma, "sigma")
        validate_cev_beta(self.beta)
    
    @property
    def dimension(self) -> ProcessDimension:
        return ProcessDimension(1)
    
    def drift(self, time: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """CEV drift μ(S) = μS."""
        self.validate_state(state)
        
        if state.ndim == 1:
            return np.array([self.mu * state[0]])
        else:
            return self.mu * state[:, 0:1]
    
    def covariance(self, time: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """CEV covariance Σ(S) = σ²S^(2β)."""
        self.validate_state(state)
        
        if state.ndim == 1:
            s = max(state[0], 1e-10)  # Ensure positive for numerical stability
            return np.array([[self.sigma**2 * s**(2 * self.beta)]])
        else:
            batch_size = state.shape[0]
            result = np.zeros((batch_size, 1, 1))
            s_vals = np.maximum(state[:, 0], 1e-10)
            result[:, 0, 0] = self.sigma**2 * s_vals**(2 * self.beta)
            return result


@dataclass
class SABRModel(NonAffineProcess):
    """SABR (Stochastic Alpha Beta Rho) model.
    
    dF = σF^β dW₁
    dσ = ασ dW₂
    
    with correlation ρ between W₁ and W₂.
    
    Parameters
    ----------
    alpha : float
        Volatility of volatility.
    beta : float
        CEV exponent (0 ≤ β ≤ 1).
    rho : float
        Correlation between forward and volatility.
    """
    
    alpha: float
    beta: float
    rho: float
    
    def __post_init__(self) -> None:
        validate_sabr_parameters(self.alpha, self.beta, self.rho)
    
    @property
    def dimension(self) -> ProcessDimension:
        return ProcessDimension(2)
    
    def drift(self, time: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """SABR drift (zero drift for martingale processes)."""
        self.validate_state(state)
        
        if state.ndim == 1:
            return np.zeros(2)
        else:
            batch_size = state.shape[0]
            return np.zeros((batch_size, 2))
    
    def covariance(self, time: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """SABR covariance matrix."""
        self.validate_state(state)
        
        if state.ndim == 1:
            f, sigma = max(state[0], 1e-10), max(state[1], 1e-10)
            
            # Variance components
            var_f = sigma**2 * f**(2 * self.beta)
            var_sigma = self.alpha**2 * sigma**2
            cov_f_sigma = self.rho * self.alpha * sigma * f**self.beta * sigma
            
            return np.array([
                [var_f, cov_f_sigma],
                [cov_f_sigma, var_sigma]
            ])
        else:
            batch_size = state.shape[0]
            result = np.zeros((batch_size, 2, 2))
            
            f_vals = np.maximum(state[:, 0], 1e-10)
            sigma_vals = np.maximum(state[:, 1], 1e-10)
            
            # Diagonal elements
            result[:, 0, 0] = sigma_vals**2 * f_vals**(2 * self.beta)
            result[:, 1, 1] = self.alpha**2 * sigma_vals**2
            
            # Off-diagonal elements
            cov_fs = self.rho * self.alpha * sigma_vals * f_vals**self.beta * sigma_vals
            result[:, 0, 1] = cov_fs
            result[:, 1, 0] = cov_fs
            
            return result


# Convenience functions
def create_cev_process(mu: float, sigma: float, beta: float) -> ConstantElasticityVariance:
    """Create CEV process."""
    return ConstantElasticityVariance(mu=mu, sigma=sigma, beta=beta)


def create_sabr_model(alpha: float, beta: float, rho: float) -> SABRModel:
    """Create SABR model."""
    return SABRModel(alpha=alpha, beta=beta, rho=rho)
