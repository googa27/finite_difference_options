"""Affine stochastic processes implementation.

This module contains implementations of common affine stochastic processes
used in quantitative finance, utilizing the new validation utilities.
"""
from __future__ import annotations

from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from pydantic import Field, validator

from .base import AffineProcess, ProcessDimension
from ..utils.validation import validate_positive, validate_non_negative
from ..utils.process_validators import (
    validate_feller_condition, 
    validate_correlation_parameter,
    validate_cev_beta,
    validate_heston_parameters
)
from ..utils.state_handling import validate_positive_state_components


class GeometricBrownianMotion(AffineProcess):
    """Geometric Brownian Motion process.
    
    dS = μS dt + σS dW
    
    Parameters
    ----------
    mu : float
        Drift parameter.
    sigma : float
        Volatility parameter.
    """
    
    mu: float
    sigma: float
    
    class Config:
        """Pydantic configuration."""
        allow_mutation = False  # Make it immutable like a dataclass
        extra = "forbid"  # Prevent extra fields
    
    @validator('sigma')
    def validate_sigma(cls, v):
        """Validate sigma parameter."""
        validate_positive(v, "sigma")
        return v
    
    @property
    def dimension(self) -> ProcessDimension:
        return ProcessDimension(value=1)
    
    def affine_drift_coefficients(self, time: float = 0.0) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """μ(S) = μS, so α=0, β=μ."""
        return np.array([0.0]), np.array([self.mu])
    
    def affine_covariance_coefficients(self, time: float = 0.0) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Σ(S) = σ²S², so γ=0, δ=σ²."""
        return np.array([[0.0]]), np.array([[self.sigma**2]])
    
    def covariance(self, time: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Covariance matrix Σ(S) = σ²S²."""
        self.validate_state(state)
        if state.ndim == 1:
            return np.array([[self.sigma**2 * state[0]**2]])
        else:
            batch_size = state.shape[0]
            result = np.zeros((batch_size, 1, 1))
            result[:, 0, 0] = self.sigma**2 * state[:, 0]**2
            return result


class OrnsteinUhlenbeck(AffineProcess):
    """Ornstein-Uhlenbeck process.
    
    dr = κ(θ - r) dt + σ dW
    
    Parameters
    ----------
    kappa : float
        Mean reversion speed.
    theta : float
        Long-term mean.
    sigma : float
        Volatility parameter.
    """
    
    kappa: float
    theta: float
    sigma: float
    
    class Config:
        """Pydantic configuration."""
        allow_mutation = False  # Make it immutable like a dataclass
        extra = "forbid"  # Prevent extra fields
    
    @validator('kappa')
    def validate_kappa(cls, v):
        """Validate kappa parameter."""
        validate_positive(v, "kappa")
        return v
    
    @validator('sigma')
    def validate_sigma(cls, v):
        """Validate sigma parameter."""
        validate_positive(v, "sigma")
        return v
    
    @property
    def dimension(self) -> ProcessDimension:
        return ProcessDimension(value=1)
    
    def affine_drift_coefficients(self, time: float = 0.0) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """μ(r) = κθ - κr, so α=κθ, β=-κ."""
        return np.array([self.kappa * self.theta]), np.array([-self.kappa])
    
    def affine_covariance_coefficients(self, time: float = 0.0) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Σ = σ², so γ=σ², δ=0."""
        return np.array([[self.sigma**2]]), np.array([[0.0]])
    
    def covariance(self, time: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Constant covariance matrix Σ = σ²."""
        self.validate_state(state)
        if state.ndim == 1:
            return np.array([[self.sigma**2]])
        else:
            batch_size = state.shape[0]
            result = np.full((batch_size, 1, 1), self.sigma**2)
            return result


class CoxIngersollRoss(AffineProcess):
    """Cox-Ingersoll-Ross process.
    
    dr = κ(θ - r) dt + σ√r dW
    
    Parameters
    ----------
    kappa : float
        Mean reversion speed.
    theta : float
        Long-term mean.
    sigma : float
        Volatility parameter.
    """
    
    kappa: float
    theta: float
    sigma: float
    
    class Config:
        """Pydantic configuration."""
        allow_mutation = False  # Make it immutable like a dataclass
        extra = "forbid"  # Prevent extra fields
    
    @validator('kappa')
    def validate_kappa(cls, v):
        """Validate kappa parameter."""
        validate_positive(v, "kappa")
        return v
    
    @validator('theta')
    def validate_theta(cls, v):
        """Validate theta parameter."""
        validate_positive(v, "theta")
        return v
    
    @validator('sigma')
    def validate_sigma(cls, v):
        """Validate sigma parameter."""
        validate_positive(v, "sigma")
        return v
    
    def __init__(self, **data):
        """Initialize and validate CIR parameters."""
        super().__init__(**data)
        validate_feller_condition(self.kappa, self.theta, self.sigma, "CIR")
    
    @property
    def dimension(self) -> ProcessDimension:
        return ProcessDimension(value=1)
    
    def affine_drift_coefficients(self, time: float = 0.0) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """μ(r) = κθ - κr, so α=κθ, β=-κ."""
        return np.array([self.kappa * self.theta]), np.array([-self.kappa])
    
    def affine_covariance_coefficients(self, time: float = 0.0) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Σ(r) = σ²r, so γ=0, δ=σ²."""
        return np.array([[0.0]]), np.array([[self.sigma**2]])
    
    def covariance(self, time: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Covariance matrix Σ(r) = σ²r."""
        self.validate_state(state)
        if state.ndim == 1:
            # Ensure non-negative for numerical stability
            r = max(state[0], 1e-10)
            return np.array([[self.sigma**2 * r]])
        else:
            batch_size = state.shape[0]
            result = np.zeros((batch_size, 1, 1))
            r_vals = np.maximum(state[:, 0], 1e-10)
            result[:, 0, 0] = self.sigma**2 * r_vals
            return result


class HestonModel(AffineProcess):
    """Heston stochastic volatility model.
    
    dS = rS dt + √V S dW₁
    dV = κ(θ - V) dt + σ√V dW₂
    
    with correlation ρ between W₁ and W₂.
    
    Parameters
    ----------
    risk_free_rate : float
        Risk-free interest rate.
    kappa : float
        Mean reversion speed of variance.
    theta : float
        Long-term variance.
    sigma : float
        Volatility of volatility.
    rho : float
        Correlation between asset and variance.
    dividend_yield : float, optional
        Dividend yield.
    """
    
    risk_free_rate: float
    kappa: float
    theta: float
    sigma: float
    rho: float
    dividend_yield: float = 0.0
    
    class Config:
        """Pydantic configuration."""
        allow_mutation = False  # Make it immutable like a dataclass
        extra = "forbid"  # Prevent extra fields
    
    def __init__(self, **data):
        """Initialize and validate Heston parameters."""
        super().__init__(**data)
        validate_heston_parameters(self.kappa, self.theta, self.sigma, self.rho)
        validate_non_negative(self.dividend_yield, "dividend_yield")
    
    @property
    def dimension(self) -> ProcessDimension:
        return ProcessDimension(value=2)
    
    def affine_drift_coefficients(self, time: float = 0.0) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Heston drift coefficients."""
        alpha = np.array([0.0, self.kappa * self.theta])
        beta = np.array([
            [self.risk_free_rate - self.dividend_yield, 0.0],
            [0.0, -self.kappa]
        ])
        return alpha, beta
    
    def affine_covariance_coefficients(self, time: float = 0.0) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Heston covariance coefficients."""
        gamma = np.zeros((2, 2))
        delta = np.array([
            [[0.0, 0.0], [0.0, 1.0]],  # S component: V * S²
            [[0.0, 0.0], [0.0, self.sigma**2]]  # V component: σ² * V
        ])
        return gamma, delta
    
    def covariance(self, time: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute Heston covariance matrix."""
        self.validate_state(state)
        
        if state.ndim == 1:
            s, v = state[0], max(state[1], 1e-10)  # Ensure positive variance
            
            # Covariance matrix elements
            var_s = v * s**2  # Variance of S
            var_v = self.sigma**2 * v  # Variance of V
            cov_sv = self.rho * self.sigma * s * v  # Covariance S,V
            
            return np.array([
                [var_s, cov_sv],
                [cov_sv, var_v]
            ])
        else:
            batch_size = state.shape[0]
            result = np.zeros((batch_size, 2, 2))
            
            s_vals = state[:, 0]
            v_vals = np.maximum(state[:, 1], 1e-10)
            
            # Diagonal elements
            result[:, 0, 0] = v_vals * s_vals**2  # Var(S)
            result[:, 1, 1] = self.sigma**2 * v_vals  # Var(V)
            
            # Off-diagonal elements
            cov_sv = self.rho * self.sigma * s_vals * v_vals
            result[:, 0, 1] = cov_sv
            result[:, 1, 0] = cov_sv
            
            return result


# Convenience functions for creating common processes
def create_black_scholes_process(mu: float, sigma: float) -> GeometricBrownianMotion:
    """Create Black-Scholes (GBM) process."""
    return GeometricBrownianMotion(mu=mu, sigma=sigma)


def create_vasicek_process(kappa: float, theta: float, sigma: float) -> OrnsteinUhlenbeck:
    """Create Vasicek (OU) process."""
    return OrnsteinUhlenbeck(kappa=kappa, theta=theta, sigma=sigma)


def create_cir_process(kappa: float, theta: float, sigma: float) -> CoxIngersollRoss:
    """Create CIR process."""
    return CoxIngersollRoss(kappa=kappa, theta=theta, sigma=sigma)


def create_standard_heston(
    r: float = 0.05,
    kappa: float = 2.0,
    theta: float = 0.04,
    sigma: float = 0.3,
    rho: float = -0.7,
    dividend_yield: float = 0.0
) -> HestonModel:
    """Create standard Heston model with typical parameters."""
    return HestonModel(
        risk_free_rate=r,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        rho=rho,
        dividend_yield=dividend_yield
    )