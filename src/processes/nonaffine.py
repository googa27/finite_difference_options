"""Non-affine stochastic process models.

This module provides two-factor and one-factor process implementations with
state-dependent coefficients that are not representable as globally affine forms.

These models are primarily used when closed-form affine drift/covariance
factorisations are unavailable but finite-difference numerics can still be
constructed.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, field_validator, ConfigDict

from .base import FactorRole, NonAffineProcess, ProcessDimension, ProcessFactorMetadata
from ..validation import validate_positive
from ..utils.process_validators import validate_cev_beta, validate_sabr_parameters



class ConstantElasticityVariance(NonAffineProcess, BaseModel):
    r"""Constant Elasticity of Variance (CEV) process.

    .. math::

       dS_t = \mu S_t dt + \sigma S_t^{\beta} dW_t

    Notes
    -----
    ``beta=1`` recovers geometric Brownian motion; ``beta<1`` increases leverage
    in the diffusion term as spot decreases.
    """
    
    mu: float
    sigma: float
    beta: float
    
    model_config = ConfigDict(frozen=True, extra='forbid')
    
    @field_validator('sigma')
    @classmethod
    def validate_sigma(cls, v):
        """Validate sigma parameter."""
        validate_positive(v, "sigma")
        return v
    
    @field_validator('beta')
    @classmethod
    def validate_beta(cls, v):
        """Validate beta parameter."""
        validate_cev_beta(v)
        return v
    
    @property
    def dimension(self) -> ProcessDimension:
        return ProcessDimension(value=1)

    def factor_metadata(self) -> tuple[ProcessFactorMetadata, ...]:
        """CEV's sole state coordinate is a tradable spot."""

        return (
            ProcessFactorMetadata(
                name="spot",
                role=FactorRole.TRADABLE_SPOT,
                coordinate="spot",
                asset_id="spot",
            ),
        )
    
    def drift(self, time: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Compute CEV drift :math:`\mu S_t` for each input state point."""
        self.validate_state(state)
        
        if state.ndim == 1:
            return np.array([self.mu * state[0]])
        else:
            return self.mu * state[:, 0:1]
    
    def covariance(self, time: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Compute CEV covariance :math:`\sigma^2 S_t^{2\beta}`.

        Returns
        -------
        NDArray[np.float64]
            Matrix of shape ``(1,1)`` or ``(n, 1, 1)``.
        """
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


class SABRModel(NonAffineProcess, BaseModel):
    r"""SABR (Stochastic Alpha Beta Rho) model.

    In two-factor form with state ``(F, sigma)``:

    .. math::

       dF_t = \sigma_t F_t^{\beta} dW_t^{(1)}
       d\sigma_t = \alpha \sigma_t dW_t^{(2)}
       dW_t^{(1)} dW_t^{(2)} = \rho dt

    Notes
    -----
    The implementation assumes risk-neutral forward drift for the chosen
    convention and validates ``alpha``, ``beta`` and ``rho`` at construction.
    """
    
    alpha: float
    beta: float
    rho: float
    
    model_config = ConfigDict(frozen=True, extra='forbid')
    
    def __init__(self, **data):
        """Initialize and validate SABR parameters."""
        super().__init__(**data)
        validate_sabr_parameters(self.alpha, self.beta, self.rho)
    
    @property
    def dimension(self) -> ProcessDimension:
        return ProcessDimension(value=2)

    def factor_metadata(self) -> tuple[ProcessFactorMetadata, ...]:
        """SABR state is ``(forward, volatility)``; only the first factor is tradable."""

        return (
            ProcessFactorMetadata(
                name="forward",
                role=FactorRole.TRADABLE_SPOT,
                coordinate="forward",
                asset_id="forward",
            ),
            ProcessFactorMetadata(name="volatility", role=FactorRole.VOLATILITY, coordinate="volatility"),
        )
    
    def drift(self, time: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return SABR model drift vector.

        In this simplified implementation, risk-neutral forward drift is assumed
        to be zero for the two state coordinates.
        """
        self.validate_state(state)
        
        if state.ndim == 1:
            return np.zeros(2)
        else:
            batch_size = state.shape[0]
            return np.zeros((batch_size, 2))
    
    def covariance(self, time: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute SABR diffusion matrix for each state point."""
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
    """Create a single-factor CEV process instance."""
    return ConstantElasticityVariance(mu=mu, sigma=sigma, beta=beta)


def create_sabr_model(alpha: float, beta: float, rho: float) -> SABRModel:
    """Create a two-factor SABR model instance."""
    return SABRModel(alpha=alpha, beta=beta, rho=rho)