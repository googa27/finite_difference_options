"""Affine stochastic process models.

The classes and helpers in this module define common affine SDE models used by
pricing solvers.  All models expose dimensioned drift and covariance terms that
are compatible with the finite-difference operator builders.

Each model follows the ``StochasticProcess`` contract:

- ``drift`` and ``covariance`` use the shared calendar-time convention
- state is ``(d,)`` for a single point or ``(n, d)`` for a batch
- validated parameters are stored immutably (Pydantic ``frozen`` models)
"""
from __future__ import annotations

from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, field_validator, ConfigDict

from src.exceptions import ValidationError
from .base import AffineProcess, ProcessDimension
from ..validation import validate_positive, validate_non_negative
from ..utils.process_validators import (
    validate_feller_condition,
    validate_heston_parameters
)


class GeometricBrownianMotion(AffineProcess, BaseModel):
    r"""Geometric Brownian Motion:

    .. math::

       dS_t = \mu S_t dt + \sigma S_t dW_t

    Notes
    -----
    This is the Black--Scholes process used for vanilla equity dynamics.

    Examples
    --------
    >>> from src.processes.affine import GeometricBrownianMotion
    >>> gbm = GeometricBrownianMotion(mu=0.03, sigma=0.2)
    >>> gbm.dimension.value
    1
    >>> gbm.drift(0.0, [100.0]).shape
    (1,)
    """
    
    mu: float
    sigma: float
    
    model_config = ConfigDict(frozen=True, extra='forbid')
    
    @field_validator('sigma')
    @classmethod
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
        """Fail closed because ``Σ(S)=σ²S²`` is quadratic in native spot."""
        raise ValidationError(
            "GeometricBrownianMotion does not have exact affine covariance in native state coordinates; "
            "use covariance(...) or evaluate_coefficients(...) for native-state coefficients"
        )
    
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


class OrnsteinUhlenbeck(AffineProcess, BaseModel):
    r"""Ornstein--Uhlenbeck (Vasicek) process.

    .. math::

       dr_t = \kappa(\theta - r_t)dt + \sigma dW_t

    Notes
    -----
    The state is interpreted as short rate in calendar time.
    """
    
    kappa: float
    theta: float
    sigma: float
    
    model_config = ConfigDict(frozen=True, extra='forbid')
    
    @field_validator('kappa')
    @classmethod
    def validate_kappa(cls, v):
        """Validate kappa parameter."""
        validate_positive(v, "kappa")
        return v
    
    @field_validator('sigma')
    @classmethod
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


class CoxIngersollRoss(AffineProcess, BaseModel):
    r"""Cox--Ingersoll--Ross (CIR) short-rate model.

    .. math::

       dr_t = \kappa(\theta-r_t)dt + \sigma\sqrt{r_t} dW_t

    The Feller condition is validated at construction time to avoid negative
    variance under idealized analytical assumptions.
    """
    
    kappa: float
    theta: float
    sigma: float
    
    model_config = ConfigDict(frozen=True, extra='forbid')
    
    @field_validator('kappa')
    @classmethod
    def validate_kappa(cls, v):
        """Validate kappa parameter."""
        validate_positive(v, "kappa")
        return v
    
    @field_validator('theta')
    @classmethod
    def validate_theta(cls, v):
        """Validate theta parameter."""
        validate_positive(v, "theta")
        return v
    
    @field_validator('sigma')
    @classmethod
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


class HestonModel(AffineProcess, BaseModel):
    r"""Two-factor Heston stochastic-volatility model.

    Equations
    ---------

    .. math::

       dS_t = (r - q)S_t dt + \sqrt{v_t} S_t dW_t^{(1)}
       dv_t = \kappa(\theta-v_t)dt + \sigma\sqrt{v_t} dW_t^{(2)}
       dW_t^{(1)} dW_t^{(2)} = \rho dt

    Notes
    -----
    The state vector is ``(S, v)`` and both coordinates are validated for
    non-negativity where applicable.
    """
    
    risk_free_rate: float
    kappa: float
    theta: float
    sigma: float
    rho: float
    dividend_yield: float = 0.0
    
    model_config = ConfigDict(frozen=True, extra='forbid')
    
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
        """Fail closed because native-state Heston covariance is quadratic/bilinear."""
        raise ValidationError(
            "HestonModel does not have exact affine covariance in native state coordinates; "
            "use covariance(...) or evaluate_coefficients(...) for native-state coefficients"
        )
    
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
    """Create a one-factor Black--Scholes process.

    Parameters
    ----------
    mu : float
        Drift under pricing measure.
    sigma : float
        Spot volatility.

    Returns
    -------
    GeometricBrownianMotion
        Configured process instance.
    """
    return GeometricBrownianMotion(mu=mu, sigma=sigma)


def create_vasicek_process(kappa: float, theta: float, sigma: float) -> OrnsteinUhlenbeck:
    """Create a Vasicek short-rate process.

    This is the canonical one-factor affine short-rate configuration used for
    simple fixed-income sanity checks.
    """
    return OrnsteinUhlenbeck(kappa=kappa, theta=theta, sigma=sigma)


def create_cir_process(kappa: float, theta: float, sigma: float) -> CoxIngersollRoss:
    """Create a CIR short-rate process.

    Parameters are validated against the Feller condition; the caller receives an
    exception early if positivity constraints are not satisfied.
    """
    return CoxIngersollRoss(kappa=kappa, theta=theta, sigma=sigma)


def create_standard_heston(
    r: float = 0.05,
    kappa: float = 2.0,
    theta: float = 0.04,
    sigma: float = 0.3,
    rho: float = -0.7,
    dividend_yield: float = 0.0
) -> HestonModel:
    """Create a standard Heston model with conservative defaults.

    The defaults are a common starting point for regression tests and documentation
    examples, not a calibrated production set.
    """
    return HestonModel(
        risk_free_rate=r,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        rho=rho,
        dividend_yield=dividend_yield
    )