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
from .base import AffineCovarianceForm, AffineProcess, FactorRole, ProcessDimension, ProcessFactorMetadata
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

    def factor_metadata(self) -> tuple[ProcessFactorMetadata, ...]:
        """GBM's sole state coordinate is a tradable spot."""

        return (
            ProcessFactorMetadata(
                name="spot",
                role=FactorRole.TRADABLE_SPOT,
                coordinate="spot",
                asset_id="spot",
            ),
        )
    
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

    def factor_metadata(self) -> tuple[ProcessFactorMetadata, ...]:
        """Vasicek's state coordinate is a short rate, not an equity spot."""

        return (
            ProcessFactorMetadata(
                name="short_rate",
                role=FactorRole.SHORT_RATE,
                coordinate="short_rate",
            ),
        )
    
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

    def factor_metadata(self) -> tuple[ProcessFactorMetadata, ...]:
        """CIR's state coordinate is a short rate."""

        return (
            ProcessFactorMetadata(
                name="short_rate",
                role=FactorRole.SHORT_RATE,
                coordinate="short_rate",
            ),
        )
    
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

    The executable process state is ``(x, v) = (log(S), variance)``:

    .. math::

       dx_t = (r - q - \tfrac12 v_t)dt + \sqrt{v_t} dW_t^{(1)}

    Payoffs receive spot through the explicit ``exp`` transform declared in
    :meth:`factor_metadata`; solver/operator coefficients stay in log-spot
    coordinates.  Negative variance states fail closed rather than being
    silently clipped.
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

    def factor_metadata(self) -> tuple[ProcessFactorMetadata, ...]:
        """Return explicit state-factor roles for ``(log_spot, variance)``."""

        return (
            ProcessFactorMetadata(
                name="log_spot",
                role=FactorRole.TRADABLE_SPOT,
                coordinate="log_spot",
                asset_id="spot",
                payoff_transform="exp",
            ),
            ProcessFactorMetadata(name="variance", role=FactorRole.VARIANCE, coordinate="variance"),
        )
    
    def affine_drift_coefficients(self, time: float = 0.0) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Drift in log-spot state: ``(r-q-0.5v, kappa(theta-v))``."""
        alpha = np.array([self.risk_free_rate - self.dividend_yield, self.kappa * self.theta])
        beta = np.array([
            [0.0, -0.5],
            [0.0, -self.kappa]
        ])
        return alpha, beta
    
    def affine_covariance_coefficients(self, time: float = 0.0) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Exact covariance tensor in ``(log_spot, variance)`` coordinates."""
        constant = np.zeros((2, 2), dtype=float)
        linear = np.zeros((2, 2, 2), dtype=float)
        linear[1] = np.array(
            [
                [1.0, self.rho * self.sigma],
                [self.rho * self.sigma, self.sigma**2],
            ],
            dtype=float,
        )
        return constant, linear

    def validate_state(self, state: NDArray[np.float64]) -> None:
        """Validate Heston log-spot/variance state coordinates."""

        super().validate_state(state)
        states = np.asarray(state, dtype=float)
        variance = states[1] if states.ndim == 1 else states[:, 1]
        if np.any(variance < 0.0):
            raise ValidationError("Heston variance coordinate must be non-negative")
    
    def covariance(self, time: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute Heston covariance in ``(log_spot, variance)`` coordinates."""

        state_array = np.asarray(state, dtype=float)
        self.validate_state(state_array)
        constant, linear = self.affine_covariance_coefficients(time)
        covariance = AffineCovarianceForm.from_coefficients(constant, linear).evaluate(state_array)
        return covariance[0] if state_array.ndim == 1 else covariance

    def discount(self, time: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the Heston pricing reaction field, independent of drift."""

        _ = time
        _, states, _ = self._canonical_state_batch(state)
        return np.full(states.shape[0], self.risk_free_rate, dtype=float)

    @staticmethod
    def state_from_spot(
        spot: float | NDArray[np.float64],
        variance: float | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Map spot/variance inputs to Heston's ``(log_spot, variance)`` state."""

        spot_array = np.asarray(spot, dtype=float)
        variance_array = np.asarray(variance, dtype=float)
        if np.any(spot_array <= 0.0) or not np.all(np.isfinite(spot_array)):
            raise ValidationError("spot must be finite and strictly positive for log-state conversion")
        if np.any(variance_array < 0.0) or not np.all(np.isfinite(variance_array)):
            raise ValidationError("variance must be finite and non-negative for Heston state conversion")
        return np.stack(np.broadcast_arrays(np.log(spot_array), variance_array), axis=-1)

    @staticmethod
    def spot_from_state(state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map Heston log-spot state coordinate(s) back to spot values."""

        states = np.asarray(state, dtype=float)
        log_spot = states[0] if states.ndim == 1 else states[:, 0]
        return np.exp(log_spot)


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