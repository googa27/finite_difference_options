r"""Base classes for stochastic processes.

This module defines the abstract interfaces and reusable base implementations
for all stochastic processes used by the pricing engines.

Public process objects must expose a drift :math:`\mu(t, x)` and
covariance :math:`\Sigma(t, x)` in calendar-time convention
where time increases forward from valuation (``t = 0``) to maturity.
State coordinates are represented as NumPy arrays and validated against
``ProcessDimension`` before numerical operator construction.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, field_validator, ConfigDict

from src.exceptions import ValidationError
from ..utils.state_handling import ensure_state_array, validate_state_dimensions


class ProcessDimension(BaseModel):
    """Container and validator for process dimensionality.

    The dimension is a small strongly typed value object so that process and
    solver code can reliably branch on state space shape.  A value of ``1``
    indicates a univariate process while values greater than one indicate
    multi-dimensional dynamics.

    Parameters
    ----------
    value : int
        Number of state dimensions.
    """
    
    value: int
    
    model_config = ConfigDict(frozen=True, extra='forbid')
    
    @field_validator('value')
    @classmethod
    def validate_value(cls, v):
        """Validate that process dimension is strictly positive.

        Raises
        ------
        ValidationError
            If ``value < 1``.
        """
        if v < 1:
            raise ValidationError(f"Process dimension must be positive, got {v}")
        return v
    
    @property
    def is_univariate(self) -> bool:
        """Whether the process has a single state coordinate.

        Returns
        -------
        bool
            ``True`` when ``value == 1``.
        """
        return self.value == 1
    
    @property
    def is_multivariate(self) -> bool:
        """Whether the process has multiple state coordinates.

        Returns
        -------
        bool
            ``True`` when ``value > 1``.
        """
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
    """Abstract base class for all stochastic processes.

    Implementations must provide drift and covariance in consistent array shapes.
    The canonical convention is that ``state`` is either:

    - a 1D state vector ``(d,)`` for a single point
    - a 2D array ``(n, d)`` for a batch of ``n`` points

    where ``d`` is ``self.dimension.value``.

    Both methods are defined for the *calendar-time* convention used by the rest
    of the framework.
    """
    
    @property
    @abstractmethod
    def dimension(self) -> ProcessDimension:
        """Get process dimension.

        Returns
        -------
        ProcessDimension
            Strongly-typed number of state dimensions.
        """
        ...
    
    @property
    @abstractmethod
    def process_type(self) -> ProcessType:
        """Get process type for optimization.

        Returns
        -------
        ProcessType
            Either :class:`ProcessType.AFFINE` or
            :class:`ProcessType.NON_AFFINE`.
        """
        ...
    
    @abstractmethod
    def drift(self, time: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Compute drift vector :math:`\mu(t, x)`.

        Parameters
        ----------
        time : float
            Calendar time.
        state : NDArray[np.float64]
            State vector(s) with shape ``(d,)`` or ``(n, d)``.

        Returns
        -------
        NDArray[np.float64]
            Drift evaluated at each input point. The shape follows the input shape
            by replacing state dimension with ``d``.
        """
        ...
    
    @abstractmethod
    def covariance(self, time: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Compute diffusion covariance matrix :math:`\Sigma(t, x)`.

        Parameters
        ----------
        time : float
            Calendar time.
        state : NDArray[np.float64]
            State vector(s) with shape ``(d,)`` or ``(n, d)``.

        Returns
        -------
        NDArray[np.float64]
            Covariance matrix/matrices with shape ``(d, d)`` or
            ``(n, d, d)``.
        """
        ...
    
    def validate_state(self, state: NDArray[np.float64]) -> None:
        """Validate state vector dimensions and state-space assumptions.

        Parameters
        ----------
        state : NDArray[np.float64]
            Candidate state array.

        Raises
        ------
        ValidationError
            If dimensionality does not match ``self.dimension``.
        """
        state = ensure_state_array(state)
        validate_state_dimensions(state, self.dimension.value, self.__class__.__name__)
    
    def diffusion(self, time: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Compute diffusion matrix :math:`\sigma(t, x)` from covariance.

        This convenience method applies the matrix square root to the covariance
        returned by :meth:`covariance`.  For a batched state array, the square
        root is returned per batch point.

        Parameters
        ----------
        time : float
            Calendar time.
        state : NDArray[np.float64]
            State vector(s).

        Returns
        -------
        NDArray[np.float64]
            Diffusion matrix/matrices with matching batch shape to covariance.
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
    """Abstract base class for non-affine stochastic processes.

    Non-affine processes do not admit a global affine decomposition of drift and
    covariance. They must implement :meth:`drift` and :meth:`covariance` directly
    and are typically used for models where state-dependence is non-linear.
    """
    
    @property
    def process_type(self) -> ProcessType:
        return ProcessType.NON_AFFINE