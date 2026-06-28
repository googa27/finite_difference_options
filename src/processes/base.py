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
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ProcessCoefficientEvaluation:
    """Canonical batched coefficient evaluation for a stochastic process."""

    time: float
    states: NDArray[np.float64]
    is_single_point: bool
    drift: NDArray[np.float64]
    covariance: NDArray[np.float64]
    discount: NDArray[np.float64]


@dataclass(frozen=True)
class CovarianceValidationResult:
    """Diagnostics for covariance symmetry and PSD validation."""

    min_eigenvalue: float
    max_asymmetry: float
    batch_size: int


@dataclass(frozen=True)
class AffineCovarianceForm:
    r"""Affine covariance tensor ``a(x,t)=a0(t)+sum_k x_k a_k(t)``."""

    constant: NDArray[np.float64]
    linear: NDArray[np.float64]

    def __post_init__(self) -> None:
        constant = np.asarray(self.constant, dtype=float)
        linear = np.asarray(self.linear, dtype=float)
        if constant.ndim != 2 or constant.shape[0] != constant.shape[1]:
            raise ValidationError(
                "constant affine covariance term must be a square matrix, "
                f"got shape {constant.shape}"
            )
        dimension = constant.shape[0]
        if linear.shape != (dimension, dimension, dimension):
            raise ValidationError(
                "linear affine covariance tensor must have shape "
                f"({dimension}, {dimension}, {dimension}), got {linear.shape}"
            )
        if not np.allclose(constant, constant.T, rtol=1e-10, atol=1e-12):
            raise ValidationError("constant affine covariance term must be symmetric")
        if not np.allclose(linear, np.swapaxes(linear, -1, -2), rtol=1e-10, atol=1e-12):
            raise ValidationError("linear affine covariance tensor slices must be symmetric")
        object.__setattr__(self, "constant", constant)
        object.__setattr__(self, "linear", linear)

    @classmethod
    def from_coefficients(
        cls,
        constant: NDArray[np.float64],
        linear: NDArray[np.float64],
    ) -> "AffineCovarianceForm":
        """Build and validate an affine covariance form from raw coefficients."""

        constant_array = np.asarray(constant, dtype=float)
        linear_array = np.asarray(linear, dtype=float)
        if constant_array.shape == (1, 1) and linear_array.shape == (1, 1):
            linear_array = linear_array.reshape(1, 1, 1)
        return cls(constant=constant_array, linear=linear_array)

    def evaluate(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the affine tensor at one or more state points."""

        states = ensure_state_array(state)
        if states.ndim == 1:
            states = states.reshape(1, -1)
        if states.ndim != 2 or states.shape[1] != self.linear.shape[0]:
            raise ValidationError(
                "state for affine covariance evaluation must have shape "
                f"(n, {self.linear.shape[0]}), got {states.shape}"
            )
        return self.constant[None, :, :] + np.einsum("nk,kij->nij", states, self.linear)


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

    def _canonical_state_batch(
        self,
        state: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], bool]:
        """Return ``(original_state, batched_state, is_single_point)``."""

        original = ensure_state_array(state)
        self.validate_state(original)
        is_single_point = original.ndim == 1
        states = original.reshape(1, -1) if is_single_point else original
        if not np.all(np.isfinite(states)):
            raise ValidationError(f"{self.__class__.__name__} state contains non-finite values")
        return original, states, is_single_point

    def discount(self, time: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the reaction/discount field ``c(t,x)``.

        The base process contract deliberately defaults to zero discount. A
        pricing adapter must pass an explicit curve/rate when the PDE includes a
        reaction term; it is never inferred from drift.
        """

        _ = time
        _, states, _ = self._canonical_state_batch(state)
        return np.zeros(states.shape[0], dtype=float)

    def _normalise_vector_field(
        self,
        values: NDArray[np.float64],
        field_name: str,
        batch_size: int,
    ) -> NDArray[np.float64]:
        dimension = self.dimension.value
        array = np.asarray(values, dtype=float)
        if array.shape == (dimension,):
            array = array.reshape(1, dimension)
        if array.shape != (batch_size, dimension):
            raise ValidationError(
                f"{field_name} must have shape ({batch_size}, {dimension}) after batching, got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValidationError(f"{field_name} contains non-finite values")
        return array

    def _normalise_matrix_field(
        self,
        values: NDArray[np.float64],
        field_name: str,
        batch_size: int,
    ) -> NDArray[np.float64]:
        dimension = self.dimension.value
        array = np.asarray(values, dtype=float)
        if array.shape == (dimension, dimension):
            array = array.reshape(1, dimension, dimension)
        if array.shape != (batch_size, dimension, dimension):
            raise ValidationError(
                f"{field_name} must have shape ({batch_size}, {dimension}, {dimension}) after batching, "
                f"got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValidationError(f"{field_name} contains non-finite values")
        return array

    @staticmethod
    def _normalise_scalar_field(
        values: NDArray[np.float64] | float,
        field_name: str,
        batch_size: int,
    ) -> NDArray[np.float64]:
        array = np.asarray(values, dtype=float)
        if array.ndim == 0:
            array = np.full(batch_size, float(array), dtype=float)
        elif array.shape == (1,) and batch_size != 1:
            array = np.full(batch_size, float(array[0]), dtype=float)
        if array.shape != (batch_size,):
            raise ValidationError(f"{field_name} must have shape ({batch_size},), got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValidationError(f"{field_name} contains non-finite values")
        return array

    def evaluate_coefficients(
        self,
        time: float,
        state: NDArray[np.float64],
    ) -> ProcessCoefficientEvaluation:
        """Evaluate drift, covariance and discount using canonical batch shapes."""

        original, states, is_single_point = self._canonical_state_batch(state)
        batch_size = states.shape[0]
        drift = self._normalise_vector_field(self.drift(time, original), "drift", batch_size)
        covariance = self._normalise_matrix_field(self.covariance(time, original), "covariance", batch_size)
        discount = self._normalise_scalar_field(self.discount(time, original), "discount", batch_size)
        return ProcessCoefficientEvaluation(
            time=time,
            states=states,
            is_single_point=is_single_point,
            drift=drift,
            covariance=covariance,
            discount=discount,
        )

    def validate_covariance(
        self,
        time: float,
        state: NDArray[np.float64],
        *,
        tolerance: float = 1e-10,
    ) -> CovarianceValidationResult:
        """Validate covariance symmetry and positive semidefiniteness."""

        coefficients = self.evaluate_coefficients(time, state)
        covariance = coefficients.covariance
        asymmetry = covariance - np.swapaxes(covariance, -1, -2)
        max_asymmetry = float(np.max(np.abs(asymmetry)))
        if max_asymmetry > tolerance:
            raise ValidationError(f"process covariance must be symmetric; max asymmetry is {max_asymmetry:.2e}")
        min_eigenvalue = float(np.min(np.linalg.eigvalsh(covariance)))
        if min_eigenvalue < -tolerance:
            raise ValidationError(
                "process covariance must be positive semi-definite; "
                f"minimum eigenvalue is {min_eigenvalue:.2e}"
            )
        return CovarianceValidationResult(
            min_eigenvalue=min_eigenvalue,
            max_asymmetry=max_asymmetry,
            batch_size=covariance.shape[0],
        )

    def apply_generator(
        self,
        time: float,
        state: NDArray[np.float64],
        *,
        value: NDArray[np.float64] | float,
        gradient: NDArray[np.float64],
        hessian: NDArray[np.float64],
        discount: NDArray[np.float64] | float | None = None,
        source: NDArray[np.float64] | float = 0.0,
    ) -> NDArray[np.float64]:
        r"""Apply ``0.5 tr(a Hessian) + b·grad - c value + source``."""

        coefficients = self.evaluate_coefficients(time, state)
        batch_size = coefficients.states.shape[0]
        gradient_array = self._normalise_vector_field(gradient, "gradient", batch_size)
        hessian_array = self._normalise_matrix_field(hessian, "hessian", batch_size)
        value_array = self._normalise_scalar_field(value, "value", batch_size)
        discount_array = (
            coefficients.discount
            if discount is None
            else self._normalise_scalar_field(discount, "discount", batch_size)
        )
        source_array = self._normalise_scalar_field(source, "source", batch_size)
        diffusion_term = 0.5 * np.einsum("nij,nij->n", coefficients.covariance, hessian_array)
        drift_term = np.einsum("ni,ni->n", coefficients.drift, gradient_array)
        return diffusion_term + drift_term - discount_array * value_array + source_array

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
        r"""Get affine covariance coefficients ``a0(t), a_linear(t)``.

        Returns
        -------
        Tuple[NDArray[np.float64], NDArray[np.float64]]
            ``(a0, a_linear)`` where ``a0`` has shape ``(d, d)`` and
            ``a_linear`` has shape ``(d, d, d)`` with convention
            ``a(x,t) = a0(t) + sum_k x_k a_linear[k](t)``.
        """
        ...

    def affine_covariance_form(self, time: float = 0.0) -> AffineCovarianceForm:
        """Return a validated affine covariance tensor form."""

        constant, linear = self.affine_covariance_coefficients(time)
        form = AffineCovarianceForm.from_coefficients(constant, linear)
        if type(self).covariance is not AffineProcess.covariance:
            self._validate_affine_covariance_form_matches_covariance(form, time)
        return form

    def _validate_affine_covariance_form_matches_covariance(
        self,
        form: AffineCovarianceForm,
        time: float,
    ) -> None:
        """Fail closed when advertised affine coefficients are not exact."""

        dimension = self.dimension.value
        probes = [np.ones(dimension), 2.0 * np.ones(dimension)]
        for index in range(dimension):
            point = np.ones(dimension)
            point[index] = 3.0
            probes.append(point)
        states = np.asarray(probes, dtype=float)
        expected = form.evaluate(states)
        actual = self._normalise_matrix_field(
            self.covariance(time, states),
            "covariance",
            states.shape[0],
        )
        if not np.allclose(actual, expected, rtol=1e-10, atol=1e-12):
            max_error = float(np.max(np.abs(actual - expected)))
            raise ValidationError(
                "affine covariance coefficients do not exactly reproduce process covariance; "
                f"max error on certification probes is {max_error:.2e}"
            )

    def drift(
        self,
        time: float,
        state: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute affine drift μ(x,t) = α(t) + β(t)x."""
        state = ensure_state_array(state)
        self.validate_state(state)
        alpha, beta = self.affine_drift_coefficients(time)
        alpha = np.asarray(alpha, dtype=float)
        beta = np.asarray(beta, dtype=float)
        dimension = self.dimension.value
        if alpha.shape != (dimension,):
            raise ValidationError(f"affine drift alpha must have shape ({dimension},), got {alpha.shape}")
        if beta.shape == (dimension,):
            if dimension != 1:
                raise ValidationError(
                    "one-dimensional affine drift beta shorthand is only valid for one-factor processes"
                )
            beta = beta.reshape(1, 1)
        if beta.shape != (dimension, dimension):
            raise ValidationError(
                f"affine drift beta must have shape ({dimension}, {dimension}), got {beta.shape}"
            )

        if state.ndim == 1:
            return alpha + beta @ state
        return state @ beta.T + alpha

    def covariance(
        self,
        time: float,
        state: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r"""Compute affine covariance ``a0(t)+sum_k x_k a_k(t)``."""
        state = ensure_state_array(state)
        self.validate_state(state)
        covariance = self.affine_covariance_form(time).evaluate(state)
        return covariance[0] if state.ndim == 1 else covariance


class NonAffineProcess(StochasticProcess):
    """Abstract base class for non-affine stochastic processes.

    Non-affine processes do not admit a global affine decomposition of drift and
    covariance. They must implement :meth:`drift` and :meth:`covariance` directly
    and are typically used for models where state-dependence is non-linear.
    """
    
    @property
    def process_type(self) -> ProcessType:
        return ProcessType.NON_AFFINE