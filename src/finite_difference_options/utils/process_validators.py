"""Common validation patterns for stochastic processes.

This module provides reusable validation functions for common patterns
found across different stochastic process implementations.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np
from numpy.typing import NDArray

from ..validation import validate_positive, validate_non_negative
from finite_difference_options.exceptions import ValidationError


class FellerPolicy(str, Enum):
    """Policy for using the CIR/Heston Feller condition in validation.

    The Feller inequality is a boundary-classification diagnostic.  It is not a
    universal statement that a square-root diffusion or Heston pricing problem
    is mathematically invalid when violated.  Selectable numerical routes may
    still require strict positivity, but that is a route/governance policy.
    """

    REQUIRE_STRICT_POSITIVITY = "require_strict_positivity"
    WARN_AND_ALLOW_IF_BACKEND_VALIDATED = "warn_and_allow_if_backend_validated"
    ALLOW_WITH_EXPLICIT_BOUNDARY_AND_SCHEME = "allow_with_explicit_boundary_and_scheme"


class ZeroBoundaryClassification(str, Enum):
    """Classification of the square-root variance zero boundary."""

    DETERMINISTIC = "deterministic_variance"
    INACCESSIBLE = "inaccessible"
    ATTAINABLE = "attainable"


@dataclass(frozen=True)
class FellerDiagnostics:
    """Stable diagnostics for a square-root variance boundary check."""

    process_name: str
    feller_lhs: float
    feller_rhs: float
    feller_margin: float
    feller_ratio: float
    cir_dimension: float
    is_satisfied: bool
    zero_boundary: ZeroBoundaryClassification
    policy: FellerPolicy
    requires_explicit_boundary_policy: bool
    route_capability_required: Optional[str]
    correlation_degeneracy: str = "none"


def _validate_finite_real(value: float, name: str) -> None:
    """Validate a scalar finite real value without imposing sign."""

    if not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValidationError(f"{name} must be a finite real number, got {value}")


def _coerce_feller_policy(policy: FellerPolicy | str) -> FellerPolicy:
    """Normalize user-supplied policy strings into the policy enum."""

    try:
        return policy if isinstance(policy, FellerPolicy) else FellerPolicy(policy)
    except ValueError as exc:
        valid = ", ".join(item.value for item in FellerPolicy)
        raise ValidationError(f"feller_policy must be one of {{{valid}}}, got {policy!r}") from exc


def diagnose_feller_condition(
    kappa: float,
    theta: float,
    sigma: float,
    process_name: str = "process",
    *,
    rho: Optional[float] = None,
    policy: FellerPolicy | str = FellerPolicy.WARN_AND_ALLOW_IF_BACKEND_VALIDATED,
    tolerance: float = 1e-12,
) -> FellerDiagnostics:
    """Compute Feller and variance-zero diagnostics without hidden rejection.

    The square-root variance domain checks are separated from the Feller
    inequality.  Invalid domains still fail immediately, while a negative Feller
    margin is reported as an attainable-boundary finding that downstream routes
    can accept or reject according to declared numerical capability.
    """

    _validate_finite_real(kappa, "kappa")
    _validate_finite_real(theta, "theta")
    _validate_finite_real(sigma, "sigma")
    if kappa <= 0.0:
        raise ValidationError(f"kappa must be a positive number, got {kappa}")
    if theta < 0.0:
        raise ValidationError(f"theta must be non-negative, got {theta}")
    if sigma < 0.0:
        raise ValidationError(f"sigma must be non-negative, got {sigma}")

    correlation_degeneracy = "none"
    if rho is not None:
        validate_correlation_parameter(rho, "rho")
        if np.isclose(rho, 1.0, rtol=0.0, atol=tolerance):
            correlation_degeneracy = "perfect_positive"
        elif np.isclose(rho, -1.0, rtol=0.0, atol=tolerance):
            correlation_degeneracy = "perfect_negative"

    normalized_policy = _coerce_feller_policy(policy)
    feller_lhs = 2.0 * kappa * theta
    feller_rhs = sigma**2
    feller_margin = feller_lhs - feller_rhs

    if sigma == 0.0:
        feller_ratio = float("inf")
        cir_dimension = float("inf")
        zero_boundary = ZeroBoundaryClassification.DETERMINISTIC
        is_satisfied = True
    else:
        feller_ratio = feller_lhs / feller_rhs
        cir_dimension = 4.0 * kappa * theta / feller_rhs
        is_satisfied = feller_margin >= -tolerance
        zero_boundary = (
            ZeroBoundaryClassification.INACCESSIBLE
            if is_satisfied
            else ZeroBoundaryClassification.ATTAINABLE
        )

    requires_boundary_policy = zero_boundary is ZeroBoundaryClassification.ATTAINABLE
    return FellerDiagnostics(
        process_name=process_name,
        feller_lhs=feller_lhs,
        feller_rhs=feller_rhs,
        feller_margin=feller_margin,
        feller_ratio=feller_ratio,
        cir_dimension=cir_dimension,
        is_satisfied=is_satisfied,
        zero_boundary=zero_boundary,
        policy=normalized_policy,
        requires_explicit_boundary_policy=requires_boundary_policy,
        route_capability_required=(
            "attainable_variance_boundary" if requires_boundary_policy else None
        ),
        correlation_degeneracy=correlation_degeneracy,
    )


def validate_feller_condition(
    kappa: float, theta: float, sigma: float, process_name: str = "process"
) -> None:
    """Validate strict Feller condition: 2κθ ≥ σ².

    This strict helper is retained for routes/models that explicitly require an
    inaccessible zero boundary. Use :func:`diagnose_feller_condition` when the
    Feller rule should be treated as a route-capability diagnostic instead of a
    universal model-domain rejection.
    """
    diagnostics = diagnose_feller_condition(
        kappa,
        theta,
        sigma,
        process_name=process_name,
        policy=FellerPolicy.REQUIRE_STRICT_POSITIVITY,
    )

    if not diagnostics.is_satisfied:
        raise ValidationError(
            f"Feller condition violated in {process_name}: "
            f"2κθ = {diagnostics.feller_lhs:.4f} < σ² = {diagnostics.feller_rhs:.4f}"
        )


def validate_correlation_parameter(rho: float, param_name: str = "rho") -> None:
    """Validate correlation parameter is in [-1, 1].

    Parameters
    ----------
    rho : float
        Correlation parameter.
    param_name : str
        Parameter name for error messages.

    Raises
    ------
    ValidationError
        If correlation is outside [-1, 1].
    """
    if not (-1.0 <= rho <= 1.0):
        raise ValidationError(f"{param_name} must be in [-1, 1], got {rho}")


def validate_cev_beta(beta: float) -> None:
    """Validate CEV beta parameter is in [0, 1].

    Parameters
    ----------
    beta : float
        CEV elasticity parameter.

    Raises
    ------
    ValidationError
        If beta is outside [0, 1].
    """
    if not (0.0 <= beta <= 1.0):
        raise ValidationError(f"CEV beta must be in [0, 1], got {beta}")


def validate_jump_parameters(
    jump_intensity: float, jump_mean: float, jump_volatility: float
) -> None:
    """Validate jump process parameters.

    Parameters
    ----------
    jump_intensity : float
        Jump arrival rate (must be non-negative).
    jump_mean : float
        Mean jump size.
    jump_volatility : float
        Jump size volatility (must be positive).

    Raises
    ------
    ValidationError
        If any parameter is invalid.
    """
    validate_non_negative(jump_intensity, "jump_intensity")
    validate_positive(jump_volatility, "jump_volatility")
    # jump_mean can be any real number (no validation needed)


def validate_heston_parameters(
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    feller_policy: FellerPolicy | str = FellerPolicy.WARN_AND_ALLOW_IF_BACKEND_VALIDATED,
) -> FellerDiagnostics:
    """Validate Heston model parameters and return Feller diagnostics.

    Hard domain checks are separated from the Feller boundary-policy diagnostic.
    A Feller violation is rejected only when ``feller_policy`` explicitly
    requires strict positivity.
    """
    diagnostics = diagnose_feller_condition(
        kappa,
        theta,
        sigma,
        process_name="Heston model",
        rho=rho,
        policy=feller_policy,
    )
    if (
        diagnostics.policy is FellerPolicy.REQUIRE_STRICT_POSITIVITY
        and not diagnostics.is_satisfied
    ):
        raise ValidationError(
            "Feller condition violated in Heston model: "
            f"2κθ = {diagnostics.feller_lhs:.4f} < σ² = {diagnostics.feller_rhs:.4f}. "
            "Use an explicit non-strict feller_policy only with a route that records "
            "its variance-zero boundary and numerical scheme."
        )
    return diagnostics


def validate_sabr_parameters(alpha: float, beta: float, rho: float) -> None:
    """Validate SABR model parameters.

    Parameters
    ----------
    alpha : float
        Volatility of volatility.
    beta : float
        CEV exponent.
    rho : float
        Correlation parameter.

    Raises
    ------
    ValidationError
        If any parameter is invalid.
    """
    validate_positive(alpha, "alpha")
    validate_cev_beta(beta)
    validate_correlation_parameter(rho, "rho")


def validate_array_shape(
    array: NDArray[np.float64], expected_shape: tuple, param_name: str
) -> None:
    """Validate array has expected shape.

    Parameters
    ----------
    array : NDArray[np.float64]
        Array to validate.
    expected_shape : tuple
        Expected shape.
    param_name : str
        Parameter name for error messages.

    Raises
    ------
    ValidationError
        If shape doesn't match.
    """
    if array.shape != expected_shape:
        raise ValidationError(
            f"{param_name} must have shape {expected_shape}, got {array.shape}"
        )


def validate_weights_sum_to_one(
    weights: NDArray[np.float64], tolerance: float = 1e-10
) -> None:
    """Validate weights sum to 1.0.

    Parameters
    ----------
    weights : NDArray[np.float64]
        Weight array.
    tolerance : float
        Numerical tolerance.

    Raises
    ------
    ValidationError
        If weights don't sum to 1.0.
    """
    weight_sum = np.sum(weights)
    if not np.isclose(weight_sum, 1.0, atol=tolerance):
        raise ValidationError(f"Weights must sum to 1.0, got {weight_sum:.6f}")
