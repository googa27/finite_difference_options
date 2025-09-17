"""Common validation patterns for stochastic processes.

This module provides reusable validation functions for common patterns
found across different stochastic process implementations.
"""
from typing import Optional
import numpy as np
from numpy.typing import NDArray

from ..validation import validate_positive, validate_non_negative, validate_probability
from src.exceptions import ValidationError


def validate_feller_condition(
    kappa: float, 
    theta: float, 
    sigma: float,
    process_name: str = "process"
) -> None:
    """Validate Feller condition: 2κθ ≥ σ².
    
    Parameters
    ----------
    kappa : float
        Mean reversion speed.
    theta : float
        Long-term mean.
    sigma : float
        Volatility parameter.
    process_name : str
        Name of the process for error messages.
        
    Raises
    ------
    ValidationError
        If Feller condition is violated.
    """
    feller_lhs = 2 * kappa * theta
    feller_rhs = sigma**2
    
    if feller_lhs < feller_rhs:
        raise ValidationError(
            f"Feller condition violated in {process_name}: "
            f"2κθ = {feller_lhs:.4f} < σ² = {feller_rhs:.4f}"
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
    jump_intensity: float,
    jump_mean: float,
    jump_volatility: float
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
    rho: float
) -> None:
    """Validate Heston model parameters.
    
    Parameters
    ----------
    kappa : float
        Mean reversion speed.
    theta : float
        Long-term variance.
    sigma : float
        Volatility of volatility.
    rho : float
        Correlation between asset and variance.
        
    Raises
    ------
    ValidationError
        If any parameter is invalid.
    """
    validate_positive(kappa, "kappa")
    validate_positive(theta, "theta")
    validate_positive(sigma, "sigma")
    validate_correlation_parameter(rho, "rho")
    validate_feller_condition(kappa, theta, sigma, "Heston model")


def validate_sabr_parameters(
    alpha: float,
    beta: float,
    rho: float
) -> None:
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
    array: NDArray[np.float64],
    expected_shape: tuple,
    param_name: str
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
    weights: NDArray[np.float64],
    tolerance: float = 1e-10
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
        raise ValidationError(
            f"Weights must sum to 1.0, got {weight_sum:.6f}"
        )