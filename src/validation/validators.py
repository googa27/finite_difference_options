"""Input validation utilities for finite difference pricing.

This module provides validation functions to ensure input parameters
are valid for financial computations and numerical methods.
"""
from __future__ import annotations

# No typing imports needed

import numpy as np
from numpy.typing import NDArray

from src.exceptions import ValidationError, GridError, ModelError, InstrumentError


def validate_positive(value: float, name: str) -> None:
    """Validate that a value is positive.
    
    Parameters
    ----------
    value : float
        Value to validate.
    name : str
        Name of the parameter for error messages.
        
    Raises
    ------
    ValidationError
        If value is not positive.
    """
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValidationError(f"{name} must be a positive number, got {value}")


def validate_non_negative(value: float, name: str) -> None:
    """Validate that a value is non-negative.
    
    Parameters
    ----------
    value : float
        Value to validate.
    name : str
        Name of the parameter for error messages.
        
    Raises
    ------
    ValidationError
        If value is negative.
    """
    if not isinstance(value, (int, float)) or value < 0:
        raise ValidationError(f"{name} must be non-negative, got {value}")


def validate_probability(value: float, name: str) -> None:
    """Validate that a value is a valid probability (between 0 and 1).
    
    Parameters
    ----------
    value : float
        Value to validate.
    name : str
        Name of the parameter for error messages.
        
    Raises
    ------
    ValidationError
        If value is not between 0 and 1.
    """
    if not isinstance(value, (int, float)) or not (0 <= value <= 1):
        raise ValidationError(f"{name} must be between 0 and 1, got {value}")


def validate_grid_parameters(s_max: float, s_steps: int, t_steps: int) -> None:
    """Validate grid generation parameters.
    
    Parameters
    ----------
    s_max : float
        Maximum asset price.
    s_steps : int
        Number of spatial grid points.
    t_steps : int
        Number of time steps.
        
    Raises
    ------
    GridError
        If any parameter is invalid.
    """
    try:
        validate_positive(s_max, "s_max")
    except ValidationError as e:
        raise GridError(f"Invalid spatial grid parameter: {e}") from e
    
    if not isinstance(s_steps, int) or s_steps < 3:
        raise GridError(f"s_steps must be an integer >= 3, got {s_steps}")
    
    if not isinstance(t_steps, int) or t_steps < 2:
        raise GridError(f"t_steps must be an integer >= 2, got {t_steps}")


def validate_option_parameters(strike: float, maturity: float) -> None:
    """Validate European option parameters.
    
    Parameters
    ----------
    strike : float
        Strike price of the option.
    maturity : float
        Time to maturity.
        
    Raises
    ------
    InstrumentError
        If any parameter is invalid.
    """
    try:
        validate_positive(strike, "strike")
        validate_positive(maturity, "maturity")
    except ValidationError as e:
        raise InstrumentError(f"Invalid option parameter: {e}") from e


def validate_model_parameters(
    risk_free_rate: float, 
    volatility: float, 
    dividend_yield: float = 0.0
) -> None:
    """Validate geometric Brownian motion model parameters.
    
    Parameters
    ----------
    risk_free_rate : float
        Risk-free interest rate.
    volatility : float
        Asset volatility.
    dividend_yield : float, optional
        Dividend yield, by default 0.0.
        
    Raises
    ------
    ModelError
        If any parameter is invalid.
    """
    try:
        # Risk-free rate can be negative in some markets
        if not isinstance(risk_free_rate, (int, float)):
            raise ValidationError("risk_free_rate must be a number")
        
        validate_positive(volatility, "volatility")
        validate_non_negative(dividend_yield, "dividend_yield")
    except ValidationError as e:
        raise ModelError(f"Invalid model parameter: {e}") from e


def validate_array(
    array: NDArray[np.float64], 
    name: str, 
    min_length: int = 1,
    require_positive: bool = False,
    require_monotonic: bool = False
) -> None:
    """Validate numpy array properties.
    
    Parameters
    ----------
    array : NDArray[np.float64]
        Array to validate.
    name : str
        Name of the array for error messages.
    min_length : int, optional
        Minimum required length, by default 1.
    require_positive : bool, optional
        Whether all values must be positive, by default False.
    require_monotonic : bool, optional
        Whether array must be monotonically increasing, by default False.
        
    Raises
    ------
    ValidationError
        If array validation fails.
    """
    if not isinstance(array, np.ndarray):
        raise ValidationError(f"{name} must be a numpy array")
    
    if array.size < min_length:
        raise ValidationError(f"{name} must have at least {min_length} elements, got {array.size}")
    
    if not np.isfinite(array).all():
        raise ValidationError(f"{name} contains non-finite values")
    
    if require_positive and (array <= 0).any():
        raise ValidationError(f"{name} must contain only positive values")
    
    if require_monotonic and len(array) > 1:
        if not np.all(np.diff(array) > 0):
            raise ValidationError(f"{name} must be monotonically increasing")


def validate_spot_price(spot_price: float, spatial_grid: NDArray[np.float64]) -> None:
    """Validate that spot price is within the spatial grid range.
    
    Parameters
    ----------
    spot_price : float
        Current asset price.
    spatial_grid : NDArray[np.float64]
        Spatial grid points.
        
    Raises
    ------
    ValidationError
        If spot price is outside grid range.
    """
    validate_positive(spot_price, "spot_price")
    
    if spot_price < spatial_grid[0] or spot_price > spatial_grid[-1]:
        raise ValidationError(
            f"spot_price {spot_price} is outside spatial grid range "
            f"[{spatial_grid[0]}, {spatial_grid[-1]}]"
        )
