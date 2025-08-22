"""Tests for input validation functionality."""
import pathlib
import sys

# Ensure project root on path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
import numpy as np
from src.exceptions import ValidationError, GridError, InstrumentError, ModelError
from src.validation import (
    validate_positive, validate_non_negative, validate_probability,
    validate_grid_parameters, validate_option_parameters, validate_model_parameters
)
from src.models import GeometricBrownianMotion
from src.options import EuropeanCall


def test_validate_positive():
    """Test positive value validation."""
    validate_positive(1.0, "test_param")  # Should not raise
    
    with pytest.raises(ValidationError, match="test_param must be a positive number"):
        validate_positive(-1.0, "test_param")
    
    with pytest.raises(ValidationError, match="test_param must be a positive number"):
        validate_positive(0.0, "test_param")


def test_validate_grid_parameters():
    """Test grid parameter validation."""
    validate_grid_parameters(100.0, 50, 100)  # Should not raise
    
    with pytest.raises(GridError, match="Invalid spatial grid parameter"):
        validate_grid_parameters(-100.0, 50, 100)
    
    with pytest.raises(GridError, match="s_steps must be an integer >= 3"):
        validate_grid_parameters(100.0, 2, 100)
    
    with pytest.raises(GridError, match="t_steps must be an integer >= 2"):
        validate_grid_parameters(100.0, 50, 1)


def test_validate_option_parameters():
    """Test option parameter validation."""
    validate_option_parameters(100.0, 1.0)  # Should not raise
    
    with pytest.raises(InstrumentError, match="Invalid option parameter"):
        validate_option_parameters(-100.0, 1.0)
    
    with pytest.raises(InstrumentError, match="Invalid option parameter"):
        validate_option_parameters(100.0, -1.0)


def test_validate_model_parameters():
    """Test model parameter validation."""
    validate_model_parameters(0.05, 0.2, 0.0)  # Should not raise
    validate_model_parameters(-0.01, 0.2, 0.0)  # Negative rates allowed
    
    with pytest.raises(ModelError, match="Invalid model parameter"):
        validate_model_parameters(0.05, -0.2, 0.0)  # Negative volatility not allowed
    
    with pytest.raises(ModelError, match="Invalid model parameter"):
        validate_model_parameters(0.05, 0.2, -0.01)  # Negative dividend yield not allowed


def test_option_validation_in_constructor():
    """Test that option validation works in constructor."""
    model = GeometricBrownianMotion(rate=0.05, sigma=0.2)
    
    # Valid option should work
    call = EuropeanCall(strike=100.0, maturity=1.0, model=model)
    assert call.strike == 100.0
    
    # Invalid strike should raise
    with pytest.raises(InstrumentError):
        EuropeanCall(strike=-100.0, maturity=1.0, model=model)
    
    # Invalid maturity should raise
    with pytest.raises(InstrumentError):
        EuropeanCall(strike=100.0, maturity=-1.0, model=model)
    
    # Invalid model parameters should raise
    bad_model = GeometricBrownianMotion(rate=0.05, sigma=-0.2)
    with pytest.raises(ModelError):
        EuropeanCall(strike=100.0, maturity=1.0, model=bad_model)
