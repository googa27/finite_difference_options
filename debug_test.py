#!/usr/bin/env python3
"""Debug test for parameter validation."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
from src.pricing.instruments.options import UnifiedEuropeanOption
from src.utils.exceptions import ValidationError

def test_parameter_validation():
    """Test parameter validation."""
    print("Testing parameter validation...")
    
    # Invalid strike
    print("Testing invalid strike...")
    with pytest.raises(ValidationError, match="strike"):
        UnifiedEuropeanOption(strike=-100.0, maturity=1.0)
    print("Invalid strike test passed")
    
    # Invalid maturity
    print("Testing invalid maturity...")
    with pytest.raises(ValidationError, match="maturity"):
        UnifiedEuropeanOption(strike=100.0, maturity=-1.0)
    print("Invalid maturity test passed")
    
    # Invalid option type
    print("Testing invalid option type...")
    with pytest.raises(ValidationError, match="option_type must be"):
        UnifiedEuropeanOption(strike=100.0, maturity=1.0, option_type='invalid')
    print("Invalid option type test passed")
    
    print("All tests passed!")

if __name__ == "__main__":
    test_parameter_validation()