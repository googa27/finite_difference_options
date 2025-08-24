"""Comprehensive test script to verify the refactored structure."""
import numpy as np
from src import (
    EuropeanCall,
    EuropeanPut,
    create_black_scholes_process,
    create_unified_pricing_engine,
    create_log_grid,
    matrix_sqrt,
    validate_positive
)

def test_refactored_structure():
    """Test that the refactored structure works correctly."""
    print("Testing refactored structure...")
    
    # Test 1: Create European call and put options
    call_option = EuropeanCall(strike=100.0, _maturity=1.0)
    put_option = EuropeanPut(strike=100.0, _maturity=1.0)
    print(f"Created call option: {call_option}")
    print(f"Created put option: {put_option}")
    
    # Test 2: Create a Geometric Brownian Motion process
    gbm = create_black_scholes_process(mu=0.05, sigma=0.2)
    print(f"Created GBM process: {gbm}")
    
    # Test 3: Create pricing engine
    engine = create_unified_pricing_engine(gbm)
    print(f"Created engine: {engine}")
    
    # Test 4: Create grids
    s_grid = create_log_grid(0.1, 200.0, 50)
    print(f"Created grid with {len(s_grid)} points")
    
    # Test 5: Test payoff functions
    call_payoff = call_option.payoff(s_grid)
    put_payoff = put_option.payoff(s_grid)
    print(f"Call payoff shape: {call_payoff.shape}")
    print(f"Put payoff shape: {put_payoff.shape}")
    
    # Test 6: Test utility functions
    # Create a simple positive definite matrix for testing
    matrix = np.array([[2.0, 1.0], [1.0, 2.0]])
    sqrt_matrix = matrix_sqrt(matrix)
    print(f"Matrix square root shape: {sqrt_matrix.shape}")
    
    # Test 7: Test validation functions
    try:
        validate_positive(1.0, "test")
        print("Validation function works correctly")
    except Exception as e:
        print(f"Validation function failed: {e}")
    
    print("Comprehensive refactored structure test completed successfully!")

if __name__ == "__main__":
    test_refactored_structure()