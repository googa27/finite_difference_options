"""Test script to verify the refactored structure."""
import numpy as np
from src import (
    EuropeanCall,
    create_black_scholes_process,
    create_unified_pricing_engine,
    create_log_grid
)

def test_refactored_structure():
    """Test that the refactored structure works correctly."""
    print("Testing refactored structure...")
    
    # Create a simple European call option
    option = EuropeanCall(strike=100.0, _maturity=1.0)
    print(f"Created option: {option}")
    
    # Create a Geometric Brownian Motion process
    gbm = create_black_scholes_process(mu=0.05, sigma=0.2)
    print(f"Created process: {gbm}")
    
    # Create pricing engine
    engine = create_unified_pricing_engine(gbm)
    print(f"Created engine: {engine}")
    
    # Create grids
    s_grid = create_log_grid(0.1, 200.0, 50)
    print(f"Created grid with {len(s_grid)} points")
    
    print("Refactored structure test completed successfully!")

if __name__ == "__main__":
    test_refactored_structure()