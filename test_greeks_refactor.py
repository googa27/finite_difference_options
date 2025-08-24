#!/usr/bin/env python3
"""Test the Greeks calculator abstraction refactor."""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def test_greeks_calculator_factory():
    """Test Greeks calculator factory creation."""
    print("Testing Greeks calculator factory...")
    
    # Import only what we need directly
    from src.processes.base import ProcessDimension
    from src.greeks.base import GreeksCalculatorFactory, FDCalculator1D, FDCalculator2D
    
    # Create mock processes with different dimensions
    class Mock1DProcess:
        @property
        def dimension(self):
            return ProcessDimension(1)
    
    class Mock2DProcess:
        @property
        def dimension(self):
            return ProcessDimension(2)
    
    # Test factory creation
    process_1d = Mock1DProcess()
    process_2d = Mock2DProcess()
    
    calculator_1d = GreeksCalculatorFactory.create_calculator(process_1d)
    calculator_2d = GreeksCalculatorFactory.create_calculator(process_2d)
    
    assert isinstance(calculator_1d, FDCalculator1D), "Factory should create FDCalculator1D for 1D process"
    assert isinstance(calculator_2d, FDCalculator2D), "Factory should create FDCalculator2D for 2D process"
    
    print("✓ Greeks calculator factory test passed")


def test_greeks_calculator_interface():
    """Test Greeks calculator interface."""
    print("\nTesting Greeks calculator interface...")
    
    from src.greeks.base import GreeksCalculator, FDCalculator1D, FDCalculator2D
    
    # Test that calculators implement the GreeksCalculator interface
    calc_1d = FDCalculator1D()
    calc_2d = FDCalculator2D()
    
    assert isinstance(calc_1d, GreeksCalculator), "FDCalculator1D should implement GreeksCalculator interface"
    assert isinstance(calc_2d, GreeksCalculator), "FDCalculator2D should implement GreeksCalculator interface"
    
    print("✓ Greeks calculator interface test passed")


def test_1d_greeks_calculation():
    """Test 1D Greeks calculation."""
    print("\nTesting 1D Greeks calculation...")
    
    from src.greeks.base import FDCalculator1D
    
    # Create calculator
    calculator = FDCalculator1D()
    
    # Create mock price data (time, asset)
    prices = np.array([
        [0.0, 0.0, 1.0, 4.0, 9.0],  # Prices at different asset levels at time 0
        [0.0, 0.0, 0.9, 3.6, 8.1],  # Prices at different asset levels at time 0.1
    ])
    
    # Create asset grid
    s_grid = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    
    # Create time grid
    t_grid = np.array([0.0, 0.1])
    
    # Calculate Greeks
    greeks = calculator.calculate(prices, s_grid, time_grid=t_grid)
    
    # Check that we have the expected Greeks
    assert 'delta' in greeks, "Should have delta"
    assert 'gamma' in greeks, "Should have gamma"
    assert 'theta' in greeks, "Should have theta"
    
    # Check shapes
    assert greeks['delta'].shape == (5,), "Delta should have shape (5,)"
    assert greeks['gamma'].shape == (5,), "Gamma should have shape (5,)"
    assert greeks['theta'].shape == (2, 5), "Theta should have shape (2, 5)"
    
    print("✓ 1D Greeks calculation test passed")


def main():
    """Run all tests."""
    print("Testing Greeks calculator abstraction refactor...")
    
    try:
        test_greeks_calculator_factory()
        test_greeks_calculator_interface()
        test_1d_greeks_calculation()
        print("\n✓ All Greeks calculator abstraction tests passed!")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()