#!/usr/bin/env python3
"""Test the payoff calculator refactor."""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from src.pricing.instruments import (
    create_unified_european_call,
    create_unified_european_put,
    create_unified_basket_call,
    create_unified_basket_put,
    PayoffCalculatorFactory,
    EuropeanPayoffCalculator,
    BasketPayoffCalculator
)

def test_european_payoff():
    """Test European option payoff calculation."""
    print("Testing European option payoff...")
    
    # Create European call option
    call_option = create_unified_european_call(strike=100.0, maturity=1.0)
    
    # Create price grid
    s_grid = np.linspace(80, 120, 5)
    
    # Calculate payoff using the refactored approach
    payoff = call_option.payoff(s_grid)
    
    # Expected payoff (manual calculation)
    expected = np.maximum(s_grid - 100.0, 0.0)
    
    print(f"Price grid: {s_grid}")
    print(f"Payoff: {payoff}")
    print(f"Expected: {expected}")
    
    # Check if they match
    assert np.allclose(payoff, expected), "European call payoff calculation failed"
    print("✓ European call payoff test passed")
    
    # Test put option
    put_option = create_unified_european_put(strike=100.0, maturity=1.0)
    put_payoff = put_option.payoff(s_grid)
    expected_put = np.maximum(100.0 - s_grid, 0.0)
    
    print(f"Put payoff: {put_payoff}")
    print(f"Expected put: {expected_put}")
    
    assert np.allclose(put_payoff, expected_put), "European put payoff calculation failed"
    print("✓ European put payoff test passed")


def test_basket_payoff():
    """Test basket option payoff calculation."""
    print("\nTesting basket option payoff...")
    
    # Create basket call option (2 assets)
    strikes = np.array([100.0, 110.0])
    weights = np.array([0.6, 0.4])
    basket_call = create_unified_basket_call(strikes, weights, maturity=1.0)
    
    # Create price grids
    s1_grid = np.linspace(90, 110, 3)
    s2_grid = np.linspace(100, 120, 3)
    
    # Calculate payoff using the refactored approach
    payoff = basket_call.payoff(s1_grid, s2_grid)
    
    # Manual calculation of expected payoff
    s1_mesh, s2_mesh = np.meshgrid(s1_grid, s2_grid, indexing='ij')
    basket_value = 0.6 * s1_mesh + 0.4 * s2_mesh
    weighted_strike = np.sum(weights * strikes)
    expected = np.maximum(basket_value - weighted_strike, 0.0)
    
    print(f"S1 grid: {s1_grid}")
    print(f"S2 grid: {s2_grid}")
    print(f"Payoff shape: {payoff.shape}")
    print(f"Payoff: {payoff}")
    print(f"Expected shape: {expected.shape}")
    print(f"Expected: {expected}")
    
    # Check if they match
    assert np.allclose(payoff, expected), "Basket call payoff calculation failed"
    print("✓ Basket call payoff test passed")
    
    # Test put option
    basket_put = create_unified_basket_put(strikes, weights, maturity=1.0)
    put_payoff = basket_put.payoff(s1_grid, s2_grid)
    expected_put = np.maximum(weighted_strike - basket_value, 0.0)
    
    print(f"Put payoff: {put_payoff}")
    print(f"Expected put: {expected_put}")
    
    assert np.allclose(put_payoff, expected_put), "Basket put payoff calculation failed"
    print("✓ Basket put payoff test passed")


def test_payoff_calculator_factory():
    """Test payoff calculator factory."""
    print("\nTesting payoff calculator factory...")
    
    # Create options
    call_option = create_unified_european_call(strike=100.0, maturity=1.0)
    basket_option = create_unified_basket_call(
        strikes=np.array([100.0, 110.0]),
        weights=np.array([0.6, 0.4]),
        maturity=1.0
    )
    
    # Test factory creation
    call_calculator = PayoffCalculatorFactory.create_calculator(call_option)
    basket_calculator = PayoffCalculatorFactory.create_calculator(basket_option)
    
    assert isinstance(call_calculator, EuropeanPayoffCalculator), "Factory should create EuropeanPayoffCalculator"
    assert isinstance(basket_calculator, BasketPayoffCalculator), "Factory should create BasketPayoffCalculator"
    print("✓ Payoff calculator factory test passed")


def main():
    """Run all tests."""
    print("Testing payoff calculator refactor...")
    
    try:
        test_european_payoff()
        test_basket_payoff()
        test_payoff_calculator_factory()
        print("\n✓ All tests passed!")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()