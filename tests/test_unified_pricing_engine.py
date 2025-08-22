"""Tests for unified pricing engine."""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from src.pricing import (
    UnifiedInstrument,
    UnifiedEuropeanOption,
    UnifiedBasketOption,
    UnifiedPricingEngine,
    create_unified_european_call,
    create_unified_european_put,
    create_unified_basket_call,
    create_unified_pricing_engine,
    create_log_grid,
    create_linear_grid,
)
from src.processes import (
    GeometricBrownianMotion,
    OrnsteinUhlenbeck,
    CoxIngersollRoss,
    HestonModel,
    create_black_scholes_process,
    create_vasicek_process,
    create_cir_process,
    create_standard_heston,
)
from src.exceptions import ValidationError


class TestUnifiedEuropeanOption:
    """Test unified European option."""
    
    def test_european_call_initialization(self):
        """Test European call initialization."""
        option = UnifiedEuropeanOption(strike=100.0, maturity=1.0, option_type='call')
        assert option.strike == 100.0
        assert option.maturity == 1.0
        assert option.option_type == 'call'
    
    def test_european_put_initialization(self):
        """Test European put initialization."""
        option = UnifiedEuropeanOption(strike=100.0, maturity=1.0, option_type='put')
        assert option.option_type == 'put'
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Invalid strike
        with pytest.raises(ValidationError, match="strike"):
            UnifiedEuropeanOption(strike=-100.0, maturity=1.0)
        
        # Invalid maturity
        with pytest.raises(ValidationError, match="maturity"):
            UnifiedEuropeanOption(strike=100.0, maturity=-1.0)
        
        # Invalid option type
        with pytest.raises(ValidationError, match="option_type must be"):
            UnifiedEuropeanOption(strike=100.0, maturity=1.0, option_type='invalid')
    
    def test_call_payoff_1d(self):
        """Test call option payoff computation."""
        option = UnifiedEuropeanOption(strike=100.0, maturity=1.0, option_type='call')
        price_grid = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        
        payoff = option.payoff(price_grid)
        expected = np.array([0.0, 0.0, 0.0, 10.0, 20.0])
        assert_allclose(payoff, expected)
    
    def test_put_payoff_1d(self):
        """Test put option payoff computation."""
        option = UnifiedEuropeanOption(strike=100.0, maturity=1.0, option_type='put')
        price_grid = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        
        payoff = option.payoff(price_grid)
        expected = np.array([20.0, 10.0, 0.0, 0.0, 0.0])
        assert_allclose(payoff, expected)
    
    def test_payoff_2d_grid(self):
        """Test payoff with 2D price grid."""
        option = UnifiedEuropeanOption(strike=100.0, maturity=1.0, option_type='call')
        
        # Create 2D grid (e.g., for Heston model: price x volatility)
        s_grid = np.array([90.0, 100.0, 110.0])
        v_grid = np.array([0.04, 0.09])
        
        # For European option, only first grid (price) matters
        payoff = option.payoff(s_grid, v_grid)
        expected = np.array([0.0, 0.0, 10.0])  # Based on s_grid only
        assert_allclose(payoff, expected)
    
    def test_no_grids_error(self):
        """Test error when no grids provided."""
        option = UnifiedEuropeanOption(strike=100.0, maturity=1.0)
        
        with pytest.raises(ValidationError, match="At least one grid required"):
            option.payoff()


class TestUnifiedBasketOption:
    """Test unified basket option."""
    
    def test_basket_call_initialization(self):
        """Test basket call initialization."""
        strikes = np.array([100.0, 110.0])
        weights = np.array([0.6, 0.4])
        
        option = UnifiedBasketOption(
            strikes=strikes, weights=weights, maturity=1.0, option_type='call'
        )
        assert_array_equal(option.strikes, strikes)
        assert_array_equal(option.weights, weights)
        assert option.maturity == 1.0
        assert option.option_type == 'call'
    
    def test_parameter_validation(self):
        """Test basket option parameter validation."""
        # Mismatched strikes and weights
        with pytest.raises(ValidationError, match="strikes and weights must have same length"):
            UnifiedBasketOption(
                strikes=np.array([100.0]), weights=np.array([0.6, 0.4]), maturity=1.0
            )
        
        # Negative strike
        with pytest.raises(ValidationError, match="All strikes must be positive"):
            UnifiedBasketOption(
                strikes=np.array([-100.0, 110.0]), weights=np.array([0.6, 0.4]), maturity=1.0
            )
    
    def test_basket_call_payoff_2d(self):
        """Test basket call payoff computation."""
        strikes = np.array([100.0, 110.0])
        weights = np.array([0.6, 0.4])
        
        option = UnifiedBasketOption(
            strikes=strikes, weights=weights, maturity=1.0, option_type='call'
        )
        
        # Create 2D grids
        s1_grid = np.array([90.0, 100.0, 110.0])
        s2_grid = np.array([100.0, 120.0])
        
        payoff = option.payoff(s1_grid, s2_grid)
        
        # Basket strike = 0.6 * 100 + 0.4 * 110 = 104
        # Basket values at each grid point:
        # (90, 100): 0.6*90 + 0.4*100 = 94 -> max(94-104, 0) = 0
        # (90, 120): 0.6*90 + 0.4*120 = 102 -> max(102-104, 0) = 0
        # (100, 100): 0.6*100 + 0.4*100 = 100 -> max(100-104, 0) = 0
        # (100, 120): 0.6*100 + 0.4*120 = 108 -> max(108-104, 0) = 4
        # (110, 100): 0.6*110 + 0.4*100 = 106 -> max(106-104, 0) = 2
        # (110, 120): 0.6*110 + 0.4*120 = 114 -> max(114-104, 0) = 10
        
        expected = np.array([[0.0, 0.0], [0.0, 4.0], [2.0, 10.0]])
        assert_allclose(payoff, expected)
    
    def test_basket_put_payoff_2d(self):
        """Test basket put payoff computation."""
        strikes = np.array([100.0, 110.0])
        weights = np.array([0.6, 0.4])
        
        option = UnifiedBasketOption(
            strikes=strikes, weights=weights, maturity=1.0, option_type='put'
        )
        
        s1_grid = np.array([90.0, 100.0])
        s2_grid = np.array([100.0])
        
        payoff = option.payoff(s1_grid, s2_grid)
        
        # Basket strike = 104
        # (90, 100): basket = 94 -> max(104-94, 0) = 10
        # (100, 100): basket = 100 -> max(104-100, 0) = 4
        
        expected = np.array([[10.0], [4.0]])
        assert_allclose(payoff, expected)
    
    def test_wrong_number_of_grids(self):
        """Test error with wrong number of grids."""
        option = UnifiedBasketOption(
            strikes=np.array([100.0, 110.0]), weights=np.array([0.6, 0.4]), maturity=1.0
        )
        
        with pytest.raises(ValidationError, match="Expected 2 grids, got 1"):
            option.payoff(np.array([100.0]))


class TestUnifiedPricingEngine:
    """Test unified pricing engine."""
    
    def test_engine_initialization_1d(self):
        """Test engine initialization with 1D process."""
        process = GeometricBrownianMotion(mu=0.05, sigma=0.2)
        engine = UnifiedPricingEngine(process)
        
        assert engine.process is process
        assert engine.solver is not None
    
    def test_engine_initialization_2d(self):
        """Test engine initialization with 2D process."""
        process = create_standard_heston()
        engine = UnifiedPricingEngine(process)
        
        assert engine.process is process
        assert engine.solver is not None
    
    def test_price_option_1d_gbm(self):
        """Test option pricing with 1D GBM."""
        process = create_black_scholes_process(0.05, 0.2)
        engine = UnifiedPricingEngine(process)
        
        option = create_unified_european_call(100.0, 0.25)
        s_grid = create_log_grid(50.0, 150.0, 21, center=100.0)
        time_grid = np.linspace(0, 0.25, 10)
        
        prices = engine.price_option(option, s_grid, time_grid=time_grid)
        
        # Check dimensions
        assert prices.shape == (len(time_grid), len(s_grid))
        
        # Check that prices are non-negative
        assert np.all(prices >= 0)
        
        # Check terminal condition matches payoff
        terminal_payoff = option.payoff(s_grid)
        assert_allclose(prices[-1], terminal_payoff, rtol=1e-10)
    
    def test_price_option_2d_heston(self):
        """Test option pricing with 2D Heston."""
        process = create_standard_heston()
        engine = UnifiedPricingEngine(process)
        
        option = create_unified_european_call(100.0, 0.25)
        s_grid = create_log_grid(50.0, 150.0, 11, center=100.0)
        v_grid = create_linear_grid(0.01, 0.5, 6)
        time_grid = np.linspace(0, 0.25, 5)
        
        prices = engine.price_option(option, s_grid, v_grid, time_grid=time_grid)
        
        # Check dimensions
        assert prices.shape == (len(time_grid), len(s_grid), len(v_grid))
        
        # Check that prices are non-negative
        assert np.all(prices >= 0)
        
        # Check terminal condition
        s_mesh, v_mesh = np.meshgrid(s_grid, v_grid, indexing='ij')
        terminal_payoff = option.payoff(s_grid, v_grid)
        assert_allclose(prices[-1], terminal_payoff, rtol=1e-10)
    
    def test_price_option_wrong_dimensions(self):
        """Test error with wrong number of grids."""
        process = create_standard_heston()  # 2D process
        engine = UnifiedPricingEngine(process)
        
        option = create_unified_european_call(100.0, 0.25)
        s_grid = create_log_grid(50.0, 150.0, 11)
        
        # Only provide 1 grid for 2D process
        with pytest.raises(ValidationError, match="Expected 2 grids, got 1"):
            engine.price_option(option, s_grid)
    
    def test_compute_greeks_1d(self):
        """Test Greeks computation for 1D process."""
        process = create_black_scholes_process(0.05, 0.2)
        engine = UnifiedPricingEngine(process)
        
        # Create simple price array
        s_grid = np.linspace(80, 120, 21)
        time_grid = np.linspace(0, 0.25, 5)
        
        # Mock price array (increasing with spot)
        prices = np.zeros((len(time_grid), len(s_grid)))
        for i, s in enumerate(s_grid):
            prices[:, i] = max(s - 100, 0)  # Call payoff
        
        greeks = engine.compute_greeks(prices, s_grid)
        
        assert 'delta' in greeks
        assert 'gamma' in greeks
        assert 'theta' in greeks
        
        # Check shapes
        assert greeks['delta'].shape == (len(s_grid),)
        assert greeks['gamma'].shape == (len(s_grid),)
        assert greeks['theta'].shape == prices.shape
    
    def test_compute_greeks_2d(self):
        """Test Greeks computation for 2D process."""
        process = create_standard_heston()
        engine = UnifiedPricingEngine(process)
        
        s_grid = np.linspace(80, 120, 11)
        v_grid = np.linspace(0.01, 0.5, 6)
        time_grid = np.linspace(0, 0.25, 3)
        
        # Mock price array
        prices = np.zeros((len(time_grid), len(s_grid), len(v_grid)))
        for i, s in enumerate(s_grid):
            for j, v in enumerate(v_grid):
                prices[:, i, j] = max(s - 100, 0)
        
        greeks = engine.compute_greeks(prices, s_grid, v_grid)
        
        assert 'delta' in greeks
        assert 'gamma' in greeks
        assert 'vega' in greeks
        assert 'theta' in greeks
        
        # Check shapes
        assert greeks['delta'].shape == (len(s_grid), len(v_grid))
        assert greeks['gamma'].shape == (len(s_grid), len(v_grid))
        assert greeks['vega'].shape == (len(s_grid), len(v_grid))
        assert greeks['theta'].shape == prices.shape


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_unified_european_call(self):
        """Test unified European call creation."""
        option = create_unified_european_call(100.0, 1.0)
        assert isinstance(option, UnifiedEuropeanOption)
        assert option.strike == 100.0
        assert option.maturity == 1.0
        assert option.option_type == 'call'
    
    def test_create_unified_european_put(self):
        """Test unified European put creation."""
        option = create_unified_european_put(100.0, 1.0)
        assert isinstance(option, UnifiedEuropeanOption)
        assert option.option_type == 'put'
    
    def test_create_unified_basket_call(self):
        """Test unified basket call creation."""
        strikes = np.array([100.0, 110.0])
        weights = np.array([0.6, 0.4])
        
        option = create_unified_basket_call(strikes, weights, 1.0)
        assert isinstance(option, UnifiedBasketOption)
        assert_array_equal(option.strikes, strikes)
        assert_array_equal(option.weights, weights)
        assert option.option_type == 'call'
    
    def test_create_unified_pricing_engine(self):
        """Test unified pricing engine creation."""
        process = create_black_scholes_process(0.05, 0.2)
        engine = create_unified_pricing_engine(process)
        
        assert isinstance(engine, UnifiedPricingEngine)
        assert engine.process is process
    
    def test_create_log_grid(self):
        """Test logarithmic grid creation."""
        grid = create_log_grid(50.0, 200.0, 11)
        
        assert len(grid) == 11
        assert grid[0] == 50.0
        assert grid[-1] == 200.0
        assert np.all(np.diff(grid) > 0)  # Increasing
    
    def test_create_log_grid_centered(self):
        """Test centered logarithmic grid creation."""
        grid = create_log_grid(50.0, 200.0, 11, center=100.0)
        
        assert len(grid) == 11
        assert 50.0 <= grid[0] <= 200.0
        assert 50.0 <= grid[-1] <= 200.0
        
        # Should be roughly centered around 100
        center_idx = len(grid) // 2
        assert 90.0 <= grid[center_idx] <= 110.0
    
    def test_create_linear_grid(self):
        """Test linear grid creation."""
        grid = create_linear_grid(0.0, 1.0, 11)
        
        assert len(grid) == 11
        assert grid[0] == 0.0
        assert grid[-1] == 1.0
        assert_allclose(np.diff(grid), 0.1, rtol=1e-10)


class TestPricingEngineIntegration:
    """Integration tests for unified pricing engine."""
    
    def test_gbm_call_pricing_integration(self):
        """Test complete GBM call pricing workflow."""
        # Set up Black-Scholes parameters
        process = create_black_scholes_process(0.05, 0.2)
        engine = create_unified_pricing_engine(process)
        
        # Create option
        option = create_unified_european_call(100.0, 0.25)
        
        # Create grids
        s_grid = create_log_grid(50.0, 150.0, 21, center=100.0)
        time_grid = np.linspace(0, 0.25, 10)
        
        # Price option
        prices = engine.price_option(option, s_grid, time_grid=time_grid)
        
        # Extract price at spot = 100
        s_idx = np.argmin(np.abs(s_grid - 100.0))
        option_price = prices[0, s_idx]  # At t=0
        
        # Should be reasonable for ATM call
        assert 5.0 <= option_price <= 15.0
        
        # Compute Greeks
        greeks = engine.compute_greeks(prices, s_grid)
        
        # Delta should be around 0.5 for ATM call
        delta_atm = greeks['delta'][s_idx]
        assert 0.3 <= delta_atm <= 0.7
    
    def test_heston_call_pricing_integration(self):
        """Test complete Heston call pricing workflow."""
        # Set up Heston parameters
        process = create_standard_heston()
        engine = create_unified_pricing_engine(process)
        
        # Create option
        option = create_unified_european_call(100.0, 0.25)
        
        # Create grids
        s_grid = create_log_grid(50.0, 150.0, 15, center=100.0)
        v_grid = create_linear_grid(0.01, 0.3, 8)
        time_grid = np.linspace(0, 0.25, 6)
        
        # Price option
        prices = engine.price_option(option, s_grid, v_grid, time_grid=time_grid)
        
        # Extract price at spot = 100, vol = 0.04
        s_idx = np.argmin(np.abs(s_grid - 100.0))
        v_idx = np.argmin(np.abs(v_grid - 0.04))
        option_price = prices[0, s_idx, v_idx]
        
        # Should be reasonable for ATM call
        assert 3.0 <= option_price <= 20.0
        
        # Compute Greeks
        greeks = engine.compute_greeks(prices, s_grid, v_grid)
        
        # Check that Greeks have reasonable values
        delta_atm = greeks['delta'][s_idx, v_idx]
        vega_atm = greeks['vega'][s_idx, v_idx]
        
        assert 0.2 <= delta_atm <= 0.8
        assert vega_atm >= 0  # Vega should be positive for calls
    
    def test_basket_option_pricing_integration(self):
        """Test basket option pricing with 2D process."""
        # Use simple 2D process (could be extended to proper multi-asset model)
        process = create_standard_heston()
        engine = create_unified_pricing_engine(process)
        
        # Create basket option
        strikes = np.array([100.0, 100.0])
        weights = np.array([0.5, 0.5])
        option = create_unified_basket_call(strikes, weights, 0.25)
        
        # Create grids (treating as two assets for simplicity)
        s1_grid = create_log_grid(80.0, 120.0, 9)
        s2_grid = create_log_grid(80.0, 120.0, 9)
        time_grid = np.linspace(0, 0.25, 5)
        
        # Price option
        prices = engine.price_option(option, s1_grid, s2_grid, time_grid=time_grid)
        
        # Check basic properties
        assert prices.shape == (len(time_grid), len(s1_grid), len(s2_grid))
        assert np.all(prices >= 0)
        
        # Terminal condition should match payoff
        terminal_payoff = option.payoff(s1_grid, s2_grid)
        assert_allclose(prices[-1], terminal_payoff, rtol=1e-10)


class TestErrorHandling:
    """Test error handling in unified pricing engine."""
    
    def test_unsupported_dimension_error(self):
        """Test error for unsupported process dimensions."""
        # This would require creating a 4D+ process, which we don't have
        # Skip for now as our framework supports up to 3D
        pass
    
    def test_invalid_time_grid(self):
        """Test handling of invalid time grids."""
        process = create_black_scholes_process(0.05, 0.2)
        engine = create_unified_pricing_engine(process)
        option = create_unified_european_call(100.0, 0.25)
        s_grid = create_log_grid(50.0, 150.0, 11)
        
        # Empty time grid should use default
        prices = engine.price_option(option, s_grid, time_grid=np.array([]))
        assert prices.shape[0] > 0  # Should create default time grid
    
    def test_boundary_condition_defaults(self):
        """Test that default boundary conditions are created."""
        process = create_black_scholes_process(0.05, 0.2)
        engine = create_unified_pricing_engine(process)
        option = create_unified_european_call(100.0, 0.25)
        s_grid = create_log_grid(50.0, 150.0, 11)
        
        # Should work without explicit boundary conditions
        prices = engine.price_option(option, s_grid)
        assert prices.shape == (50, len(s_grid))  # Default time grid size
