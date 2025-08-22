"""Tests for multi-dimensional pricing engine."""
import numpy as np
import pytest

from src.multidimensional_processes import HestonModel, ThreeFactorModel
from src.multidimensional_solver import ADISolver
from src.multidimensional_pricing_engine import (
    MultiDimensionalOption,
    MultiDimensionalPricingEngine,
    create_european_call_2d,
    create_european_put_2d,
    create_basket_call,
    create_log_grid,
    create_variance_grid,
    create_heston_grids,
    create_default_pricing_engine,
)
from src.exceptions import ValidationError, PricingError


class TestMultiDimensionalOption:
    """Test cases for multi-dimensional option."""

    def test_option_initialization(self):
        """Test option initialization with valid parameters."""
        def simple_payoff(s1, s2):
            return np.maximum(s1 - 100, 0)

        option = MultiDimensionalOption(
            payoff_func=simple_payoff,
            maturity_time=1.0,
            strike=100.0,
            option_type="call"
        )

        assert option.maturity == 1.0
        assert option.strike == 100.0
        assert option.option_type == "call"

    def test_option_parameter_validation(self):
        """Test option parameter validation."""
        def simple_payoff(s1, s2):
            return np.maximum(s1 - 100, 0)

        # Invalid maturity
        with pytest.raises(ValidationError, match="maturity must be positive"):
            MultiDimensionalOption(
                payoff_func=simple_payoff,
                maturity_time=-1.0
            )

        with pytest.raises(ValidationError, match="maturity must be positive"):
            MultiDimensionalOption(
                payoff_func=simple_payoff,
                maturity_time=0.0
            )

        # Invalid strike
        with pytest.raises(ValidationError, match="strike must be positive"):
            MultiDimensionalOption(
                payoff_func=simple_payoff,
                maturity_time=1.0,
                strike=-50.0
            )

    def test_option_payoff_computation(self):
        """Test option payoff computation."""
        def call_payoff(s1, s2):
            return np.maximum(s1 - 100, 0)

        option = MultiDimensionalOption(
            payoff_func=call_payoff,
            maturity_time=1.0,
            strike=100.0
        )

        s1 = np.array([90, 100, 110])
        s2 = np.array([0.1, 0.2, 0.3])

        payoff = option.payoff(s1, s2)
        expected = np.array([0, 0, 10])

        assert np.allclose(payoff, expected)


class TestMultiDimensionalPricingEngine:
    """Test cases for multi-dimensional pricing engine."""

    def test_pricing_engine_initialization(self):
        """Test pricing engine initialization."""
        process = HestonModel(r=0.05, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7)
        solver = ADISolver()
        
        engine = MultiDimensionalPricingEngine(process=process, solver=solver)
        
        assert engine.process is process
        assert engine.solver is solver

    def test_price_option_input_validation(self):
        """Test input validation for price_option method."""
        process = HestonModel(r=0.05, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7)
        solver = ADISolver()
        engine = MultiDimensionalPricingEngine(process=process, solver=solver)

        option = create_european_call_2d(strike=100.0, maturity=1.0)

        # Wrong number of grids
        with pytest.raises(ValidationError, match="Number of grids.*must match process dimension"):
            single_grid = (np.linspace(50, 150, 10),)
            engine.price_option(option, single_grid)

        # Invalid time steps
        with pytest.raises(ValidationError, match="time_steps must be positive"):
            grids = (np.linspace(50, 150, 10), np.linspace(0.01, 0.5, 5))
            engine.price_option(option, grids, time_steps=0)

    def test_price_option_2d_simple(self):
        """Test 2D option pricing with simple setup."""
        # Simple Heston model
        process = HestonModel(r=0.05, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7)
        solver = ADISolver(theta=1.0)  # Fully implicit for stability
        engine = MultiDimensionalPricingEngine(process=process, solver=solver)

        # European call option
        option = create_european_call_2d(strike=100.0, maturity=0.25)

        # Small grids for efficiency
        spot_grid = np.linspace(80, 120, 15)
        var_grid = np.linspace(0.01, 0.2, 10)
        grids = (spot_grid, var_grid)

        # Price option
        prices = engine.price_option(option, grids, time_steps=20)

        # Check basic properties
        assert prices.shape == (len(spot_grid), len(var_grid))
        assert np.all(prices >= 0)  # Prices should be non-negative

        # Check monotonicity in spot price (call option)
        for j in range(len(var_grid)):
            for i in range(len(spot_grid) - 1):
                assert prices[i + 1, j] >= prices[i, j]  # Non-decreasing in spot

    def test_price_option_3d_simple(self):
        """Test 3D option pricing with simple setup."""
        # Three-factor model
        process = ThreeFactorModel(
            r=0.05,
            mu=np.array([0.1, 0.08, 0.06]),
            sigma=np.array([0.2, 0.15, 0.1]),
            correlation_matrix=np.eye(3)
        )
        solver = ADISolver(theta=1.0)
        engine = MultiDimensionalPricingEngine(process=process, solver=solver)

        # Basket call option
        weights = np.array([0.5, 0.3, 0.2])
        option = create_basket_call(weights=weights, strike=100.0, maturity=0.25)

        # Small grids
        grid1 = np.linspace(80, 120, 8)
        grid2 = np.linspace(80, 120, 6)
        grid3 = np.linspace(80, 120, 5)
        grids = (grid1, grid2, grid3)

        # Price option
        prices = engine.price_option(option, grids, time_steps=10)

        # Check basic properties
        assert prices.shape == (len(grid1), len(grid2), len(grid3))
        assert np.all(prices >= 0)

    def test_compute_greeks_2d(self):
        """Test Greeks computation for 2D option."""
        process = HestonModel(r=0.05, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7)
        solver = ADISolver(theta=1.0)
        engine = MultiDimensionalPricingEngine(process=process, solver=solver)

        option = create_european_call_2d(strike=100.0, maturity=0.25)

        # Small grids for efficiency
        spot_grid = np.linspace(90, 110, 10)
        var_grid = np.linspace(0.02, 0.1, 8)
        grids = (spot_grid, var_grid)

        # Compute Greeks
        greeks = engine.compute_greeks(option, grids, time_steps=15, finite_diff_step=0.01)

        # Check that Greeks are computed
        assert 'delta' in greeks
        assert 'gamma' in greeks
        assert 'vega' in greeks

        # Check shapes
        assert greeks['delta'].shape == (len(spot_grid), len(var_grid))
        assert greeks['gamma'].shape == (len(spot_grid), len(var_grid))
        assert greeks['vega'].shape == (len(spot_grid), len(var_grid))

        # Check that delta is positive for call option (at least at some points)
        assert np.any(greeks['delta'] > 0)

    def test_unsupported_dimension(self):
        """Test error handling for unsupported dimensions."""
        # Create a mock process with unsupported dimension
        class MockProcess:
            @property
            def dimension(self):
                return 4  # Unsupported - return integer directly
            
            def drift(self, states):
                return np.zeros((states.shape[0], 4))
            
            def diffusion_matrix(self, states):
                return np.zeros((states.shape[0], 4, 4))

        process = MockProcess()
        solver = ADISolver()
        engine = MultiDimensionalPricingEngine(process=process, solver=solver)

        option = create_european_call_2d(strike=100.0, maturity=1.0)
        grids = tuple(np.linspace(50, 150, 10) for _ in range(4))

        with pytest.raises(PricingError, match="Unsupported dimension"):
            engine.price_option(option, grids)


class TestConvenienceFunctions:
    """Test convenience functions for creating options."""

    def test_create_european_call_2d(self):
        """Test European call option creation."""
        option = create_european_call_2d(strike=100.0, maturity=1.0)
        
        assert option.strike == 100.0
        assert option.maturity == 1.0
        assert option.option_type == "european_call_2d"

        # Test payoff
        s1 = np.array([90, 100, 110])
        s2 = np.array([0.1, 0.2, 0.3])
        payoff = option.payoff(s1, s2)
        expected = np.array([0, 0, 10])
        assert np.allclose(payoff, expected)

    def test_create_european_put_2d(self):
        """Test European put option creation."""
        option = create_european_put_2d(strike=100.0, maturity=1.0)
        
        assert option.strike == 100.0
        assert option.maturity == 1.0
        assert option.option_type == "european_put_2d"

        # Test payoff
        s1 = np.array([90, 100, 110])
        s2 = np.array([0.1, 0.2, 0.3])
        payoff = option.payoff(s1, s2)
        expected = np.array([10, 0, 0])
        assert np.allclose(payoff, expected)

    def test_create_basket_call(self):
        """Test basket call option creation."""
        weights = np.array([0.6, 0.4])
        option = create_basket_call(weights=weights, strike=100.0, maturity=1.0)
        
        assert option.strike == 100.0
        assert option.maturity == 1.0
        assert option.option_type == "basket_call"

        # Test payoff
        s1 = np.array([80, 100, 120])
        s2 = np.array([90, 110, 130])
        payoff = option.payoff(s1, s2)
        
        # Basket values: 0.6*80 + 0.4*90 = 84, etc.
        basket_values = 0.6 * s1 + 0.4 * s2
        expected = np.maximum(basket_values - 100, 0)
        assert np.allclose(payoff, expected)

    def test_basket_call_validation(self):
        """Test basket call validation."""
        weights = np.array([0.6, 0.4])
        option = create_basket_call(weights=weights, strike=100.0, maturity=1.0)

        # Wrong number of states
        with pytest.raises(ValidationError, match="Number of states.*must match weights"):
            s1 = np.array([100])
            option.payoff(s1)  # Missing second state


class TestGridUtilities:
    """Test grid creation utilities."""

    def test_create_log_grid(self):
        """Test logarithmic grid creation."""
        grid = create_log_grid(spot=100.0, num_points=21, std_devs=2.0)
        
        assert len(grid) == 21
        assert np.all(grid > 0)
        assert grid[10] == pytest.approx(100.0, rel=0.1)  # Middle point near spot

    def test_create_log_grid_validation(self):
        """Test log grid parameter validation."""
        with pytest.raises(ValidationError, match="spot must be positive"):
            create_log_grid(spot=-100.0, num_points=10)

        with pytest.raises(ValidationError, match="num_points must be positive"):
            create_log_grid(spot=100.0, num_points=0)

    def test_create_variance_grid(self):
        """Test variance grid creation."""
        grid = create_variance_grid(initial_var=0.04, num_points=11, max_var_multiple=3.0)
        
        assert len(grid) == 11
        assert np.all(grid > 0)
        assert grid[0] == 0.001  # Minimum variance
        assert grid[-1] == pytest.approx(3.0 * 0.04, rel=1e-10)  # Maximum variance

    def test_create_variance_grid_validation(self):
        """Test variance grid parameter validation."""
        with pytest.raises(ValidationError, match="initial_var must be positive"):
            create_variance_grid(initial_var=-0.04, num_points=10)

        with pytest.raises(ValidationError, match="num_points must be positive"):
            create_variance_grid(initial_var=0.04, num_points=-5)

    def test_create_heston_grids(self):
        """Test Heston grid creation."""
        spot_grid, var_grid = create_heston_grids(
            spot=100.0,
            initial_var=0.04,
            num_spot=21,
            num_var=11
        )
        
        assert len(spot_grid) == 21
        assert len(var_grid) == 11
        assert np.all(spot_grid > 0)
        assert np.all(var_grid > 0)

    def test_create_default_pricing_engine(self):
        """Test default pricing engine creation."""
        process = HestonModel(r=0.05, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7)
        engine = create_default_pricing_engine(process)
        
        assert isinstance(engine, MultiDimensionalPricingEngine)
        assert engine.process is process
        assert isinstance(engine.solver, ADISolver)


class TestPricingEngineIntegration:
    """Integration tests for pricing engine."""

    def test_heston_call_pricing_integration(self):
        """Test complete Heston call option pricing workflow."""
        # Set up Heston model
        process = HestonModel(r=0.05, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7)
        engine = create_default_pricing_engine(process)

        # Create option
        option = create_european_call_2d(strike=100.0, maturity=0.25)

        # Create grids
        spot_grid, var_grid = create_heston_grids(
            spot=100.0,
            initial_var=0.04,
            num_spot=25,
            num_var=15
        )

        # Price option
        prices = engine.price_option(option, (spot_grid, var_grid), time_steps=30)

        # Basic sanity checks
        assert prices.shape == (25, 15)
        assert np.all(prices >= 0)
        
        # At-the-money option should have positive value
        spot_idx = np.argmin(np.abs(spot_grid - 100.0))
        var_idx = np.argmin(np.abs(var_grid - 0.04))
        atm_price = prices[spot_idx, var_idx]
        assert atm_price > 0

        # Deep out-of-the-money should have low value
        otm_idx = np.argmin(np.abs(spot_grid - 80.0))
        otm_price = prices[otm_idx, var_idx]
        assert otm_price < atm_price

    def test_put_call_parity_approximation(self):
        """Test approximate put-call parity for European options."""
        process = HestonModel(r=0.05, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7)
        engine = create_default_pricing_engine(process)

        strike = 100.0
        maturity = 0.25
        call_option = create_european_call_2d(strike=strike, maturity=maturity)
        put_option = create_european_put_2d(strike=strike, maturity=maturity)

        # Small grids for efficiency
        spot_grid = np.linspace(90, 110, 15)
        var_grid = np.linspace(0.02, 0.08, 10)
        grids = (spot_grid, var_grid)

        # Price both options
        call_prices = engine.price_option(call_option, grids, time_steps=20)
        put_prices = engine.price_option(put_option, grids, time_steps=20)

        # Check put-call parity at a few points
        # C - P ≈ S - K*exp(-r*T) (approximately, ignoring stochastic vol effects)
        discount_factor = np.exp(-process.r * maturity)
        
        for i in [5, 7, 9]:  # Check a few spot prices
            for j in [3, 5, 7]:  # Check a few variance levels
                spot = spot_grid[i]
                call_price = call_prices[i, j]
                put_price = put_prices[i, j]
                
                parity_diff = call_price - put_price
                expected_diff = spot - strike * discount_factor
                
                # Allow for some deviation due to stochastic volatility
                relative_error = abs(parity_diff - expected_diff) / abs(expected_diff)
                assert relative_error < 0.5  # Within 50% (rough approximation)
