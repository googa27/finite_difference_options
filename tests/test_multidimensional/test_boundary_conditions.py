"""Tests for multi-dimensional boundary conditions module."""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from src.multidimensional_boundary_conditions import (
    BoundaryType,
    BoundaryLocation,
    BoundaryCondition,
    BoundaryManager2D,
    BoundaryManager3D,
    create_dirichlet_boundaries_2d,
    create_zero_gradient_boundaries_2d,
    create_heston_boundaries,
    create_mixed_boundaries_2d,
)
from src.exceptions import ValidationError


class TestBoundaryCondition:
    """Test BoundaryCondition class."""
    
    def test_dirichlet_condition(self):
        """Test Dirichlet boundary condition creation."""
        bc = BoundaryCondition(BoundaryType.DIRICHLET, 1.0)
        assert bc.boundary_type == BoundaryType.DIRICHLET
        assert bc.value == 1.0
        assert bc.alpha == 1.0
        assert bc.beta == 0.0
    
    def test_neumann_condition(self):
        """Test Neumann boundary condition creation."""
        bc = BoundaryCondition(BoundaryType.NEUMANN, 0.5)
        assert bc.boundary_type == BoundaryType.NEUMANN
        assert bc.value == 0.5
    
    def test_robin_condition_valid(self):
        """Test valid Robin boundary condition."""
        bc = BoundaryCondition(BoundaryType.ROBIN, 1.0, alpha=2.0, beta=1.0)
        assert bc.boundary_type == BoundaryType.ROBIN
        assert bc.alpha == 2.0
        assert bc.beta == 1.0
    
    def test_robin_condition_invalid(self):
        """Test invalid Robin boundary condition with zero coefficients."""
        with pytest.raises(ValidationError, match="Robin condition requires"):
            BoundaryCondition(BoundaryType.ROBIN, 1.0, alpha=0.0, beta=0.0)
    
    def test_zero_gradient_condition(self):
        """Test zero gradient boundary condition."""
        bc = BoundaryCondition(BoundaryType.ZERO_GRADIENT)
        assert bc.boundary_type == BoundaryType.ZERO_GRADIENT
        assert bc.value == 0.0


class TestBoundaryManager2D:
    """Test BoundaryManager2D class."""
    
    @pytest.fixture
    def simple_boundaries(self):
        """Create simple 2D boundary conditions."""
        return {
            BoundaryLocation.LEFT: BoundaryCondition(BoundaryType.DIRICHLET, 0.0),
            BoundaryLocation.RIGHT: BoundaryCondition(BoundaryType.DIRICHLET, 1.0),
            BoundaryLocation.BOTTOM: BoundaryCondition(BoundaryType.ZERO_GRADIENT),
            BoundaryLocation.TOP: BoundaryCondition(BoundaryType.ZERO_GRADIENT),
        }
    
    @pytest.fixture
    def test_grids(self):
        """Create test grids."""
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 4)
        return x, y
    
    def test_initialization_valid(self, simple_boundaries):
        """Test valid boundary manager initialization."""
        manager = BoundaryManager2D(simple_boundaries)
        assert len(manager.boundaries) == 4
    
    def test_initialization_invalid_location(self):
        """Test invalid boundary location."""
        boundaries = {
            BoundaryLocation.FRONT: BoundaryCondition(BoundaryType.DIRICHLET, 0.0)
        }
        with pytest.raises(ValidationError, match="Invalid 2D boundary location"):
            BoundaryManager2D(boundaries)
    
    def test_apply_dirichlet_boundaries(self, test_grids):
        """Test application of Dirichlet boundary conditions."""
        x, y = test_grids
        u = np.ones((len(x), len(y)))
        
        boundaries = {
            BoundaryLocation.LEFT: BoundaryCondition(BoundaryType.DIRICHLET, 2.0),
            BoundaryLocation.RIGHT: BoundaryCondition(BoundaryType.DIRICHLET, 3.0),
            BoundaryLocation.BOTTOM: BoundaryCondition(BoundaryType.DIRICHLET, 4.0),
            BoundaryLocation.TOP: BoundaryCondition(BoundaryType.DIRICHLET, 5.0),
        }
        
        manager = BoundaryManager2D(boundaries)
        result = manager.apply_boundaries(u, (x, y))
        
        # Check boundary values (excluding corners which may be overwritten)
        assert_allclose(result[0, 1:-1], 2.0)  # Left (excluding corners)
        assert_allclose(result[-1, 1:-1], 3.0)  # Right (excluding corners)
        assert_allclose(result[1:-1, 0], 4.0)  # Bottom (excluding corners)
        assert_allclose(result[1:-1, -1], 5.0)  # Top (excluding corners)
        
        # Check that boundaries are applied (corners will have last applied value)
        assert result[0, 0] in [2.0, 4.0]  # Bottom-left corner
        assert result[0, -1] in [2.0, 5.0]  # Top-left corner
        assert result[-1, 0] in [3.0, 4.0]  # Bottom-right corner
        assert result[-1, -1] in [3.0, 5.0]  # Top-right corner
    
    def test_apply_zero_gradient_boundaries(self, test_grids):
        """Test zero gradient boundary conditions."""
        x, y = test_grids
        u = np.random.rand(len(x), len(y))
        
        boundaries = {
            BoundaryLocation.LEFT: BoundaryCondition(BoundaryType.ZERO_GRADIENT),
            BoundaryLocation.RIGHT: BoundaryCondition(BoundaryType.ZERO_GRADIENT),
            BoundaryLocation.BOTTOM: BoundaryCondition(BoundaryType.ZERO_GRADIENT),
            BoundaryLocation.TOP: BoundaryCondition(BoundaryType.ZERO_GRADIENT),
        }
        
        manager = BoundaryManager2D(boundaries)
        result = manager.apply_boundaries(u, (x, y))
        
        # Check zero gradient conditions
        assert_allclose(result[0, :], result[1, :])  # Left
        assert_allclose(result[-1, :], result[-2, :])  # Right
        assert_allclose(result[:, 0], result[:, 1])  # Bottom
        assert_allclose(result[:, -1], result[:, -2])  # Top
    
    def test_apply_neumann_boundaries(self, test_grids):
        """Test Neumann boundary conditions."""
        x, y = test_grids
        u = np.ones((len(x), len(y))) * 2.0
        
        boundaries = {
            BoundaryLocation.LEFT: BoundaryCondition(BoundaryType.NEUMANN, 1.0),
            BoundaryLocation.RIGHT: BoundaryCondition(BoundaryType.NEUMANN, -1.0),
            BoundaryLocation.BOTTOM: BoundaryCondition(BoundaryType.NEUMANN, 0.5),
            BoundaryLocation.TOP: BoundaryCondition(BoundaryType.NEUMANN, -0.5),
        }
        
        manager = BoundaryManager2D(boundaries)
        result = manager.apply_boundaries(u, (x, y))
        
        # Check Neumann conditions (approximate due to finite differences)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        # Left: u[0] = u[1] - dx * value
        expected_left = result[1, :] - dx * 1.0
        assert_allclose(result[0, :], expected_left)
        
        # Right: u[-1] = u[-2] + dx * value
        expected_right = result[-2, :] + dx * (-1.0)
        assert_allclose(result[-1, :], expected_right)
    
    def test_apply_robin_boundaries(self, test_grids):
        """Test Robin boundary conditions."""
        x, y = test_grids
        u = np.ones((len(x), len(y))) * 2.0
        
        # Simple Robin condition: u + ∂u/∂x = 1
        boundaries = {
            BoundaryLocation.LEFT: BoundaryCondition(
                BoundaryType.ROBIN, 1.0, alpha=1.0, beta=1.0
            ),
        }
        
        manager = BoundaryManager2D(boundaries)
        result = manager.apply_boundaries(u, (x, y))
        
        # Check that Robin condition is satisfied approximately
        dx = x[1] - x[0]
        # α*u[0] + β*(u[1] - u[0])/dx = value
        # u[0] = (value - β*u[1]/dx) / (α - β/dx)
        expected = (1.0 - 1.0 * result[1, :] / dx) / (1.0 - 1.0 / dx)
        assert_allclose(result[0, :], expected, rtol=1e-10)
    
    def test_wrong_grid_dimensions(self, simple_boundaries):
        """Test error with wrong number of grids."""
        manager = BoundaryManager2D(simple_boundaries)
        u = np.ones((5, 4))
        
        with pytest.raises(ValidationError, match="2D boundary manager requires 2 grids"):
            manager.apply_boundaries(u, (np.linspace(0, 1, 5),))
    
    def test_wrong_solution_shape(self, simple_boundaries, test_grids):
        """Test error with wrong solution shape."""
        manager = BoundaryManager2D(simple_boundaries)
        x, y = test_grids
        u = np.ones((3, 3))  # Wrong shape
        
        with pytest.raises(ValidationError, match="Solution shape .* doesn't match grid"):
            manager.apply_boundaries(u, (x, y))
    
    def test_callable_boundary_value(self, test_grids):
        """Test boundary condition with callable value."""
        x, y = test_grids
        u = np.ones((len(x), len(y)))
        
        def boundary_func(x_val, y_val, time):
            if isinstance(x_val, np.ndarray):
                return np.sin(y_val) + time
            else:
                return np.sin(y_val) + time
        
        boundaries = {
            BoundaryLocation.LEFT: BoundaryCondition(
                BoundaryType.DIRICHLET, boundary_func
            ),
        }
        
        manager = BoundaryManager2D(boundaries)
        result = manager.apply_boundaries(u, (x, y), time=0.5)
        
        expected = np.sin(y) + 0.5
        assert_allclose(result[0, :], expected)


class TestBoundaryManager3D:
    """Test BoundaryManager3D class."""
    
    def test_initialization_and_basic_application(self):
        """Test 3D boundary manager basic functionality."""
        boundaries = {
            BoundaryLocation.LEFT: BoundaryCondition(BoundaryType.DIRICHLET, 0.0),
            BoundaryLocation.RIGHT: BoundaryCondition(BoundaryType.ZERO_GRADIENT),
        }
        
        manager = BoundaryManager3D(boundaries)
        
        x = np.linspace(0, 1, 3)
        y = np.linspace(0, 1, 3)
        z = np.linspace(0, 1, 3)
        u = np.ones((3, 3, 3))
        
        result = manager.apply_boundaries(u, (x, y, z))
        
        # Check basic boundary application
        assert_allclose(result[0, :, :], 0.0)  # Left Dirichlet
        assert_allclose(result[-1, :, :], result[-2, :, :])  # Right zero gradient
    
    def test_wrong_grid_dimensions_3d(self):
        """Test error with wrong number of grids for 3D."""
        boundaries = {
            BoundaryLocation.LEFT: BoundaryCondition(BoundaryType.DIRICHLET, 0.0),
        }
        manager = BoundaryManager3D(boundaries)
        u = np.ones((3, 3, 3))
        
        with pytest.raises(ValidationError, match="3D boundary manager requires 3 grids"):
            manager.apply_boundaries(u, (np.linspace(0, 1, 3), np.linspace(0, 1, 3)))


class TestConvenienceFunctions:
    """Test convenience functions for boundary creation."""
    
    def test_create_dirichlet_boundaries_2d(self):
        """Test Dirichlet boundary creation function."""
        manager = create_dirichlet_boundaries_2d(1.0, 2.0, 3.0, 4.0)
        
        assert isinstance(manager, BoundaryManager2D)
        assert len(manager.boundaries) == 4
        assert manager.boundaries[BoundaryLocation.LEFT].value == 1.0
        assert manager.boundaries[BoundaryLocation.RIGHT].value == 2.0
        assert manager.boundaries[BoundaryLocation.BOTTOM].value == 3.0
        assert manager.boundaries[BoundaryLocation.TOP].value == 4.0
        
        for bc in manager.boundaries.values():
            assert bc.boundary_type == BoundaryType.DIRICHLET
    
    def test_create_zero_gradient_boundaries_2d(self):
        """Test zero gradient boundary creation function."""
        manager = create_zero_gradient_boundaries_2d()
        
        assert isinstance(manager, BoundaryManager2D)
        assert len(manager.boundaries) == 4
        
        for bc in manager.boundaries.values():
            assert bc.boundary_type == BoundaryType.ZERO_GRADIENT
    
    def test_create_heston_boundaries_call(self):
        """Test Heston boundary creation for call option."""
        manager = create_heston_boundaries(
            spot_min=0.0, spot_max=200.0, var_max=1.0, strike=100.0, is_call=True
        )
        
        assert isinstance(manager, BoundaryManager2D)
        
        # Check call option boundaries
        left_bc = manager.boundaries[BoundaryLocation.LEFT]
        right_bc = manager.boundaries[BoundaryLocation.RIGHT]
        bottom_bc = manager.boundaries[BoundaryLocation.BOTTOM]
        top_bc = manager.boundaries[BoundaryLocation.TOP]
        
        assert left_bc.boundary_type == BoundaryType.DIRICHLET
        assert left_bc.value == 0.0  # S=0 -> call worth 0
        
        assert right_bc.boundary_type == BoundaryType.DIRICHLET
        assert right_bc.value == 100.0  # S=200, K=100 -> call worth 100
        
        assert bottom_bc.boundary_type == BoundaryType.ZERO_GRADIENT
        assert top_bc.boundary_type == BoundaryType.ZERO_GRADIENT
    
    def test_create_heston_boundaries_put(self):
        """Test Heston boundary creation for put option."""
        manager = create_heston_boundaries(
            spot_min=0.0, spot_max=200.0, var_max=1.0, strike=100.0, is_call=False
        )
        
        # Check put option boundaries
        left_bc = manager.boundaries[BoundaryLocation.LEFT]
        right_bc = manager.boundaries[BoundaryLocation.RIGHT]
        
        assert left_bc.boundary_type == BoundaryType.DIRICHLET
        assert left_bc.value == 100.0  # S=0 -> put worth K
        
        assert right_bc.boundary_type == BoundaryType.DIRICHLET
        assert right_bc.value == 0.0  # S=200, K=100 -> put worth 0
    
    def test_create_mixed_boundaries_2d(self):
        """Test mixed boundary creation function."""
        manager = create_mixed_boundaries_2d(
            left_type=BoundaryType.DIRICHLET,
            left_value=1.0,
            right_type=BoundaryType.NEUMANN,
            right_value=0.5,
            bottom_type=BoundaryType.ZERO_GRADIENT,
            top_type=BoundaryType.ROBIN,
            top_value=2.0
        )
        
        assert isinstance(manager, BoundaryManager2D)
        
        left_bc = manager.boundaries[BoundaryLocation.LEFT]
        right_bc = manager.boundaries[BoundaryLocation.RIGHT]
        bottom_bc = manager.boundaries[BoundaryLocation.BOTTOM]
        top_bc = manager.boundaries[BoundaryLocation.TOP]
        
        assert left_bc.boundary_type == BoundaryType.DIRICHLET
        assert left_bc.value == 1.0
        
        assert right_bc.boundary_type == BoundaryType.NEUMANN
        assert right_bc.value == 0.5
        
        assert bottom_bc.boundary_type == BoundaryType.ZERO_GRADIENT
        
        assert top_bc.boundary_type == BoundaryType.ROBIN
        assert top_bc.value == 2.0


class TestBoundaryIntegration:
    """Integration tests for boundary conditions."""
    
    def test_heat_equation_boundaries(self):
        """Test boundary conditions on simple heat equation solution."""
        # Create a simple 2D grid
        x = np.linspace(0, 1, 11)
        y = np.linspace(0, 1, 11)
        
        # Initial temperature distribution
        u = np.zeros((len(x), len(y)))
        u[5, 5] = 1.0  # Hot spot in center
        
        # Apply fixed temperature boundaries
        boundaries = {
            BoundaryLocation.LEFT: BoundaryCondition(BoundaryType.DIRICHLET, 0.0),
            BoundaryLocation.RIGHT: BoundaryCondition(BoundaryType.DIRICHLET, 0.0),
            BoundaryLocation.BOTTOM: BoundaryCondition(BoundaryType.DIRICHLET, 0.0),
            BoundaryLocation.TOP: BoundaryCondition(BoundaryType.DIRICHLET, 0.0),
        }
        
        manager = BoundaryManager2D(boundaries)
        result = manager.apply_boundaries(u, (x, y))
        
        # Check that boundaries are fixed at 0
        assert_allclose(result[0, :], 0.0)
        assert_allclose(result[-1, :], 0.0)
        assert_allclose(result[:, 0], 0.0)
        assert_allclose(result[:, -1], 0.0)
        
        # Check that interior is preserved
        assert result[5, 5] == 1.0
    
    def test_option_pricing_boundaries(self):
        """Test realistic option pricing boundary setup."""
        # Spot and volatility grids for Heston model
        s_grid = np.linspace(0, 200, 21)
        v_grid = np.linspace(0, 1, 11)
        
        # Initial option values (at maturity)
        strike = 100.0
        s_mesh, v_mesh = np.meshgrid(s_grid, v_grid, indexing='ij')
        payoff = np.maximum(s_mesh - strike, 0)  # Call payoff
        
        # Apply Heston boundaries
        manager = create_heston_boundaries(
            spot_min=s_grid[0], spot_max=s_grid[-1], 
            var_max=v_grid[-1], strike=strike, is_call=True
        )
        
        result = manager.apply_boundaries(payoff, (s_grid, v_grid))
        
        # Check boundary conditions
        assert_allclose(result[0, :], 0.0)  # S=0 boundary
        assert_allclose(result[-1, :], s_grid[-1] - strike)  # S=S_max boundary
        
        # Variance boundaries should be zero gradient
        assert_allclose(result[:, 0], result[:, 1])  # V=0 boundary
        assert_allclose(result[:, -1], result[:, -2])  # V=V_max boundary
    
    @pytest.mark.parametrize("boundary_type", [
        BoundaryType.DIRICHLET,
        BoundaryType.NEUMANN,
        BoundaryType.ZERO_GRADIENT,
        BoundaryType.ROBIN
    ])
    def test_boundary_types_consistency(self, boundary_type):
        """Test that all boundary types work consistently."""
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
        u = np.random.rand(5, 5)
        
        if boundary_type == BoundaryType.ROBIN:
            bc = BoundaryCondition(boundary_type, 1.0, alpha=1.0, beta=0.5)
        else:
            bc = BoundaryCondition(boundary_type, 1.0)
        
        boundaries = {BoundaryLocation.LEFT: bc}
        manager = BoundaryManager2D(boundaries)
        
        # Should not raise any errors
        result = manager.apply_boundaries(u, (x, y))
        assert result.shape == u.shape
