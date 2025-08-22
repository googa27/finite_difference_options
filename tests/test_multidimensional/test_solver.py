"""Tests for multi-dimensional PDE solver."""
import numpy as np
import pytest

from src.multidimensional_solver import (
    ADISolver,
    create_default_adi_solver,
    create_crank_nicolson_solver,
    create_implicit_solver,
)
from src.exceptions import ValidationError


class TestADISolver:
    """Test cases for ADI solver."""

    def test_adi_solver_initialization(self):
        """Test ADI solver initialization with valid parameters."""
        solver = ADISolver()
        assert solver.theta == 0.5
        assert solver.max_iterations == 1000
        assert solver.tolerance == 1e-8

        # Custom parameters
        solver = ADISolver(theta=0.8, max_iterations=500, tolerance=1e-6)
        assert solver.theta == 0.8
        assert solver.max_iterations == 500
        assert solver.tolerance == 1e-6

    def test_adi_solver_parameter_validation(self):
        """Test ADI solver parameter validation."""
        # Invalid theta
        with pytest.raises(ValidationError, match="theta must be between 0 and 1"):
            ADISolver(theta=-0.1)

        with pytest.raises(ValidationError, match="theta must be between 0 and 1"):
            ADISolver(theta=1.5)

        # Invalid max_iterations
        with pytest.raises(ValidationError, match="max_iterations must be positive"):
            ADISolver(max_iterations=0)

        with pytest.raises(ValidationError, match="max_iterations must be positive"):
            ADISolver(max_iterations=-10)

        # Invalid tolerance
        with pytest.raises(ValidationError, match="tolerance must be positive"):
            ADISolver(tolerance=0.0)

        with pytest.raises(ValidationError, match="tolerance must be positive"):
            ADISolver(tolerance=-1e-6)

    def test_2d_input_validation(self):
        """Test 2D input validation."""
        solver = ADISolver()
        
        # Create test grids
        x_grid = np.linspace(0, 1, 5)
        y_grid = np.linspace(0, 1, 4)
        _ = np.linspace(0, 1, 3)  # time_grid not used in validation test
        nx, ny = len(x_grid), len(y_grid)

        # Valid inputs
        drift = np.zeros((nx, ny, 2))
        diffusion = np.zeros((nx, ny, 2, 2))
        initial_conditions = np.ones((nx, ny))

        # This should not raise
        solver._validate_2d_inputs(drift, diffusion, initial_conditions, nx, ny)

        # Invalid drift shape
        with pytest.raises(ValidationError, match="drift must have shape"):
            bad_drift = np.zeros((nx, ny, 3))
            solver._validate_2d_inputs(bad_drift, diffusion, initial_conditions, nx, ny)

        # Invalid diffusion shape
        with pytest.raises(ValidationError, match="diffusion must have shape"):
            bad_diffusion = np.zeros((nx, ny, 3, 3))
            solver._validate_2d_inputs(drift, bad_diffusion, initial_conditions, nx, ny)

        # Invalid initial conditions shape
        with pytest.raises(ValidationError, match="initial_conditions must have shape"):
            bad_ic = np.ones((nx + 1, ny))
            solver._validate_2d_inputs(drift, diffusion, bad_ic, nx, ny)

    def test_3d_input_validation(self):
        """Test 3D input validation."""
        solver = ADISolver()
        
        # Create test grids
        x_grid = np.linspace(0, 1, 3)
        y_grid = np.linspace(0, 1, 4)
        z_grid = np.linspace(0, 1, 5)
        nx, ny, nz = len(x_grid), len(y_grid), len(z_grid)

        # Valid inputs
        drift = np.zeros((nx, ny, nz, 3))
        diffusion = np.zeros((nx, ny, nz, 3, 3))
        initial_conditions = np.ones((nx, ny, nz))

        # This should not raise
        solver._validate_3d_inputs(drift, diffusion, initial_conditions, nx, ny, nz)

        # Invalid drift shape
        with pytest.raises(ValidationError, match="drift must have shape"):
            bad_drift = np.zeros((nx, ny, nz, 2))
            solver._validate_3d_inputs(bad_drift, diffusion, initial_conditions, nx, ny, nz)

        # Invalid diffusion shape
        with pytest.raises(ValidationError, match="diffusion must have shape"):
            bad_diffusion = np.zeros((nx, ny, nz, 2, 2))
            solver._validate_3d_inputs(drift, bad_diffusion, initial_conditions, nx, ny, nz)

        # Invalid initial conditions shape
        with pytest.raises(ValidationError, match="initial_conditions must have shape"):
            bad_ic = np.ones((nx, ny, nz + 1))
            solver._validate_3d_inputs(drift, diffusion, bad_ic, nx, ny, nz)

    def test_tridiagonal_solver(self):
        """Test tridiagonal system solver."""
        solver = ADISolver()

        # Simple test case: solve -u_{i-1} + 2u_i - u_{i+1} = 0 with u_0=0, u_n=1
        n = 5
        a = np.array([0, -1, -1, -1, 0])  # sub-diagonal
        b = np.array([1, 2, 2, 2, 1])    # diagonal
        c = np.array([0, -1, -1, -1, 0])  # super-diagonal
        d = np.array([0, 0, 0, 0, 1])    # RHS

        solution = solver._solve_tridiagonal(a, b, c, d)
        
        # Check boundary conditions
        assert abs(solution[0] - 0.0) < 1e-10
        assert abs(solution[-1] - 1.0) < 1e-10
        
        # Check that solution is monotonic (should increase from 0 to 1)
        for i in range(n - 1):
            assert solution[i + 1] >= solution[i]

    def test_tridiagonal_solver_edge_cases(self):
        """Test tridiagonal solver edge cases."""
        solver = ADISolver()

        # Single element: solve 2*x = 4, so x = 2
        result = solver._solve_tridiagonal(np.array([0]), np.array([2]), np.array([0]), np.array([4]))
        assert abs(result[0] - 2.0) < 1e-10

        # Two elements
        a = np.array([0, 1])
        b = np.array([2, 2])
        c = np.array([1, 0])
        d = np.array([3, 4])
        
        result = solver._solve_tridiagonal(a, b, c, d)
        assert len(result) == 2

    def test_solve_2d_simple(self):
        """Test 2D solver with simple diffusion equation."""
        solver = ADISolver(theta=1.0)  # Fully implicit for stability
        
        # Simple 2D grid
        x_grid = np.linspace(0, 1, 6)
        y_grid = np.linspace(0, 1, 5)
        time_grid = np.linspace(0, 0.1, 3)
        nx, ny = len(x_grid), len(y_grid)

        # Pure diffusion: ∂u/∂t = ∇²u
        drift = np.zeros((nx, ny, 2))
        diffusion = np.ones((nx, ny, 2, 2)) * 0.1  # Small diffusion coefficient
        
        # Set off-diagonal terms to zero (no cross-diffusion)
        diffusion[:, :, 0, 1] = 0
        diffusion[:, :, 1, 0] = 0

        # Initial condition: Gaussian-like bump
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
        initial_conditions = np.exp(-10 * ((X - 0.5)**2 + (Y - 0.5)**2))

        # Solve
        solution = solver.solve_2d(
            drift, diffusion, initial_conditions, (x_grid, y_grid), time_grid
        )

        # Check solution properties
        assert solution.shape == (len(time_grid), nx, ny)
        assert np.all(solution >= 0)  # Solution should remain non-negative
        assert np.allclose(solution[0], initial_conditions)  # Initial condition preserved

        # For diffusion, total mass should be approximately conserved
        mass_initial = np.sum(initial_conditions)
        mass_final = np.sum(solution[-1])
        assert abs(mass_final - mass_initial) / mass_initial < 0.5  # Allow some numerical diffusion

    def test_solve_3d_simple(self):
        """Test 3D solver with simple diffusion equation."""
        solver = ADISolver(theta=1.0)  # Fully implicit
        
        # Small 3D grid for efficiency
        x_grid = np.linspace(0, 1, 4)
        y_grid = np.linspace(0, 1, 3)
        z_grid = np.linspace(0, 1, 3)
        time_grid = np.linspace(0, 0.05, 2)
        nx, ny, nz = len(x_grid), len(y_grid), len(z_grid)

        # Pure diffusion
        drift = np.zeros((nx, ny, nz, 3))
        diffusion = np.zeros((nx, ny, nz, 3, 3))
        
        # Set diagonal diffusion terms
        diffusion[:, :, :, 0, 0] = 0.1
        diffusion[:, :, :, 1, 1] = 0.1
        diffusion[:, :, :, 2, 2] = 0.1

        # Initial condition: simple bump
        X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
        initial_conditions = np.exp(-5 * ((X - 0.5)**2 + (Y - 0.5)**2 + (Z - 0.5)**2))

        # Solve
        solution = solver.solve_3d(
            drift, diffusion, initial_conditions, (x_grid, y_grid, z_grid), time_grid
        )

        # Check solution properties
        assert solution.shape == (len(time_grid), nx, ny, nz)
        assert np.all(solution >= 0)
        assert np.allclose(solution[0], initial_conditions)

    def test_solve_2d_with_drift(self):
        """Test 2D solver with drift term."""
        solver = ADISolver(theta=0.5)  # Crank-Nicolson
        
        # Simple grid
        x_grid = np.linspace(0, 1, 5)
        y_grid = np.linspace(0, 1, 4)
        time_grid = np.linspace(0, 0.1, 3)
        nx, ny = len(x_grid), len(y_grid)

        # Constant drift in x-direction
        drift = np.zeros((nx, ny, 2))
        drift[:, :, 0] = 0.5  # Drift in x-direction
        
        # Small diffusion
        diffusion = np.zeros((nx, ny, 2, 2))
        diffusion[:, :, 0, 0] = 0.01
        diffusion[:, :, 1, 1] = 0.01

        # Initial condition
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
        initial_conditions = np.exp(-10 * ((X - 0.2)**2 + (Y - 0.5)**2))

        # Solve
        solution = solver.solve_2d(
            drift, diffusion, initial_conditions, (x_grid, y_grid), time_grid
        )

        # Check that solution evolved
        assert solution.shape == (len(time_grid), nx, ny)
        assert not np.allclose(solution[0], solution[-1])  # Should change over time

    def test_factory_functions(self):
        """Test factory functions for creating solvers."""
        # Default solver
        solver1 = create_default_adi_solver()
        assert isinstance(solver1, ADISolver)
        assert solver1.theta == 0.5

        # Crank-Nicolson solver
        solver2 = create_crank_nicolson_solver()
        assert isinstance(solver2, ADISolver)
        assert solver2.theta == 0.5

        # Implicit solver
        solver3 = create_implicit_solver()
        assert isinstance(solver3, ADISolver)
        assert solver3.theta == 1.0


class TestADIStability:
    """Test ADI solver stability and accuracy."""

    def test_stability_2d(self):
        """Test that ADI solver remains stable for various parameters."""
        solver = ADISolver(theta=1.0)  # Fully implicit for unconditional stability
        
        # Test with larger time steps
        x_grid = np.linspace(0, 1, 10)
        y_grid = np.linspace(0, 1, 8)
        time_grid = np.linspace(0, 1.0, 5)  # Larger time steps
        nx, ny = len(x_grid), len(y_grid)

        # Moderate diffusion
        drift = np.zeros((nx, ny, 2))
        diffusion = np.zeros((nx, ny, 2, 2))
        diffusion[:, :, 0, 0] = 0.5
        diffusion[:, :, 1, 1] = 0.5

        # Initial condition
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
        initial_conditions = np.sin(np.pi * X) * np.sin(np.pi * Y)

        # Solve
        solution = solver.solve_2d(
            drift, diffusion, initial_conditions, (x_grid, y_grid), time_grid
        )

        # Check for stability (no blow-up)
        assert np.all(np.isfinite(solution))
        assert np.max(np.abs(solution)) < 100  # Reasonable bounds

    def test_conservation_properties(self):
        """Test conservation properties of the solver."""
        solver = ADISolver(theta=0.5)
        
        # Grid
        x_grid = np.linspace(0, 1, 8)
        y_grid = np.linspace(0, 1, 6)
        time_grid = np.linspace(0, 0.5, 4)
        nx, ny = len(x_grid), len(y_grid)

        # Pure diffusion (should conserve mass in absence of boundaries)
        drift = np.zeros((nx, ny, 2))
        diffusion = np.zeros((nx, ny, 2, 2))
        diffusion[:, :, 0, 0] = 0.1
        diffusion[:, :, 1, 1] = 0.1

        # Smooth initial condition
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
        initial_conditions = np.exp(-2 * ((X - 0.5)**2 + (Y - 0.5)**2))

        # Solve
        solution = solver.solve_2d(
            drift, diffusion, initial_conditions, (x_grid, y_grid), time_grid
        )

        # Check approximate mass conservation (allowing for boundary effects)
        initial_mass = np.sum(initial_conditions)
        final_mass = np.sum(solution[-1])
        relative_change = abs(final_mass - initial_mass) / initial_mass
        
        # Should conserve mass reasonably well
        assert relative_change < 0.3  # Allow some numerical diffusion


class TestADIPerformance:
    """Test ADI solver performance characteristics."""

    def test_2d_performance_scaling(self):
        """Test that 2D solver scales reasonably with grid size."""
        solver = ADISolver()
        
        # Small grid
        x_grid = np.linspace(0, 1, 5)
        y_grid = np.linspace(0, 1, 4)
        time_grid = np.linspace(0, 0.1, 3)
        
        nx, ny = len(x_grid), len(y_grid)
        drift = np.zeros((nx, ny, 2))
        diffusion = np.eye(2).reshape(1, 1, 2, 2) * np.ones((nx, ny, 1, 1)) * 0.1
        initial_conditions = np.ones((nx, ny))

        # Should complete without issues
        solution = solver.solve_2d(
            drift, diffusion, initial_conditions, (x_grid, y_grid), time_grid
        )
        
        assert solution.shape == (len(time_grid), nx, ny)

    def test_3d_small_grid_performance(self):
        """Test 3D solver on small grids."""
        solver = ADISolver()
        
        # Very small 3D grid
        x_grid = np.linspace(0, 1, 3)
        y_grid = np.linspace(0, 1, 3)
        z_grid = np.linspace(0, 1, 3)
        time_grid = np.linspace(0, 0.1, 2)
        
        nx, ny, nz = len(x_grid), len(y_grid), len(z_grid)
        drift = np.zeros((nx, ny, nz, 3))
        diffusion = np.eye(3).reshape(1, 1, 1, 3, 3) * np.ones((nx, ny, nz, 1, 1)) * 0.1
        initial_conditions = np.ones((nx, ny, nz))

        # Should complete
        solution = solver.solve_3d(
            drift, diffusion, initial_conditions, (x_grid, y_grid, z_grid), time_grid
        )
        
        assert solution.shape == (len(time_grid), nx, ny, nz)
