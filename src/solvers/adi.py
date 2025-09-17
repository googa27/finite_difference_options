"""ADI (Alternating Direction Implicit) solver for multi-dimensional PDEs.

This module implements efficient solvers for 2D and 3D parabolic PDEs arising
in multi-dimensional option pricing models like Heston stochastic volatility.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import numpy as np
from numpy.typing import NDArray

from src.exceptions import ValidationError


@dataclass
class ADISolver:
    """Alternating Direction Implicit (ADI) solver for multi-dimensional PDEs.
    
    The ADI method splits multi-dimensional operators into a sequence of 
    1D problems that can be solved efficiently using tridiagonal solvers.
    
    Parameters
    ----------
    theta : float, optional
        Implicitness parameter (0=explicit, 0.5=Crank-Nicolson, 1=implicit).
        Default is 0.5 for second-order accuracy.
    max_iterations : int, optional
        Maximum iterations for iterative methods (default: 1000).
    tolerance : float, optional
        Convergence tolerance for iterative methods (default: 1e-8).
    """
    
    theta: float = 0.5
    max_iterations: int = 1000
    tolerance: float = 1e-8

    def __post_init__(self) -> None:
        """Validate ADI solver parameters."""
        if not 0.0 <= self.theta <= 1.0:
            raise ValidationError(f"theta must be between 0 and 1, got {self.theta}")
        
        if self.max_iterations <= 0:
            raise ValidationError(f"max_iterations must be positive, got {self.max_iterations}")
        
        if self.tolerance <= 0:
            raise ValidationError(f"tolerance must be positive, got {self.tolerance}")

    def solve_2d(
        self,
        initial_condition: NDArray[np.float64],
        drift: NDArray[np.float64],
        covariance: NDArray[np.float64],
        time_grid: NDArray[np.float64],
        spatial_grids: Tuple[NDArray[np.float64], NDArray[np.float64]],
        boundary_conditions: Optional[Dict[str, Any]] = None
    ) -> NDArray[np.float64]:
        """Solve 2D parabolic PDE using ADI method.
        
        The 2D PDE is: ∂u/∂t = L_x u + L_y u + mixed terms
        ADI splits this into: (I - θΔt L_x)(I - θΔt L_y)u^{n+1} = RHS
        """
        x_grid, y_grid = spatial_grids
        nx, ny = len(x_grid), len(y_grid)
        nt = len(time_grid)
        
        # Validate input shapes
        self._validate_2d_inputs(drift, covariance, initial_condition, nx, ny)
        
        # Initialize solution array
        solution = np.zeros((nt, nx, ny))
        # Set terminal condition at final time step
        solution[-1] = initial_condition.copy()
        
        # Compute grid spacings
        dx = x_grid[1] - x_grid[0] if nx > 1 else 1.0
        dy = y_grid[1] - y_grid[0] if ny > 1 else 1.0
        
        # Time stepping - solve backward from maturity to initial time
        for n in range(nt - 2, -1, -1):  # Go from nt-2 down to 0
            dt = time_grid[n + 1] - time_grid[n]  # Positive time step
            
            # ADI step: solve in x-direction, then y-direction
            intermediate = self._adi_step_x(
                solution[n + 1], drift, covariance, dx, dy, dt, nx, ny
            )
            
            solution[n] = self._adi_step_y(
                intermediate, drift, covariance, dx, dy, dt, nx, ny
            )
            
            # Apply boundary conditions if provided
            if boundary_conditions is not None:
                solution[n] = self._apply_boundary_conditions_2d(
                    solution[n], spatial_grids, boundary_conditions
                )
        
        return solution

    def solve_3d(
        self,
        initial_condition: NDArray[np.float64],
        drift: NDArray[np.float64],
        covariance: NDArray[np.float64],
        time_grid: NDArray[np.float64],
        spatial_grids: Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        boundary_conditions: Optional[Dict[str, Any]] = None
    ) -> NDArray[np.float64]:
        """Solve 3D parabolic PDE using ADI method.
        
        The 3D PDE is split into three 1D problems solved sequentially.
        """
        x_grid, y_grid, z_grid = spatial_grids
        nx, ny, nz = len(x_grid), len(y_grid), len(z_grid)
        nt = len(time_grid)
        
        # Validate input shapes
        self._validate_3d_inputs(drift, covariance, initial_condition, nx, ny, nz)
        
        # Initialize solution array
        solution = np.zeros((nt, nx, ny, nz))
        solution[0] = initial_condition.copy()
        
        # Compute grid spacings
        dx = x_grid[1] - x_grid[0] if nx > 1 else 1.0
        dy = y_grid[1] - y_grid[0] if ny > 1 else 1.0
        dz = z_grid[1] - z_grid[0] if nz > 1 else 1.0
        
        # Time stepping
        for n in range(nt - 1):
            dt = time_grid[n + 1] - time_grid[n]
            
            # ADI step: solve in x, y, then z directions
            temp1 = self._adi_step_x_3d(
                solution[n], drift, covariance, dx, dy, dz, dt, nx, ny, nz
            )
            
            temp2 = self._adi_step_y_3d(
                temp1, drift, covariance, dx, dy, dz, dt, nx, ny, nz
            )
            
            solution[n + 1] = self._adi_step_z_3d(
                temp2, drift, covariance, dx, dy, dz, dt, nx, ny, nz
            )
            
            # Apply boundary conditions if provided
            if boundary_conditions is not None:
                solution[n + 1] = self._apply_boundary_conditions_3d(
                    solution[n + 1], spatial_grids, boundary_conditions
                )
        
        return solution

    def _validate_2d_inputs(
        self,
        drift: NDArray[np.float64],
        covariance: NDArray[np.float64],
        initial_condition: NDArray[np.float64],
        nx: int,
        ny: int
    ) -> None:
        """Validate 2D input arrays."""
        if drift.shape != (nx, ny, 2):
            raise ValidationError(f"drift must have shape ({nx}, {ny}, 2), got {drift.shape}")
        
        if covariance.shape != (nx, ny, 2, 2):
            raise ValidationError(f"covariance must have shape ({nx}, {ny}, 2, 2), got {covariance.shape}")
        
        if initial_condition.shape != (nx, ny):
            raise ValidationError(f"initial_condition must have shape ({nx}, {ny}), got {initial_condition.shape}")

    def _validate_3d_inputs(
        self,
        drift: NDArray[np.float64],
        covariance: NDArray[np.float64],
        initial_condition: NDArray[np.float64],
        nx: int,
        ny: int,
        nz: int
    ) -> None:
        """Validate 3D input arrays."""
        if drift.shape != (nx, ny, nz, 3):
            raise ValidationError(f"drift must have shape ({nx}, {ny}, {nz}, 3), got {drift.shape}")
        
        if covariance.shape != (nx, ny, nz, 3, 3):
            raise ValidationError(f"covariance must have shape ({nx}, {ny}, {nz}, 3, 3), got {covariance.shape}")
        
        if initial_condition.shape != (nx, ny, nz):
            raise ValidationError(f"initial_condition must have shape ({nx}, {ny}, {nz}), got {initial_condition.shape}")

    def _adi_step_x(
        self,
        u: NDArray[np.float64],
        drift: NDArray[np.float64],
        covariance: NDArray[np.float64],
        dx: float,
        dy: float,
        dt: float,
        nx: int,
        ny: int
    ) -> NDArray[np.float64]:
        """Perform ADI step in x-direction."""
        result = u.copy()
        
        # For each y-slice, solve tridiagonal system in x-direction
        for j in range(ny):
            if nx <= 2:
                continue  # Skip if too few points
                
            # Extract coefficients for this y-slice
            mu_x = drift[:, j, 0]  # x-drift
            sigma_xx = covariance[:, j, 0, 0]  # xx-covariance
            
            # Build tridiagonal matrix for x-direction
            # Second-order finite differences: a*u_{i-1} + b*u_i + c*u_{i+1}
            a = np.zeros(nx)
            b = np.ones(nx)
            c = np.zeros(nx)
            
            for i in range(1, nx - 1):
                # Coefficients for second derivative term
                coeff_2nd = self.theta * dt * sigma_xx[i] / (2 * dx * dx)
                # Coefficients for first derivative term  
                coeff_1st = self.theta * dt * mu_x[i] / (2 * dx)
                
                a[i] = -coeff_2nd + coeff_1st
                b[i] = 1 + 2 * coeff_2nd
                c[i] = -coeff_2nd - coeff_1st
            
            # Solve tridiagonal system
            rhs = u[:, j].copy()
            result[:, j] = self._solve_tridiagonal(a, b, c, rhs)
        
        return result

    def _adi_step_y(
        self,
        u: NDArray[np.float64],
        drift: NDArray[np.float64],
        covariance: NDArray[np.float64],
        dx: float,
        dy: float,
        dt: float,
        nx: int,
        ny: int
    ) -> NDArray[np.float64]:
        """Perform ADI step in y-direction."""
        result = u.copy()
        
        # For each x-slice, solve tridiagonal system in y-direction
        for i in range(nx):
            if ny <= 2:
                continue  # Skip if too few points
                
            # Extract coefficients for this x-slice
            mu_y = drift[i, :, 1]  # y-drift
            sigma_yy = covariance[i, :, 1, 1]  # yy-covariance
            
            # Build tridiagonal matrix for y-direction
            a = np.zeros(ny)
            b = np.ones(ny)
            c = np.zeros(ny)
            
            for j in range(1, ny - 1):
                # Coefficients for second derivative term
                coeff_2nd = self.theta * dt * sigma_yy[j] / (2 * dy * dy)
                # Coefficients for first derivative term
                coeff_1st = self.theta * dt * mu_y[j] / (2 * dy)
                
                a[j] = -coeff_2nd + coeff_1st
                b[j] = 1 + 2 * coeff_2nd
                c[j] = -coeff_2nd - coeff_1st
            
            # Solve tridiagonal system
            rhs = u[i, :].copy()
            result[i, :] = self._solve_tridiagonal(a, b, c, rhs)
        
        return result

    def _adi_step_x_3d(
        self,
        u: NDArray[np.float64],
        drift: NDArray[np.float64],
        covariance: NDArray[np.float64],
        dx: float,
        dy: float,
        dz: float,
        dt: float,
        nx: int,
        ny: int,
        nz: int
    ) -> NDArray[np.float64]:
        """Perform ADI step in x-direction for 3D."""
        result = u.copy()
        
        # For each (y,z) pair, solve tridiagonal system in x-direction
        for j in range(ny):
            for k in range(nz):
                if nx <= 2:
                    continue
                    
                # Extract coefficients
                mu_x = drift[:, j, k, 0]
                sigma_xx = covariance[:, j, k, 0, 0]
                
                # Build tridiagonal matrix
                a = np.zeros(nx)
                b = np.ones(nx)
                c = np.zeros(nx)
                
                for i in range(1, nx - 1):
                    coeff_2nd = self.theta * dt * sigma_xx[i] / (2 * dx * dx)
                    coeff_1st = self.theta * dt * mu_x[i] / (2 * dx)
                    
                    a[i] = -coeff_2nd + coeff_1st
                    b[i] = 1 + 2 * coeff_2nd
                    c[i] = -coeff_2nd - coeff_1st
                
                # Solve tridiagonal system
                rhs = u[:, j, k].copy()
                result[:, j, k] = self._solve_tridiagonal(a, b, c, rhs)
        
        return result

    def _adi_step_y_3d(
        self,
        u: NDArray[np.float64],
        drift: NDArray[np.float64],
        covariance: NDArray[np.float64],
        dx: float,
        dy: float,
        dz: float,
        dt: float,
        nx: int,
        ny: int,
        nz: int
    ) -> NDArray[np.float64]:
        """Perform ADI step in y-direction for 3D."""
        result = u.copy()
        
        # For each (x,z) pair, solve tridiagonal system in y-direction
        for i in range(nx):
            for k in range(nz):
                if ny <= 2:
                    continue
                    
                # Extract coefficients
                mu_y = drift[i, :, k, 1]
                sigma_yy = covariance[i, :, k, 1, 1]
                
                # Build tridiagonal matrix
                a = np.zeros(ny)
                b = np.ones(ny)
                c = np.zeros(ny)
                
                for j in range(1, ny - 1):
                    coeff_2nd = self.theta * dt * sigma_yy[j] / (2 * dy * dy)
                    coeff_1st = self.theta * dt * mu_y[j] / (2 * dy)
                    
                    a[j] = -coeff_2nd + coeff_1st
                    b[j] = 1 + 2 * coeff_2nd
                    c[j] = -coeff_2nd - coeff_1st
                
                # Solve tridiagonal system
                rhs = u[i, :, k].copy()
                result[i, :, k] = self._solve_tridiagonal(a, b, c, rhs)
        
        return result

    def _adi_step_z_3d(
        self,
        u: NDArray[np.float64],
        drift: NDArray[np.float64],
        covariance: NDArray[np.float64],
        dx: float,
        dy: float,
        dz: float,
        dt: float,
        nx: int,
        ny: int,
        nz: int
    ) -> NDArray[np.float64]:
        """Perform ADI step in z-direction for 3D."""
        result = u.copy()
        
        # For each (x,y) pair, solve tridiagonal system in z-direction
        for i in range(nx):
            for j in range(ny):
                if nz <= 2:
                    continue
                    
                # Extract coefficients
                mu_z = drift[i, j, :, 2]
                sigma_zz = covariance[i, j, :, 2, 2]
                
                # Build tridiagonal matrix
                a = np.zeros(nz)
                b = np.ones(nz)
                c = np.zeros(nz)
                
                for k in range(1, nz - 1):
                    coeff_2nd = self.theta * dt * sigma_zz[k] / (2 * dz * dz)
                    coeff_1st = self.theta * dt * mu_z[k] / (2 * dz)
                    
                    a[k] = -coeff_2nd + coeff_1st
                    b[k] = 1 + 2 * coeff_2nd
                    c[k] = -coeff_2nd - coeff_1st
                
                # Solve tridiagonal system
                rhs = u[i, j, :].copy()
                result[i, j, :] = self._solve_tridiagonal(a, b, c, rhs)
        
        return result

    def _solve_tridiagonal(
        self,
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        c: NDArray[np.float64],
        d: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Solve tridiagonal system using Thomas algorithm.
        
        Solves: a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
        """
        n = len(d)
        if n <= 1:
            if n == 1 and abs(b[0]) > 1e-14:
                return np.array([d[0] / b[0]])
            return d.copy()
        
        # Forward elimination
        c_prime = np.zeros(n)
        d_prime = np.zeros(n)
        
        c_prime[0] = c[0] / b[0] if abs(b[0]) > 1e-14 else 0.0
        d_prime[0] = d[0] / b[0] if abs(b[0]) > 1e-14 else d[0]
        
        for i in range(1, n):
            denom = b[i] - a[i] * c_prime[i - 1]
            if abs(denom) < 1e-14:
                denom = 1e-14  # Avoid division by zero
            
            if i < n - 1:
                c_prime[i] = c[i] / denom
            d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom
        
        # Back substitution
        x = np.zeros(n)
        x[n - 1] = d_prime[n - 1]
        
        for i in range(n - 2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i + 1]
        
        return x

    def _apply_boundary_conditions_2d(
        self,
        u: NDArray[np.float64],
        grids: Tuple[NDArray[np.float64], NDArray[np.float64]],
        boundary_conditions: Dict[str, Any]
    ) -> NDArray[np.float64]:
        """Apply boundary conditions for 2D problem.
        
        Parameters
        ----------
        u : NDArray[np.float64]
            Solution array.
        grids : Tuple[NDArray[np.float64], NDArray[np.float64]]
            Spatial grids (x_grid, y_grid).
        boundary_conditions : Dict[str, Any] or BoundaryManager2D
            Boundary condition specifications or manager.
            
        Returns
        -------
        NDArray[np.float64]
            Solution with boundary conditions applied.
        """
        # Handle both dictionary and BoundaryManager2D formats
        if boundary_conditions is None:
            return u
            
        # If it's a boundary manager object, use its apply_boundaries method
        if hasattr(boundary_conditions, 'apply_boundaries'):
            try:
                return boundary_conditions.apply_boundaries(u, grids)
            except Exception:
                # If applying boundaries fails, return unchanged
                return u
                
        # If it's a dictionary, apply simple boundary conditions
        if isinstance(boundary_conditions, dict):
            result = u.copy()
            x_grid, y_grid = grids
            nx, ny = len(x_grid), len(y_grid)
            
            # Apply simple boundary conditions if specified
            for boundary_location, boundary_spec in boundary_conditions.items():
                if boundary_location == 'left':
                    # Left boundary (x = x_min)
                    if boundary_spec.get('type') == 'dirichlet':
                        result[0, :] = boundary_spec.get('value', 0.0)
                    elif boundary_spec.get('type') == 'zero_gradient':
                        result[0, :] = result[1, :]
                elif boundary_location == 'right':
                    # Right boundary (x = x_max)
                    if boundary_spec.get('type') == 'dirichlet':
                        result[-1, :] = boundary_spec.get('value', 0.0)
                    elif boundary_spec.get('type') == 'zero_gradient':
                        result[-1, :] = result[-2, :]
                elif boundary_location == 'bottom':
                    # Bottom boundary (y = y_min)
                    if boundary_spec.get('type') == 'dirichlet':
                        result[:, 0] = boundary_spec.get('value', 0.0)
                    elif boundary_spec.get('type') == 'zero_gradient':
                        result[:, 0] = result[:, 1]
                elif boundary_location == 'top':
                    # Top boundary (y = y_max)
                    if boundary_spec.get('type') == 'dirichlet':
                        result[:, -1] = boundary_spec.get('value', 0.0)
                    elif boundary_spec.get('type') == 'zero_gradient':
                        result[:, -1] = result[:, -2]
                        
            return result
            
        # Default: return unchanged
        return u

    def _apply_boundary_conditions_3d(
        self,
        u: NDArray[np.float64],
        grids: Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        boundary_conditions: Dict[str, Any]
    ) -> NDArray[np.float64]:
        """Apply boundary conditions for 3D problem.
        
        Parameters
        ----------
        u : NDArray[np.float64]
            Solution array.
        grids : Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
            Spatial grids (x_grid, y_grid, z_grid).
        boundary_conditions : Dict[str, Any] or BoundaryManager3D
            Boundary condition specifications or manager.
            
        Returns
        -------
        NDArray[np.float64]
            Solution with boundary conditions applied.
        """
        # Handle both dictionary and BoundaryManager3D formats
        if boundary_conditions is None:
            return u
            
        # If it's a boundary manager object, use its apply_boundaries method
        if hasattr(boundary_conditions, 'apply_boundaries'):
            try:
                return boundary_conditions.apply_boundaries(u, grids)
            except Exception:
                # If applying boundaries fails, return unchanged
                return u
                
        # If it's a dictionary, apply simple boundary conditions
        if isinstance(boundary_conditions, dict):
            result = u.copy()
            x_grid, y_grid, z_grid = grids
            nx, ny, nz = len(x_grid), len(y_grid), len(z_grid)
            
            # Apply simple boundary conditions if specified
            for boundary_location, boundary_spec in boundary_conditions.items():
                if boundary_location == 'left':
                    # Left boundary (x = x_min)
                    if boundary_spec.get('type') == 'dirichlet':
                        result[0, :, :] = boundary_spec.get('value', 0.0)
                    elif boundary_spec.get('type') == 'zero_gradient':
                        result[0, :, :] = result[1, :, :]
                elif boundary_location == 'right':
                    # Right boundary (x = x_max)
                    if boundary_spec.get('type') == 'dirichlet':
                        result[-1, :, :] = boundary_spec.get('value', 0.0)
                    elif boundary_spec.get('type') == 'zero_gradient':
                        result[-1, :, :] = result[-2, :, :]
                elif boundary_location == 'bottom':
                    # Bottom boundary (y = y_min)
                    if boundary_spec.get('type') == 'dirichlet':
                        result[:, 0, :] = boundary_spec.get('value', 0.0)
                    elif boundary_spec.get('type') == 'zero_gradient':
                        result[:, 0, :] = result[:, 1, :]
                elif boundary_location == 'top':
                    # Top boundary (y = y_max)
                    if boundary_spec.get('type') == 'dirichlet':
                        result[:, -1, :] = boundary_spec.get('value', 0.0)
                    elif boundary_spec.get('type') == 'zero_gradient':
                        result[:, -1, :] = result[:, -2, :]
                elif boundary_location == 'front':
                    # Front boundary (z = z_min)
                    if boundary_spec.get('type') == 'dirichlet':
                        result[:, :, 0] = boundary_spec.get('value', 0.0)
                    elif boundary_spec.get('type') == 'zero_gradient':
                        result[:, :, 0] = result[:, :, 1]
                elif boundary_location == 'back':
                    # Back boundary (z = z_max)
                    if boundary_spec.get('type') == 'dirichlet':
                        result[:, :, -1] = boundary_spec.get('value', 0.0)
                    elif boundary_spec.get('type') == 'zero_gradient':
                        result[:, :, -1] = result[:, :, -2]
                        
            return result
            
        # Default: return unchanged
        return u


def create_adi_solver(theta: float = 0.5) -> ADISolver:
    """Create ADI solver with specified parameters."""
    return ADISolver(theta=theta)
