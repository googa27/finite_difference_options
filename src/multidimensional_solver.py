"""Multi-dimensional PDE solver using Alternating Direction Implicit (ADI) method.

This module implements efficient solvers for 2D and 3D parabolic PDEs arising
in multi-dimensional option pricing models like Heston stochastic volatility.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import numpy as np
from numpy.typing import NDArray

from .exceptions import ValidationError


class MultiDimensionalPDESolver(ABC):
    """Abstract base class for multi-dimensional PDE solvers."""
    
    @abstractmethod
    def solve_2d(
        self,
        drift: NDArray[np.float64],
        diffusion: NDArray[np.float64],
        initial_conditions: NDArray[np.float64],
        grids: Tuple[NDArray[np.float64], NDArray[np.float64]],
        time_grid: NDArray[np.float64],
        boundary_conditions: Optional[Dict[str, Any]] = None
    ) -> NDArray[np.float64]:
        """Solve 2D parabolic PDE.
        
        Parameters
        ----------
        drift : NDArray[np.float64]
            Drift coefficients with shape (nx, ny, 2).
        diffusion : NDArray[np.float64]
            Diffusion matrix with shape (nx, ny, 2, 2).
        initial_conditions : NDArray[np.float64]
            Initial values with shape (nx, ny).
        grids : Tuple[NDArray[np.float64], NDArray[np.float64]]
            Spatial grids (x_grid, y_grid).
        time_grid : NDArray[np.float64]
            Time grid points.
        boundary_conditions : Dict[str, Any], optional
            Boundary condition specifications.
            
        Returns
        -------
        NDArray[np.float64]
            Solution with shape (nt, nx, ny).
        """
        ...
    
    @abstractmethod
    def solve_3d(
        self,
        drift: NDArray[np.float64],
        diffusion: NDArray[np.float64],
        initial_conditions: NDArray[np.float64],
        grids: Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        time_grid: NDArray[np.float64],
        boundary_conditions: Optional[Dict[str, Any]] = None
    ) -> NDArray[np.float64]:
        """Solve 3D parabolic PDE.
        
        Parameters
        ----------
        drift : NDArray[np.float64]
            Drift coefficients with shape (nx, ny, nz, 3).
        diffusion : NDArray[np.float64]
            Diffusion matrix with shape (nx, ny, nz, 3, 3).
        initial_conditions : NDArray[np.float64]
            Initial values with shape (nx, ny, nz).
        grids : Tuple[NDArray[np.float64], ...]
            Spatial grids (x_grid, y_grid, z_grid).
        time_grid : NDArray[np.float64]
            Time grid points.
        boundary_conditions : Dict[str, Any], optional
            Boundary condition specifications.
            
        Returns
        -------
        NDArray[np.float64]
            Solution with shape (nt, nx, ny, nz).
        """
        ...


@dataclass
class ADISolver(MultiDimensionalPDESolver):
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
        drift: NDArray[np.float64],
        diffusion: NDArray[np.float64],
        initial_conditions: NDArray[np.float64],
        grids: Tuple[NDArray[np.float64], NDArray[np.float64]],
        time_grid: NDArray[np.float64],
        boundary_conditions: Optional[Dict[str, Any]] = None
    ) -> NDArray[np.float64]:
        """Solve 2D parabolic PDE using ADI method.
        
        The 2D PDE is: ∂u/∂t = L_x u + L_y u + mixed terms
        ADI splits this into: (I - θΔt L_x)(I - θΔt L_y)u^{n+1} = RHS
        """
        x_grid, y_grid = grids
        nx, ny = len(x_grid), len(y_grid)
        nt = len(time_grid)
        
        # Validate input shapes
        self._validate_2d_inputs(drift, diffusion, initial_conditions, nx, ny)
        
        # Initialize solution array
        solution = np.zeros((nt, nx, ny))
        solution[0] = initial_conditions.copy()
        
        # Compute grid spacings
        dx = x_grid[1] - x_grid[0] if nx > 1 else 1.0
        dy = y_grid[1] - y_grid[0] if ny > 1 else 1.0
        
        # Time stepping
        for n in range(nt - 1):
            dt = time_grid[n + 1] - time_grid[n]
            
            # ADI step: solve in x-direction, then y-direction
            intermediate = self._adi_step_x(
                solution[n], drift, diffusion, dx, dy, dt, nx, ny
            )
            
            solution[n + 1] = self._adi_step_y(
                intermediate, drift, diffusion, dx, dy, dt, nx, ny
            )
            
            # Apply boundary conditions if provided
            if boundary_conditions is not None:
                solution[n + 1] = self._apply_boundary_conditions_2d(
                    solution[n + 1], grids, boundary_conditions
                )
        
        return solution

    def solve_3d(
        self,
        drift: NDArray[np.float64],
        diffusion: NDArray[np.float64],
        initial_conditions: NDArray[np.float64],
        grids: Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        time_grid: NDArray[np.float64],
        boundary_conditions: Optional[Dict[str, Any]] = None
    ) -> NDArray[np.float64]:
        """Solve 3D parabolic PDE using ADI method.
        
        The 3D PDE is split into three 1D problems solved sequentially.
        """
        x_grid, y_grid, z_grid = grids
        nx, ny, nz = len(x_grid), len(y_grid), len(z_grid)
        nt = len(time_grid)
        
        # Validate input shapes
        self._validate_3d_inputs(drift, diffusion, initial_conditions, nx, ny, nz)
        
        # Initialize solution array
        solution = np.zeros((nt, nx, ny, nz))
        solution[0] = initial_conditions.copy()
        
        # Compute grid spacings
        dx = x_grid[1] - x_grid[0] if nx > 1 else 1.0
        dy = y_grid[1] - y_grid[0] if ny > 1 else 1.0
        dz = z_grid[1] - z_grid[0] if nz > 1 else 1.0
        
        # Time stepping
        for n in range(nt - 1):
            dt = time_grid[n + 1] - time_grid[n]
            
            # ADI step: solve in x, y, then z directions
            temp1 = self._adi_step_x_3d(
                solution[n], drift, diffusion, dx, dy, dz, dt, nx, ny, nz
            )
            
            temp2 = self._adi_step_y_3d(
                temp1, drift, diffusion, dx, dy, dz, dt, nx, ny, nz
            )
            
            solution[n + 1] = self._adi_step_z_3d(
                temp2, drift, diffusion, dx, dy, dz, dt, nx, ny, nz
            )
            
            # Apply boundary conditions if provided
            if boundary_conditions is not None:
                solution[n + 1] = self._apply_boundary_conditions_3d(
                    solution[n + 1], grids, boundary_conditions
                )
        
        return solution

    def _validate_2d_inputs(
        self,
        drift: NDArray[np.float64],
        diffusion: NDArray[np.float64],
        initial_conditions: NDArray[np.float64],
        nx: int,
        ny: int
    ) -> None:
        """Validate 2D input arrays."""
        if drift.shape != (nx, ny, 2):
            raise ValidationError(f"drift must have shape ({nx}, {ny}, 2), got {drift.shape}")
        
        if diffusion.shape != (nx, ny, 2, 2):
            raise ValidationError(f"diffusion must have shape ({nx}, {ny}, 2, 2), got {diffusion.shape}")
        
        if initial_conditions.shape != (nx, ny):
            raise ValidationError(f"initial_conditions must have shape ({nx}, {ny}), got {initial_conditions.shape}")

    def _validate_3d_inputs(
        self,
        drift: NDArray[np.float64],
        diffusion: NDArray[np.float64],
        initial_conditions: NDArray[np.float64],
        nx: int,
        ny: int,
        nz: int
    ) -> None:
        """Validate 3D input arrays."""
        if drift.shape != (nx, ny, nz, 3):
            raise ValidationError(f"drift must have shape ({nx}, {ny}, {nz}, 3), got {drift.shape}")
        
        if diffusion.shape != (nx, ny, nz, 3, 3):
            raise ValidationError(f"diffusion must have shape ({nx}, {ny}, {nz}, 3, 3), got {diffusion.shape}")
        
        if initial_conditions.shape != (nx, ny, nz):
            raise ValidationError(f"initial_conditions must have shape ({nx}, {ny}, {nz}), got {initial_conditions.shape}")

    def _adi_step_x(
        self,
        u: NDArray[np.float64],
        drift: NDArray[np.float64],
        diffusion: NDArray[np.float64],
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
            sigma_xx = diffusion[:, j, 0, 0]  # xx-diffusion
            
            # Build tridiagonal matrix for x-direction
            # Second-order finite differences: a*u_{i-1} + b*u_i + c*u_{i+1}
            a = np.zeros(nx)
            b = np.ones(nx)
            c = np.zeros(nx)
            
            for i in range(1, nx - 1):
                # Coefficients for second derivative term
                coeff_2nd = self.theta * dt * sigma_xx[i] / (dx * dx)
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
        diffusion: NDArray[np.float64],
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
            sigma_yy = diffusion[i, :, 1, 1]  # yy-diffusion
            
            # Build tridiagonal matrix for y-direction
            a = np.zeros(ny)
            b = np.ones(ny)
            c = np.zeros(ny)
            
            for j in range(1, ny - 1):
                # Coefficients for second derivative term
                coeff_2nd = self.theta * dt * sigma_yy[j] / (dy * dy)
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
        diffusion: NDArray[np.float64],
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
                sigma_xx = diffusion[:, j, k, 0, 0]
                
                # Build tridiagonal matrix
                a = np.zeros(nx)
                b = np.ones(nx)
                c = np.zeros(nx)
                
                for i in range(1, nx - 1):
                    coeff_2nd = self.theta * dt * sigma_xx[i] / (3 * dx * dx)
                    coeff_1st = self.theta * dt * mu_x[i] / (6 * dx)
                    
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
        diffusion: NDArray[np.float64],
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
                sigma_yy = diffusion[i, :, k, 1, 1]
                
                # Build tridiagonal matrix
                a = np.zeros(ny)
                b = np.ones(ny)
                c = np.zeros(ny)
                
                for j in range(1, ny - 1):
                    coeff_2nd = self.theta * dt * sigma_yy[j] / (3 * dy * dy)
                    coeff_1st = self.theta * dt * mu_y[j] / (6 * dy)
                    
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
        diffusion: NDArray[np.float64],
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
                sigma_zz = diffusion[i, j, :, 2, 2]
                
                # Build tridiagonal matrix
                a = np.zeros(nz)
                b = np.ones(nz)
                c = np.zeros(nz)
                
                for k in range(1, nz - 1):
                    coeff_2nd = self.theta * dt * sigma_zz[k] / (3 * dz * dz)
                    coeff_1st = self.theta * dt * mu_z[k] / (6 * dz)
                    
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
        """Apply boundary conditions for 2D problem."""
        # Simple implementation - just return unchanged for now
        # In a full implementation, this would apply specific boundary conditions
        return u

    def _apply_boundary_conditions_3d(
        self,
        u: NDArray[np.float64],
        grids: Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        boundary_conditions: Dict[str, Any]
    ) -> NDArray[np.float64]:
        """Apply boundary conditions for 3D problem."""
        # Simple implementation - just return unchanged for now
        return u


def create_default_adi_solver() -> ADISolver:
    """Create ADI solver with default parameters."""
    return ADISolver()


def create_crank_nicolson_solver() -> ADISolver:
    """Create ADI solver with Crank-Nicolson scheme (theta=0.5)."""
    return ADISolver(theta=0.5)


def create_implicit_solver() -> ADISolver:
    """Create fully implicit ADI solver (theta=1.0)."""
    return ADISolver(theta=1.0)


def create_adi_solver(theta: float = 0.5) -> ADISolver:
    """Create ADI solver with specified theta parameter."""
    return ADISolver(theta=theta)
