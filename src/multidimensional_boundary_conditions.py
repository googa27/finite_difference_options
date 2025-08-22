"""Multi-dimensional boundary conditions for PDE solvers.

This module provides boundary condition management for 2D and 3D PDEs
arising in multi-dimensional option pricing models.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional, Dict, Any, Callable

import numpy as np
from numpy.typing import NDArray

from .exceptions import ValidationError


class BoundaryType(Enum):
    """Types of boundary conditions."""
    DIRICHLET = "dirichlet"      # u = value
    NEUMANN = "neumann"          # ∂u/∂n = value  
    ROBIN = "robin"              # αu + β∂u/∂n = value
    ZERO_GRADIENT = "zero_grad"  # ∂u/∂n = 0


class BoundaryLocation(Enum):
    """Boundary locations for multi-dimensional domains."""
    # 2D boundaries
    LEFT = "left"       # x = x_min
    RIGHT = "right"     # x = x_max
    BOTTOM = "bottom"   # y = y_min
    TOP = "top"         # y = y_max
    
    # 3D boundaries (additional)
    FRONT = "front"     # z = z_min
    BACK = "back"       # z = z_max


@dataclass
class BoundaryCondition:
    """Single boundary condition specification.
    
    Parameters
    ----------
    boundary_type : BoundaryType
        Type of boundary condition.
    value : float or callable
        Boundary value or function.
    alpha : float, optional
        Robin condition coefficient for u term (default: 1.0).
    beta : float, optional
        Robin condition coefficient for ∂u/∂n term (default: 0.0).
    """
    
    boundary_type: BoundaryType
    value: float | Callable[..., float] = 0.0
    alpha: float = 1.0
    beta: float = 0.0
    
    def __post_init__(self) -> None:
        """Validate boundary condition parameters."""
        if self.boundary_type == BoundaryType.ROBIN:
            if abs(self.alpha) < 1e-14 and abs(self.beta) < 1e-14:
                raise ValidationError("Robin condition requires at least one non-zero coefficient")


class MultiDimensionalBoundaryManager(ABC):
    """Abstract base class for managing multi-dimensional boundary conditions."""
    
    @abstractmethod
    def apply_boundaries(
        self,
        u: NDArray[np.float64],
        grids: Tuple[NDArray[np.float64], ...],
        time: float = 0.0
    ) -> NDArray[np.float64]:
        """Apply boundary conditions to solution array.
        
        Parameters
        ----------
        u : NDArray[np.float64]
            Solution array.
        grids : Tuple[NDArray[np.float64], ...]
            Spatial grids.
        time : float, optional
            Current time (default: 0.0).
            
        Returns
        -------
        NDArray[np.float64]
            Solution with boundary conditions applied.
        """
        ...


@dataclass
class BoundaryManager2D(MultiDimensionalBoundaryManager):
    """Boundary condition manager for 2D problems.
    
    Parameters
    ----------
    boundaries : Dict[BoundaryLocation, BoundaryCondition]
        Boundary conditions for each location.
    """
    
    boundaries: Dict[BoundaryLocation, BoundaryCondition]
    
    def __post_init__(self) -> None:
        """Validate 2D boundary specifications."""
        valid_2d_locations = {BoundaryLocation.LEFT, BoundaryLocation.RIGHT, 
                             BoundaryLocation.BOTTOM, BoundaryLocation.TOP}
        
        for location in self.boundaries:
            if location not in valid_2d_locations:
                raise ValidationError(f"Invalid 2D boundary location: {location}")
    
    def apply_boundaries(
        self,
        u: NDArray[np.float64],
        grids: Tuple[NDArray[np.float64], NDArray[np.float64]],
        time: float = 0.0
    ) -> NDArray[np.float64]:
        """Apply 2D boundary conditions."""
        if len(grids) != 2:
            raise ValidationError(f"2D boundary manager requires 2 grids, got {len(grids)}")
        
        x_grid, y_grid = grids
        nx, ny = len(x_grid), len(y_grid)
        
        if u.shape != (nx, ny):
            raise ValidationError(f"Solution shape {u.shape} doesn't match grid ({nx}, {ny})")
        
        result = u.copy()
        
        # Apply boundary conditions
        for location, bc in self.boundaries.items():
            if location == BoundaryLocation.LEFT:
                self._apply_left_boundary(result, x_grid, y_grid, bc, time)
            elif location == BoundaryLocation.RIGHT:
                self._apply_right_boundary(result, x_grid, y_grid, bc, time)
            elif location == BoundaryLocation.BOTTOM:
                self._apply_bottom_boundary(result, x_grid, y_grid, bc, time)
            elif location == BoundaryLocation.TOP:
                self._apply_top_boundary(result, x_grid, y_grid, bc, time)
        
        return result
    
    def _apply_left_boundary(
        self,
        u: NDArray[np.float64],
        x_grid: NDArray[np.float64],
        y_grid: NDArray[np.float64],
        bc: BoundaryCondition,
        time: float
    ) -> None:
        """Apply boundary condition at left edge (x = x_min)."""
        if bc.boundary_type == BoundaryType.DIRICHLET:
            value = self._evaluate_boundary_value(bc.value, x_grid[0], y_grid, time)
            u[0, :] = value
        elif bc.boundary_type == BoundaryType.NEUMANN:
            # ∂u/∂x = value at x = x_min
            # Use forward difference: u[1] - u[0] = dx * value
            dx = x_grid[1] - x_grid[0]
            value = self._evaluate_boundary_value(bc.value, x_grid[0], y_grid, time)
            u[0, :] = u[1, :] - dx * value
        elif bc.boundary_type == BoundaryType.ZERO_GRADIENT:
            # ∂u/∂x = 0 at x = x_min
            u[0, :] = u[1, :]
        elif bc.boundary_type == BoundaryType.ROBIN:
            # αu + β∂u/∂x = value at x = x_min
            dx = x_grid[1] - x_grid[0]
            value = self._evaluate_boundary_value(bc.value, x_grid[0], y_grid, time)
            # α*u[0] + β*(u[1] - u[0])/dx = value
            # u[0] = (value - β*u[1]/dx) / (α - β/dx)
            denom = bc.alpha - bc.beta / dx
            if abs(denom) < 1e-14:
                raise ValidationError("Robin boundary condition is singular")
            u[0, :] = (value - bc.beta * u[1, :] / dx) / denom
    
    def _apply_right_boundary(
        self,
        u: NDArray[np.float64],
        x_grid: NDArray[np.float64],
        y_grid: NDArray[np.float64],
        bc: BoundaryCondition,
        time: float
    ) -> None:
        """Apply boundary condition at right edge (x = x_max)."""
        nx = len(x_grid)
        if bc.boundary_type == BoundaryType.DIRICHLET:
            value = self._evaluate_boundary_value(bc.value, x_grid[-1], y_grid, time)
            u[-1, :] = value
        elif bc.boundary_type == BoundaryType.NEUMANN:
            # ∂u/∂x = value at x = x_max
            # Use backward difference: u[-1] - u[-2] = dx * value
            dx = x_grid[-1] - x_grid[-2]
            value = self._evaluate_boundary_value(bc.value, x_grid[-1], y_grid, time)
            u[-1, :] = u[-2, :] + dx * value
        elif bc.boundary_type == BoundaryType.ZERO_GRADIENT:
            u[-1, :] = u[-2, :]
        elif bc.boundary_type == BoundaryType.ROBIN:
            dx = x_grid[-1] - x_grid[-2]
            value = self._evaluate_boundary_value(bc.value, x_grid[-1], y_grid, time)
            # α*u[-1] + β*(u[-1] - u[-2])/dx = value
            # u[-1] = (value + β*u[-2]/dx) / (α + β/dx)
            denom = bc.alpha + bc.beta / dx
            if abs(denom) < 1e-14:
                raise ValidationError("Robin boundary condition is singular")
            u[-1, :] = (value + bc.beta * u[-2, :] / dx) / denom
    
    def _apply_bottom_boundary(
        self,
        u: NDArray[np.float64],
        x_grid: NDArray[np.float64],
        y_grid: NDArray[np.float64],
        bc: BoundaryCondition,
        time: float
    ) -> None:
        """Apply boundary condition at bottom edge (y = y_min)."""
        if bc.boundary_type == BoundaryType.DIRICHLET:
            value = self._evaluate_boundary_value(bc.value, x_grid, y_grid[0], time)
            u[:, 0] = value
        elif bc.boundary_type == BoundaryType.NEUMANN:
            dy = y_grid[1] - y_grid[0]
            value = self._evaluate_boundary_value(bc.value, x_grid, y_grid[0], time)
            u[:, 0] = u[:, 1] - dy * value
        elif bc.boundary_type == BoundaryType.ZERO_GRADIENT:
            u[:, 0] = u[:, 1]
        elif bc.boundary_type == BoundaryType.ROBIN:
            dy = y_grid[1] - y_grid[0]
            value = self._evaluate_boundary_value(bc.value, x_grid, y_grid[0], time)
            denom = bc.alpha - bc.beta / dy
            if abs(denom) < 1e-14:
                raise ValidationError("Robin boundary condition is singular")
            u[:, 0] = (value - bc.beta * u[:, 1] / dy) / denom
    
    def _apply_top_boundary(
        self,
        u: NDArray[np.float64],
        x_grid: NDArray[np.float64],
        y_grid: NDArray[np.float64],
        bc: BoundaryCondition,
        time: float
    ) -> None:
        """Apply boundary condition at top edge (y = y_max)."""
        if bc.boundary_type == BoundaryType.DIRICHLET:
            value = self._evaluate_boundary_value(bc.value, x_grid, y_grid[-1], time)
            u[:, -1] = value
        elif bc.boundary_type == BoundaryType.NEUMANN:
            dy = y_grid[-1] - y_grid[-2]
            value = self._evaluate_boundary_value(bc.value, x_grid, y_grid[-1], time)
            u[:, -1] = u[:, -2] + dy * value
        elif bc.boundary_type == BoundaryType.ZERO_GRADIENT:
            u[:, -1] = u[:, -2]
        elif bc.boundary_type == BoundaryType.ROBIN:
            dy = y_grid[-1] - y_grid[-2]
            value = self._evaluate_boundary_value(bc.value, x_grid, y_grid[-1], time)
            denom = bc.alpha + bc.beta / dy
            if abs(denom) < 1e-14:
                raise ValidationError("Robin boundary condition is singular")
            u[:, -1] = (value + bc.beta * u[:, -2] / dy) / denom
    
    def _evaluate_boundary_value(
        self,
        value: float | Callable[..., float],
        x: NDArray[np.float64] | float,
        y: NDArray[np.float64] | float,
        time: float
    ) -> float | NDArray[np.float64]:
        """Evaluate boundary value (constant or function)."""
        if callable(value):
            return value(x, y, time)
        else:
            return value


@dataclass
class BoundaryManager3D(MultiDimensionalBoundaryManager):
    """Boundary condition manager for 3D problems.
    
    Parameters
    ----------
    boundaries : Dict[BoundaryLocation, BoundaryCondition]
        Boundary conditions for each location.
    """
    
    boundaries: Dict[BoundaryLocation, BoundaryCondition]
    
    def apply_boundaries(
        self,
        u: NDArray[np.float64],
        grids: Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        time: float = 0.0
    ) -> NDArray[np.float64]:
        """Apply 3D boundary conditions."""
        if len(grids) != 3:
            raise ValidationError(f"3D boundary manager requires 3 grids, got {len(grids)}")
        
        x_grid, y_grid, z_grid = grids
        nx, ny, nz = len(x_grid), len(y_grid), len(z_grid)
        
        if u.shape != (nx, ny, nz):
            raise ValidationError(f"Solution shape {u.shape} doesn't match grid ({nx}, {ny}, {nz})")
        
        result = u.copy()
        
        # Apply boundary conditions (simplified implementation)
        for location, bc in self.boundaries.items():
            if location == BoundaryLocation.LEFT:
                if bc.boundary_type == BoundaryType.DIRICHLET:
                    result[0, :, :] = bc.value
                elif bc.boundary_type == BoundaryType.ZERO_GRADIENT:
                    result[0, :, :] = result[1, :, :]
            elif location == BoundaryLocation.RIGHT:
                if bc.boundary_type == BoundaryType.DIRICHLET:
                    result[-1, :, :] = bc.value
                elif bc.boundary_type == BoundaryType.ZERO_GRADIENT:
                    result[-1, :, :] = result[-2, :, :]
            # Add similar logic for other boundaries...
        
        return result


# Convenience functions for common boundary setups

def create_dirichlet_boundaries_2d(
    left: float = 0.0,
    right: float = 0.0,
    bottom: float = 0.0,
    top: float = 0.0
) -> BoundaryManager2D:
    """Create 2D boundary manager with Dirichlet conditions."""
    boundaries = {
        BoundaryLocation.LEFT: BoundaryCondition(BoundaryType.DIRICHLET, left),
        BoundaryLocation.RIGHT: BoundaryCondition(BoundaryType.DIRICHLET, right),
        BoundaryLocation.BOTTOM: BoundaryCondition(BoundaryType.DIRICHLET, bottom),
        BoundaryLocation.TOP: BoundaryCondition(BoundaryType.DIRICHLET, top),
    }
    return BoundaryManager2D(boundaries)


def create_zero_gradient_boundaries_2d() -> BoundaryManager2D:
    """Create 2D boundary manager with zero gradient conditions."""
    boundaries = {
        BoundaryLocation.LEFT: BoundaryCondition(BoundaryType.ZERO_GRADIENT),
        BoundaryLocation.RIGHT: BoundaryCondition(BoundaryType.ZERO_GRADIENT),
        BoundaryLocation.BOTTOM: BoundaryCondition(BoundaryType.ZERO_GRADIENT),
        BoundaryLocation.TOP: BoundaryCondition(BoundaryType.ZERO_GRADIENT),
    }
    return BoundaryManager2D(boundaries)


def create_heston_boundaries(
    spot_min: float,
    spot_max: float,
    var_max: float,
    strike: float,
    is_call: bool = True
) -> BoundaryManager2D:
    """Create boundary conditions typical for Heston model option pricing.
    
    Parameters
    ----------
    spot_min : float
        Minimum spot price.
    spot_max : float
        Maximum spot price.
    var_max : float
        Maximum variance.
    strike : float
        Option strike price.
    is_call : bool, optional
        True for call option, False for put (default: True).
        
    Returns
    -------
    BoundaryManager2D
        Configured boundary manager.
    """
    if is_call:
        # Call option boundaries
        boundaries = {
            # S = 0: option worth 0
            BoundaryLocation.LEFT: BoundaryCondition(BoundaryType.DIRICHLET, 0.0),
            # S = S_max: option worth approximately S_max - K
            BoundaryLocation.RIGHT: BoundaryCondition(
                BoundaryType.DIRICHLET, 
                max(spot_max - strike, 0.0)
            ),
            # V = 0: zero gradient (no volatility)
            BoundaryLocation.BOTTOM: BoundaryCondition(BoundaryType.ZERO_GRADIENT),
            # V = V_max: zero gradient (high volatility)
            BoundaryLocation.TOP: BoundaryCondition(BoundaryType.ZERO_GRADIENT),
        }
    else:
        # Put option boundaries
        boundaries = {
            # S = 0: option worth K
            BoundaryLocation.LEFT: BoundaryCondition(BoundaryType.DIRICHLET, strike),
            # S = S_max: option worth 0
            BoundaryLocation.RIGHT: BoundaryCondition(BoundaryType.DIRICHLET, 0.0),
            # V = 0: zero gradient
            BoundaryLocation.BOTTOM: BoundaryCondition(BoundaryType.ZERO_GRADIENT),
            # V = V_max: zero gradient
            BoundaryLocation.TOP: BoundaryCondition(BoundaryType.ZERO_GRADIENT),
        }
    
    return BoundaryManager2D(boundaries)


def create_mixed_boundaries_2d(
    left_type: BoundaryType = BoundaryType.DIRICHLET,
    left_value: float = 0.0,
    right_type: BoundaryType = BoundaryType.ZERO_GRADIENT,
    right_value: float = 0.0,
    bottom_type: BoundaryType = BoundaryType.ZERO_GRADIENT,
    bottom_value: float = 0.0,
    top_type: BoundaryType = BoundaryType.ZERO_GRADIENT,
    top_value: float = 0.0
) -> BoundaryManager2D:
    """Create 2D boundary manager with mixed boundary types."""
    boundaries = {
        BoundaryLocation.LEFT: BoundaryCondition(left_type, left_value),
        BoundaryLocation.RIGHT: BoundaryCondition(right_type, right_value),
        BoundaryLocation.BOTTOM: BoundaryCondition(bottom_type, bottom_value),
        BoundaryLocation.TOP: BoundaryCondition(top_type, top_value),
    }
    return BoundaryManager2D(boundaries)
