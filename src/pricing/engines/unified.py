"""Unified pricing engine for multi-dimensional option pricing.

This module provides a unified framework for pricing options using various
stochastic processes through a dimension-agnostic interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np
from numpy.typing import NDArray

from ...processes.base import StochasticProcess
from ...pricing.instruments.base import UnifiedInstrument
from ...utils.exceptions import ValidationError


@dataclass
class UnifiedPricingEngine:
    """Unified pricing engine for multi-dimensional processes.
    
    This engine automatically selects appropriate solvers based on
    the process dimension and provides a consistent interface for
    pricing various financial instruments.
    
    Parameters
    ----------
    process : StochasticProcess
        Stochastic process for the underlying asset(s).
    solver : Optional
        PDE solver (auto-selected if None).
    """
    
    process: StochasticProcess
    solver: Optional = None
    
    def __post_init__(self) -> None:
        """Initialize pricing engine."""
        if self.solver is None:
            if self.process.dimension.is_univariate:
                # For 1D, we could use existing solvers or create simple ADI
                self.solver = self._create_1d_solver()
            else:
                # For multi-D, use ADI solver
                from ...solvers.adi import create_adi_solver
                self.solver = create_adi_solver(theta=0.5)
    
    def _create_1d_solver(self):
        """Create 1D solver (placeholder for now)."""
        # This would create a 1D finite difference solver
        # For now, return a simple placeholder
        return None
    
    def price_option(
        self,
        instrument: UnifiedInstrument,
        *grids: NDArray[np.float64],
        time_grid: Optional[NDArray[np.float64]] = None,
        boundary_manager: Optional = None
    ) -> NDArray[np.float64]:
        """Price option using unified framework.
        
        Parameters
        ----------
        instrument : UnifiedInstrument
            Financial instrument to price.
        *grids : NDArray[np.float64]
            Spatial grids for each dimension.
        time_grid : NDArray[np.float64], optional
            Time grid for evolution.
        boundary_manager : Optional
            Boundary condition manager.
            
        Returns
        -------
        NDArray[np.float64]
            Option prices on the spatial grid.
        """
        if len(grids) == 0:
            raise ValidationError("At least one spatial grid required")
        
        # Validate grid dimensions match process dimension
        if len(grids) != self.process.dimension.value:
            raise ValidationError(
                f"Number of grids ({len(grids)}) must match process dimension "
                f"({self.process.dimension.value})"
            )
        
        # Set default time grid if not provided
        if time_grid is None:
            n_time_steps = 100
            time_grid = np.linspace(0, instrument.maturity, n_time_steps + 1)
        
        # Compute PDE coefficients
        drift, covariance = self._compute_pde_coefficients(grids)
        
        # Get initial condition (payoff at maturity)
        initial_condition = instrument.payoff(*grids)
        
        # Solve PDE backward in time (placeholder implementation)
        # In a full implementation, this would use the appropriate solver
        return initial_condition  # Simplified for now
    
    def _compute_pde_coefficients(
        self, 
        grids: tuple
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute PDE coefficients on spatial grids.
        
        Parameters
        ----------
        grids : tuple
            Spatial grids.
            
        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.float64]]
            (drift, covariance) coefficients on the grid.
        """
        from ...utils.state_handling import create_state_matrix
        
        # Create state matrix from grids
        state_matrix = create_state_matrix(*grids)
        
        # Evaluate drift and covariance for all points at once
        drift_flat = self.process.drift(0.0, state_matrix)
        covariance_flat = self.process.covariance(0.0, state_matrix)
        
        # Reshape back to grid
        if len(grids) == 1:
            grid_shape = grids[0].shape
            drift = drift_flat.reshape(grid_shape + (1,))
            covariance = covariance_flat.reshape(grid_shape + (1, 1))
        else:
            mesh_grids = np.meshgrid(*grids, indexing='ij')
            grid_shape = mesh_grids[0].shape
            drift = drift_flat.reshape(grid_shape + (len(grids),))
            covariance = covariance_flat.reshape(grid_shape + (len(grids), len(grids)))
        
        return drift, covariance
    
    def compute_greeks(
        self,
        instrument: UnifiedInstrument,
        *grids: NDArray[np.float64],
        time_grid: Optional[NDArray[np.float64]] = None,
        boundary_manager: Optional = None,
        epsilon: float = 1e-4
    ) -> Dict[str, NDArray[np.float64]]:
        """Compute option Greeks using finite differences.
        
        Parameters
        ----------
        instrument : UnifiedInstrument
            Financial instrument.
        *grids : NDArray[np.float64]
            Spatial grids.
        time_grid : NDArray[np.float64], optional
            Time grid.
        boundary_manager : Optional
            Boundary condition manager.
        epsilon : float
            Finite difference step size.
            
        Returns
        -------
        Dict[str, NDArray[np.float64]]
            Dictionary of Greeks.
        """
        greeks = {}
        
        # Base price
        base_price = self.price_option(
            instrument, *grids, 
            time_grid=time_grid, boundary_manager=boundary_manager
        )
        
        # Delta and Gamma (w.r.t. first underlying)
        if len(grids) >= 1:
            grid_up = grids[0] + epsilon * grids[0]
            grid_down = grids[0] - epsilon * grids[0]
            
            price_up = self.price_option(
                instrument, grid_up, *grids[1:], 
                time_grid=time_grid, boundary_manager=boundary_manager
            )
            price_down = self.price_option(
                instrument, grid_down, *grids[1:],
                time_grid=time_grid, boundary_manager=boundary_manager
            )
            
            greeks['delta'] = (price_up - price_down) / (2 * epsilon * grids[0])
            greeks['gamma'] = (price_up - 2 * base_price + price_down) / (epsilon * grids[0])**2
        
        return greeks


# Convenience functions
def create_unified_pricing_engine(process: StochasticProcess) -> UnifiedPricingEngine:
    """Create unified pricing engine with auto-selected solver."""
    return UnifiedPricingEngine(process=process)


# Grid utility functions
def create_log_grid(s_min: float, s_max: float, n_points: int) -> NDArray[np.float64]:
    """Create logarithmically spaced grid."""
    log_min = np.log(s_min)
    log_max = np.log(s_max)
    log_grid = np.linspace(log_min, log_max, n_points)
    return np.exp(log_grid)


def create_linear_grid(x_min: float, x_max: float, n_points: int) -> NDArray[np.float64]:
    """Create linearly spaced grid."""
    return np.linspace(x_min, x_max, n_points)
