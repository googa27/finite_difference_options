"""Unified pricing engine for multi-dimensional option pricing.

This module provides a unified framework for pricing options using various
stochastic processes through a dimension-agnostic interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
from numpy.typing import NDArray

from ...processes.base import StochasticProcess
from ...pricing.instruments.base import UnifiedInstrument
from ...utils.exceptions import ValidationError
from ...solvers.base import Solver, SolverFactory


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
    solver: Optional[Solver] = None
    
    def __post_init__(self) -> None:
        """Initialize pricing engine."""
        if self.solver is None:
            self.solver = SolverFactory.create_solver(self.process)
    
    def price_option(
        self,
        instrument: UnifiedInstrument,
        *grids: NDArray[np.float64],
        time_grid: Optional[NDArray[np.float64]] = None,
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
                f"Expected {self.process.dimension.value} grids, got {len(grids)}"
            )
        
        # Get initial condition (payoff at maturity)
        # For multi-dimensional case, we need to create a meshgrid
        if len(grids) == 1:
            initial_condition = instrument.payoff(grids[0])
        else:
            mesh_grids = np.meshgrid(*grids, indexing='ij')
            # Flatten the meshgrid to pass to payoff function
            flattened_grids = [grid.flatten() for grid in mesh_grids]
            # Reshape back to grid shape
            grid_shape = mesh_grids[0].shape
            initial_condition = instrument.payoff(*flattened_grids).reshape(grid_shape)
        
        # Solve PDE using the abstract solver interface
        solution = self.solver.solve(
            initial_condition,
            instrument,
            *grids,
            time_grid=time_grid
        )
        
        return solution
    
    def compute_greeks(
        self,
        prices: NDArray[np.float64],
        *grids: NDArray[np.float64],
        time_grid: Optional[NDArray[np.float64]] = None,
    ) -> Dict[str, NDArray[np.float64]]:
        """Compute option Greeks using finite differences.
        
        Parameters
        ----------
        prices : NDArray[np.float64]
            Option prices on the grid.
        *grids : NDArray[np.float64]
            Spatial grids.
        time_grid : NDArray[np.float64], optional
            Time grid.
            
        Returns
        -------
        Dict[str, NDArray[np.float64]]
            Dictionary of Greeks.
        """
        from ...greeks.base import GreeksCalculatorFactory
        
        # Create appropriate Greeks calculator
        calculator = GreeksCalculatorFactory.create_calculator(self.process)
        
        # Calculate Greeks
        greeks = calculator.calculate(prices, *grids, time_grid=time_grid)
        
        return greeks


# Convenience functions
def create_unified_pricing_engine(process: StochasticProcess) -> UnifiedPricingEngine:
    """Create unified pricing engine with auto-selected solver."""
    return UnifiedPricingEngine(process=process)


# Grid utility functions
def create_log_grid(s_min: float, s_max: float, n_points: int, center: Optional[float] = None) -> NDArray[np.float64]:
    """Create logarithmically spaced grid.
    
    Parameters
    ----------
    s_min : float
        Minimum value of the grid.
    s_max : float
        Maximum value of the grid.
    n_points : int
        Number of points in the grid.
    center : float, optional
        Center point for the grid. If provided, the grid will be centered around this point.
        
    Returns
    -------
    NDArray[np.float64]
        Logarithmically spaced grid.
    """
    if center is None:
        log_min = np.log(s_min)
        log_max = np.log(s_max)
        log_grid = np.linspace(log_min, log_max, n_points)
        return np.exp(log_grid)
    else:
        # Create grid centered around center point
        log_center = np.log(center)
        log_range = max(np.log(center) - np.log(s_min), np.log(s_max) - np.log(center))
        log_min = log_center - log_range
        log_max = log_center + log_range
        log_grid = np.linspace(log_min, log_max, n_points)
        return np.exp(log_grid)


def create_linear_grid(x_min: float, x_max: float, n_points: int) -> NDArray[np.float64]:
    """Create linearly spaced grid."""
    return np.linspace(x_min, x_max, n_points)