"""Pricing engine that coordinates PDE solving with financial instruments.

This module provides the high-level interface for pricing financial instruments
using PDE methods, coordinating between instruments, solvers, and grid generation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numpy.typing import NDArray

from .validation import validate_grid_parameters, validate_spot_price
from .exceptions import PricingError

if TYPE_CHECKING:
    from .instruments import Instrument
    from .pde_solver import PDESolver


class PricingResult(NamedTuple):
    """Result of pricing computation with grids and values.
    
    Attributes
    ----------
    spatial_grid : NDArray[np.float64]
        Asset price grid points.
    time_grid : NDArray[np.float64]
        Time grid points from 0 to maturity.
    values : NDArray[np.float64]
        Option values with shape (len(time_grid), len(spatial_grid)).
    """
    spatial_grid: NDArray[np.float64]
    time_grid: NDArray[np.float64]
    values: NDArray[np.float64]


@dataclass
class GridParameters:
    """Parameters for grid generation.
    
    Parameters
    ----------
    s_max : float
        Maximum asset price for spatial grid.
    s_steps : int
        Number of spatial grid points.
    t_steps : int
        Number of time steps.
    """
    s_max: float
    s_steps: int
    t_steps: int


@dataclass
class PricingEngine:
    """High-level pricing engine for financial instruments.
    
    This engine coordinates between instruments, PDE solvers, and grid generation
    to provide a clean interface for pricing financial derivatives.
    
    Parameters
    ----------
    solver : PDESolver
        The PDE solver to use for numerical computation.
    """
    
    solver: PDESolver

    def price_instrument(
        self,
        instrument: Instrument,
        grid_params: GridParameters,
    ) -> PricingResult:
        """Price a financial instrument using PDE methods.
        
        Parameters
        ----------
        instrument : Instrument
            The financial instrument to price (e.g., EuropeanCall).
        grid_params : GridParameters
            Grid generation parameters.
            
        Returns
        -------
        PricingResult
            Pricing result with grids and computed values.
            
        Raises
        ------
        GridError
            If grid parameters are invalid.
        PricingError
            If pricing computation fails.
        """
        # Validate grid parameters
        validate_grid_parameters(
            grid_params.s_max, 
            grid_params.s_steps, 
            grid_params.t_steps
        )
        
        try:
            # Generate grids
            s = np.linspace(0, grid_params.s_max, grid_params.s_steps)
            t = np.linspace(0, instrument.maturity, grid_params.t_steps)
        
            # Get PDE components from instrument
            generator = instrument.generator(s)
            boundary_conditions = instrument.boundary_conditions(s)
            initial_conditions = instrument.payoff(s)
            
            # Solve PDE
            values = self.solver.solve(
                generator=generator,
                boundary_conditions=boundary_conditions,
                initial_conditions=initial_conditions,
                time_grid=t,
            )
            
            return PricingResult(
                spatial_grid=s,
                time_grid=t,
                values=values,
            )
        except Exception as e:
            raise PricingError(f"Failed to price instrument: {e}") from e

    def compute_spot_price(
        self,
        instrument: Instrument,
        spot_price: float,
        grid_params: GridParameters,
    ) -> float:
        """Compute option value at a specific spot price.
        
        Parameters
        ----------
        instrument : Instrument
            The financial instrument to price.
        spot_price : float
            Current asset price.
        grid_params : GridParameters
            Grid generation parameters.
            
        Returns
        -------
        float
            Option value at the spot price and current time (t=0).
            
        Raises
        ------
        ValidationError
            If spot price is invalid or outside grid range.
        """
        result = self.price_instrument(instrument, grid_params)
        
        # Validate spot price is within grid range
        validate_spot_price(spot_price, result.spatial_grid)
        
        # Find closest grid point to spot price
        idx = np.searchsorted(result.spatial_grid, spot_price)
        if idx >= len(result.spatial_grid):
            idx = len(result.spatial_grid) - 1
        elif idx > 0:
            # Choose closer of the two adjacent points
            if abs(result.spatial_grid[idx-1] - spot_price) < abs(result.spatial_grid[idx] - spot_price):
                idx = idx - 1
        
        # Return value at t=0 (present time)
        return result.values[-1, idx]


def create_default_pricing_engine() -> PricingEngine:
    """Create a default pricing engine with standard solver.
    
    Returns
    -------
    PricingEngine
        A PricingEngine with FiniteDifferenceSolver using Crank-Nicolson method.
    """
    from .pde_solver import create_default_solver
    return PricingEngine(solver=create_default_solver())
