"""Multi-dimensional pricing engine for options using PDE methods.

This module provides a high-level interface for pricing options with 
multi-dimensional underlying processes like Heston stochastic volatility.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, Callable

import numpy as np
from numpy.typing import NDArray

from .multidimensional_processes import MultiDimensionalProcess
from .multidimensional_solver import MultiDimensionalPDESolver, ADISolver
from .exceptions import ValidationError, PricingError


class MultiDimensionalInstrument(ABC):
    """Abstract base class for multi-dimensional financial instruments."""
    
    @abstractmethod
    def payoff(self, *states: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute payoff function at maturity.
        
        Parameters
        ----------
        *states : NDArray[np.float64]
            State variables (e.g., spot price, volatility for Heston).
            
        Returns
        -------
        NDArray[np.float64]
            Payoff values.
        """
        ...
    
    @property
    @abstractmethod
    def maturity(self) -> float:
        """Time to maturity."""
        ...


@dataclass
class MultiDimensionalOption(MultiDimensionalInstrument):
    """Multi-dimensional option with customizable payoff.
    
    Parameters
    ----------
    payoff_func : Callable
        Function that computes payoff given state variables.
    maturity_time : float
        Time to maturity.
    strike : float, optional
        Strike price (if applicable).
    option_type : str, optional
        Option type description.
    """
    
    payoff_func: Callable[..., NDArray[np.float64]]
    maturity_time: float
    strike: Optional[float] = None
    option_type: str = "custom"
    
    def __post_init__(self) -> None:
        """Validate option parameters."""
        if self.maturity_time <= 0:
            raise ValidationError(f"maturity must be positive, got {self.maturity_time}")
        
        if self.strike is not None and self.strike <= 0:
            raise ValidationError(f"strike must be positive, got {self.strike}")
    
    def payoff(self, *states: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute option payoff."""
        return self.payoff_func(*states)
    
    @property
    def maturity(self) -> float:
        """Time to maturity."""
        return self.maturity_time


@dataclass
class MultiDimensionalPricingEngine:
    """Pricing engine for multi-dimensional options.
    
    Coordinates multi-dimensional stochastic processes and PDE solvers
    to price options with multiple underlying factors.
    
    Parameters
    ----------
    process : MultiDimensionalProcess
        Stochastic process model.
    solver : MultiDimensionalPDESolver
        PDE solver for time evolution.
    """
    
    process: MultiDimensionalProcess
    solver: MultiDimensionalPDESolver
    
    def price_option(
        self,
        option: MultiDimensionalInstrument,
        grids: Tuple[NDArray[np.float64], ...],
        time_steps: int = 100,
        boundary_conditions: Optional[Dict[str, Any]] = None
    ) -> NDArray[np.float64]:
        """Price multi-dimensional option using PDE methods.
        
        Parameters
        ----------
        option : MultiDimensionalInstrument
            Option to price.
        grids : Tuple[NDArray[np.float64], ...]
            Spatial grids for each dimension.
        time_steps : int, optional
            Number of time steps (default: 100).
        boundary_conditions : Dict[str, Any], optional
            Boundary condition specifications.
            
        Returns
        -------
        NDArray[np.float64]
            Option values with shape matching spatial grids.
        """
        process_dim = (self.process.dimension.value 
                      if hasattr(self.process.dimension, 'value') 
                      else self.process.dimension)
        if len(grids) != process_dim:
            raise ValidationError(
                f"Number of grids ({len(grids)}) must match process dimension "
                f"({process_dim})"
            )
        
        if time_steps <= 0:
            raise ValidationError(f"time_steps must be positive, got {time_steps}")
        
        # Create time grid (backward in time from maturity to 0)
        time_grid = np.linspace(0, option.maturity, time_steps + 1)
        
        # Compute drift and diffusion on grids
        drift, diffusion = self._compute_coefficients(grids)
        
        # Set up initial conditions (payoff at maturity)
        # Create meshgrids for proper broadcasting
        process_dim = (self.process.dimension.value 
                      if hasattr(self.process.dimension, 'value') 
                      else self.process.dimension)
        if process_dim == 2:
            S, V = np.meshgrid(grids[0], grids[1], indexing='ij')
            initial_conditions = option.payoff(S, V)
        elif process_dim == 3:
            S1, S2, S3 = np.meshgrid(grids[0], grids[1], grids[2], indexing='ij')
            initial_conditions = option.payoff(S1, S2, S3)
        else:
            raise PricingError(f"Unsupported dimension: {process_dim}")
        
        # Solve PDE
        process_dim = (self.process.dimension.value 
                      if hasattr(self.process.dimension, 'value') 
                      else self.process.dimension)
        if process_dim == 2:
            if len(grids) != 2:
                raise ValidationError("2D process requires exactly 2 grids")
            solution = self.solver.solve_2d(
                drift, diffusion, initial_conditions, 
                grids, time_grid, boundary_conditions
            )
        elif process_dim == 3:
            if len(grids) != 3:
                raise ValidationError("3D process requires exactly 3 grids")
            solution = self.solver.solve_3d(
                drift, diffusion, initial_conditions,
                grids, time_grid, boundary_conditions
            )
        else:
            raise PricingError(
                f"Unsupported dimension: {self.process.dimension.value}"
            )
        
        # Return option values at t=0 (present time)
        return solution[-1]  # Last time step corresponds to t=0
    
    def compute_greeks(
        self,
        option: MultiDimensionalInstrument,
        grids: Tuple[NDArray[np.float64], ...],
        time_steps: int = 100,
        finite_diff_step: float = 0.01
    ) -> Dict[str, NDArray[np.float64]]:
        """Compute option Greeks using finite differences.
        
        Parameters
        ----------
        option : MultiDimensionalInstrument
            Option to analyze.
        grids : Tuple[NDArray[np.float64], ...]
            Spatial grids for each dimension.
        time_steps : int, optional
            Number of time steps.
        finite_diff_step : float, optional
            Step size for finite differences.
            
        Returns
        -------
        Dict[str, NDArray[np.float64]]
            Dictionary containing computed Greeks.
        """
        # Base price
        base_price = self.price_option(option, grids, time_steps)
        greeks = {}
        
        # Delta (first derivative w.r.t. first state variable)
        if len(grids) >= 1:
            shifted_grids_up = list(grids)
            shifted_grids_down = list(grids)
            
            step = finite_diff_step * grids[0]
            shifted_grids_up[0] = grids[0] + step
            shifted_grids_down[0] = grids[0] - step
            
            price_up = self.price_option(option, tuple(shifted_grids_up), time_steps)
            price_down = self.price_option(option, tuple(shifted_grids_down), time_steps)
            
            # Create step matrix with proper broadcasting
            if len(grids) == 2:
                step_matrix = step[:, np.newaxis]  # Shape (nx, 1) for broadcasting to (nx, ny)
            else:
                step_matrix = step
            
            greeks['delta'] = (price_up - price_down) / (2 * step_matrix)
        
        # Gamma (second derivative w.r.t. first state variable)
        if len(grids) >= 1:
            shifted_grids_up = list(grids)
            shifted_grids_down = list(grids)
            
            step = finite_diff_step * grids[0]
            shifted_grids_up[0] = grids[0] + step
            shifted_grids_down[0] = grids[0] - step
            
            price_up = self.price_option(option, tuple(shifted_grids_up), time_steps)
            price_down = self.price_option(option, tuple(shifted_grids_down), time_steps)
            
            # Create step matrix with proper broadcasting
            if len(grids) == 2:
                step_matrix = step[:, np.newaxis]  # Shape (nx, 1) for broadcasting to (nx, ny)
            else:
                step_matrix = step
            
            greeks['gamma'] = (price_up - 2 * base_price + price_down) / (step_matrix ** 2)
        
        # Vega (derivative w.r.t. second state variable, if exists)
        if len(grids) >= 2:
            shifted_grids_up = list(grids)
            shifted_grids_down = list(grids)
            
            step = finite_diff_step * grids[1]
            shifted_grids_up[1] = grids[1] + step
            shifted_grids_down[1] = grids[1] - step
            
            price_up = self.price_option(option, tuple(shifted_grids_up), time_steps)
            price_down = self.price_option(option, tuple(shifted_grids_down), time_steps)
            
            # Create step matrix with proper broadcasting
            if len(grids) == 2:
                step_matrix = step[np.newaxis, :]  # Shape (1, ny) for broadcasting to (nx, ny)
            else:
                step_matrix = step
            
            greeks['vega'] = (price_up - price_down) / (2 * step_matrix)
        
        return greeks
    
    def _compute_coefficients(
        self, 
        grids: Tuple[NDArray[np.float64], ...]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute drift and diffusion coefficients on grid points."""
        process_dim = (self.process.dimension.value 
                      if hasattr(self.process.dimension, 'value') 
                      else self.process.dimension)
        if process_dim == 2:
            x_grid, y_grid = grids
            nx, ny = len(x_grid), len(y_grid)
            
            # Create meshgrid
            X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
            states = np.stack([X.flatten(), Y.flatten()], axis=1)
            
            # Compute coefficients
            drift_flat = self.process.drift(states)
            diffusion_flat = self.process.diffusion_matrix(states)
            
            # Reshape to grid format
            drift = drift_flat.reshape(nx, ny, 2)
            diffusion = diffusion_flat.reshape(nx, ny, 2, 2)
            
            return drift, diffusion
            
        elif process_dim == 3:
            x_grid, y_grid, z_grid = grids
            nx, ny, nz = len(x_grid), len(y_grid), len(z_grid)
            
            # Create meshgrid
            X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
            states = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
            
            # Compute coefficients
            drift_flat = self.process.drift(states)
            diffusion_flat = self.process.diffusion_matrix(states)
            
            # Reshape to grid format
            drift = drift_flat.reshape(nx, ny, nz, 3)
            diffusion = diffusion_flat.reshape(nx, ny, nz, 3, 3)
            
            return drift, diffusion
            
        else:
            raise PricingError(f"Unsupported dimension: {process_dim}")


# Convenience functions for common option types

def create_european_call_2d(strike: float, maturity: float) -> MultiDimensionalOption:
    """Create 2D European call option (payoff depends on first state variable)."""
    def call_payoff(s1: NDArray[np.float64], s2: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.maximum(s1 - strike, 0.0)
    
    return MultiDimensionalOption(
        payoff_func=call_payoff,
        maturity_time=maturity,
        strike=strike,
        option_type="european_call_2d"
    )


def create_european_put_2d(strike: float, maturity: float) -> MultiDimensionalOption:
    """Create 2D European put option (payoff depends on first state variable)."""
    def put_payoff(s1: NDArray[np.float64], s2: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.maximum(strike - s1, 0.0)
    
    return MultiDimensionalOption(
        payoff_func=put_payoff,
        maturity_time=maturity,
        strike=strike,
        option_type="european_put_2d"
    )


def create_basket_call(
    weights: NDArray[np.float64], 
    strike: float, 
    maturity: float
) -> MultiDimensionalOption:
    """Create basket call option with weighted average payoff."""
    def basket_payoff(*states: NDArray[np.float64]) -> NDArray[np.float64]:
        if len(states) != len(weights):
            raise ValidationError(f"Number of states ({len(states)}) must match weights ({len(weights)})")
        
        basket_value = sum(w * s for w, s in zip(weights, states, strict=True))
        return np.maximum(basket_value - strike, 0.0)
    
    return MultiDimensionalOption(
        payoff_func=basket_payoff,
        maturity_time=maturity,
        strike=strike,
        option_type="basket_call"
    )


# Grid creation utilities

def create_log_grid(
    spot: float, 
    num_points: int, 
    std_devs: float = 4.0
) -> NDArray[np.float64]:
    """Create logarithmically spaced grid around spot price.
    
    Parameters
    ----------
    spot : float
        Current spot price.
    num_points : int
        Number of grid points.
    std_devs : float, optional
        Number of standard deviations to cover (default: 4).
        
    Returns
    -------
    NDArray[np.float64]
        Grid points.
    """
    if spot <= 0:
        raise ValidationError(f"spot must be positive, got {spot}")
    if num_points <= 0:
        raise ValidationError(f"num_points must be positive, got {num_points}")
    
    # Create symmetric grid in log space
    log_spot = np.log(spot)
    log_range = std_devs * 0.3  # Approximate volatility scaling
    
    log_grid = np.linspace(
        log_spot - log_range, 
        log_spot + log_range, 
        num_points
    )
    
    return np.exp(log_grid)


def create_variance_grid(
    initial_var: float, 
    num_points: int, 
    max_var_multiple: float = 5.0
) -> NDArray[np.float64]:
    """Create grid for variance dimension (e.g., in Heston model).
    
    Parameters
    ----------
    initial_var : float
        Initial variance level.
    num_points : int
        Number of grid points.
    max_var_multiple : float, optional
        Maximum variance as multiple of initial (default: 5).
        
    Returns
    -------
    NDArray[np.float64]
        Variance grid points.
    """
    if initial_var <= 0:
        raise ValidationError(f"initial_var must be positive, got {initial_var}")
    if num_points <= 0:
        raise ValidationError(f"num_points must be positive, got {num_points}")
    
    return np.linspace(0.001, max_var_multiple * initial_var, num_points)


def create_heston_grids(
    spot: float,
    initial_var: float,
    num_spot: int = 100,
    num_var: int = 50,
    spot_std_devs: float = 4.0,
    max_var_multiple: float = 5.0
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create grids for Heston model pricing.
    
    Parameters
    ----------
    spot : float
        Current spot price.
    initial_var : float
        Current variance level.
    num_spot : int, optional
        Number of spot price points (default: 100).
    num_var : int, optional
        Number of variance points (default: 50).
    spot_std_devs : float, optional
        Spot price grid range in standard deviations (default: 4).
    max_var_multiple : float, optional
        Maximum variance as multiple of initial (default: 5).
        
    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64]]
        Spot and variance grids.
    """
    spot_grid = create_log_grid(spot, num_spot, spot_std_devs)
    var_grid = create_variance_grid(initial_var, num_var, max_var_multiple)
    
    return spot_grid, var_grid


def create_default_pricing_engine(
    process: MultiDimensionalProcess
) -> MultiDimensionalPricingEngine:
    """Create pricing engine with default ADI solver."""
    solver = ADISolver()
    return MultiDimensionalPricingEngine(process=process, solver=solver)
