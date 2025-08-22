"""PDE solving engine for finite difference methods.

This module provides the core solving logic separated from PDE definition,
following the single responsibility principle for better modularity.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from findiff import FinDiff, BoundaryConditions

if TYPE_CHECKING:
    from .time_steppers import TimeStepper


class PDESolver(ABC):
    """Abstract base class for PDE solving engines.
    
    This class handles the numerical solution of PDEs using finite difference
    methods, separated from the PDE definition itself for better modularity.
    """

    @abstractmethod
    def solve(
        self,
        generator: FinDiff,
        boundary_conditions: BoundaryConditions,
        initial_conditions: NDArray[np.float64],
        time_grid: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Solve the PDE with given conditions.
        
        Parameters
        ----------
        generator : FinDiff
            The discretized infinitesimal generator (spatial operator).
        boundary_conditions : BoundaryConditions
            Boundary conditions for the spatial domain.
        initial_conditions : NDArray[np.float64]
            Initial values at t=0 (typically payoff at maturity).
        time_grid : NDArray[np.float64]
            Time discretization grid.
            
        Returns
        -------
        NDArray[np.float64]
            Solution grid with shape (len(time_grid), len(spatial_grid)).
        """
        ...


@dataclass
class FiniteDifferenceSolver(PDESolver):
    """Finite difference PDE solver using time stepping methods.
    
    This solver uses explicit or implicit time stepping schemes to evolve
    the PDE solution forward in time from initial conditions.
    
    Parameters
    ----------
    time_stepper : TimeStepper
        The time stepping method to use (e.g., ThetaMethod, ExplicitEuler).
    """
    
    time_stepper: TimeStepper

    def solve(
        self,
        generator: FinDiff,
        boundary_conditions: BoundaryConditions,
        initial_conditions: NDArray[np.float64],
        time_grid: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Solve the PDE using finite difference time stepping.
        
        The solution evolves backward in time from maturity (t=T) to present (t=0),
        which is why we reverse the time grid and initial conditions represent
        the payoff at maturity.
        
        Parameters
        ----------
        generator : FinDiff
            The discretized spatial operator L.
        boundary_conditions : BoundaryConditions
            Spatial boundary conditions.
        initial_conditions : NDArray[np.float64]
            Values at maturity (payoff).
        time_grid : NDArray[np.float64]
            Time points from 0 to T.
            
        Returns
        -------
        NDArray[np.float64]
            Solution values with shape (len(time_grid), len(spatial_grid)).
            values[i, j] is the solution at time_grid[i] and spatial_grid[j].
        """
        dt = time_grid[1] - time_grid[0]
        n_time_steps = len(time_grid)
        n_spatial_points = len(initial_conditions)
        
        # Initialize solution array
        values = np.empty((n_time_steps, n_spatial_points))
        
        # Set initial condition (payoff at maturity)
        values[0] = initial_conditions
        
        # Time step forward from t=0 to t=T
        for i in range(n_time_steps - 1):
            values[i + 1] = self.time_stepper.step(
                values[i], generator, boundary_conditions, dt
            )
            
        return values


def create_default_solver() -> PDESolver:
    """Create a default PDE solver with theta method.
    
    Returns
    -------
    PDESolver
        A FiniteDifferenceSolver with ThetaMethod (theta=0.5, Crank-Nicolson).
    """
    from .time_steppers import ThetaMethod
    return FiniteDifferenceSolver(time_stepper=ThetaMethod(theta=0.5))
