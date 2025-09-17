"""Solver interface for PDE pricing.

This module defines the abstract interface for PDE solvers
in the unified pricing framework.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from findiff import BoundaryConditions, FinDiff
from numpy.typing import NDArray

from ..pricing.instruments.base import UnifiedInstrument
from ..processes.base import StochasticProcess
from ..spatial_operator import SpatialOperator
from src.exceptions import ValidationError

from .finite_difference import (
    FiniteDifferenceSolver,
    ThetaMethod,
    TimeStepper,
)


class Solver(ABC):
    """Abstract base class for PDE solvers."""
    
    @abstractmethod
    def solve(
        self,
        initial_condition: NDArray[np.float64],
        instrument: UnifiedInstrument,
        *grids: NDArray[np.float64],
        time_grid: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        """Solve PDE for instrument pricing.
        
        Parameters
        ----------
        initial_condition : NDArray[np.float64]
            Initial condition (payoff at maturity).
        instrument : UnifiedInstrument
            Financial instrument to price.
        *grids : NDArray[np.float64]
            Spatial grids for each dimension.
        time_grid : NDArray[np.float64], optional
            Time grid for evolution.
            
        Returns
        -------
        NDArray[np.float64]
            Solution on the spatial grid.
        """
        ...


class FiniteDifferenceSolverAdapter(Solver):
    """Adapter exposing the finite difference solver through the unified API."""

    def __init__(
        self,
        process: StochasticProcess,
        *,
        time_stepper: TimeStepper | None = None,
        theta: float = 0.5,
    ) -> None:
        if time_stepper is None:
            time_stepper = ThetaMethod(theta)

        self._process = process
        self._time_stepper = time_stepper
        self._solver = FiniteDifferenceSolver(time_stepper=time_stepper)

    def solve(
        self,
        initial_condition: NDArray[np.float64],
        instrument: UnifiedInstrument,
        *grids: NDArray[np.float64],
        time_grid: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        if len(grids) != 1:
            raise ValidationError("1D finite difference solver expects a single spatial grid")

        if time_grid is None:
            raise ValidationError("time_grid must be provided for 1D finite difference solver")

        spatial_grid = grids[0]
        generator = self._build_generator(spatial_grid)
        boundary_conditions = self._build_boundary_conditions(spatial_grid, instrument)

        return self._solver.solve(
            generator=generator,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_condition,
            time_grid=time_grid,
        )

    def _build_generator(self, spatial_grid: NDArray[np.float64]) -> FinDiff:
        operator = SpatialOperator(self._process)
        return operator.build(spatial_grid)

    def _build_boundary_conditions(
        self,
        spatial_grid: NDArray[np.float64],
        instrument: UnifiedInstrument,
    ) -> BoundaryConditions:
        if hasattr(instrument, "boundary_conditions"):
            candidate = instrument.boundary_conditions(spatial_grid)  # type: ignore[attr-defined]
            if isinstance(candidate, BoundaryConditions):
                return candidate

        ds = spatial_grid[1] - spatial_grid[0] if len(spatial_grid) > 1 else 1.0
        bc = BoundaryConditions(spatial_grid.shape)
        d1 = FinDiff(0, ds, 1)
        d2 = FinDiff(0, ds, 2)

        bc[0] = d2, 0.0

        option_type = getattr(instrument, "option_type", "").lower()
        if option_type == "call":
            bc[-1] = d1, 1.0
        elif option_type == "put":
            bc[-1] = d1, 0.0
        else:
            bc[-1] = d2, 0.0

        return bc


class ADISolverWrapper(Solver):
    """Wrapper for ADI solver to conform to Solver interface."""
    
    def __init__(self, adi_solver):
        """Initialize wrapper with ADI solver."""
        self._adi_solver = adi_solver
    
    def solve(
        self,
        initial_condition: NDArray[np.float64],
        instrument: UnifiedInstrument,
        *grids: NDArray[np.float64],
        time_grid: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        """Solve multi-dimensional PDE using ADI method."""
        if time_grid is None:
            n_time_steps = 100
            time_grid = np.linspace(0, instrument.maturity, n_time_steps + 1)
        
        # For this example, we'll use the existing ADI solver methods
        if len(grids) == 2:
            # Create dummy drift and covariance for 2D
            grid_shape = np.meshgrid(*grids, indexing='ij')[0].shape
            drift = np.zeros(grid_shape + (2,))
            covariance = np.zeros(grid_shape + (2, 2))
            # Set some basic values to make it work
            covariance[..., 0, 0] = 0.04  # sigma^2 for first dimension
            covariance[..., 1, 1] = 0.01  # sigma^2 for second dimension
            
            solution = self._adi_solver.solve_2d(
                initial_condition=initial_condition,
                drift=drift,
                covariance=covariance,
                time_grid=time_grid,
                spatial_grids=grids,
            )
            return solution
        elif len(grids) == 3:
            # Create dummy drift and covariance for 3D
            grid_shape = np.meshgrid(*grids, indexing='ij')[0].shape
            drift = np.zeros(grid_shape + (3,))
            covariance = np.zeros(grid_shape + (3, 3))
            # Set some basic values to make it work
            covariance[..., 0, 0] = 0.04  # sigma^2 for first dimension
            covariance[..., 1, 1] = 0.01  # sigma^2 for second dimension
            covariance[..., 2, 2] = 0.005  # sigma^2 for third dimension
            
            solution = self._adi_solver.solve_3d(
                initial_condition=initial_condition,
                drift=drift,
                covariance=covariance,
                time_grid=time_grid,
                spatial_grids=grids,
            )
            return solution
        else:
            raise ValidationError(f"Unsupported dimension: {len(grids)}")


class SolverFactory:
    """Factory for creating appropriate solvers."""
    
    @staticmethod
    def create_solver(
        process: StochasticProcess,
        theta: float = 0.5,
        time_stepper: TimeStepper | None = None,
    ) -> Solver:
        """Create appropriate solver for process.
        
        Parameters
        ----------
        process : StochasticProcess
            Stochastic process for the underlying asset(s).
        theta : float, optional
            Implicitness parameter for finite difference methods.
            
        Returns
        -------
        Solver
            Appropriate solver for the process.
        """
        from ..solvers.adi import ADISolver
        
        if process.dimension.is_univariate:
            return FiniteDifferenceSolverAdapter(
                process,
                time_stepper=time_stepper,
                theta=theta,
            )
        else:
            adi_solver = ADISolver(theta=theta)
            return ADISolverWrapper(adi_solver)
