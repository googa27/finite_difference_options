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

from ..instruments.operators import SpatialOperator
from ..pricing.instruments.base import UnifiedInstrument
from ..processes.base import StochasticProcess
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

        if time_grid is None or len(time_grid) == 0:
            time_grid = np.linspace(0.0, instrument.maturity, 50)

        spatial_grid = grids[0]
        generator = self._build_generator(spatial_grid)
        boundary_conditions = self._build_boundary_conditions(spatial_grid, instrument)

        solution = self._solver.solve(
            generator=generator,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_condition,
            time_grid=time_grid,
        )
        return solution[::-1]

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
    """Wrapper for ADI solver to conform to Solver interface.

    The wrapper owns coefficient evaluation for the selected stochastic process.
    It deliberately does not invent placeholder drift or covariance arrays: if a
    process cannot provide coefficients with the expected shape, the route fails
    before calling the ADI solver.
    """

    def __init__(self, adi_solver, *, process: StochasticProcess) -> None:
        """Initialize wrapper with ADI solver and selected process."""
        self._adi_solver = adi_solver
        self._process = process

    def solve(
        self,
        initial_condition: NDArray[np.float64],
        instrument: UnifiedInstrument,
        *grids: NDArray[np.float64],
        time_grid: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        """Solve a multi-dimensional PDE using process-derived coefficients."""
        if time_grid is None or len(time_grid) == 0:
            n_time_steps = 100
            time_grid = np.linspace(0, instrument.maturity, n_time_steps + 1)

        dimension = self._process.dimension.value
        if dimension not in {2, 3}:
            raise ValidationError(f"ADI wrapper supports only 2D and 3D processes, got {dimension}D")
        if len(grids) != dimension:
            raise ValidationError(f"Expected {dimension} grids for ADI solve, got {len(grids)}")

        drift, covariance = self._build_process_coefficients(float(time_grid[-1]), grids)

        if dimension == 2:
            return self._adi_solver.solve_2d(
                initial_condition=initial_condition,
                drift=drift,
                covariance=covariance,
                time_grid=time_grid,
                spatial_grids=grids,
            )

        return self._adi_solver.solve_3d(
            initial_condition=initial_condition,
            drift=drift,
            covariance=covariance,
            time_grid=time_grid,
            spatial_grids=grids,
        )

    def _build_process_coefficients(
        self,
        time: float,
        grids: tuple[NDArray[np.float64], ...],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Evaluate drift and covariance on the full tensor product grid."""
        dimension = self._process.dimension.value
        grid_shape = tuple(len(grid) for grid in grids)
        mesh = np.meshgrid(*grids, indexing="ij")
        states = np.stack([axis.reshape(-1) for axis in mesh], axis=-1)

        drift = np.asarray(self._process.drift(time, states), dtype=float)
        covariance = np.asarray(self._process.covariance(time, states), dtype=float)

        expected_drift_shape = (states.shape[0], dimension)
        expected_covariance_shape = (states.shape[0], dimension, dimension)
        if drift.shape != expected_drift_shape:
            raise ValidationError(
                "process drift must have shape "
                f"{expected_drift_shape} on the ADI state grid, got {drift.shape}"
            )
        if covariance.shape != expected_covariance_shape:
            raise ValidationError(
                "process covariance must have shape "
                f"{expected_covariance_shape} on the ADI state grid, got {covariance.shape}"
            )
        if not np.all(np.isfinite(drift)):
            raise ValidationError("process drift contains non-finite values on the ADI state grid")
        if not np.all(np.isfinite(covariance)):
            raise ValidationError("process covariance contains non-finite values on the ADI state grid")
        if not np.allclose(covariance, np.swapaxes(covariance, -1, -2), rtol=1e-10, atol=1e-12):
            raise ValidationError("process covariance must be symmetric on the ADI state grid")
        min_eigenvalue = float(np.min(np.linalg.eigvalsh(covariance)))
        if min_eigenvalue < -1e-10:
            raise ValidationError(
                "process covariance must be positive semi-definite on the ADI state grid; "
                f"minimum eigenvalue is {min_eigenvalue:.2e}"
            )

        return (
            drift.reshape(*grid_shape, dimension),
            covariance.reshape(*grid_shape, dimension, dimension),
        )


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
            return ADISolverWrapper(adi_solver, process=process)
