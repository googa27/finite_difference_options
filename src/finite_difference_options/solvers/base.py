"""Solver interface for PDE pricing.

This module defines the unified solver abstraction and concrete adapters used by
pricing engines. Adapters translate the generic ":class:`Solver`" contract into
underlying numerical implementations for finite differences and ADI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from findiff import BoundaryConditions, FinDiff
from numpy.typing import NDArray

from ..boundary_conditions import BlackScholesBoundaryBuilder
from ..instruments.operators import SpatialOperator
from ..pricing.instruments.base import UnifiedInstrument
from ..processes.base import StochasticProcess
from finite_difference_options.exceptions import ValidationError

from .finite_difference import (
    FiniteDifferenceSolver,
    LCPDiagnostics,
    ProjectedSORLCP,
    ThetaMethod,
    TimeStepper,
)


class Solver(ABC):
    """Abstract base class for PDE solvers.

    Implementations should accept a terminal/initial payoff and return a full time
    profile in the canonical orientation used by the rest of the stack.
    """

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
    """Adapter exposing the finite-difference solver through the unified API.

    This adapter is intentionally conservative: only univariate (1D) spatial
    problems are supported in this path. A :class:`ValidationError` is raised for
    other dimensions rather than silently degrading behavior.
    """

    def __init__(
        self,
        process: StochasticProcess,
        *,
        time_stepper: TimeStepper | None = None,
        theta: float = 0.5,
    ) -> None:
        """Create the 1D finite-difference adapter.

        Parameters
        ----------
        process : StochasticProcess
            Underlying model. It is used to build the spatial generator.
        time_stepper : TimeStepper, optional
            Optional custom time stepper. If omitted, defaults to
            ``ThetaMethod(theta)``.
        theta : float
            Theta parameter used when ``time_stepper`` is not supplied.
        """
        if time_stepper is None:
            time_stepper = ThetaMethod(theta)

        self._process = process
        self._time_stepper = time_stepper
        self._solver = FiniteDifferenceSolver(time_stepper=time_stepper)
        self.last_lcp_diagnostics = LCPDiagnostics(
            exercise_style="european",
            levels=(),
            tolerance=0.0,
            relaxation=0.0,
            max_iterations=0,
        )

    def solve(
        self,
        initial_condition: NDArray[np.float64],
        instrument: UnifiedInstrument,
        *grids: NDArray[np.float64],
        time_grid: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        """Solve a 1D PDE and return solution ordered by valuation to maturity.

        The wrapped :class:`FiniteDifferenceSolver` emits values in forward time,
        so this adapter reverses the axis to keep ``prices[0]`` at valuation time
        and ``prices[-1]`` at maturity, consistent with unified API docs.
        """
        if len(grids) != 1:
            raise ValidationError(
                "1D finite difference solver expects a single spatial grid"
            )

        if time_grid is None or len(time_grid) == 0:
            time_grid = np.linspace(0.0, instrument.maturity, 50)

        spatial_grid = grids[0]
        exercise_style = str(getattr(instrument, "exercise_style", "european"))
        if exercise_style in {"american", "bermudan"}:
            return self._solve_obstacle_lcp(
                initial_condition,
                instrument,
                spatial_grid,
                time_grid,
                exercise_style=exercise_style,
            )

        generator = self._build_generator(spatial_grid, instrument)
        boundary_conditions = self._build_boundary_conditions(spatial_grid, instrument)

        solution = self._solver.solve(
            generator=generator,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_condition,
            time_grid=time_grid,
        )
        self.last_step_schedule = self._solver.last_step_schedule
        return solution[::-1]


    def _solve_obstacle_lcp(
        self,
        initial_condition: NDArray[np.float64],
        instrument: UnifiedInstrument,
        spatial_grid: NDArray[np.float64],
        time_grid: NDArray[np.float64],
        *,
        exercise_style: str,
    ) -> NDArray[np.float64]:
        """Solve a 1D Black-Scholes obstacle problem and return calendar order."""
        if not hasattr(instrument, "strike") or not hasattr(instrument, "option_type"):
            raise ValidationError(
                "American/Bermudan LCP route requires a vanilla strike and option_type"
            )
        if not hasattr(self._process, "sigma"):
            raise ValidationError(
                "American/Bermudan LCP route currently supports one-factor Black-Scholes/GBM processes"
            )
        exercise_dates = tuple(float(x) for x in getattr(instrument, "exercise_dates", ()))
        lcp_solver = ProjectedSORLCP(
            tolerance=float(getattr(instrument, "lcp_tolerance", 1.0e-8)),
            max_iterations=int(getattr(instrument, "lcp_max_iterations", 10_000)),
            relaxation=float(getattr(instrument, "lcp_relaxation", 1.2)),
        )
        try:
            tau_solution = lcp_solver.solve_black_scholes(
                spot_grid=spatial_grid,
                payoff=initial_condition,
                time_grid=time_grid,
                strike=float(instrument.strike),  # type: ignore[attr-defined]
                option_type=str(instrument.option_type),  # type: ignore[attr-defined]
                risk_free_rate=self._risk_free_rate_for(instrument),
                dividend_yield=self._dividend_yield_for(instrument),
                volatility=self._volatility_for(instrument),
                exercise_style=exercise_style,
                exercise_dates=exercise_dates,
            )
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        self.last_lcp_diagnostics = lcp_solver.last_diagnostics
        self.last_step_schedule = ()
        return tau_solution[::-1]

    def _build_generator(
        self,
        spatial_grid: NDArray[np.float64],
        instrument: UnifiedInstrument,
    ) -> FinDiff:
        operator = SpatialOperator(
            self._process,
            discount_rate=self._risk_free_rate_for(instrument),
        )
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

        builder = BlackScholesBoundaryBuilder()
        return builder.build(
            spatial_grid,
            instrument,
            time_to_maturity=getattr(instrument, "maturity", None),
            risk_free_rate=self._risk_free_rate_for(instrument),
            dividend_yield=self._dividend_yield_for(instrument),
        )

    def _risk_free_rate_for(self, instrument: UnifiedInstrument) -> float:
        explicit = getattr(instrument, "risk_free_rate", None)
        if explicit is None:
            explicit = getattr(instrument, "discount_rate", None)
        if explicit is not None:
            return float(explicit)
        model_rate = getattr(self._process, "risk_free_rate", None)
        if model_rate is not None:
            return float(model_rate)
        legacy_mu = getattr(self._process, "mu", None)
        if legacy_mu is not None:
            return float(legacy_mu)
        return 0.0

    def _dividend_yield_for(self, instrument: UnifiedInstrument) -> float:
        explicit = getattr(instrument, "dividend_yield", None)
        if explicit is not None:
            return float(explicit)
        return float(getattr(self._process, "dividend_yield", 0.0))

    def _volatility_for(self, instrument: UnifiedInstrument) -> float:
        explicit = getattr(instrument, "volatility", None)
        if explicit is None:
            explicit = getattr(instrument, "sigma", None)
        if explicit is None:
            explicit = getattr(self._process, "sigma", None)
        if explicit is None:
            raise ValidationError("American/Bermudan LCP route requires volatility/sigma")
        sigma = float(explicit)
        if sigma <= 0.0 or not np.isfinite(sigma):
            raise ValidationError(
                "American/Bermudan LCP volatility must be finite and positive"
            )
        return sigma


class ADISolverWrapper(Solver):
    """Wrapper for ADI solvers to conform to the unified :class:`Solver` interface.

    Supported dimensions are 2D/3D. The wrapper validates shape consistency and
    covariance definiteness before dispatching to the underlying ADI implementation.
    """

    def __init__(self, adi_solver, *, process: StochasticProcess) -> None:
        """Initialize wrapper with ADI solver and selected process.

        Parameters
        ----------
        adi_solver
            Concrete ADI implementation.
        process
            Multi-dimensional stochastic process providing drift and covariance.
        """
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
            raise ValidationError(
                f"ADI wrapper supports only 2D and 3D processes, got {dimension}D"
            )
        if len(grids) != dimension:
            raise ValidationError(
                f"Expected {dimension} grids for ADI solve, got {len(grids)}"
            )

        drift, covariance, reaction = self._build_process_coefficients(
            float(time_grid[-1]), grids
        )

        if dimension == 2:
            return self._adi_solver.solve_2d(
                initial_condition=initial_condition,
                drift=drift,
                covariance=covariance,
                time_grid=time_grid,
                spatial_grids=grids,
                reaction=reaction,
            )

        return self._adi_solver.solve_3d(
            initial_condition=initial_condition,
            drift=drift,
            covariance=covariance,
            time_grid=time_grid,
            spatial_grids=grids,
            reaction=reaction,
        )

    def _build_process_coefficients(
        self,
        time: float,
        grids: tuple[NDArray[np.float64], ...],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Evaluate drift, covariance, and reaction on the full tensor grid."""
        dimension = self._process.dimension.value
        grid_shape = tuple(len(grid) for grid in grids)
        mesh = np.meshgrid(*grids, indexing="ij")
        states = np.stack([axis.reshape(-1) for axis in mesh], axis=-1)

        drift = np.asarray(self._process.drift(time, states), dtype=float)
        covariance = np.asarray(self._process.covariance(time, states), dtype=float)
        reaction = np.asarray(self._process.discount(time, states), dtype=float)

        expected_drift_shape = (states.shape[0], dimension)
        expected_covariance_shape = (states.shape[0], dimension, dimension)
        expected_reaction_shape = (states.shape[0],)
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
        if reaction.shape != expected_reaction_shape:
            raise ValidationError(
                "process reaction/discount must have shape "
                f"{expected_reaction_shape} on the ADI state grid, got {reaction.shape}"
            )
        if not np.all(np.isfinite(drift)):
            raise ValidationError(
                "process drift contains non-finite values on the ADI state grid"
            )
        if not np.all(np.isfinite(covariance)):
            raise ValidationError(
                "process covariance contains non-finite values on the ADI state grid"
            )
        if not np.all(np.isfinite(reaction)):
            raise ValidationError(
                "process reaction/discount contains non-finite values on the ADI state grid"
            )
        if not np.allclose(
            covariance, np.swapaxes(covariance, -1, -2), rtol=1e-10, atol=1e-12
        ):
            raise ValidationError(
                "process covariance must be symmetric on the ADI state grid"
            )
        min_eigenvalue = float(np.min(np.linalg.eigvalsh(covariance)))
        if min_eigenvalue < -1e-10:
            raise ValidationError(
                "process covariance must be positive semi-definite on the ADI state grid; "
                f"minimum eigenvalue is {min_eigenvalue:.2e}"
            )

        return (
            drift.reshape(*grid_shape, dimension),
            covariance.reshape(*grid_shape, dimension, dimension),
            reaction.reshape(*grid_shape),
        )


class SolverFactory:
    """Factory for creating appropriate solvers.

    Current production recommendation:
    - 1D processes use the finite-difference adapter.
    - 2D/3D processes route to ADI.
    - Any unsupported dimension intentionally fails with a typed validation error.
    """

    @staticmethod
    def create_solver(
        process: StochasticProcess,
        theta: float = 0.5,
        time_stepper: TimeStepper | None = None,
    ) -> Solver:
        """Create an appropriate solver for the supplied process.

        Parameters
        ----------
        process : StochasticProcess
            Stochastic process for the underlying asset(s).
        theta : float, optional
            Implicitness parameter for finite difference methods.
        time_stepper : TimeStepper | None
            Optional time stepper override for the 1D finite-difference adapter.

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
