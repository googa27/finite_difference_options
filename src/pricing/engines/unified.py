"""Unified pricing engine for multi-dimensional option pricing.

The unified engine handles one- to three-dimensional processes via a single
interface and normalises time/space input before dispatching to the selected
solver.

Current production guidance
--------------------------

- Univariate problems route to the finite-difference adapter.
- Multi-dimensional problems route to ADI implementations where available.
- Solver output is in calendar-time order where index 0 is valuation time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Iterable, cast

import numpy as np
from numpy.typing import NDArray

from ...processes.base import FactorRole, StochasticProcess
from ...pricing.instruments.base import UnifiedInstrument
from src.exceptions import ValidationError
from ...solvers.base import Solver, SolverFactory


@dataclass
class UnifiedPricingEngine:
    """Unified pricing engine for multi-dimensional processes.

    The engine performs lightweight input normalisation and delegates numerical
    work to a concrete solver from :mod:`src.solvers.base`.

    Notes
    -----
    The returned price arrays are in calendar-time order:

    - index 0: valuation time (time 0)
    - index -1: maturity/terminal time

    Use ``create_unified_pricing_engine`` to obtain a default engine configured
    with the auto-selected solver.

    Parameters
    ----------
    process : StochasticProcess
        Stochastic process for the underlying asset(s).
    solver : Optional[Solver]
        PDE solver to use (defaults to auto-selection by process dimension).
    """

    process: StochasticProcess
    solver: Optional[Solver] = None

    def __post_init__(self) -> None:
        """Initialise engine and auto-select the solver if missing."""
        if self.solver is None:
            self.solver = SolverFactory.create_solver(self.process)

    def price_option(
        self,
        instrument: UnifiedInstrument,
        *grids: NDArray[np.float64],
        time_grid: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        """Price an option via the configured solver.

        Examples
        --------
        Price a Black--Scholes call with a default grid:

        >>> from src.processes.affine import create_black_scholes_process
        >>> from src.pricing.engines.unified import (
        ...     create_unified_pricing_engine,
        ...     create_log_grid,
        ... )
        >>> from src.pricing.instruments.options import create_unified_european_call
        >>> process = create_black_scholes_process(mu=0.03, sigma=0.2)
        >>> engine = create_unified_pricing_engine(process)
        >>> option = create_unified_european_call(strike=100.0, maturity=1.0)
        >>> grid = create_log_grid(20.0, 200.0, 51)
        >>> times = np.linspace(0, 1.0, 8)
        >>> value_slice = engine.price_option(option, grid, time_grid=times)[-1]

        Parameters
        ----------
        instrument : UnifiedInstrument
            Option contract to price.
        *grids : NDArray[np.float64]
            One spatial grid per process dimension.
        time_grid : NDArray[np.float64], optional
            Monotone calendar-time grid on ``[0, maturity]``.

        Returns
        -------
        NDArray[np.float64]
            Prices on all time slices (calendar-time order).
        """
        if len(grids) == 0:
            raise ValidationError("At least one spatial grid required")

        time_grid = self._normalise_time_grid(time_grid, instrument.maturity)

        if len(grids) != self.process.dimension.value:
            raise ValidationError(
                f"Expected {self.process.dimension.value} grids, got {len(grids)}"
            )

        self._validate_factor_compatibility(instrument)

        if len(grids) == 1:
            initial_condition = instrument.payoff(grids[0])
        else:
            mesh_grids = np.meshgrid(*grids, indexing="ij")
            flattened_grids = [grid.flatten() for grid in mesh_grids]
            grid_shape = mesh_grids[0].shape
            initial_condition = instrument.payoff(*flattened_grids).reshape(grid_shape)

        solution = self.solver.solve(
            initial_condition,
            instrument,
            *grids,
            time_grid=time_grid,
        )

        return solution

    def _validate_factor_compatibility(self, instrument: UnifiedInstrument) -> None:
        """Validate instrument payoff dependencies against process factor roles."""

        required_roles_getter = getattr(instrument, "required_factor_roles", None)
        if not callable(required_roles_getter):
            return
        required_roles = tuple(cast(Iterable[FactorRole], required_roles_getter()))
        if not required_roles:
            return

        factors = tuple(self.process.factor_metadata())
        if len(required_roles) > len(factors):
            raise ValidationError(
                f"{type(instrument).__name__} requires {len(required_roles)} factors, "
                f"but {type(self.process).__name__} exposes {len(factors)}"
            )
        for index, required_role in enumerate(required_roles):
            actual = factors[index]
            if actual.role != required_role:
                required_label = required_role.value.replace("_", " ")
                actual_label = actual.role.value.replace("_", " ")
                raise ValidationError(
                    f"{type(instrument).__name__} requires factor {index} to be {required_label}, "
                    f"but {type(self.process).__name__} factor {index} ({actual.name}) is {actual_label}. "
                    "A basket payoff cannot consume variance or other non-tradable factors by accident."
                )

    def compute_greeks(
        self,
        prices: NDArray[np.float64],
        *grids: NDArray[np.float64],
        time_grid: Optional[NDArray[np.float64]] = None,
    ) -> Dict[str, NDArray[np.float64]]:
        """Compute option Greeks from a precomputed price surface.

        Parameters
        ----------
        prices : NDArray[np.float64]
            Option prices on the corresponding spatial/time grid.
        *grids : NDArray[np.float64]
            Spatial grids used for pricing.
        time_grid : NDArray[np.float64], optional
            Calendar-time grid. If omitted, defaults to the default solver grid.

        Returns
        -------
        Dict[str, NDArray[np.float64]]
            Mapping with common Greeks ``delta``, ``gamma``, ``theta`` etc.
        """
        from ...greeks.base import GreeksCalculatorFactory

        calculator = GreeksCalculatorFactory.create_calculator(self.process)
        return calculator.calculate(prices, *grids, time_grid=time_grid)

    @staticmethod
    def _normalise_time_grid(
        time_grid: Optional[NDArray[np.float64]],
        maturity: float,
    ) -> NDArray[np.float64]:
        """Normalise an optional user-supplied time grid.

        The helper enforces

        - one-dimensional input,
        - finite values,
        - strict monotonicity,
        - boundaries ``0`` and ``maturity``.
        """
        if time_grid is None or len(time_grid) == 0:
            return np.linspace(0.0, maturity, 50)

        normalised = np.asarray(time_grid, dtype=np.float64)
        if normalised.ndim != 1:
            raise ValidationError("time_grid must be one-dimensional")
        if len(normalised) < 2:
            raise ValidationError("time_grid must contain at least two points")
        if not np.all(np.isfinite(normalised)):
            raise ValidationError("time_grid must contain only finite values")
        if not np.all(np.diff(normalised) > 0):
            raise ValidationError("time_grid must be strictly increasing")
        if not np.isclose(normalised[0], 0.0, rtol=0.0, atol=1e-12) or not np.isclose(
            normalised[-1], maturity, rtol=1e-12, atol=1e-12
        ):
            raise ValidationError("time_grid must span [0, maturity]")
        return normalised


# Convenience functions

def create_unified_pricing_engine(process: StochasticProcess) -> UnifiedPricingEngine:
    """Create a unified pricing engine with auto-selected solver."""

    return UnifiedPricingEngine(process=process)


def create_log_grid(
    s_min: float, s_max: float, n_points: int, center: Optional[float] = None
) -> NDArray[np.float64]:
    """Create a logarithmically spaced positive grid.

    Parameters
    ----------
    s_min : float
        Lower grid bound, must be positive.
    s_max : float
        Upper grid bound, greater than ``s_min``.
    n_points : int
        Number of grid points.
    center : float, optional
        Optional central point inserted at the midpoint index.

    Returns
    -------
    NDArray[np.float64]
        Monotone log-spaced grid.
    """
    if s_min <= 0:
        raise ValidationError("s_min must be positive")
    if s_max <= s_min:
        raise ValidationError("s_max must be greater than s_min")
    if n_points < 2:
        raise ValidationError("n_points must be at least 2")

    if center is None:
        log_min = np.log(s_min)
        log_max = np.log(s_max)
        log_grid = np.linspace(log_min, log_max, n_points)
        grid = np.exp(log_grid)
    else:
        if not s_min < center < s_max:
            raise ValidationError("center must lie strictly between s_min and s_max")

        center_idx = n_points // 2
        left = np.exp(np.linspace(np.log(s_min), np.log(center), center_idx + 1))
        right = np.exp(np.linspace(np.log(center), np.log(s_max), n_points - center_idx))
        grid = np.concatenate([left[:-1], right])

    grid[0] = s_min
    grid[-1] = s_max
    return grid


def create_linear_grid(x_min: float, x_max: float, n_points: int) -> NDArray[np.float64]:
    """Create a linearly spaced grid.

    Parameters
    ----------
    x_min : float
        Lower bound.
    x_max : float
        Upper bound.
    n_points : int
        Number of points.

    Returns
    -------
    NDArray[np.float64]
        Uniformly spaced values between ``x_min`` and ``x_max``.
    """
    if n_points < 2:
        raise ValidationError("n_points must be at least 2")
    if x_max <= x_min:
        raise ValidationError("x_max must be greater than x_min")
    return np.linspace(x_min, x_max, n_points)
