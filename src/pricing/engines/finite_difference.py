"""Finite difference pricing engines and legacy PDE models.

This module contains both the modern pricing engine used throughout the
unified codebase and the legacy ``PDEModel`` abstractions that older callers
still depend on. Keeping these implementations co-located simplifies import
paths and avoids duplicating functionality while ensuring backward
compatibility.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import numpy as np
from findiff import BoundaryConditions, FinDiff
from numpy.typing import NDArray

from src.exceptions import PricingError
from src.instruments.base import EuropeanOption, Instrument
from src.market import Market
from src.solvers.finite_difference import (
    FiniteDifferenceSolver,
    PDESolver,
    ThetaMethod,
    TimeStepper,
    create_default_solver,
)
from src.processes.affine import GeometricBrownianMotion
from src.spatial_operator import SpatialOperator
from src.validation import validate_grid_parameters, validate_spot_price


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
            grid_params.t_steps,
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
        except Exception as exc:  # pragma: no cover - defensive programming
            raise PricingError(f"Failed to price instrument: {exc}") from exc

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
            if abs(result.spatial_grid[idx - 1] - spot_price) < abs(
                result.spatial_grid[idx] - spot_price
            ):
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

    return PricingEngine(solver=create_default_solver())


# ---------------------------------------------------------------------------
# Legacy PDE abstractions
# ---------------------------------------------------------------------------


class PDEModel(ABC):
    """Abstract base class for PDE pricing models."""

    time_stepper: TimeStepper

    @abstractmethod
    def generator(self, s: NDArray[np.float64]) -> FinDiff:
        """Return the discretised generator on the spatial grid."""

    @abstractmethod
    def payoff(
        self, s: NDArray[np.float64], option: Optional[EuropeanOption]
    ) -> NDArray[np.float64]:
        """Return payoff at maturity for the spatial grid."""

    @abstractmethod
    def boundary_conditions(
        self, s: NDArray[np.float64], option: Optional[EuropeanOption]
    ) -> BoundaryConditions:
        """Return boundary conditions for the spatial grid."""

    def price(
        self,
        option: Optional[EuropeanOption],
        s: NDArray[np.float64],
        t: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return grid with instrument values."""

        solver = FiniteDifferenceSolver(time_stepper=self.time_stepper)

        generator = self.generator(s)
        boundary_conditions = self.boundary_conditions(s, option)
        initial_conditions = self.payoff(s, option)

        return solver.solve(
            generator=generator,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            time_grid=t,
        )


@dataclass
class BlackScholesPDE(PDEModel):
    """Price European options by solving the Black--Scholes PDE."""

    instrument: Instrument
    theta: float = 0.5  # retained for backward compatibility
    time_stepper: TimeStepper | None = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.time_stepper is None:
            self.time_stepper = ThetaMethod(self.theta)
        else:  # keep ``theta`` in sync for backward compatibility
            self.theta = getattr(self.time_stepper, "theta", self.theta)

    def generator(self, s: NDArray[np.float64]) -> FinDiff:
        return self.instrument.generator(s)

    def payoff(
        self, s: NDArray[np.float64], option: Optional[EuropeanOption] = None
    ) -> NDArray[np.float64]:
        return self.instrument.payoff(s)

    def boundary_conditions(
        self, s: NDArray[np.float64], option: Optional[EuropeanOption] = None
    ) -> BoundaryConditions:
        return self.instrument.boundary_conditions(s)


@dataclass
class CallableBondPDEModel(PDEModel):
    """PDE model for pricing simple callable bonds."""

    face_value: float
    call_price: float
    market: Market
    model: GeometricBrownianMotion
    _maturity: float
    time_stepper: TimeStepper = field(default_factory=lambda: ThetaMethod(0.5))

    @property
    def maturity(self) -> float:
        return self._maturity

    @property
    def strike(self) -> Optional[float]:
        return None

    def generator(self, s: NDArray[np.float64]) -> FinDiff:
        return SpatialOperator(self.model).build(s)

    def payoff(
        self, s: NDArray[np.float64], option: Optional[EuropeanOption] = None
    ) -> NDArray[np.float64]:
        return np.full_like(s, min(self.face_value, self.call_price))

    def boundary_conditions(
        self, s: NDArray[np.float64], option: Optional[EuropeanOption] = None
    ) -> BoundaryConditions:
        bc = BoundaryConditions(s.shape)
        bc[0] = 0.0
        bc[-1] = self.call_price
        return bc

    def price(
        self,
        s: NDArray[np.float64],
        t: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return grid of callable bond prices respecting the call cap."""
        solver = FiniteDifferenceSolver(time_stepper=self.time_stepper)

        generator = self.generator(s)
        boundary_conditions = self.boundary_conditions(s)
        initial_conditions = self.payoff(s)

        values = solver.solve(
            generator=generator,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            time_grid=t,
        )

        return np.minimum(values, self.call_price)


__all__ = [
    "CallableBondPDEModel",
    "GridParameters",
    "PDEModel",
    "PricingEngine",
    "PricingResult",
    "BlackScholesPDE",
    "create_default_pricing_engine",
]
