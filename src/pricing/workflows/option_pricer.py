"""High-level option pricing workflows built on the unified engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple, Optional

import numpy as np
from numpy.typing import NDArray

from src.exceptions import ValidationError
from src.greeks import FiniteDifferenceGreeks
from src.pricing.engines import UnifiedPricingEngine
from src.pricing.instruments.base import UnifiedInstrument
from src.solvers.finite_difference import PDESolver, create_default_solver
from src.solvers.base import Solver


class GridResult(NamedTuple):
    """Named result for pricing grid and Greeks.

    Orientation convention: ``values`` and Greeks are shaped ``(t, s)`` where the
    first axis is time ascending from 0 to maturity, and the second axis is the
    spatial asset-price grid from 0 to ``s_max``.
    """

    s: NDArray[np.float64]
    t: NDArray[np.float64]
    values: NDArray[np.float64]
    delta: Optional[NDArray[np.float64]]
    gamma: Optional[NDArray[np.float64]]
    theta: Optional[NDArray[np.float64]]


class _LegacyInstrumentAdapter(UnifiedInstrument):
    """Adapter exposing legacy instruments through the unified interface."""

    def __init__(self, legacy_instrument):
        self.legacy_instrument = legacy_instrument

    @property
    def maturity(self) -> float:
        return self.legacy_instrument.maturity

    def payoff(self, *grids: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.legacy_instrument.payoff(*grids)

    # Optional hooks used by the legacy solver.
    def generator(self, grid: NDArray[np.float64]):  # pragma: no cover - simple delegation
        return self.legacy_instrument.generator(grid)

    def boundary_conditions(self, grid: NDArray[np.float64]):  # pragma: no cover - simple delegation
        return self.legacy_instrument.boundary_conditions(grid)


class _LegacyFiniteDifferenceSolver(Solver):
    """Bridge between the unified solver interface and the legacy PDE solver."""

    def __init__(self, solver: PDESolver | None = None) -> None:
        self._solver = solver or create_default_solver()

    def solve(
        self,
        initial_condition: NDArray[np.float64],
        instrument: UnifiedInstrument,
        *grids: NDArray[np.float64],
        time_grid: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        if len(grids) != 1:
            raise ValidationError(
                "Legacy finite difference solver only supports one spatial grid"
            )
        if time_grid is None:
            raise ValidationError("time_grid must be provided for legacy solver")

        spatial_grid = grids[0]

        # The adapter exposes the underlying legacy instrument so we can reuse
        # its generator and boundary definitions.
        legacy = getattr(instrument, "legacy_instrument", instrument)

        try:
            generator = legacy.generator(spatial_grid)
            boundary_conditions = legacy.boundary_conditions(spatial_grid)
        except AttributeError as exc:  # pragma: no cover - defensive programming
            raise ValidationError(
                "Legacy instrument must define generator and boundary_conditions"
            ) from exc

        return self._solver.solve(
            generator=generator,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_condition,
            time_grid=time_grid,
        )


@dataclass
class OptionPricer:
    """Compute value grids using finite difference methods."""

    instrument: Any
    _engine: UnifiedPricingEngine = field(init=False)
    _adapter: _LegacyInstrumentAdapter = field(init=False)
    _greeks: FiniteDifferenceGreeks = field(init=False, default_factory=FiniteDifferenceGreeks)

    def __post_init__(self) -> None:
        """Initialise the unified pricing engine from the instrument."""
        process = getattr(self.instrument, "model", None)
        if process is None:
            raise ValidationError("Instrument must define a stochastic model for pricing")

        self._adapter = _LegacyInstrumentAdapter(self.instrument)
        self._engine = UnifiedPricingEngine(
            process=process,
            solver=_LegacyFiniteDifferenceSolver(),
        )

    def compute_grid(
        self,
        *,
        s_max: float,
        s_steps: int,
        t_steps: int,
        return_greeks: bool = False,
    ) -> GridResult:
        """Return grids and values with optional Greeks as a NamedTuple."""

        s = np.linspace(0, s_max, s_steps)
        t = np.linspace(0, self.instrument.maturity, t_steps)

        values = self._engine.price_option(
            self._adapter,
            s,
            time_grid=t,
        )

        if not return_greeks:
            return GridResult(
                s=s,
                t=t,
                values=values,
                delta=None,
                gamma=None,
                theta=None,
            )

        delta = self._greeks.delta(values, s)
        gamma = self._greeks.gamma(values, s)
        theta = self._greeks.theta(values, t)
        return GridResult(s=s, t=t, values=values, delta=delta, gamma=gamma, theta=theta)
