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
from typing import Any, NamedTuple, Optional

import numpy as np
from findiff import BoundaryConditions, FinDiff
from numpy.typing import NDArray

from finite_difference_options.exceptions import PricingError
from finite_difference_options.instruments.base import EuropeanOption, Instrument
from finite_difference_options.models import Market
from finite_difference_options.solvers.finite_difference import (
    FiniteDifferenceSolver,
    PDESolver,
    ThetaMethod,
    TimeStepper,
    create_default_solver,
)
from finite_difference_options.validation import (
    validate_grid_parameters,
    validate_spot_price,
)


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
            if abs(result.spatial_grid[idx - 1] - spot_price) < abs(result.spatial_grid[idx] - spot_price):
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
    def payoff(self, s: NDArray[np.float64], option: Optional[EuropeanOption]) -> NDArray[np.float64]:
        """Return payoff at maturity for the spatial grid."""

    @abstractmethod
    def boundary_conditions(self, s: NDArray[np.float64], option: Optional[EuropeanOption]) -> BoundaryConditions:
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

        values = solver.solve(
            generator=generator,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            time_grid=t,
        )
        self.last_step_schedule = solver.last_step_schedule
        return values


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

    def payoff(self, s: NDArray[np.float64], option: Optional[EuropeanOption] = None) -> NDArray[np.float64]:
        return self.instrument.payoff(s)

    def boundary_conditions(
        self, s: NDArray[np.float64], option: Optional[EuropeanOption] = None
    ) -> BoundaryConditions:
        return self.instrument.boundary_conditions(s)


@dataclass(frozen=True)
class BondCashFlow:
    """Contractual fixed-income cash flow in valuation-year units."""

    time: float
    amount: float
    label: str = "cash_flow"

    def __post_init__(self) -> None:
        if not np.isfinite(self.time) or self.time < 0.0:
            raise PricingError("cash-flow time must be finite and non-negative")
        if not np.isfinite(self.amount):
            raise PricingError("cash-flow amount must be finite")


@dataclass(frozen=True)
class CallScheduleEntry:
    """Issuer exercise right on one contractual call date."""

    time: float
    price: float
    quote_convention: str = "dirty"
    label: str = "call"

    def __post_init__(self) -> None:
        if not np.isfinite(self.time) or self.time < 0.0:
            raise PricingError("call time must be finite and non-negative")
        if not np.isfinite(self.price) or self.price < 0.0:
            raise PricingError("call price must be finite and non-negative")
        convention = self.quote_convention.lower()
        if convention not in {"clean", "dirty"}:
            raise PricingError("call quote_convention must be 'clean' or 'dirty'")
        object.__setattr__(self, "quote_convention", convention)


@dataclass(frozen=True)
class CallableBondExerciseRecord:
    """Diagnostics for one realised issuer exercise projection."""

    time: float
    call_price: float
    settlement_value: float
    exercised_nodes: int
    continuation_min: float
    continuation_max: float


@dataclass
class CallableBondPDEModel(PDEModel):
    """Schedule-aware one-factor callable fixed-rate bond model.

    The legacy implementation applied ``np.minimum(values, call_price)`` to the
    whole grid after solving, which silently treated every time slice as
    callable.  This implementation uses an explicit cash-flow and call schedule
    and applies issuer exercise only when the backward-induction clock crosses a
    contractual call date.  The executable route is a short-rate-state reference
    induction with local discounting; the generic ``generator`` path fails
    closed so the old global-cap route cannot be selected accidentally.
    """

    face_value: float
    call_price: float | None
    market: Market
    model: Any
    _maturity: float
    coupon_rate: float = 0.0
    coupon_times: tuple[float, ...] = ()
    call_schedule: tuple[CallScheduleEntry | tuple[float, float], ...] = ()
    redemption_amount: float | None = None
    settlement_time: float = 0.0
    day_count: str = "ACT/365F"
    currency: str = "synthetic"
    curve_id: str = "flat_market_rate"
    time_stepper: TimeStepper = field(default_factory=lambda: ThetaMethod(0.5))
    cash_flows: tuple[BondCashFlow, ...] = field(init=False)
    normalized_call_schedule: tuple[CallScheduleEntry, ...] = field(init=False)
    last_exercise_diagnostics: tuple[CallableBondExerciseRecord, ...] = field(init=False, default=())

    def __post_init__(self) -> None:
        self._validate_contract_scalars()
        object.__setattr__(self, "cash_flows", self._build_cash_flows())
        object.__setattr__(
            self,
            "normalized_call_schedule",
            self._normalise_call_schedule(),
        )

    def _validate_contract_scalars(self) -> None:
        if not np.isfinite(self.face_value) or self.face_value <= 0.0:
            raise PricingError("face_value must be finite and positive")
        if self.call_price is not None and (not np.isfinite(self.call_price) or self.call_price < 0.0):
            raise PricingError("call_price must be finite and non-negative when set")
        if not np.isfinite(self._maturity) or self._maturity <= 0.0:
            raise PricingError("maturity must be finite and positive")
        if not np.isfinite(self.coupon_rate) or self.coupon_rate < 0.0:
            raise PricingError("coupon_rate must be finite and non-negative")
        if self.coupon_rate > 0.0 and not self.coupon_times:
            raise PricingError("coupon_times are required for coupon-bearing bonds")
        if not np.isfinite(self.settlement_time) or self.settlement_time < 0.0:
            raise PricingError("settlement_time must be finite and non-negative")
        if self.settlement_time > self._maturity:
            raise PricingError("settlement_time cannot exceed maturity")

    def _build_cash_flows(self) -> tuple[BondCashFlow, ...]:
        redemption = self.face_value if self.redemption_amount is None else self.redemption_amount
        if not np.isfinite(redemption) or redemption < 0.0:
            raise PricingError("redemption_amount must be finite and non-negative")

        coupon_times = self._sorted_unique_times(self.coupon_times, "coupon")
        flows: list[BondCashFlow] = []
        previous = self.settlement_time
        for index, payment_time in enumerate(coupon_times, start=1):
            year_fraction = payment_time - previous
            if year_fraction <= 0.0:
                raise PricingError("coupon_times must be strictly after settlement_time")
            coupon_amount = self.face_value * self.coupon_rate * year_fraction
            if coupon_amount:
                flows.append(BondCashFlow(payment_time, coupon_amount, f"coupon_{index}"))
            previous = payment_time

        flows.append(BondCashFlow(self._maturity, redemption, "redemption"))
        return tuple(sorted(flows, key=lambda item: (item.time, item.label)))

    def _normalise_call_schedule(self) -> tuple[CallScheduleEntry, ...]:
        entries: list[CallScheduleEntry] = []
        for raw in self.call_schedule:
            if isinstance(raw, CallScheduleEntry):
                entry = raw
            else:
                time, price = raw
                entry = CallScheduleEntry(time=float(time), price=float(price))
            if entry.time < self.settlement_time or entry.time > self._maturity:
                raise PricingError("call dates must lie between settlement and maturity")
            entries.append(entry)

        if self.call_price is not None and not entries:
            raise PricingError("call_price alone is not a callable-bond contract; provide explicit call_schedule")
        if entries:
            self._sorted_unique_times((entry.time for entry in entries), "call")
        return tuple(sorted(entries, key=lambda item: item.time))

    def _sorted_unique_times(self, times: Any, label: str) -> tuple[float, ...]:
        cleaned = tuple(float(item) for item in times)
        previous: float | None = None
        for item in cleaned:
            if not np.isfinite(item) or item < self.settlement_time or item > self._maturity:
                raise PricingError(f"{label} times must be finite and lie between settlement and maturity")
            if previous is not None and item <= previous:
                raise PricingError(f"{label} times must be strictly increasing")
            previous = item
        return cleaned

    @property
    def maturity(self) -> float:
        return self._maturity

    @property
    def pricing_horizon(self) -> float:
        """Return remaining time from settlement to maturity."""

        return self._maturity - self.settlement_time

    @property
    def strike(self) -> Optional[float]:
        return None

    @property
    def call_dates(self) -> tuple[float, ...]:
        """Return contractual issuer exercise dates."""

        return tuple(entry.time for entry in self.normalized_call_schedule)

    def accrued_interest(self, time: float) -> float:
        """Return simple linear accrued coupon at ``time`` for clean calls."""

        if self.coupon_rate == 0.0:
            return 0.0
        coupon_times = self._sorted_unique_times(self.coupon_times, "coupon")
        previous = self.settlement_time
        for payment_time in coupon_times:
            if np.isclose(time, payment_time):
                return 0.0
            if time < payment_time:
                return self.face_value * self.coupon_rate * max(time - previous, 0.0)
            previous = payment_time
        return 0.0

    def generator(self, s: NDArray[np.float64]) -> FinDiff:
        raise PricingError(
            "CallableBondPDEModel uses explicit cash-flow/call-date backward induction; "
            "the legacy global-cap PDE generator is intentionally unavailable"
        )

    def payoff(self, s: NDArray[np.float64], option: Optional[EuropeanOption] = None) -> NDArray[np.float64]:
        terminal_amount = sum(flow.amount for flow in self.cash_flows if np.isclose(flow.time, self._maturity))
        return np.full_like(np.asarray(s, dtype=float), terminal_amount, dtype=float)

    def boundary_conditions(
        self, s: NDArray[np.float64], option: Optional[EuropeanOption] = None
    ) -> BoundaryConditions:
        raise PricingError("CallableBondPDEModel boundaries are schedule/model dependent; use price_grid")

    def price_grid(
        self,
        s: NDArray[np.float64],
        t: NDArray[np.float64],
        *,
        include_calls: bool = True,
    ) -> NDArray[np.float64]:
        """Return value grid with exercise applied only on call dates.

        The spatial grid is interpreted as short-rate states.  ``t`` is the
        repository's legacy time-to-maturity grid: index 0 corresponds to the
        maturity payoff and the last row corresponds to valuation time.
        """

        rate_grid = self._validate_rate_grid(s)
        tau_grid = self._validate_time_grid(t)
        values, records = self._backward_induction_grid(rate_grid, tau_grid, include_calls=include_calls)
        self.last_exercise_diagnostics = records
        return values

    def straight_bond_grid(self, s: NDArray[np.float64], t: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the otherwise identical noncallable fixed-rate bond grid."""

        return self.price_grid(s, t, include_calls=False)

    def price(
        self,
        option: Optional[EuropeanOption] | NDArray[np.float64] = None,
        s: NDArray[np.float64] | None = None,
        t: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Return the schedule-aware callable bond value grid.

        Supports both the legacy ``price(option, s, t)`` abstract signature and
        the historical callable-bond shortcut ``price(s, t)``.
        """

        if t is None and isinstance(option, np.ndarray) and isinstance(s, np.ndarray):
            return self.price_grid(option, s)
        if s is None or t is None:
            raise PricingError("CallableBondPDEModel.price requires spatial and time grids")
        return self.price_grid(s, t)

    def _validate_rate_grid(self, s: NDArray[np.float64]) -> NDArray[np.float64]:
        grid = np.asarray(s, dtype=float)
        if grid.ndim != 1 or len(grid) < 2:
            raise PricingError("callable bond rate grid must be one-dimensional")
        if not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0):
            raise PricingError("callable bond rate grid must be finite and increasing")
        return grid

    def _validate_time_grid(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        grid = np.asarray(t, dtype=float)
        if grid.ndim != 1 or len(grid) < 2:
            raise PricingError("callable bond time grid must be one-dimensional")
        if not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0):
            raise PricingError("callable bond time grid must be finite and increasing")
        if not np.isclose(grid[0], 0.0):
            raise PricingError("callable bond time grid must start at maturity tau=0")
        if not np.isclose(grid[-1], self.pricing_horizon):
            raise PricingError("callable bond time grid must end at remaining maturity")
        return grid

    def _backward_induction_grid(
        self,
        rate_grid: NDArray[np.float64],
        tau_grid: NDArray[np.float64],
        *,
        include_calls: bool,
    ) -> tuple[NDArray[np.float64], tuple[CallableBondExerciseRecord, ...]]:
        values = np.empty((len(tau_grid), len(rate_grid)))
        current = self.payoff(rate_grid)
        records: list[CallableBondExerciseRecord] = []
        if include_calls:
            current = self._apply_calls(current, self._maturity, records)
        values[0] = current

        for index in range(len(tau_grid) - 1):
            start_tau = tau_grid[index]
            end_tau = tau_grid[index + 1]
            current_tau = start_tau
            next_value = values[index].copy()
            for event_tau in self._event_taus_between(start_tau, end_tau, include_calls):
                next_value *= np.exp(-rate_grid * (event_tau - current_tau))
                event_time = self._maturity - event_tau
                if include_calls:
                    next_value = self._apply_calls(next_value, event_time, records)
                next_value = self._apply_cash_flows(next_value, event_time)
                current_tau = event_tau
            next_value *= np.exp(-rate_grid * (end_tau - current_tau))
            values[index + 1] = next_value
        return values, tuple(records)

    def _event_taus_between(self, start_tau: float, end_tau: float, include_calls: bool) -> tuple[float, ...]:
        event_taus: list[float] = []
        for flow in self.cash_flows:
            tau = self._maturity - flow.time
            if start_tau < tau <= end_tau and not np.isclose(tau, 0.0):
                event_taus.append(tau)
        if include_calls:
            for entry in self.normalized_call_schedule:
                tau = self._maturity - entry.time
                if start_tau < tau <= end_tau:
                    event_taus.append(tau)
        return tuple(sorted(set(event_taus)))

    def _apply_cash_flows(self, values: NDArray[np.float64], event_time: float) -> NDArray[np.float64]:
        amount = sum(flow.amount for flow in self.cash_flows if np.isclose(flow.time, event_time))
        if amount == 0.0:
            return values
        return values + amount

    def _apply_calls(
        self,
        values: NDArray[np.float64],
        event_time: float,
        records: list[CallableBondExerciseRecord],
    ) -> NDArray[np.float64]:
        next_values = values
        for entry in self.normalized_call_schedule:
            if not np.isclose(entry.time, event_time):
                continue
            settlement = entry.price
            if entry.quote_convention == "clean":
                settlement += self.accrued_interest(event_time)
            continuation = next_values.copy()
            exercised = continuation > settlement
            next_values = np.minimum(continuation, settlement)
            records.append(
                CallableBondExerciseRecord(
                    time=entry.time,
                    call_price=entry.price,
                    settlement_value=float(settlement),
                    exercised_nodes=int(np.count_nonzero(exercised)),
                    continuation_min=float(np.min(continuation)),
                    continuation_max=float(np.max(continuation)),
                )
            )
        return next_values


__all__ = [
    "BondCashFlow",
    "CallScheduleEntry",
    "CallableBondExerciseRecord",
    "CallableBondPDEModel",
    "GridParameters",
    "PDEModel",
    "PricingEngine",
    "PricingResult",
    "BlackScholesPDE",
    "create_default_pricing_engine",
]
