"""Finite difference models for solving parabolic PDEs.

This module defines an abstract :class:`PDEModel` base class together with
concrete implementations for vanilla options and callable bonds.  The models
encapsulate the construction of the infinitesimal generator and the boundary
conditions required by the finite difference solver.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from findiff import BoundaryConditions, FinDiff

from .models import GeometricBrownianMotion, Market
from .options import EuropeanOption
from .boundary_conditions import BlackScholesBoundaryBuilder
from .spatial_operator import SpatialOperator
from .time_steppers import TimeStepper, ThetaMethod


class PDEModel(ABC):
    """Abstract base class for PDE pricing models."""

    time_stepper: TimeStepper

    @abstractmethod
    def generator(self, s: NDArray[np.float64]) -> FinDiff:
        """Return the discretised generator on the spatial grid."""

    @abstractmethod
    def payoff(
        self, s: NDArray[np.float64], option: EuropeanOption | None
    ) -> NDArray[np.float64]:
        """Return payoff at maturity for the spatial grid."""

    @abstractmethod
    def boundary_conditions(
        self, s: NDArray[np.float64], option: EuropeanOption | None
    ) -> BoundaryConditions:
        """Return boundary conditions for the spatial grid."""

    def price(
        self,
        option: EuropeanOption | None,
        s: NDArray[np.float64],
        t: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return grid with instrument values."""

        dt = t[1] - t[0]
        L = self.generator(s)

        values = np.empty((len(t), len(s)))
        values[0] = self.payoff(s, option)

        bc = self.boundary_conditions(s, option)
        for i in range(len(t) - 1):
            values[i + 1] = self.time_stepper.step(values[i], L, bc, dt)
        return values


@dataclass
class BlackScholesPDE(PDEModel):
    """Price European options by solving the Black–Scholes PDE."""

    model: GeometricBrownianMotion
    market: Market
    theta: float = 0.5  # retained for backward compatibility
    boundary_builder: BlackScholesBoundaryBuilder = field(
        default_factory=BlackScholesBoundaryBuilder
    )
    time_stepper: TimeStepper | None = None

    def __post_init__(self) -> None:
        """Initialise default time stepper if not supplied."""
        if self.time_stepper is None:
            self.time_stepper = ThetaMethod(self.theta)
        else:  # keep ``theta`` in sync for backward compatibility
            self.theta = getattr(self.time_stepper, "theta", self.theta)

    def generator(self, s: NDArray[np.float64]) -> SpatialOperator:
        """Return the Black--Scholes infinitesimal generator."""

        return SpatialOperator(self.model).build(s)

    def payoff(
        self, s: NDArray[np.float64], option: EuropeanOption | None
    ) -> NDArray[np.float64]:
        """Return option payoff at maturity."""

        if option is None:
            raise ValueError("Option contract must be provided for BlackScholesPDE")
        return option.payoff(s)

    def boundary_conditions(
        self, s: NDArray[np.float64], option: EuropeanOption | None
    ) -> BoundaryConditions:
        """Return model-specific boundary conditions."""

        if option is None:
            raise ValueError("Option contract must be provided for BlackScholesPDE")
        return self.boundary_builder.build(s, option)


@dataclass
class CallableBondPDEModel(PDEModel):
    """PDE model for pricing simple callable bonds.

    The implementation is intentionally simplified: the underlying short rate
    follows a geometric Brownian motion and the bond may be called at a fixed
    price at any time.
    """

    face_value: float
    call_price: float
    market: Market
    model: GeometricBrownianMotion
    time_stepper: TimeStepper = field(default_factory=lambda: ThetaMethod(0.5))

    def generator(self, s: NDArray[np.float64]) -> FinDiff:
        """Return the infinitesimal generator for the short rate."""

        return SpatialOperator(self.model).build(s)

    def payoff(
        self, s: NDArray[np.float64], option: EuropeanOption | None
    ) -> NDArray[np.float64]:
        """Face value paid at maturity."""

        return np.full_like(s, min(self.face_value, self.call_price))

    def boundary_conditions(
        self, s: NDArray[np.float64], option: EuropeanOption | None
    ) -> BoundaryConditions:
        """Dirichlet boundaries enforcing the call price."""

        bc = BoundaryConditions(s.shape)
        # Value is zero when the bond price approaches zero
        bc[0] = 0, 0.0
        # Bond cannot exceed the call price
        bc[-1] = 0, self.call_price
        return bc

    def price(
        self,
        option: EuropeanOption | None,
        s: NDArray[np.float64],
        t: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return grid of callable bond prices.

        After each time step the value is capped at the call price to emulate
        the early redemption feature.
        """

        dt = t[1] - t[0]
        L = self.generator(s)

        values = np.empty((len(t), len(s)))
        values[0] = self.payoff(s, option)

        bc = self.boundary_conditions(s, option)
        for i in range(len(t) - 1):
            values[i + 1] = self.time_stepper.step(values[i], L, bc, dt)
            values[i + 1] = np.minimum(values[i + 1], self.call_price)
        values = np.minimum(values, self.call_price)
        return values

