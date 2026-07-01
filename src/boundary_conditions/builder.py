"""Utilities for constructing typed, model-aware boundary conditions.

The boundary layer is intentionally separate from solver code: solvers consume
already resolved facet conditions and must not infer option or model semantics
from class names at solve time.  The concrete findiff adapter remains here so
legacy callers can still obtain ``BoundaryConditions`` objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Any, Literal

import findiff as fd
import numpy as np
from findiff import BoundaryConditions, FinDiff
from numpy.typing import NDArray

from src.exceptions import BoundaryConditionError

BoundaryKind = Literal[
    "dirichlet", "neumann", "second_derivative", "degenerate", "extrapolated"
]
BoundarySide = Literal["lower", "upper"]


@dataclass(frozen=True)
class BoundarySpec:
    """Resolved finite-difference condition on one grid facet."""

    axis: str
    side: BoundarySide
    kind: BoundaryKind
    value: float | str
    expression: str
    time_to_maturity: float
    coordinate: str = "spot"
    model: str = "black_scholes"

    def as_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "side": self.side,
            "kind": self.kind,
            "value": self.value,
            "expression": self.expression,
            "time_to_maturity": self.time_to_maturity,
            "coordinate": self.coordinate,
            "model": self.model,
        }


@dataclass(frozen=True)
class BoundaryResolution:
    """Boundary set plus provenance for solver evidence."""

    specs: tuple[BoundarySpec, ...]
    risk_free_rate: float
    dividend_yield: float
    discount_source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "risk_free_rate": self.risk_free_rate,
            "dividend_yield": self.dividend_yield,
            "discount_source": self.discount_source,
            "specs": [spec.as_dict() for spec in self.specs],
        }


@dataclass
class BlackScholesBoundaryBuilder:
    """Build model-aware one-dimensional Black--Scholes boundaries.

    For backward compatibility, a legacy ``GeometricBrownianMotion(mu=...)``
    instrument may still use ``mu`` as a rate when no explicit rate is supplied.
    New solver routes should pass ``risk_free_rate`` explicitly so discount is not
    inferred from drift.
    """

    allow_legacy_mu_rate: bool = True

    def resolve(
        self,
        s: NDArray[np.float64],
        option: Any,
        *,
        time_to_maturity: float | None = None,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
    ) -> BoundaryResolution:
        grid = np.asarray(s, dtype=float)
        self._validate_grid(grid)
        tau = self._time_to_maturity(option, time_to_maturity)
        option_type = self._option_type(option)
        strike = self._strike(option)
        rate, rate_source = self._risk_free_rate(option, risk_free_rate)
        carry = self._dividend_yield(option, dividend_yield)

        if option_type == "call":
            lower_value = 0.0
            upper_value = max(
                float(grid[-1]) * exp(-carry * tau) - strike * exp(-rate * tau), 0.0
            )
            lower_expr = "V(0,tau)=0 for a vanilla call"
            upper_expr = "V(Smax,tau)=Smax*exp(-q*tau)-K*exp(-r*tau)"
        elif option_type == "put":
            lower_value = strike * exp(-rate * tau)
            upper_value = 0.0
            lower_expr = "V(0,tau)=K*exp(-r*tau) for a vanilla put"
            upper_expr = "V(Smax,tau)=0 for a far-out-of-the-money put"
        else:  # pragma: no cover - _option_type already fails closed
            raise BoundaryConditionError(
                f"unsupported Black-Scholes option type {option_type!r}"
            )

        specs = (
            BoundarySpec(
                axis="S",
                side="lower",
                kind="dirichlet",
                value=float(lower_value),
                expression=lower_expr,
                time_to_maturity=tau,
            ),
            BoundarySpec(
                axis="S",
                side="upper",
                kind="dirichlet",
                value=float(upper_value),
                expression=upper_expr,
                time_to_maturity=tau,
            ),
        )
        return BoundaryResolution(
            specs=specs,
            risk_free_rate=rate,
            dividend_yield=carry,
            discount_source=rate_source,
        )

    def build(
        self,
        s: NDArray[np.float64],
        option: Any,
        *,
        time_to_maturity: float | None = None,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
    ) -> BoundaryConditions:
        """Return a findiff ``BoundaryConditions`` object for resolved specs."""

        resolution = self.resolve(
            s,
            option,
            time_to_maturity=time_to_maturity,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
        )
        bc = BoundaryConditions(np.asarray(s, dtype=float).shape)
        identity = fd.Identity()
        bc[0] = identity, float(resolution.specs[0].value)
        bc[-1] = identity, float(resolution.specs[1].value)
        return bc

    @staticmethod
    def _validate_grid(s: NDArray[np.float64]) -> None:
        if s.ndim != 1 or len(s) < 3:
            raise BoundaryConditionError(
                "Black-Scholes boundaries require a one-dimensional grid with at least 3 nodes"
            )
        if not np.all(np.isfinite(s)):
            raise BoundaryConditionError("spatial grid contains non-finite values")
        if np.any(np.diff(s) <= 0.0):
            raise BoundaryConditionError("spatial grid must be strictly increasing")
        if s[0] < 0.0:
            raise BoundaryConditionError(
                "spot grid lower boundary must be non-negative"
            )

    @staticmethod
    def _time_to_maturity(option: Any, override: float | None) -> float:
        value = getattr(option, "maturity", None) if override is None else override
        if value is None:
            raise BoundaryConditionError(
                "time_to_maturity or option.maturity is required"
            )
        tau = float(value)
        if tau < 0.0 or not np.isfinite(tau):
            raise BoundaryConditionError(
                "time_to_maturity must be finite and non-negative"
            )
        return tau

    @staticmethod
    def _strike(option: Any) -> float:
        value = getattr(option, "strike", None)
        if value is None:
            raise BoundaryConditionError(
                "vanilla boundary construction requires option.strike"
            )
        strike = float(value)
        if strike <= 0.0 or not np.isfinite(strike):
            raise BoundaryConditionError("option.strike must be finite and positive")
        return strike

    @staticmethod
    def _option_type(option: Any) -> str:
        explicit = getattr(option, "option_type", None)
        if explicit is not None:
            value = str(explicit).lower()
        else:
            name = type(option).__name__.lower()
            if "call" in name:
                value = "call"
            elif "put" in name:
                value = "put"
            else:
                value = ""
        if value not in {"call", "put"}:
            raise BoundaryConditionError(
                "only vanilla call/put Black-Scholes boundaries are supported"
            )
        return value

    def _risk_free_rate(self, option: Any, override: float | None) -> tuple[float, str]:
        if override is not None:
            return self._finite_rate(override, "explicit"), "explicit"
        explicit = getattr(option, "risk_free_rate", None)
        if explicit is None:
            explicit = getattr(option, "discount_rate", None)
        if explicit is not None:
            return self._finite_rate(explicit, "instrument"), "instrument"

        model = getattr(option, "model", None)
        model_rate = getattr(model, "risk_free_rate", None) if model is not None else None
        if model_rate is not None:
            return (
                self._finite_rate(model_rate, "model.risk_free_rate"),
                "model.risk_free_rate",
            )
        if (
            self.allow_legacy_mu_rate
            and model is not None
            and getattr(model, "mu", None) is not None
        ):
            return (
                self._finite_rate(getattr(model, "mu"), "legacy model.mu"),
                "legacy model.mu",
            )
        raise BoundaryConditionError(
            "risk_free_rate must be supplied explicitly for vanilla boundaries"
        )

    @staticmethod
    def _dividend_yield(option: Any, override: float | None) -> float:
        if override is not None:
            value = override
        else:
            value = getattr(option, "dividend_yield", None)
            if value is None:
                model = getattr(option, "model", None)
                value = getattr(model, "dividend_yield", 0.0)
        carry = float(value)
        if carry < 0.0 or not np.isfinite(carry):
            raise BoundaryConditionError(
                "dividend_yield must be finite and non-negative"
            )
        return carry

    @staticmethod
    def _finite_rate(value: float, source: str) -> float:
        rate = float(value)
        if not np.isfinite(rate):
            raise BoundaryConditionError(f"{source} risk-free rate must be finite")
        return rate


@dataclass
class HestonBoundaryBuilder:
    """Resolve Heston state-coordinate boundary semantics for solver provenance."""

    def resolve(
        self,
        log_spot_grid: NDArray[np.float64],
        variance_grid: NDArray[np.float64],
        option: Any,
        *,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
        time_to_maturity: float | None = None,
    ) -> BoundaryResolution:
        x = np.asarray(log_spot_grid, dtype=float)
        v = np.asarray(variance_grid, dtype=float)
        BlackScholesBoundaryBuilder._validate_grid(np.exp(x))
        if v.ndim != 1 or len(v) < 2 or np.any(np.diff(v) <= 0.0) or v[0] < 0.0:
            raise BoundaryConditionError(
                "Heston variance grid must be one-dimensional, increasing, and non-negative"
            )
        tau = BlackScholesBoundaryBuilder._time_to_maturity(option, time_to_maturity)
        strike = BlackScholesBoundaryBuilder._strike(option)
        option_type = BlackScholesBoundaryBuilder._option_type(option)
        rate = BlackScholesBoundaryBuilder._finite_rate(risk_free_rate, "explicit")
        carry = BlackScholesBoundaryBuilder._dividend_yield(option, dividend_yield)
        s_min = float(np.exp(x[0]))
        s_max = float(np.exp(x[-1]))

        lower_spot = 0.0 if option_type == "call" else strike * exp(-rate * tau)
        upper_spot = (
            max(s_max * exp(-carry * tau) - strike * exp(-rate * tau), 0.0)
            if option_type == "call"
            else 0.0
        )
        return BoundaryResolution(
            risk_free_rate=rate,
            dividend_yield=carry,
            discount_source="explicit",
            specs=(
                BoundarySpec(
                    "x",
                    "lower",
                    "dirichlet",
                    lower_spot,
                    "spot boundary after exp(log_spot) transform",
                    tau,
                    "log_spot",
                    "heston",
                ),
                BoundarySpec(
                    "x",
                    "upper",
                    "dirichlet",
                    upper_spot,
                    "far spot Black-Scholes asymptotic in log coordinate",
                    tau,
                    "log_spot",
                    "heston",
                ),
                BoundarySpec(
                    "v",
                    "lower",
                    "degenerate",
                    "drop v-second derivative at variance=0",
                    "CIR/Heston variance degeneracy facet",
                    tau,
                    "variance",
                    "heston",
                ),
                BoundarySpec(
                    "v",
                    "upper",
                    "extrapolated",
                    "zero normal variance-gradient",
                    "far-variance extrapolation facet",
                    tau,
                    "variance",
                    "heston",
                ),
            ),
        )
