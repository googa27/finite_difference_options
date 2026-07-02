"""Option instruments for the unified pricing framework.

The module exposes dataclass-style option objects compatible with the unified
engine and small helpers for constructing common products.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator

from .base import UnifiedInstrument
from .payoff_calculators import PayoffCalculatorFactory
from ...processes.base import FactorRole
from ...validation import validate_positive
from ...utils.process_validators import validate_weights_sum_to_one
from finite_difference_options.exceptions import ValidationError


class UnifiedEuropeanOption(UnifiedInstrument, BaseModel):
    """Plain-vanilla European option.

    Parameters
    ----------
    strike : float
        Strike level, must be strictly positive.
    maturity : float
        Time to maturity in year fraction.
    option_type : str
        Either ``"call"`` or ``"put"``.

    See Also
    --------
    create_unified_european_call : convenience constructor for call options.
    create_unified_european_put : convenience constructor for put options.
    """

    strike: float
    maturity: float
    option_type: str = "call"  # Default to 'call'

    model_config = ConfigDict(frozen=True, extra="forbid")

    @field_validator("strike")
    @classmethod
    def validate_strike(cls, v: float) -> float:
        """Validate strike price and require strictly positive values."""
        validate_positive(v, "strike")
        return v

    @field_validator("maturity")
    @classmethod
    def validate_maturity(cls, v: float) -> float:
        """Validate maturity is strictly positive."""
        validate_positive(v, "maturity")
        return v

    @field_validator("option_type")
    @classmethod
    def validate_option_type(cls, v: str) -> str:
        """Validate option type and normalise invalid values."""
        if v not in ["call", "put"]:
            raise ValidationError(f"option_type must be 'call' or 'put', got {v}")
        return v

    def payoff(self, *grids: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute European payoff for one or more spot grids.

        The call/put payoff is evaluated at the terminal grid via
        :class:`PayoffCalculatorFactory`.
        """
        calculator = PayoffCalculatorFactory.create_calculator(self)
        return calculator.calculate_payoff(self, *grids)

    def required_factor_roles(self) -> tuple[FactorRole, ...]:
        """European vanilla options consume one tradable spot factor."""

        return (FactorRole.TRADABLE_SPOT,)


class UnifiedBasketOption(UnifiedInstrument, BaseModel):
    """Basket option payoff on a weighted combination of underlying states.

    Parameters
    ----------
    strikes : array-like
        Per-asset strike values.
    weights : array-like
        Portfolio weights (must sum to one).
    maturity : float
        Time to maturity.
    option_type : str
        Either ``"call"`` or ``"put"``.
    """

    strikes: Any  # Using Any to avoid Pydantic issues with NDArray
    weights: Any  # Using Any to avoid Pydantic issues with NDArray
    maturity: float
    option_type: str = "call"  # Default to 'call'
    asset_ids: tuple[str, ...] | None = None
    product_type: Literal["leg_strike_basket"] = "leg_strike_basket"

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    @field_validator("maturity")
    @classmethod
    def validate_maturity(cls, v: float) -> float:
        """Validate maturity is strictly positive."""
        validate_positive(v, "maturity")
        return v

    @field_validator("strikes")
    @classmethod
    def validate_strikes(cls, v: NDArray[np.float64]) -> NDArray[np.float64]:
        """Validate all strikes are strictly positive."""
        v = np.asarray(v, dtype=np.float64)
        if not np.all(v > 0):
            raise ValidationError("All strikes must be positive")
        return v

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, v: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert weights to ndarray (additional checks done in ``__init__``)."""
        return np.asarray(v, dtype=np.float64)

    @field_validator("option_type")
    @classmethod
    def validate_option_type(cls, v: str) -> str:
        """Validate option type."""
        if v not in ["call", "put"]:
            raise ValidationError(f"option_type must be 'call' or 'put', got {v}")
        return v

    def __init__(self, **data):
        """Initialise and perform cross-field checks.

        Raises
        ------
        ValidationError
            If strike/weight dimensions mismatch or weights do not sum to one.
        """
        super().__init__(**data)

        if self.strikes.ndim != 1:
            raise ValidationError("strikes must be one-dimensional")
        if self.weights.ndim != 1:
            raise ValidationError("weights must be one-dimensional")
        if len(self.strikes) == 0:
            raise ValidationError("strikes and weights must be nonempty")
        if len(self.strikes) != len(self.weights):
            raise ValidationError("strikes and weights must have same length")
        if not np.all(np.isfinite(self.strikes)) or not np.all(
            np.isfinite(self.weights)
        ):
            raise ValidationError("strikes and weights must contain only finite values")
        if self.asset_ids is not None and len(self.asset_ids) != len(self.weights):
            raise ValidationError("asset_ids must have the same length as weights")

        validate_weights_sum_to_one(self.weights)

    def payoff(self, *grids: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute basket payoff from the supplied state grid(s)."""
        calculator = PayoffCalculatorFactory.create_calculator(self)
        return calculator.calculate_payoff(self, *grids)

    def required_factor_roles(self) -> tuple[FactorRole, ...]:
        """Legacy per-leg basket options require every leg to be tradable."""

        return tuple(FactorRole.TRADABLE_SPOT for _ in self.weights)

    def required_asset_ids(self) -> tuple[str | None, ...]:
        """Return expected asset identifiers for payoff legs, if supplied."""

        if self.asset_ids is None:
            return tuple(None for _ in self.weights)
        return self.asset_ids


class StandardBasketOption(UnifiedInstrument, BaseModel):
    """Standard basket option with one basket strike.

    ``UnifiedBasketOption`` is retained as the legacy per-leg-strike product.
    This contract represents the common payoff
    ``max(sum_i w_i S_i - K, 0)`` for calls and the reverse for puts.
    """

    strike: float
    weights: Any
    maturity: float
    option_type: str = "call"
    asset_ids: tuple[str, ...] | None = None
    basket_currency: str | None = None
    asset_currencies: tuple[str, ...] | None = None
    unit: str = "price"
    carry_convention: Literal["none", "external"] = "none"
    conversion_policy: Literal["single_currency_only"] = "single_currency_only"
    product_type: Literal["standard_basket"] = "standard_basket"

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    @field_validator("strike")
    @classmethod
    def validate_strike(cls, v: float) -> float:
        """Validate basket strike."""
        validate_positive(v, "strike")
        return v

    @field_validator("maturity")
    @classmethod
    def validate_maturity(cls, v: float) -> float:
        """Validate maturity is strictly positive."""
        validate_positive(v, "maturity")
        return v

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, v: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert weights to a dense floating array."""
        return np.asarray(v, dtype=np.float64)

    @field_validator("option_type")
    @classmethod
    def validate_option_type(cls, v: str) -> str:
        """Validate option type."""
        if v not in ["call", "put"]:
            raise ValidationError(f"option_type must be 'call' or 'put', got {v}")
        return v

    def __init__(self, **data):
        """Initialise and perform basket cross-field validation."""
        super().__init__(**data)

        if self.weights.ndim != 1:
            raise ValidationError("weights must be one-dimensional")
        if len(self.weights) == 0:
            raise ValidationError("weights must be nonempty")
        if not np.all(np.isfinite(self.weights)):
            raise ValidationError("weights must contain only finite values")
        validate_weights_sum_to_one(self.weights)
        if self.asset_ids is not None and len(self.asset_ids) != len(self.weights):
            raise ValidationError("asset_ids must have the same length as weights")
        if self.asset_currencies is not None and len(self.asset_currencies) != len(
            self.weights
        ):
            raise ValidationError(
                "asset_currencies must have the same length as weights"
            )
        if self.asset_currencies is not None and self.basket_currency is not None:
            mismatches = [
                currency
                for currency in self.asset_currencies
                if currency != self.basket_currency
            ]
            if mismatches:
                raise ValidationError(
                    "cross-currency basket conversion is unsupported; provide single-currency inputs "
                    "or a dedicated conversion adapter"
                )

    def payoff(self, *grids: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the standard one-strike basket payoff."""
        calculator = PayoffCalculatorFactory.create_calculator(self)
        return calculator.calculate_payoff(self, *grids)

    def required_factor_roles(self) -> tuple[FactorRole, ...]:
        """Standard basket legs must map to tradable spot factors."""

        return tuple(FactorRole.TRADABLE_SPOT for _ in self.weights)

    def required_asset_ids(self) -> tuple[str | None, ...]:
        """Return expected asset identifiers for payoff legs, if supplied."""

        if self.asset_ids is None:
            return tuple(None for _ in self.weights)
        return self.asset_ids


class SpreadOption(StandardBasketOption):
    """Two-leg spread option with a single spread strike.

    Spread coefficients are not normalized basket weights; this class keeps a
    separate product identity so spread/notional semantics cannot be confused
    with normalized standard baskets.
    """

    product_type: Literal["spread"] = "spread"

    def __init__(self, **data):
        """Validate two-leg spread shape without requiring weights sum to one."""
        BaseModel.__init__(self, **data)
        if self.weights.ndim != 1:
            raise ValidationError("weights must be one-dimensional")
        if len(self.weights) != 2:
            raise ValidationError("spread options require exactly two weights")
        if not np.all(np.isfinite(self.weights)):
            raise ValidationError("weights must contain only finite values")
        if self.asset_ids is not None and len(self.asset_ids) != 2:
            raise ValidationError("asset_ids must have length 2 for spread options")
        if self.asset_currencies is not None and len(self.asset_currencies) != 2:
            raise ValidationError(
                "asset_currencies must have length 2 for spread options"
            )
        if self.asset_currencies is not None and self.basket_currency is not None:
            mismatches = [
                currency
                for currency in self.asset_currencies
                if currency != self.basket_currency
            ]
            if mismatches:
                raise ValidationError("cross-currency spread conversion is unsupported")


# Convenience functions


def create_unified_european_call(
    strike: float,
    maturity: float,
) -> UnifiedEuropeanOption:
    """Create a plain-vanilla European call."""
    return UnifiedEuropeanOption(strike=strike, maturity=maturity, option_type="call")


def create_unified_european_put(
    strike: float,
    maturity: float,
) -> UnifiedEuropeanOption:
    """Create a plain-vanilla European put."""
    return UnifiedEuropeanOption(strike=strike, maturity=maturity, option_type="put")


def create_unified_basket_call(
    strikes: NDArray[np.float64],
    weights: NDArray[np.float64],
    maturity: float,
    asset_ids: tuple[str, ...] | None = None,
) -> UnifiedBasketOption:
    """Create a legacy leg-strike basket call with explicit per-leg strikes."""
    return UnifiedBasketOption(
        strikes=strikes,
        weights=weights,
        maturity=maturity,
        option_type="call",
        asset_ids=asset_ids,
    )


def create_standard_basket_call(
    strike: float,
    weights: NDArray[np.float64],
    maturity: float,
    asset_ids: tuple[str, ...] | None = None,
) -> StandardBasketOption:
    """Create a standard basket call with one basket strike."""
    return StandardBasketOption(
        strike=strike,
        weights=weights,
        maturity=maturity,
        option_type="call",
        asset_ids=asset_ids,
    )


def create_unified_basket_put(
    strikes: NDArray[np.float64],
    weights: NDArray[np.float64],
    maturity: float,
    asset_ids: tuple[str, ...] | None = None,
) -> UnifiedBasketOption:
    """Create a legacy leg-strike basket put with explicit per-leg strikes."""
    return UnifiedBasketOption(
        strikes=strikes,
        weights=weights,
        maturity=maturity,
        option_type="put",
        asset_ids=asset_ids,
    )


def create_standard_basket_put(
    strike: float,
    weights: NDArray[np.float64],
    maturity: float,
    asset_ids: tuple[str, ...] | None = None,
) -> StandardBasketOption:
    """Create a standard basket put with one basket strike."""
    return StandardBasketOption(
        strike=strike,
        weights=weights,
        maturity=maturity,
        option_type="put",
        asset_ids=asset_ids,
    )


def create_spread_call(
    strike: float,
    weights: NDArray[np.float64],
    maturity: float,
    asset_ids: tuple[str, ...] | None = None,
) -> SpreadOption:
    """Create a two-leg spread call with non-normalized coefficients."""
    return SpreadOption(
        strike=strike,
        weights=weights,
        maturity=maturity,
        option_type="call",
        asset_ids=asset_ids,
    )


def create_spread_put(
    strike: float,
    weights: NDArray[np.float64],
    maturity: float,
    asset_ids: tuple[str, ...] | None = None,
) -> SpreadOption:
    """Create a two-leg spread put with non-normalized coefficients."""
    return SpreadOption(
        strike=strike,
        weights=weights,
        maturity=maturity,
        option_type="put",
        asset_ids=asset_ids,
    )
