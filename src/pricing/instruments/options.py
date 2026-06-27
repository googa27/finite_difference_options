"""Option instruments for the unified pricing framework.

The module exposes dataclass-style option objects compatible with the unified
engine and small helpers for constructing common products.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator

from .base import UnifiedInstrument
from .payoff_calculators import PayoffCalculatorFactory
from ...validation import validate_positive
from ...utils.process_validators import validate_weights_sum_to_one
from src.exceptions import ValidationError


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

        if len(self.strikes) != len(self.weights):
            raise ValidationError("strikes and weights must have same length")

        validate_weights_sum_to_one(self.weights)

    def payoff(self, *grids: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute basket payoff from the supplied state grid(s)."""
        calculator = PayoffCalculatorFactory.create_calculator(self)
        return calculator.calculate_payoff(self, *grids)


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
) -> UnifiedBasketOption:
    """Create a basket call with explicit per-asset strikes/weights."""
    return UnifiedBasketOption(
        strikes=strikes,
        weights=weights,
        maturity=maturity,
        option_type="call",
    )


def create_unified_basket_put(
    strikes: NDArray[np.float64],
    weights: NDArray[np.float64],
    maturity: float,
) -> UnifiedBasketOption:
    """Create a basket put with explicit per-asset strikes/weights."""
    return UnifiedBasketOption(
        strikes=strikes,
        weights=weights,
        maturity=maturity,
        option_type="put",
    )
