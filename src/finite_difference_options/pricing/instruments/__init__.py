"""Financial instruments package.

This package contains implementations of various financial instruments
for the unified pricing framework.
"""

# Base classes
from .base import UnifiedInstrument

# Option instruments
from .options import (
    SpreadOption,
    StandardBasketOption,
    UnifiedEuropeanOption,
    UnifiedBasketOption,
    create_spread_call,
    create_spread_put,
    create_standard_basket_call,
    create_standard_basket_put,
    create_unified_european_call,
    create_unified_european_put,
    create_unified_basket_call,
    create_unified_basket_put,
)

# Payoff calculators
from .payoff_calculators import (
    PayoffCalculator,
    EuropeanPayoffCalculator,
    BasketPayoffCalculator,
    PayoffCalculatorFactory,
)

__all__ = [
    # Base classes
    "UnifiedInstrument",
    # Option instruments
    "SpreadOption",
    "StandardBasketOption",
    "UnifiedEuropeanOption",
    "UnifiedBasketOption",
    "create_spread_call",
    "create_spread_put",
    "create_standard_basket_call",
    "create_standard_basket_put",
    "create_unified_european_call",
    "create_unified_european_put",
    "create_unified_basket_call",
    "create_unified_basket_put",
    # Payoff calculators
    "PayoffCalculator",
    "EuropeanPayoffCalculator",
    "BasketPayoffCalculator",
    "PayoffCalculatorFactory",
]
