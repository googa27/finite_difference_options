"""Financial instruments package.

This package contains implementations of various financial instruments
for the unified pricing framework.
"""

# Base classes
from .base import UnifiedInstrument

# Option instruments
from .options import (
    UnifiedEuropeanOption,
    UnifiedBasketOption,
    create_unified_european_call,
    create_unified_european_put,
    create_unified_basket_call,
    create_unified_basket_put
)

# Payoff calculators
from .payoff_calculators import (
    PayoffCalculator,
    EuropeanPayoffCalculator,
    BasketPayoffCalculator,
    PayoffCalculatorFactory
)

__all__ = [
    # Base classes
    'UnifiedInstrument',
    
    # Option instruments
    'UnifiedEuropeanOption',
    'UnifiedBasketOption',
    'create_unified_european_call',
    'create_unified_european_put',
    'create_unified_basket_call',
    'create_unified_basket_put',
    
    # Payoff calculators
    'PayoffCalculator',
    'EuropeanPayoffCalculator',
    'BasketPayoffCalculator',
    'PayoffCalculatorFactory'
]