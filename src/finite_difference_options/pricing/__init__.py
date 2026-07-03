"""Unified pricing package.

This package contains the unified pricing framework for financial instruments.
"""

from .engines import (
    BlackScholesPDE,
    BondCashFlow,
    CallableBondExerciseRecord,
    CallableBondPDEModel,
    CallScheduleEntry,
    GridParameters,
    PDEModel,
    PricingEngine,
    PricingResult,
    UnifiedPricingEngine,
    create_default_pricing_engine,
    create_linear_grid,
    create_log_grid,
    create_unified_pricing_engine,
)
from .workflows import GridResult, OptionPricer

# Backward compatibility imports
from .instruments.base import UnifiedInstrument
from .instruments.options import (
    SpreadOption,
    StandardBasketOption,
    UnifiedEuropeanOption,
    UnifiedAmericanOption,
    UnifiedBermudanOption,
    UnifiedBasketOption,
    create_spread_call,
    create_spread_put,
    create_standard_basket_call,
    create_standard_basket_put,
    create_unified_european_call,
    create_unified_european_put,
    create_unified_american_call,
    create_unified_american_put,
    create_unified_bermudan_call,
    create_unified_bermudan_put,
    create_unified_basket_call,
)

__all__ = [
    "GridParameters",
    "PDEModel",
    "PricingEngine",
    "PricingResult",
    "UnifiedPricingEngine",
    "create_default_pricing_engine",
    "create_linear_grid",
    "create_log_grid",
    "create_unified_pricing_engine",
    "GridResult",
    "OptionPricer",
    # Backward compatibility
    "UnifiedInstrument",
    "SpreadOption",
    "StandardBasketOption",
    "UnifiedEuropeanOption",
    "UnifiedAmericanOption",
    "UnifiedBermudanOption",
    "UnifiedBasketOption",
    "create_spread_call",
    "create_spread_put",
    "create_standard_basket_call",
    "create_standard_basket_put",
    "create_unified_european_call",
    "create_unified_european_put",
    "create_unified_american_call",
    "create_unified_american_put",
    "create_unified_bermudan_call",
    "create_unified_bermudan_put",
    "create_unified_basket_call",
    "BlackScholesPDE",
    "BondCashFlow",
    "CallScheduleEntry",
    "CallableBondExerciseRecord",
    "CallableBondPDEModel",
]
