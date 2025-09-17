"""Unified pricing package.

This package contains the unified pricing framework for financial instruments.
"""
from .engines import (
    GridParameters,
    PricingEngine,
    PricingResult,
    UnifiedPricingEngine,
    create_default_pricing_engine,
    create_linear_grid,
    create_log_grid,
    create_unified_pricing_engine,
)

# Backward compatibility imports
from .instruments.base import UnifiedInstrument
from .instruments.options import (
    UnifiedEuropeanOption,
    UnifiedBasketOption,
    create_unified_european_call,
    create_unified_european_put,
    create_unified_basket_call,
)

__all__ = [
    "GridParameters",
    "PricingEngine",
    "PricingResult",
    "UnifiedPricingEngine",
    "create_default_pricing_engine",
    "create_linear_grid",
    "create_log_grid",
    "create_unified_pricing_engine",

    # Backward compatibility
    "UnifiedInstrument",
    "UnifiedEuropeanOption",
    "UnifiedBasketOption",
    "create_unified_european_call",
    "create_unified_european_put",
    "create_unified_basket_call",
]
