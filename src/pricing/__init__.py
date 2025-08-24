"""Unified pricing package.

This package contains the unified pricing framework for financial instruments.
"""
from .engines.unified import UnifiedPricingEngine, create_unified_pricing_engine, create_log_grid, create_linear_grid

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
    "UnifiedPricingEngine",
    "create_unified_pricing_engine",
    "create_log_grid",
    "create_linear_grid",
    
    # Backward compatibility
    "UnifiedInstrument",
    "UnifiedEuropeanOption",
    "UnifiedBasketOption",
    "create_unified_european_call",
    "create_unified_european_put",
    "create_unified_basket_call",
]