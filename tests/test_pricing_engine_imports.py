"""Tests covering the finite difference pricing engine import surface."""
import pathlib
import sys

# Ensure project root on path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.pricing import (
    GridParameters,
    PricingEngine,
    PricingResult,
    create_default_pricing_engine,
)
from src.pricing.engines import (
    GridParameters as GridParametersFromPackage,
    PricingEngine as PricingEngineFromPackage,
    PricingResult as PricingResultFromPackage,
    create_default_pricing_engine as create_default_pricing_engine_from_package,
    finite_difference,
)


def test_symbols_reexported_from_pricing_package() -> None:
    """The pricing package should expose the finite difference engine API."""
    assert PricingEngine is finite_difference.PricingEngine
    assert PricingEngine is PricingEngineFromPackage
    assert GridParameters is finite_difference.GridParameters
    assert GridParameters is GridParametersFromPackage
    assert PricingResult is finite_difference.PricingResult
    assert PricingResult is PricingResultFromPackage
    assert (
        create_default_pricing_engine
        is create_default_pricing_engine_from_package
    )


def test_create_default_pricing_engine_returns_engine() -> None:
    """Factory helper should yield a configured pricing engine instance."""
    engine = create_default_pricing_engine()
    assert isinstance(engine, PricingEngine)
