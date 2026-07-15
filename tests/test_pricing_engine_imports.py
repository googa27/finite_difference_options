"""Tests covering the finite difference pricing engine import surface."""

from __future__ import annotations

import importlib

from finite_difference_options.pricing import (
    GridParameters,
    PricingEngine,
    PricingResult,
    create_default_pricing_engine,
)
from finite_difference_options.pricing.engines import (
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
    assert create_default_pricing_engine is create_default_pricing_engine_from_package


def test_create_default_pricing_engine_returns_engine() -> None:
    """Factory helper should yield a configured pricing engine instance."""
    engine = create_default_pricing_engine()
    assert isinstance(engine, PricingEngine)


def test_historical_pricing_boundary_conditions_package_remains_importable() -> None:
    """Preserve the historical marker path while the API lives at the canonical path."""
    assert importlib.import_module("finite_difference_options.boundary_conditions")
    assert importlib.import_module("finite_difference_options.pricing.boundary_conditions")
