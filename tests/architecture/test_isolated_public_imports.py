"""Fresh-interpreter import-order fitness tests for public pricing modules."""

from __future__ import annotations

import subprocess
import sys


def test_solver_base_and_pricing_facade_import_in_fresh_interpreter() -> None:
    """Public modules must not rely on pytest collection order to initialize."""

    code = """
import finite_difference_options.solvers.base
import typing
import finite_difference_options.pricing as pricing
from finite_difference_options.solvers import base as solver_base

typing.get_type_hints(solver_base.Solver.solve)
typing.get_type_hints(solver_base.FiniteDifferenceSolverAdapter.solve)
required = (
    "UnifiedPricingEngine",
    "OptionPricer",
    "UnifiedInstrument",
    "create_unified_pricing_engine",
)
missing = [name for name in required if not hasattr(pricing, name)]
if missing:
    raise SystemExit(f"missing pricing facade exports: {missing}")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
