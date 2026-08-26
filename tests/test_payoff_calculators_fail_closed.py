"""Fail-closed tests for direct payoff calculator use."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from finite_difference_options.exceptions import ValidationError
from finite_difference_options.pricing.instruments.payoff_calculators import (
    BasketPayoffCalculator,
    EuropeanPayoffCalculator,
)


def test_european_payoff_calculator_rejects_unsupported_option_type() -> None:
    instrument: Any = SimpleNamespace(strike=100.0, option_type="digital")
    grid = np.array([90.0, 100.0, 110.0])

    with pytest.raises(ValidationError, match="option_type must be 'call' or 'put'"):
        EuropeanPayoffCalculator().calculate_payoff(instrument, grid)


def test_basket_payoff_calculator_rejects_unsupported_option_type() -> None:
    instrument: Any = SimpleNamespace(strike=100.0, weights=np.array([0.6, 0.4]), option_type="digital")
    first_grid = np.array([90.0, 110.0])
    second_grid = np.array([95.0, 105.0])

    with pytest.raises(ValidationError, match="option_type must be 'call' or 'put'"):
        BasketPayoffCalculator().calculate_payoff(instrument, first_grid, second_grid)
