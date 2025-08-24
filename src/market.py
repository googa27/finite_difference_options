"""Market model for risk neutral pricing."""
from __future__ import annotations

from pydantic import BaseModel
import numpy as np


class Market(BaseModel):
    """Risk neutral market with constant interest rate."""

    rate: float

    def discount(self, t: float) -> float:
        """Return discount factor for maturity ``t``."""
        return np.exp(-self.rate * t)