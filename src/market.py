from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass
class Market:
    """Risk neutral market with constant interest rate."""

    rate: float

    def discount(self, t: float) -> float:
        """Return discount factor for maturity ``t``."""
        return np.exp(-self.rate * t)