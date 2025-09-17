"""Market data models."""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel


class Market(BaseModel):
    """Risk-neutral market description with a constant interest rate."""

    rate: float

    def discount(self, t: float) -> float:
        """Return the discount factor for a maturity ``t`` in years."""

        return np.exp(-self.rate * t)


__all__ = ["Market"]
