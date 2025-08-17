"""Option contract definitions."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class EuropeanOption(ABC):
    """Base class for European options."""

    strike: float

    @abstractmethod
    def payoff(self, s: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        """Return the payoff at expiry for underlying prices ``s``."""
        raise NotImplementedError


@dataclass
class EuropeanCall(EuropeanOption):
    """European call option."""

    def payoff(self, s: np.ndarray) -> np.ndarray:
        return np.maximum(s - self.strike, 0.0)


@dataclass
class EuropeanPut(EuropeanOption):
    """European put option."""

    def payoff(self, s: np.ndarray) -> np.ndarray:
        return np.maximum(self.strike - s, 0.0)
