from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray
from findiff import FinDiff, BoundaryConditions


class Instrument:
    """Abstract base class for financial instruments that can be priced by PDE."""

    @abstractmethod
    def payoff(self, s: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the payoff at expiry for underlying prices ``s``."""
        ...

    @abstractmethod
    def boundary_conditions(
        self, s: NDArray[np.float64]
    ) -> BoundaryConditions:
        """Return boundary conditions for the spatial grid."""
        ...

    @abstractmethod
    def generator(self, s: NDArray[np.float64]) -> FinDiff:
        """Return the discretised infinitesimal generator for the instrument."""
        ...