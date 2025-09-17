"""Construction of spatial differential operators for instruments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import findiff as fd
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - runtime import only for typing
    from src.processes import GeometricBrownianMotion


@dataclass
class SpatialOperator:
    """Build finite difference operators for diffusion models."""

    model: "GeometricBrownianMotion"

    def build(self, s: NDArray[np.float64]) -> fd.FinDiff:
        """Return the discretised infinitesimal generator."""
        ds = s[1] - s[0]
        d1 = fd.FinDiff(0, ds, 1)
        d2 = fd.FinDiff(0, ds, 2)
        m = self.model
        return (
            fd.Coef(0.5 * m.sigma ** 2 * s ** 2) * d2
            + fd.Coef(m.mu * s) * d1
            - m.mu * fd.Identity()
        )
