"""Construction of spatial differential operators."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
import findiff as fd

if TYPE_CHECKING:  # pragma: no cover - runtime import only for typing
    from .models import GeometricBrownianMotion


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
            fd.Coef(m.diffusion * s ** 2) * d2
            + fd.Coef(m.rate * s) * d1
            - m.rate * fd.Identity()
        )
