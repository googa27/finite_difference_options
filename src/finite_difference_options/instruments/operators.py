"""Construction of spatial differential operators for instruments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import findiff as fd
import numpy as np
from numpy.typing import NDArray

from finite_difference_options.exceptions import ValidationError

if TYPE_CHECKING:  # pragma: no cover - runtime import only for typing
    from finite_difference_options.processes import GeometricBrownianMotion


@dataclass
class SpatialOperator:
    """Build finite-difference operators for one-factor diffusion models.

    ``discount_rate`` is an explicit reaction coefficient.  Leaving it as
    ``None`` preserves the repository's legacy Black-Scholes API by falling back
    to ``model.mu`` only for one-factor GBM-style callers.  New governed routes
    should pass an explicit rate and may therefore choose drift and discount
    independently.
    """

    model: "GeometricBrownianMotion"
    discount_rate: float | None = None

    def build(self, s: NDArray[np.float64]) -> fd.FinDiff:
        """Return ``0.5 sigma^2 S^2 d2 + mu S d1 - r I`` on the grid."""

        grid = np.asarray(s, dtype=float)
        if grid.ndim != 1 or len(grid) < 3:
            raise ValidationError(
                "SpatialOperator requires a one-dimensional grid with at least 3 nodes"
            )
        if not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0):
            raise ValidationError(
                "SpatialOperator grid must be finite and strictly increasing"
            )
        spacing = np.diff(grid)
        ds = float(spacing[0])
        d1 = fd.FinDiff(0, ds, 1)
        d2 = fd.FinDiff(0, ds, 2)
        m = self.model
        discount = self._resolved_discount_rate()
        return (
            fd.Coef(0.5 * m.sigma**2 * grid**2) * d2
            + fd.Coef(m.mu * grid) * d1
            - discount * fd.Identity()
        )

    def _resolved_discount_rate(self) -> float:
        if self.discount_rate is not None:
            rate = float(self.discount_rate)
            source = "explicit discount_rate"
        else:
            model_rate = getattr(self.model, "risk_free_rate", None)
            if model_rate is not None:
                rate = float(model_rate)
                source = "model.risk_free_rate"
            elif getattr(self.model, "mu", None) is not None:
                rate = float(getattr(self.model, "mu"))
                source = "legacy model.mu"
            else:
                rate = 0.0
                source = "zero default"
        if not np.isfinite(rate):
            raise ValidationError(f"{source} must be finite")
        return rate
