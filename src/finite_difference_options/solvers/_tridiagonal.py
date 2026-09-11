"""Immutable factor-once, solve-many adapter for SciPy's tridiagonal LAPACK."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg.lapack import dgttrf, dgttrs

from finite_difference_options.exceptions import ValidationError

Array = NDArray[np.float64]


@dataclass(frozen=True)
class TridiagonalLU:
    """Pivoted LU factors for private systems with at least three rows.

    Input diagonals use the FD row-aligned convention: lower[0] and upper[-1]
    are padding. LAPACK receives only lower[1:] and upper[:-1]. Factor arrays
    are owned, read-only snapshots; pivots are passed back to LAPACK unchanged.
    """

    lower: Array
    diagonal: Array
    upper: Array
    second_upper: Array
    pivots: NDArray[np.intc]

    @classmethod
    def factor(cls, lower: Array, diagonal: Array, upper: Array) -> TridiagonalLU:
        lower = _float_array(lower, "lower diagonal")
        diagonal = _float_array(diagonal, "diagonal")
        upper = _float_array(upper, "upper diagonal")
        if diagonal.ndim != 1 or len(diagonal) < 3 or lower.shape != diagonal.shape or upper.shape != diagonal.shape:
            raise ValidationError("tridiagonal diagonals must be equal-length vectors with at least three rows")
        if not all(np.all(np.isfinite(array)) for array in (lower, diagonal, upper)):
            raise ValidationError("tridiagonal coefficients must be finite")
        dl, d, du, du2, pivots, info = dgttrf(
            lower[1:].copy(),
            diagonal.copy(),
            upper[:-1].copy(),
            overwrite_dl=1,
            overwrite_d=1,
            overwrite_du=1,
        )
        if info < 0:
            raise ValidationError(f"LAPACK tridiagonal factorization rejected argument {-info}")
        if info > 0:
            raise ValidationError(f"singular tridiagonal pivot at row {info - 1}")
        if not all(np.all(np.isfinite(array)) for array in (dl, d, du, du2)):
            raise ValidationError("tridiagonal factorization must be finite")
        small_pivots = np.flatnonzero(np.abs(d) <= 1.0e-14)
        if small_pivots.size:
            raise ValidationError(f"singular tridiagonal pivot at row {int(small_pivots[0])}")
        for array in (dl, d, du, du2, pivots):
            array.setflags(write=False)
        return cls(dl, d, du, du2, pivots)

    def solve(self, rhs: Array) -> Array:
        values = _float_array(rhs, "RHS")
        if values.ndim not in (1, 2) or values.shape[0] != len(self.diagonal):
            raise ValidationError("tridiagonal RHS must have shape (rows,) or (rows, columns)")
        if not np.all(np.isfinite(values)):
            raise ValidationError("tridiagonal RHS must be finite")
        if values.ndim == 2 and values.shape[1] == 0:
            # Avoid native wrappers that mishandle LAPACK's zero-RHS quick return.
            return values.copy()
        vector = values.ndim == 1
        matrix = np.array(values[:, None] if vector else values, dtype=np.float64, order="F", copy=True)
        solved, info = dgttrs(
            self.lower,
            self.diagonal,
            self.upper,
            self.second_upper,
            self.pivots,
            matrix,
            overwrite_b=1,
        )
        if info != 0:
            raise ValidationError(f"LAPACK tridiagonal solve failed with status {info}")
        solution = np.asarray(solved, dtype=np.float64)
        if not np.all(np.isfinite(solution)):
            raise ValidationError("tridiagonal solution must be finite")
        return solution[:, 0] if vector else solution


def _float_array(value: Array, name: str) -> Array:
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"tridiagonal {name} must be a real numeric array") from exc
