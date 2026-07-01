"""ADI (Alternating Direction Implicit) solver for multi-dimensional PDEs.

This module implements a conservative Douglas ADI route for 2D/3D linear
parabolic problems on monotone tensor-product grids. The public solver API uses
calendar-time output order (``solution[0]`` valuation, ``solution[-1]``
terminal/payoff) while the numerical march is performed in forward
``tau = T - t`` time from the terminal condition.

The semi-discrete operator convention is

    u_tau = sum_i A_i u + A_mixed u - reaction * u + source

where each directional component ``A_i`` contains the drift and diagonal
covariance term for one coordinate and ``A_mixed`` contains every off-diagonal
covariance contribution exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from src.exceptions import ValidationError


Array = NDArray[np.float64]


@dataclass
class ADISolver:
    """Douglas ADI solver for 2D/3D parabolic PDEs.

    Parameters
    ----------
    theta
        Douglas implicitness parameter. ``0.5`` is the standard directional
        correction used by the repository's experimental ADI route.
    max_iterations
        Reserved for future iterative ADI/LCP variants; validated here so the
        public constructor remains fail-closed.
    tolerance
        Linear-pivot tolerance for direct tridiagonal line solves.
    scheme
        Currently only ``"douglas"`` is implemented and advertised.
    """

    theta: float = 0.5
    max_iterations: int = 1000
    tolerance: float = 1e-8
    scheme: str = "douglas"
    last_diagnostics: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.theta <= 1.0:
            raise ValidationError(f"theta must be between 0 and 1, got {self.theta}")
        if self.max_iterations <= 0:
            raise ValidationError(
                f"max_iterations must be positive, got {self.max_iterations}"
            )
        if self.tolerance <= 0:
            raise ValidationError(f"tolerance must be positive, got {self.tolerance}")
        if self.scheme != "douglas":
            raise ValidationError(
                f"unsupported ADI scheme {self.scheme!r}; only 'douglas' is implemented"
            )

    def solve_2d(
        self,
        initial_condition: Array,
        drift: Array,
        covariance: Array,
        time_grid: Array,
        spatial_grids: Tuple[Array, Array],
        boundary_conditions: Optional[Dict[str, Any]] = None,
        *,
        reaction: Optional[Array | float] = None,
        source: Optional[Array | float] = None,
    ) -> Array:
        """Solve a 2D forward-time parabolic PDE with calendar-time output.

        ``initial_condition`` is the terminal payoff at ``tau=0``. The returned
        array is reversed into public calendar order so that ``result[-1]`` is
        exactly the supplied terminal/payoff surface.
        """
        grids = self._validate_grids(spatial_grids)
        nt = self._validate_time_grid(time_grid)
        nx, ny = len(grids[0]), len(grids[1])
        self._validate_2d_inputs(drift, covariance, initial_condition, nx, ny)
        reaction_field = self._coerce_field("reaction", reaction, (nx, ny), default=0.0)
        source_field = self._coerce_field("source", source, (nx, ny), default=0.0)

        tau_solution = np.zeros((nt, nx, ny), dtype=float)
        positivity_floor = self._uses_positivity_floor(initial_condition, source_field)
        tau_solution[0] = self._apply_boundary_conditions_2d(
            np.asarray(initial_condition, dtype=float).copy(),
            grids,
            boundary_conditions,
        )

        for step in range(nt - 1):
            dt = float(time_grid[step + 1] - time_grid[step])
            next_solution = self._douglas_step_2d(
                tau_solution[step],
                drift,
                covariance,
                reaction_field,
                source_field,
                grids,
                dt,
                boundary_conditions,
            )
            tau_solution[step + 1] = self._apply_positivity_floor(
                next_solution, enabled=positivity_floor
            )

        self.last_diagnostics = self._diagnostics(
            dimension=2,
            time_grid=time_grid,
            grids=grids,
            positivity_floor=positivity_floor,
        )
        return tau_solution[::-1]

    def solve_3d(
        self,
        initial_condition: Array,
        drift: Array,
        covariance: Array,
        time_grid: Array,
        spatial_grids: Tuple[Array, Array, Array],
        boundary_conditions: Optional[Dict[str, Any]] = None,
        *,
        reaction: Optional[Array | float] = None,
        source: Optional[Array | float] = None,
    ) -> Array:
        """Solve a 3D forward-time parabolic PDE with calendar-time output."""
        grids = self._validate_grids(spatial_grids)
        nt = self._validate_time_grid(time_grid)
        nx, ny, nz = len(grids[0]), len(grids[1]), len(grids[2])
        self._validate_3d_inputs(drift, covariance, initial_condition, nx, ny, nz)
        reaction_field = self._coerce_field(
            "reaction", reaction, (nx, ny, nz), default=0.0
        )
        source_field = self._coerce_field("source", source, (nx, ny, nz), default=0.0)

        tau_solution = np.zeros((nt, nx, ny, nz), dtype=float)
        positivity_floor = self._uses_positivity_floor(initial_condition, source_field)
        tau_solution[0] = self._apply_boundary_conditions_3d(
            np.asarray(initial_condition, dtype=float).copy(),
            grids,
            boundary_conditions,
        )

        for step in range(nt - 1):
            dt = float(time_grid[step + 1] - time_grid[step])
            next_solution = self._douglas_step_3d(
                tau_solution[step],
                drift,
                covariance,
                reaction_field,
                source_field,
                grids,
                dt,
                boundary_conditions,
            )
            tau_solution[step + 1] = self._apply_positivity_floor(
                next_solution, enabled=positivity_floor
            )

        self.last_diagnostics = self._diagnostics(
            dimension=3,
            time_grid=time_grid,
            grids=grids,
            positivity_floor=positivity_floor,
        )
        return tau_solution[::-1]

    def _douglas_step_2d(
        self,
        u_old: Array,
        drift: Array,
        covariance: Array,
        reaction: Array,
        source: Array,
        grids: tuple[Array, ...],
        dt: float,
        boundary_conditions: Optional[Dict[str, Any]],
    ) -> Array:
        directional_old = [
            self._directional_operator(u_old, drift, covariance, grids, axis=0),
            self._directional_operator(u_old, drift, covariance, grids, axis=1),
        ]
        full_old = directional_old[0] + directional_old[1]
        full_old += self._mixed_operator(u_old, covariance, grids)
        full_old += -reaction * u_old + source

        predictor = self._apply_boundary_conditions_2d(
            u_old + dt * full_old, grids, boundary_conditions
        )
        rhs_x = predictor - self.theta * dt * directional_old[0]
        x_solved = self._solve_direction(rhs_x, drift, covariance, grids, dt, axis=0)
        x_solved = self._apply_boundary_conditions_2d(
            x_solved, grids, boundary_conditions
        )

        rhs_y = x_solved - self.theta * dt * directional_old[1]
        y_solved = self._solve_direction(rhs_y, drift, covariance, grids, dt, axis=1)
        return self._apply_boundary_conditions_2d(y_solved, grids, boundary_conditions)

    def _douglas_step_3d(
        self,
        u_old: Array,
        drift: Array,
        covariance: Array,
        reaction: Array,
        source: Array,
        grids: tuple[Array, ...],
        dt: float,
        boundary_conditions: Optional[Dict[str, Any]],
    ) -> Array:
        directional_old = [
            self._directional_operator(u_old, drift, covariance, grids, axis=0),
            self._directional_operator(u_old, drift, covariance, grids, axis=1),
            self._directional_operator(u_old, drift, covariance, grids, axis=2),
        ]
        full_old = directional_old[0] + directional_old[1] + directional_old[2]
        full_old += self._mixed_operator(u_old, covariance, grids)
        full_old += -reaction * u_old + source

        current = self._apply_boundary_conditions_3d(
            u_old + dt * full_old, grids, boundary_conditions
        )
        for axis, old_directional in enumerate(directional_old):
            rhs = current - self.theta * dt * old_directional
            current = self._solve_direction(
                rhs, drift, covariance, grids, dt, axis=axis
            )
            current = self._apply_boundary_conditions_3d(
                current, grids, boundary_conditions
            )
        return current

    def _validate_2d_inputs(
        self,
        drift: Array,
        covariance: Array,
        initial_condition: Array,
        nx: int,
        ny: int,
    ) -> None:
        if drift.shape != (nx, ny, 2):
            raise ValidationError(
                f"drift must have shape ({nx}, {ny}, 2), got {drift.shape}"
            )
        if covariance.shape != (nx, ny, 2, 2):
            raise ValidationError(
                f"covariance must have shape ({nx}, {ny}, 2, 2), got {covariance.shape}"
            )
        if initial_condition.shape != (nx, ny):
            raise ValidationError(
                f"initial_condition must have shape ({nx}, {ny}), got {initial_condition.shape}"
            )
        self._validate_finite_and_psd(drift, covariance)

    def _validate_3d_inputs(
        self,
        drift: Array,
        covariance: Array,
        initial_condition: Array,
        nx: int,
        ny: int,
        nz: int,
    ) -> None:
        if drift.shape != (nx, ny, nz, 3):
            raise ValidationError(
                f"drift must have shape ({nx}, {ny}, {nz}, 3), got {drift.shape}"
            )
        if covariance.shape != (nx, ny, nz, 3, 3):
            raise ValidationError(
                f"covariance must have shape ({nx}, {ny}, {nz}, 3, 3), got {covariance.shape}"
            )
        if initial_condition.shape != (nx, ny, nz):
            raise ValidationError(
                f"initial_condition must have shape ({nx}, {ny}, {nz}), got {initial_condition.shape}"
            )
        self._validate_finite_and_psd(drift, covariance)

    def _validate_finite_and_psd(self, drift: Array, covariance: Array) -> None:
        if not np.all(np.isfinite(drift)):
            raise ValidationError("ADI drift contains non-finite values")
        if not np.all(np.isfinite(covariance)):
            raise ValidationError("ADI covariance contains non-finite values")
        if not np.allclose(
            covariance, np.swapaxes(covariance, -1, -2), rtol=1e-10, atol=1e-12
        ):
            raise ValidationError("ADI covariance must be symmetric")
        eigenvalues = np.linalg.eigvalsh(
            covariance.reshape(-1, covariance.shape[-1], covariance.shape[-1])
        )
        min_eigenvalue = float(np.min(eigenvalues))
        if min_eigenvalue < -1e-10:
            raise ValidationError(
                "ADI covariance must be positive semi-definite; "
                f"minimum eigenvalue is {min_eigenvalue:.2e}"
            )

    def _validate_time_grid(self, time_grid: Array) -> int:
        if time_grid.ndim != 1 or len(time_grid) < 2:
            raise ValidationError(
                "time_grid must be a one-dimensional array with at least two nodes"
            )
        if not np.all(np.isfinite(time_grid)):
            raise ValidationError("time_grid contains non-finite values")
        if not np.all(np.diff(time_grid) > 0.0):
            raise ValidationError(
                "time_grid must be strictly increasing in calendar time"
            )
        return len(time_grid)

    def _validate_grids(self, grids: tuple[Array, ...]) -> tuple[Array, ...]:
        validated: list[Array] = []
        for axis, grid in enumerate(grids):
            candidate = np.asarray(grid, dtype=float)
            if candidate.ndim != 1 or len(candidate) < 3:
                raise ValidationError(
                    f"grid axis {axis} must be one-dimensional with at least three nodes"
                )
            if not np.all(np.isfinite(candidate)):
                raise ValidationError(f"grid axis {axis} contains non-finite values")
            if not np.all(np.diff(candidate) > 0.0):
                raise ValidationError(f"grid axis {axis} must be strictly increasing")
            validated.append(candidate)
        return tuple(validated)

    def _coerce_field(
        self,
        name: str,
        value: Optional[Array | float],
        shape: tuple[int, ...],
        *,
        default: float,
    ) -> Array:
        if value is None:
            return np.full(shape, default, dtype=float)
        field_value = np.asarray(value, dtype=float)
        try:
            result = np.broadcast_to(field_value, shape).astype(float, copy=True)
        except ValueError as exc:
            raise ValidationError(
                f"{name} must be scalar or broadcastable to shape {shape}, got {field_value.shape}"
            ) from exc
        if not np.all(np.isfinite(result)):
            raise ValidationError(f"{name} contains non-finite values")
        return result

    def _directional_operator(
        self,
        u: Array,
        drift: Array,
        covariance: Array,
        grids: tuple[Array, ...],
        *,
        axis: int,
    ) -> Array:
        first, second = self._axis_first_second(u, grids[axis], axis=axis)
        return 0.5 * covariance[..., axis, axis] * second + drift[..., axis] * first

    def _mixed_operator(
        self, u: Array, covariance: Array, grids: tuple[Array, ...]
    ) -> Array:
        result = np.zeros_like(u, dtype=float)
        for axis_a in range(len(grids)):
            first_a = self._axis_first_derivative(u, grids[axis_a], axis=axis_a)
            for axis_b in range(axis_a + 1, len(grids)):
                cross = self._axis_first_derivative(first_a, grids[axis_b], axis=axis_b)
                result += covariance[..., axis_a, axis_b] * cross
        return result

    def _axis_first_second(
        self, u: Array, grid: Array, *, axis: int
    ) -> tuple[Array, Array]:
        first = np.zeros_like(u, dtype=float)
        second = np.zeros_like(u, dtype=float)
        for idx in range(1, len(grid) - 1):
            hm = float(grid[idx] - grid[idx - 1])
            hp = float(grid[idx + 1] - grid[idx])
            wl1, wc1, wu1 = self._first_weights(hm, hp)
            wl2, wc2, wu2 = self._second_weights(hm, hp)
            lower, center, upper = self._axis_slices(axis, u.ndim, idx)
            first[center] = wl1 * u[lower] + wc1 * u[center] + wu1 * u[upper]
            second[center] = wl2 * u[lower] + wc2 * u[center] + wu2 * u[upper]
        return first, second

    def _axis_first_derivative(self, u: Array, grid: Array, *, axis: int) -> Array:
        first, _ = self._axis_first_second(u, grid, axis=axis)
        return first

    def _solve_direction(
        self,
        rhs: Array,
        drift: Array,
        covariance: Array,
        grids: tuple[Array, ...],
        dt: float,
        *,
        axis: int,
    ) -> Array:
        result = rhs.copy()
        other_axes = [range(size) for dim, size in enumerate(rhs.shape) if dim != axis]
        for fixed_indices in np.ndindex(
            *(len(axis_values) for axis_values in other_axes)
        ):
            line_selector: list[Any] = [slice(None)] * rhs.ndim
            fixed_iter = iter(fixed_indices)
            for dim in range(rhs.ndim):
                if dim != axis:
                    line_selector[dim] = next(fixed_iter)
            selector = tuple(line_selector)
            drift_line = drift[selector + (axis,)]
            covariance_line = covariance[selector + (axis, axis)]
            lower, diag, upper = self._line_system(
                drift_line, covariance_line, grids[axis], dt
            )
            result[selector] = self._solve_tridiagonal(
                lower, diag, upper, rhs[selector]
            )
        return result

    def _line_system(
        self, drift_line: Array, covariance_line: Array, grid: Array, dt: float
    ) -> tuple[Array, Array, Array]:
        n = len(grid)
        lower = np.zeros(n, dtype=float)
        diag = np.ones(n, dtype=float)
        upper = np.zeros(n, dtype=float)
        for idx in range(1, n - 1):
            hm = float(grid[idx] - grid[idx - 1])
            hp = float(grid[idx + 1] - grid[idx])
            wl1, wc1, wu1 = self._first_weights(hm, hp)
            wl2, wc2, wu2 = self._second_weights(hm, hp)
            diffusion = 0.5 * covariance_line[idx]
            advection = drift_line[idx]
            low_op = diffusion * wl2 + advection * wl1
            diag_op = diffusion * wc2 + advection * wc1
            high_op = diffusion * wu2 + advection * wu1
            lower[idx] = -self.theta * dt * low_op
            diag[idx] = 1.0 - self.theta * dt * diag_op
            upper[idx] = -self.theta * dt * high_op
        return lower, diag, upper

    def _first_weights(self, hm: float, hp: float) -> tuple[float, float, float]:
        return (
            -hp / (hm * (hm + hp)),
            (hp - hm) / (hm * hp),
            hm / (hp * (hm + hp)),
        )

    def _second_weights(self, hm: float, hp: float) -> tuple[float, float, float]:
        return (
            2.0 / (hm * (hm + hp)),
            -2.0 / (hm * hp),
            2.0 / (hp * (hm + hp)),
        )

    def _axis_slices(
        self, axis: int, ndim: int, idx: int
    ) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
        lower: list[Any] = [slice(None)] * ndim
        center: list[Any] = [slice(None)] * ndim
        upper: list[Any] = [slice(None)] * ndim
        lower[axis] = idx - 1
        center[axis] = idx
        upper[axis] = idx + 1
        return tuple(lower), tuple(center), tuple(upper)

    def _solve_tridiagonal(self, a: Array, b: Array, c: Array, d: Array) -> Array:
        """Solve a tridiagonal system using the Thomas algorithm.

        Solves ``a[i] * x[i-1] + b[i] * x[i] + c[i] * x[i+1] = d[i]`` and
        fails closed on near-singular pivots instead of returning plausible
        values from an arbitrary epsilon clamp.
        """
        n = len(d)
        if n <= 1:
            if n == 1 and abs(b[0]) > self.tolerance:
                return np.array([d[0] / b[0]])
            if n == 1:
                raise ValidationError("singular one-point tridiagonal system")
            return d.copy()

        c_prime = np.zeros(n, dtype=float)
        d_prime = np.zeros(n, dtype=float)
        if abs(b[0]) <= self.tolerance:
            raise ValidationError("singular tridiagonal pivot at row 0")
        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]

        for idx in range(1, n):
            denom = b[idx] - a[idx] * c_prime[idx - 1]
            if abs(denom) <= self.tolerance:
                raise ValidationError(f"singular tridiagonal pivot at row {idx}")
            if idx < n - 1:
                c_prime[idx] = c[idx] / denom
            d_prime[idx] = (d[idx] - a[idx] * d_prime[idx - 1]) / denom

        x = np.zeros(n, dtype=float)
        x[-1] = d_prime[-1]
        for idx in range(n - 2, -1, -1):
            x[idx] = d_prime[idx] - c_prime[idx] * x[idx + 1]
        return x

    def _apply_boundary_conditions_2d(
        self,
        u: Array,
        grids: tuple[Array, ...],
        boundary_conditions: Optional[Dict[str, Any]],
    ) -> Array:
        if boundary_conditions is None:
            return u
        if hasattr(boundary_conditions, "apply_boundaries"):
            try:
                return boundary_conditions.apply_boundaries(u, grids)
            except Exception as exc:  # pragma: no cover - manager-specific path
                raise ValidationError(
                    "2D boundary manager failed during ADI substep"
                ) from exc
        if not isinstance(boundary_conditions, dict):
            raise ValidationError(
                "2D boundary_conditions must be a dict or boundary manager"
            )

        result = u.copy()
        for boundary_location, boundary_spec in boundary_conditions.items():
            kind = boundary_spec.get("type")
            value = boundary_spec.get("value", 0.0)
            if boundary_location == "left":
                result[0, :] = self._boundary_values(kind, value, result[1, :])
            elif boundary_location == "right":
                result[-1, :] = self._boundary_values(kind, value, result[-2, :])
            elif boundary_location == "bottom":
                result[:, 0] = self._boundary_values(kind, value, result[:, 1])
            elif boundary_location == "top":
                result[:, -1] = self._boundary_values(kind, value, result[:, -2])
            else:
                raise ValidationError(
                    f"unsupported 2D boundary location {boundary_location!r}"
                )
        return result

    def _apply_boundary_conditions_3d(
        self,
        u: Array,
        grids: tuple[Array, ...],
        boundary_conditions: Optional[Dict[str, Any]],
    ) -> Array:
        if boundary_conditions is None:
            return u
        if hasattr(boundary_conditions, "apply_boundaries"):
            try:
                return boundary_conditions.apply_boundaries(u, grids)
            except Exception as exc:  # pragma: no cover - manager-specific path
                raise ValidationError(
                    "3D boundary manager failed during ADI substep"
                ) from exc
        if not isinstance(boundary_conditions, dict):
            raise ValidationError(
                "3D boundary_conditions must be a dict or boundary manager"
            )

        result = u.copy()
        for boundary_location, boundary_spec in boundary_conditions.items():
            kind = boundary_spec.get("type")
            value = boundary_spec.get("value", 0.0)
            if boundary_location == "left":
                result[0, :, :] = self._boundary_values(kind, value, result[1, :, :])
            elif boundary_location == "right":
                result[-1, :, :] = self._boundary_values(kind, value, result[-2, :, :])
            elif boundary_location == "bottom":
                result[:, 0, :] = self._boundary_values(kind, value, result[:, 1, :])
            elif boundary_location == "top":
                result[:, -1, :] = self._boundary_values(kind, value, result[:, -2, :])
            elif boundary_location == "front":
                result[:, :, 0] = self._boundary_values(kind, value, result[:, :, 1])
            elif boundary_location == "back":
                result[:, :, -1] = self._boundary_values(kind, value, result[:, :, -2])
            else:
                raise ValidationError(
                    f"unsupported 3D boundary location {boundary_location!r}"
                )
        return result

    def _boundary_values(
        self, kind: str, value: Any, neighbour: Array
    ) -> Array | float:
        if kind == "dirichlet":
            return value
        if kind == "zero_gradient":
            return neighbour
        raise ValidationError(f"unsupported ADI boundary condition type {kind!r}")

    def _uses_positivity_floor(self, initial_condition: Array, source: Array) -> bool:
        """Enable a disclosed positivity limiter for nonnegative pricing routes.

        Linear parabolic pricing equations with nonnegative payoff/source satisfy
        a maximum-principle lower bound. The finite-grid Douglas route can create
        roundoff/pre-asymptotic undershoots near nonsmooth payoffs; flooring only
        this explicitly nonnegative route prevents negative option values without
        changing signed manufactured-PDE tests.
        """
        return bool(np.all(initial_condition >= 0.0) and np.all(source >= 0.0))

    def _apply_positivity_floor(self, solution: Array, *, enabled: bool) -> Array:
        if not enabled:
            return solution
        return np.maximum(solution, 0.0)

    def _diagnostics(
        self,
        *,
        dimension: int,
        time_grid: Array,
        grids: tuple[Array, ...],
        positivity_floor: bool,
    ) -> dict[str, Any]:
        return {
            "scheme": "douglas",
            "dimension": dimension,
            "theta": self.theta,
            "time_orientation": "forward_tau_internal_calendar_output",
            "steps": len(time_grid) - 1,
            "mixed_derivative_pairs": [
                (a, b) for a in range(dimension) for b in range(a + 1, dimension)
            ],
            "grid_uniformity": [
                bool(np.allclose(np.diff(grid), np.diff(grid)[0])) for grid in grids
            ],
            "reaction_treatment": "explicit_predictor_once",
            "source_treatment": "explicit_predictor_once",
            "positivity_floor": positivity_floor,
            "boundary_treatment": "payoff_predictor_and_each_directional_substep",
        }


# Backwards-compatible helper aliases used by legacy tests/docs.
def create_adi_solver(theta: float = 0.5) -> ADISolver:
    """Create ADI solver with specified parameters."""
    return ADISolver(theta=theta)
