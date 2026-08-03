from __future__ import annotations

import numpy as np
from typing import Any

import pytest
from numpy.typing import NDArray

from finite_difference_options.solvers import FiniteDifferenceSolver, TimeStepper


class RecordingStepper(TimeStepper):
    def __init__(self) -> None:
        self.dts: list[float] = []

    def step(
        self,
        u: NDArray[np.float64],
        operator: Any,
        bc: Any,
        dt: float,
    ) -> NDArray[np.float64]:
        del operator, bc
        self.dts.append(dt)
        return u + dt


def test_finite_difference_solver_uses_each_nonuniform_time_interval() -> None:
    stepper = RecordingStepper()
    solver = FiniteDifferenceSolver(time_stepper=stepper)
    time_grid = np.array([0.0, 0.1, 0.4, 1.0], dtype=np.float64)
    initial = np.array([1.0, 2.0], dtype=np.float64)

    values = solver.solve(
        generator=None,  # type: ignore[arg-type]
        boundary_conditions=None,  # type: ignore[arg-type]
        initial_conditions=initial,
        time_grid=time_grid,
    )

    assert stepper.dts == pytest.approx([0.1, 0.3, 0.6])
    assert values[:, 0] == pytest.approx([1.0, 1.1, 1.4, 2.0])
    assert values[:, 1] == pytest.approx([2.0, 2.1, 2.4, 3.0])


@pytest.mark.parametrize(
    "time_grid",
    (
        np.array([0.0], dtype=np.float64),
        np.array([0.0, 0.2, 0.2], dtype=np.float64),
        np.array([0.0, np.nan, 1.0], dtype=np.float64),
    ),
)
def test_finite_difference_solver_rejects_invalid_time_grid(time_grid: NDArray[np.float64]) -> None:
    solver = FiniteDifferenceSolver(time_stepper=RecordingStepper())

    with pytest.raises(ValueError, match="time_grid"):
        solver.solve(
            generator=None,  # type: ignore[arg-type]
            boundary_conditions=None,  # type: ignore[arg-type]
            initial_conditions=np.array([1.0, 2.0], dtype=np.float64),
            time_grid=time_grid,
        )
