"""Solver evidence records shared by FD routing and validation fixtures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SolverEvidence:
    """Reproducibility envelope emitted by public-synthetic solver fixtures.

    The fields mirror the router evidence expected by the cross-repo quant
    platform: route identity, backend identity, code/config identity, fixture
    identity, deterministic seed where applicable, conventions, boundary
    assumptions, and bounded resource controls.
    """

    route_id: str
    backend_id: str
    code_version: str
    config_hash: str
    fixture_id: str
    seed: int | None
    valuation_date: str
    maturity_date: str
    measure: str
    numeraire: str
    units: dict[str, str]
    boundary_assumptions: tuple[str, ...]
    resource_controls: dict[str, int | float | str]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable evidence payload."""

        return asdict(self)


__all__ = ["SolverEvidence"]
