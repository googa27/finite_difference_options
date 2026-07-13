"""Shared test configuration and fixtures for the finite difference options library."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray


@dataclass(frozen=True, order=True)
class _FakeContractVersion:
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, value: object) -> "_FakeContractVersion":
        if isinstance(value, _FakeContractVersion):
            return value
        major, minor, patch = (int(part) for part in str(value).split("."))
        return cls(major=major, minor=minor, patch=patch)

    def is_compatible_with(self, other: object) -> bool:
        return self.major == self.parse(other).major

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


class _FakeBackendMaturity(Enum):
    EXPERIMENTAL = "experimental"
    VALIDATED = "validated"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    UNSUPPORTED = "unsupported"


class _FakeMethodMaturity(Enum):
    DRAFT = "draft"
    VALIDATED = "validated"
    EXPERIMENTAL = "experimental"
    PRODUCTION_GATED = "production_gated"


@dataclass(frozen=True)
class _FakeBackendIdentity:
    distribution_name: str
    distribution_version: str
    implementation_id: str
    implementation_version: str
    contract_version: _FakeContractVersion
    maturity: _FakeBackendMaturity
    license_identifier: str = "UNKNOWN"
    build_metadata: dict[str, str] = field(default_factory=dict)
    schema_version: str = "solver-backend-identity/v0"

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "distribution_name": self.distribution_name,
            "distribution_version": self.distribution_version,
            "implementation_id": self.implementation_id,
            "implementation_version": self.implementation_version,
            "contract_version": str(self.contract_version),
            "maturity": self.maturity.value,
            "license_identifier": self.license_identifier,
            "build_metadata": dict(self.build_metadata),
        }


@dataclass(frozen=True)
class _FakeMethodCapability:
    method_id: str
    backend_id: str
    family: str
    exactness: str
    maturity: _FakeMethodMaturity
    equation_families: tuple[str, ...]
    dimensions: tuple[int, ...]
    state_variables: tuple[str, ...]
    boundary_conditions: tuple[str, ...]
    smoothness_assumptions: tuple[str, ...]
    output_types: tuple[str, ...]
    runtime_controls: tuple[str, ...]
    validation_gates: tuple[str, ...]
    failure_modes: tuple[str, ...]
    fallback_route_id: str | None
    fallback_triggers: tuple[str, ...]
    fallback_policy: str
    references: tuple[str, ...]


@dataclass(frozen=True)
class _FakeBackendCapabilityManifest:
    backend_id: str
    contract_version: str
    methods: tuple[_FakeMethodCapability, ...]

    def validate(self) -> tuple[str, ...]:
        return ()


@pytest.fixture
def haircut_public_solver_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide Haircut's public solver seam shape without a source-tree dependency."""

    haircut = types.ModuleType("haircut")
    solvers = types.ModuleType("haircut.solvers")
    backend_protocol = types.ModuleType("haircut.solvers.backend_protocol")
    contracts = types.ModuleType("haircut.solvers.contracts")

    backend_protocol.__dict__["ContractVersion"] = _FakeContractVersion
    backend_protocol.__dict__["BackendMaturity"] = _FakeBackendMaturity
    backend_protocol.__dict__["BackendIdentity"] = _FakeBackendIdentity
    backend_protocol.__dict__["SOLVER_BACKEND_ENTRY_POINT_GROUP"] = "haircut.solver_backends"
    contracts.__dict__["MethodMaturity"] = _FakeMethodMaturity
    contracts.__dict__["MethodCapability"] = _FakeMethodCapability
    contracts.__dict__["BackendCapabilityManifest"] = _FakeBackendCapabilityManifest

    monkeypatch.setitem(sys.modules, "haircut", haircut)
    monkeypatch.setitem(sys.modules, "haircut.solvers", solvers)
    monkeypatch.setitem(sys.modules, "haircut.solvers.backend_protocol", backend_protocol)
    monkeypatch.setitem(sys.modules, "haircut.solvers.contracts", contracts)


# Common test data and fixtures
@pytest.fixture
def sample_spot_prices() -> NDArray[np.float64]:
    """Sample spot prices for testing."""
    return np.array([80.0, 90.0, 100.0, 110.0, 120.0])


@pytest.fixture
def sample_time_grid() -> NDArray[np.float64]:
    """Sample time grid for testing."""
    return np.linspace(0.0, 1.0, 11)


@pytest.fixture
def sample_spatial_grid() -> NDArray[np.float64]:
    """Sample spatial grid for testing."""
    return np.linspace(50.0, 150.0, 21)


@pytest.fixture
def standard_option_params() -> dict[str, float]:
    """Standard option parameters for testing."""
    return {"strike": 100.0, "maturity": 1.0, "rate": 0.05, "sigma": 0.2}


@pytest.fixture
def heston_params() -> dict[str, float]:
    """Standard Heston model parameters."""
    return {"r": 0.05, "kappa": 2.0, "theta": 0.04, "sigma_v": 0.3, "rho": -0.7}


@pytest.fixture
def small_grid_2d() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Small 2D grid for testing."""
    s_grid = np.linspace(0.0, 200.0, 11)
    v_grid = np.linspace(0.0, 0.5, 6)
    return s_grid, v_grid


@pytest.fixture
def tolerance() -> dict[str, float]:
    """Standard tolerances for numerical tests."""
    return {"rtol": 1e-10, "atol": 1e-12}
