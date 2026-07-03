"""Heston Feller-condition diagnostics and policy tests for issue #65."""

from __future__ import annotations

import math

import pytest

from finite_difference_options.exceptions import ValidationError
from finite_difference_options.processes.affine import HestonModel
from finite_difference_options.utils.process_validators import (
    FellerPolicy,
    ZeroBoundaryClassification,
    diagnose_feller_condition,
)


def test_heston_feller_violation_is_allowed_with_explicit_diagnostics() -> None:
    """A violated Feller condition is a boundary-policy finding, not invalidity."""

    model = HestonModel(
        risk_free_rate=0.05,
        kappa=1.0,
        theta=0.01,
        sigma=0.5,
        rho=-0.7,
    )

    diagnostics = model.feller_diagnostics()

    assert diagnostics.policy is FellerPolicy.WARN_AND_ALLOW_IF_BACKEND_VALIDATED
    assert diagnostics.is_satisfied is False
    assert diagnostics.requires_explicit_boundary_policy is True
    assert diagnostics.zero_boundary is ZeroBoundaryClassification.ATTAINABLE
    assert diagnostics.feller_margin == pytest.approx(-0.23)
    assert diagnostics.feller_ratio == pytest.approx(0.08)
    assert diagnostics.cir_dimension == pytest.approx(0.16)
    assert diagnostics.route_capability_required == "attainable_variance_boundary"


def test_heston_strict_feller_policy_rejects_violation() -> None:
    """Strict positivity remains available when a route or governance policy needs it."""

    with pytest.raises(ValidationError, match="Feller condition violated"):
        HestonModel(
            risk_free_rate=0.05,
            kappa=1.0,
            theta=0.01,
            sigma=0.5,
            rho=-0.7,
            feller_policy=FellerPolicy.REQUIRE_STRICT_POSITIVITY,
        )


def test_heston_zero_vol_of_vol_has_stable_feller_diagnostics() -> None:
    """Zero vol-of-vol is a deterministic variance boundary case, not 0/0."""

    model = HestonModel(
        risk_free_rate=0.05,
        kappa=1.0,
        theta=0.01,
        sigma=0.0,
        rho=-0.7,
    )

    diagnostics = model.feller_diagnostics()

    assert diagnostics.is_satisfied is True
    assert diagnostics.zero_boundary is ZeroBoundaryClassification.DETERMINISTIC
    assert math.isinf(diagnostics.feller_ratio)
    assert math.isinf(diagnostics.cir_dimension)
    assert diagnostics.requires_explicit_boundary_policy is False


def test_heston_correlation_endpoint_is_separate_from_feller_policy() -> None:
    """Perfect correlation is tracked separately from variance-boundary diagnostics."""

    model = HestonModel(
        risk_free_rate=0.05,
        kappa=2.0,
        theta=0.04,
        sigma=0.3,
        rho=1.0,
    )

    assert model.feller_diagnostics().correlation_degeneracy == "perfect_positive"

    with pytest.raises(ValidationError, match="rho must be in \\[-1, 1\\]"):
        HestonModel(
            risk_free_rate=0.05,
            kappa=2.0,
            theta=0.04,
            sigma=0.3,
            rho=1.5,
        )


def test_diagnose_feller_condition_can_be_used_without_constructing_heston() -> None:
    """Standalone diagnostics distinguish domain checks from policy rejection."""

    diagnostics = diagnose_feller_condition(
        kappa=1.0,
        theta=0.01,
        sigma=0.5,
        rho=-1.0,
        policy=FellerPolicy.ALLOW_WITH_EXPLICIT_BOUNDARY_AND_SCHEME,
    )

    assert diagnostics.is_satisfied is False
    assert diagnostics.correlation_degeneracy == "perfect_negative"
    assert diagnostics.policy is FellerPolicy.ALLOW_WITH_EXPLICIT_BOUNDARY_AND_SCHEME
