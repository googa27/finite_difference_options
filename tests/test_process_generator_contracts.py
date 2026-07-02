"""Executable process-generator contract tests for issue #44."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from finite_difference_options.exceptions import ValidationError
from finite_difference_options.processes import (
    AffineProcess,
    ProcessDimension,
    GeometricBrownianMotion,
    HestonModel,
    OrnsteinUhlenbeck,
    CoxIngersollRoss,
    ConstantElasticityVariance,
)
from finite_difference_options.processes.base import AffineCovarianceForm, ProcessType


class TwoFactorLinearCovarianceProcess(AffineProcess):
    """Minimal process with genuine affine covariance tensor."""

    @property
    def dimension(self) -> ProcessDimension:
        return ProcessDimension(value=2)

    def affine_drift_coefficients(self, time: float = 0.0):
        return np.array([0.1, -0.2]), np.array([[1.0, 0.5], [-0.25, 2.0]])

    def affine_covariance_coefficients(self, time: float = 0.0):
        a0 = np.array([[2.0, 0.1], [0.1, 1.0]])
        a_linear = np.array(
            [
                [[0.5, 0.0], [0.0, 0.2]],
                [[0.0, 0.3], [0.3, 0.4]],
            ]
        )
        return a0, a_linear


class NonPSDAffineProcess(TwoFactorLinearCovarianceProcess):
    """Process with explicit non-PSD covariance for negative tests."""

    def affine_covariance_coefficients(self, time: float = 0.0):
        return np.array([[1.0, 0.0], [0.0, -0.1]]), np.zeros((2, 2, 2))


def test_process_coefficient_evaluation_has_one_batch_shape_convention() -> None:
    process = TwoFactorLinearCovarianceProcess()

    single = process.evaluate_coefficients(0.0, np.array([1.0, 2.0]))
    batch = process.evaluate_coefficients(0.0, np.array([[1.0, 2.0], [3.0, 4.0]]))

    assert single.is_single_point
    assert single.states.shape == (1, 2)
    assert single.drift.shape == (1, 2)
    assert single.covariance.shape == (1, 2, 2)
    assert single.discount.shape == (1,)
    assert_allclose(single.discount, [0.0])

    assert not batch.is_single_point
    assert batch.states.shape == (2, 2)
    assert batch.drift.shape == (2, 2)
    assert batch.covariance.shape == (2, 2, 2)
    assert batch.discount.shape == (2,)


def test_affine_covariance_tensor_evaluates_a0_plus_sum_xk_ak_exactly() -> None:
    process = TwoFactorLinearCovarianceProcess()
    states = np.array([[1.0, 2.0], [3.0, 4.0]])
    a0, a_linear = process.affine_covariance_coefficients()

    expected = np.stack(
        [a0 + states[i, 0] * a_linear[0] + states[i, 1] * a_linear[1] for i in range(2)]
    )

    form = process.affine_covariance_form()
    assert isinstance(form, AffineCovarianceForm)
    assert_allclose(form.evaluate(states), expected)
    assert_allclose(process.covariance(0.0, states), expected)


def test_affine_covariance_rejects_ambiguous_non_tensor_shapes() -> None:
    with pytest.raises(ValidationError, match="linear affine covariance tensor"):
        AffineCovarianceForm.from_coefficients(np.eye(2), np.ones((2, 2)))


def test_affine_covariance_form_matches_builtin_models_or_fails_closed() -> None:
    short_rate_states = np.array([[0.03], [0.07]])
    for process in [
        OrnsteinUhlenbeck(kappa=2.0, theta=0.04, sigma=0.1),
        CoxIngersollRoss(kappa=2.0, theta=0.04, sigma=0.2),
    ]:
        form = process.affine_covariance_form()
        evaluated = process.evaluate_coefficients(0.0, short_rate_states)
        assert_allclose(form.evaluate(short_rate_states), evaluated.covariance)

    heston = HestonModel(
        risk_free_rate=0.05,
        kappa=2.0,
        theta=0.04,
        sigma=0.3,
        rho=-0.7,
    )
    heston_states = np.array([[np.log(90.0), 0.04], [np.log(120.0), 0.09]])
    heston_form = heston.affine_covariance_form()
    heston_evaluated = heston.evaluate_coefficients(0.0, heston_states)
    assert_allclose(heston_form.evaluate(heston_states), heston_evaluated.covariance)

    with pytest.raises(ValidationError, match="exact affine covariance"):
        GeometricBrownianMotion(mu=0.05, sigma=0.2).affine_covariance_form()


def test_raw_affine_covariance_coefficients_respect_exactness_contract() -> None:
    """Direct coefficient access fails closed only for inexact native-coordinate models."""

    with pytest.raises(ValidationError, match="exact affine covariance"):
        GeometricBrownianMotion(mu=0.05, sigma=0.2).affine_covariance_coefficients()

    heston = HestonModel(
        risk_free_rate=0.05,
        kappa=2.0,
        theta=0.04,
        sigma=0.3,
        rho=-0.7,
    )
    constant, linear = heston.affine_covariance_coefficients()
    assert constant.shape == (2, 2)
    assert linear.shape == (2, 2, 2)
    assert_allclose(linear[1, 0, 0], 1.0)


def test_generator_application_matches_analytical_one_factor_processes() -> None:
    processes = [
        GeometricBrownianMotion(mu=0.05, sigma=0.2),
        OrnsteinUhlenbeck(kappa=2.0, theta=0.04, sigma=0.1),
        CoxIngersollRoss(kappa=2.0, theta=0.04, sigma=0.2),
        ConstantElasticityVariance(mu=0.05, sigma=0.2, beta=0.5),
    ]
    states = np.array([[0.03], [0.07]])
    value = states[:, 0] ** 2
    gradient = 2.0 * states
    hessian = np.repeat(np.array([[[2.0]]]), repeats=len(states), axis=0)

    for process in processes:
        coefficients = process.evaluate_coefficients(0.0, states)
        actual = process.apply_generator(
            0.0,
            states,
            value=value,
            gradient=gradient,
            hessian=hessian,
        )
        expected = (
            coefficients.covariance[:, 0, 0]
            + 2.0 * states[:, 0] * coefficients.drift[:, 0]
        )
        assert_allclose(actual, expected)


def test_discount_is_not_inferred_from_drift_in_generator_application() -> None:
    process = GeometricBrownianMotion(mu=0.05, sigma=0.2)
    state = np.array([100.0])

    no_discount = process.apply_generator(
        0.0,
        state,
        value=1.0,
        gradient=np.array([0.0]),
        hessian=np.array([[0.0]]),
    )
    explicit_discount = process.apply_generator(
        0.0,
        state,
        value=1.0,
        gradient=np.array([0.0]),
        hessian=np.array([[0.0]]),
        discount=0.03,
    )

    assert_allclose(no_discount, [0.0])
    assert_allclose(explicit_discount, [-0.03])
    assert process.process_type == ProcessType.AFFINE
