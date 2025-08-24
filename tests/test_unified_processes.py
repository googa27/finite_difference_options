"""Tests for unified stochastic processes."""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from src.processes import (
    ProcessDimension,
    ProcessType,
    GeometricBrownianMotion,
    OrnsteinUhlenbeck,
    CoxIngersollRoss,
    HestonModel,
    ConstantElasticityVariance,
    SABRModel,
    create_black_scholes_process,
    create_vasicek_process,
    create_cir_process,
    create_standard_heston,
    create_cev_process,
    create_sabr_model,
)
from src.utils.covariance_utils import validate_covariance_matrix, diffusion_to_covariance
from src.exceptions import ValidationError


class TestProcessDimension:
    """Test ProcessDimension class."""
    
    def test_valid_dimensions(self):
        """Test valid dimension creation."""
        dim1 = ProcessDimension(value=1)
        assert dim1.value == 1
        assert dim1.is_univariate
        assert not dim1.is_multivariate
        
        dim3 = ProcessDimension(value=3)
        assert dim3.value == 3
        assert not dim3.is_univariate
        assert dim3.is_multivariate
    
    def test_invalid_dimension(self):
        """Test invalid dimension validation."""
        with pytest.raises(ValidationError, match="Process dimension must be positive"):
            ProcessDimension(value=0)
        
        with pytest.raises(ValidationError, match="Process dimension must be positive"):
            ProcessDimension(value=-1)


class TestGeometricBrownianMotion:
    """Test GBM using unified interface."""
    
    def test_gbm_initialization(self):
        """Test GBM initialization."""
        gbm = GeometricBrownianMotion(mu=0.05, sigma=0.2)
        assert gbm.mu == 0.05
        assert gbm.sigma == 0.2
        assert gbm.dimension.value == 1
        assert gbm.process_type == ProcessType.AFFINE
    
    def test_gbm_parameter_validation(self):
        """Test GBM parameter validation."""
        with pytest.raises(ValidationError, match="sigma"):
            GeometricBrownianMotion(mu=0.05, sigma=-0.1)
    
    def test_gbm_drift_single_state(self):
        """Test GBM drift computation for single state."""
        gbm = GeometricBrownianMotion(mu=0.05, sigma=0.2)
        state = np.array([100.0])
        
        drift = gbm.drift(0.0, state)
        expected = np.array([0.05 * 100.0])
        assert_allclose(drift, expected)
    
    def test_gbm_drift_multiple_states(self):
        """Test GBM drift computation for multiple states."""
        gbm = GeometricBrownianMotion(mu=0.05, sigma=0.2)
        states = np.array([[100.0], [200.0], [50.0]])
        
        drift = gbm.drift(0.0, states)
        expected = np.array([[5.0], [10.0], [2.5]])
        assert_allclose(drift, expected)
    
    def test_gbm_covariance_single_state(self):
        """Test GBM covariance computation for single state."""
        gbm = GeometricBrownianMotion(mu=0.05, sigma=0.2)
        state = np.array([100.0])
        
        cov = gbm.covariance(0.0, state)
        expected = np.array([[0.2**2 * 100.0**2]])
        assert_allclose(cov, expected)
    
    def test_gbm_covariance_multiple_states(self):
        """Test GBM covariance computation for multiple states."""
        gbm = GeometricBrownianMotion(mu=0.05, sigma=0.2)
        states = np.array([[100.0], [200.0]])
        
        cov = gbm.covariance(0.0, states)
        expected = np.array([
            [[0.04 * 10000]],
            [[0.04 * 40000]]
        ])
        assert_allclose(cov, expected)
    
    def test_gbm_affine_coefficients(self):
        """Test GBM affine coefficients."""
        gbm = GeometricBrownianMotion(mu=0.05, sigma=0.2)
        
        alpha, beta = gbm.affine_drift_coefficients()
        assert_allclose(alpha, [0.0])
        assert_allclose(beta, [0.05])
        
        gamma, delta = gbm.affine_covariance_coefficients()
        assert_allclose(gamma, [[0.0]])
        assert_allclose(delta, [[0.04]])


class TestOrnsteinUhlenbeck:
    """Test OU process using unified interface."""
    
    def test_ou_initialization(self):
        """Test OU initialization."""
        ou = OrnsteinUhlenbeck(kappa=2.0, theta=0.05, sigma=0.1)
        assert ou.kappa == 2.0
        assert ou.theta == 0.05
        assert ou.sigma == 0.1
        assert ou.dimension.value == 1
        assert ou.process_type == ProcessType.AFFINE
    
    def test_ou_parameter_validation(self):
        """Test OU parameter validation."""
        with pytest.raises(ValidationError, match="kappa"):
            OrnsteinUhlenbeck(kappa=-1.0, theta=0.05, sigma=0.1)
        
        with pytest.raises(ValidationError, match="sigma"):
            OrnsteinUhlenbeck(kappa=2.0, theta=0.05, sigma=-0.1)
    
    def test_ou_drift(self):
        """Test OU drift computation."""
        ou = OrnsteinUhlenbeck(kappa=2.0, theta=0.05, sigma=0.1)
        state = np.array([0.03])
        
        drift = ou.drift(0.0, state)
        expected = np.array([2.0 * (0.05 - 0.03)])
        assert_allclose(drift, expected)
    
    def test_ou_covariance(self):
        """Test OU covariance computation."""
        ou = OrnsteinUhlenbeck(kappa=2.0, theta=0.05, sigma=0.1)
        state = np.array([0.03])
        
        cov = ou.covariance(0.0, state)
        expected = np.array([[0.01]])
        assert_allclose(cov, expected)


class TestCoxIngersollRoss:
    """Test CIR process using unified interface."""
    
    def test_cir_initialization(self):
        """Test CIR initialization."""
        cir = CoxIngersollRoss(kappa=2.0, theta=0.04, sigma=0.3)
        assert cir.kappa == 2.0
        assert cir.theta == 0.04
        assert cir.sigma == 0.3
        assert cir.dimension.value == 1
        assert cir.process_type == ProcessType.AFFINE
    
    def test_cir_feller_condition(self):
        """Test CIR Feller condition validation."""
        # Valid case
        CoxIngersollRoss(kappa=2.0, theta=0.04, sigma=0.2)
        
        # Invalid case
        with pytest.raises(ValidationError, match="Feller condition violated"):
            CoxIngersollRoss(kappa=1.0, theta=0.01, sigma=0.5)
    
    def test_cir_drift(self):
        """Test CIR drift computation."""
        cir = CoxIngersollRoss(kappa=2.0, theta=0.04, sigma=0.3)
        state = np.array([0.03])
        
        drift = cir.drift(0.0, state)
        expected = np.array([2.0 * (0.04 - 0.03)])
        assert_allclose(drift, expected)
    
    def test_cir_covariance(self):
        """Test CIR covariance computation."""
        cir = CoxIngersollRoss(kappa=2.0, theta=0.04, sigma=0.3)
        state = np.array([0.03])
        
        cov = cir.covariance(0.0, state)
        expected = np.array([[0.09 * 0.03]])
        assert_allclose(cov, expected)
    
    def test_cir_negative_variance_handling(self):
        """Test CIR handling of negative variance."""
        cir = CoxIngersollRoss(kappa=2.0, theta=0.04, sigma=0.3)
        state = np.array([-0.01])  # Negative variance
        
        cov = cir.covariance(0.0, state)
        # Should use minimum value for numerical stability
        assert cov[0, 0] > 0


class TestConstantElasticityVariance:
    """Test CEV process using unified interface."""
    
    def test_cev_initialization(self):
        """Test CEV initialization."""
        cev = ConstantElasticityVariance(mu=0.05, sigma=0.2, beta=0.5)
        assert cev.mu == 0.05
        assert cev.sigma == 0.2
        assert cev.beta == 0.5
        assert cev.dimension.value == 1
        assert cev.process_type == ProcessType.NON_AFFINE
    
    def test_cev_parameter_validation(self):
        """Test CEV parameter validation."""
        with pytest.raises(ValidationError, match="Beta must be non-negative"):
            ConstantElasticityVariance(mu=0.05, sigma=0.2, beta=-0.1)
    
    def test_cev_drift(self):
        """Test CEV drift computation."""
        cev = ConstantElasticityVariance(mu=0.05, sigma=0.2, beta=0.5)
        state = np.array([100.0])
        
        drift = cev.drift(0.0, state)
        expected = np.array([0.05 * 100.0])
        assert_allclose(drift, expected)
    
    def test_cev_covariance(self):
        """Test CEV covariance computation."""
        cev = ConstantElasticityVariance(mu=0.05, sigma=0.2, beta=0.5)
        state = np.array([100.0])
        
        cov = cev.covariance(0.0, state)
        expected = np.array([[0.04 * 100.0]])  # σ²S^(2β) = 0.04 * 100^1
        assert_allclose(cov, expected)


class TestHestonModel:
    """Test Heston model using unified interface."""
    
    def test_heston_initialization(self):
        """Test Heston initialization."""
        heston = HestonModel(
            kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7,
            risk_free_rate=0.05, dividend_yield=0.0
        )
        assert heston.kappa == 2.0
        assert heston.theta == 0.04
        assert heston.sigma == 0.3
        assert heston.rho == -0.7
        assert heston.dimension.value == 2
        assert heston.process_type == ProcessType.AFFINE
    
    def test_heston_parameter_validation(self):
        """Test Heston parameter validation."""
        # Invalid correlation
        with pytest.raises(ValidationError, match="abs\\(rho\\)"):
            HestonModel(kappa=2.0, theta=0.04, sigma=0.3, rho=-1.5,
                       risk_free_rate=0.05)
        
        # Feller condition violation
        with pytest.raises(ValidationError, match="Feller condition violated"):
            HestonModel(kappa=1.0, theta=0.01, sigma=0.5, rho=-0.7,
                       risk_free_rate=0.05)
    
    def test_heston_drift_single_state(self):
        """Test Heston drift for single state."""
        heston = HestonModel(
            kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7,
            risk_free_rate=0.05, dividend_yield=0.01
        )
        state = np.array([100.0, 0.04])
        
        drift = heston.drift(0.0, state)
        expected = np.array([
            0.04 * 100.0,  # (r-q)S
            2.0 * (0.04 - 0.04)  # κ(θ-V)
        ])
        assert_allclose(drift, expected)
    
    def test_heston_drift_multiple_states(self):
        """Test Heston drift for multiple states."""
        heston = HestonModel(
            kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7,
            risk_free_rate=0.05, dividend_yield=0.0
        )
        states = np.array([[100.0, 0.04], [200.0, 0.09]])
        
        drift = heston.drift(0.0, states)
        expected = np.array([
            [5.0, 0.0],
            [10.0, -0.1]
        ])
        assert_allclose(drift, expected)
    
    def test_heston_covariance_single_state(self):
        """Test Heston covariance for single state."""
        heston = HestonModel(
            kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7,
            risk_free_rate=0.05, dividend_yield=0.0
        )
        state = np.array([100.0, 0.04])
        
        cov = heston.covariance(0.0, state)
        
        # Expected covariance matrix
        var_s = 0.04 * 100.0**2  # VS²
        var_v = 0.09 * 0.04  # σ²V
        cov_sv = -0.7 * 0.3 * 100.0 * 0.04  # ρσSV
        
        expected = np.array([
            [var_s, cov_sv],
            [cov_sv, var_v]
        ])
        assert_allclose(cov, expected)
    
    def test_heston_covariance_multiple_states(self):
        """Test Heston covariance for multiple states."""
        heston = HestonModel(
            kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7,
            risk_free_rate=0.05, dividend_yield=0.0
        )
        states = np.array([[100.0, 0.04], [200.0, 0.09]])
        
        cov = heston.covariance(0.0, states)
        assert cov.shape == (2, 2, 2)
        
        # Check symmetry
        assert_allclose(cov[0, 0, 1], cov[0, 1, 0])
        assert_allclose(cov[1, 0, 1], cov[1, 1, 0])


# class TestThreeFactorModel:
#     """Test three-factor model using unified interface."""
#     
#     def test_three_factor_initialization(self):
#         """Test three-factor model initialization."""
#         mu = np.array([0.02, 0.01, 0.03])
#         sigma = np.diag([0.2, 0.15, 0.25])
#         
#         model = ThreeFactorModel(r=0.05, mu=mu, sigma=sigma)
#         assert model.r == 0.05
#         assert_array_equal(model.mu, mu)
#         assert_array_equal(model.sigma, sigma)
#         assert model.dimension.value == 3
#         assert model.process_type == ProcessType.AFFINE
#     
#     def test_three_factor_parameter_validation(self):
#         """Test three-factor parameter validation."""
#         # Wrong mu shape
#         with pytest.raises(ValidationError, match="mu must be shape \\(3,\\)"):
#             ThreeFactorModel(r=0.05, mu=np.array([0.02, 0.01]), sigma=np.eye(3))
#         
#         # Wrong sigma shape
#         with pytest.raises(ValidationError, match="sigma must be shape \\(3, 3\\)"):
#             ThreeFactorModel(r=0.05, mu=np.array([0.02, 0.01, 0.03]), sigma=np.eye(2))
#     
#     def test_three_factor_correlation_validation(self):
#         """Test correlation matrix validation."""
#         mu = np.array([0.02, 0.01, 0.03])
#         sigma = np.diag([0.2, 0.15, 0.25])
#         
#         # Invalid correlation matrix (not symmetric)
#         bad_corr = np.array([[1, 0.5, 0.3], [0.4, 1, 0.2], [0.3, 0.2, 1]])
#         with pytest.raises(ValidationError, match="Correlation matrix must be symmetric"):
#             ThreeFactorModel(r=0.05, mu=mu, sigma=sigma, correlation_matrix=bad_corr)
#     
#     def test_three_factor_drift(self):
#         """Test three-factor drift computation."""
#         mu = np.array([0.02, 0.01, 0.03])
#         sigma = np.diag([0.2, 0.15, 0.25])
#         
#         model = ThreeFactorModel(r=0.05, mu=mu, sigma=sigma)
#         state = np.array([1.0, 2.0, 3.0])
#         
#         drift = model.drift(state)
#         assert_allclose(drift, mu)
#     
#     def test_three_factor_covariance(self):
#         """Test three-factor covariance computation."""
#         mu = np.array([0.02, 0.01, 0.03])
#         sigma = np.diag([0.2, 0.15, 0.25])
#         
#         model = ThreeFactorModel(r=0.05, mu=mu, sigma=sigma)
#         state = np.array([1.0, 2.0, 3.0])
#         
#         cov = model.covariance(state)
#         expected = sigma @ sigma.T
#         assert_allclose(cov, expected)


class TestSABRModel:
    """Test SABR model using unified interface."""
    
    def test_sabr_initialization(self):
        """Test SABR initialization."""
        sabr = SABRModel(alpha=0.3, beta=0.7, rho=-0.3)
        assert sabr.alpha == 0.3
        assert sabr.beta == 0.7
        assert sabr.rho == -0.3
        assert sabr.dimension.value == 2
        assert sabr.process_type == ProcessType.NON_AFFINE
    
    def test_sabr_parameter_validation(self):
        """Test SABR parameter validation."""
        # Invalid beta
        with pytest.raises(ValidationError, match="beta must be between 0 and 1"):
            SABRModel(alpha=0.3, beta=1.5, rho=-0.3)
        
        # Invalid correlation
        with pytest.raises(ValidationError, match="abs\\(rho\\)"):
            SABRModel(alpha=0.3, beta=0.7, rho=-1.5)
    
    def test_sabr_drift(self):
        """Test SABR drift (should be zero)."""
        sabr = SABRModel(alpha=0.3, beta=0.7, rho=-0.3)
        state = np.array([100.0, 0.2])
        
        drift = sabr.drift(0.0, state)
        assert_allclose(drift, [0.0, 0.0])
    
    def test_sabr_covariance(self):
        """Test SABR covariance computation."""
        sabr = SABRModel(alpha=0.3, beta=0.7, rho=-0.3)
        state = np.array([100.0, 0.2])
        
        cov = sabr.covariance(0.0, state)
        
        # Check dimensions and symmetry
        assert cov.shape == (2, 2)
        assert_allclose(cov[0, 1], cov[1, 0])


class TestConvenienceFunctions:
    """Test convenience functions for process creation."""
    
    def test_create_black_scholes_process(self):
        """Test Black-Scholes process creation."""
        process = create_black_scholes_process(0.05, 0.2)
        assert isinstance(process, GeometricBrownianMotion)
        assert process.mu == 0.05
        assert process.sigma == 0.2
    
    def test_create_vasicek_process(self):
        """Test Vasicek process creation."""
        process = create_vasicek_process(2.0, 0.05, 0.1)
        assert isinstance(process, OrnsteinUhlenbeck)
        assert process.kappa == 2.0
        assert process.theta == 0.05
        assert process.sigma == 0.1
    
    def test_create_cir_process(self):
        """Test CIR process creation."""
        process = create_cir_process(2.0, 0.04, 0.3)
        assert isinstance(process, CoxIngersollRoss)
        assert process.kappa == 2.0
        assert process.theta == 0.04
        assert process.sigma == 0.3
    
    def test_create_cev_process(self):
        """Test CEV process creation."""
        process = create_cev_process(0.05, 0.2, 0.5)
        assert isinstance(process, ConstantElasticityVariance)
        assert process.mu == 0.05
        assert process.sigma == 0.2
        assert process.beta == 0.5
    
    def test_create_standard_heston(self):
        """Test standard Heston creation."""
        heston = create_standard_heston()
        assert isinstance(heston, HestonModel)
        assert heston.kappa == 2.0
        assert heston.theta == 0.04
        assert heston.sigma == 0.3
        assert heston.rho == -0.7
    
    # def test_create_uncorrelated_three_factor(self):
    #     """Test uncorrelated three-factor creation."""
    #     model = create_uncorrelated_three_factor()
    #     assert isinstance(model, ThreeFactorModel)
    #     assert model.dimension.value == 3
    #     assert_allclose(model.correlation_matrix, np.eye(3))
    
    def test_create_sabr_model(self):
        """Test SABR model creation."""
        sabr = create_sabr_model(0.3, 0.7, -0.3)
        assert isinstance(sabr, SABRModel)
        assert sabr.alpha == 0.3
        assert sabr.beta == 0.7
        assert sabr.rho == -0.3


# class TestUtilityFunctions:
#     """Test utility functions."""
#     
#     def test_validate_covariance_matrix_valid(self):
#         """Test validation of valid covariance matrix."""
#         # Valid symmetric positive definite matrix
#         cov = np.array([[1.0, 0.5], [0.5, 1.0]])
#         validate_covariance_matrix(cov)  # Should not raise
#     
#     def test_validate_covariance_matrix_not_symmetric(self):
#         """Test validation of non-symmetric matrix."""
#         cov = np.array([[1.0, 0.5], [0.3, 1.0]])
#         with pytest.raises(ValidationError, match="Covariance matrix must be symmetric"):
#             validate_covariance_matrix(cov)
#     
#     def test_validate_covariance_matrix_not_psd(self):
#         """Test validation of non-positive semi-definite matrix."""
#         cov = np.array([[1.0, 2.0], [2.0, 1.0]])  # Negative eigenvalue
#         with pytest.raises(ValidationError, match="Covariance matrix must be positive semi-definite"):
#             validate_covariance_matrix(cov)
#     
#     def test_convert_to_covariance_form(self):
#         """Test conversion to covariance form."""
#         def drift_func(state, time=0.0):
#             return np.array([0.05 * state[0]])
#         
#         def diffusion_func(state, time=0.0):
#             return np.array([[0.2 * state[0]]])
#         
#         drift_new, cov_func = convert_to_covariance_form(drift_func, diffusion_func, 1)
#         
#         state = np.array([100.0])
#         cov = cov_func(state)
#         expected = np.array([[0.04 * 10000]])  # (0.2 * 100)²
#         assert_allclose(cov, expected)


class TestStateValidation:
    """Test state validation across processes."""
    
    def test_1d_state_validation(self):
        """Test 1D process state validation."""
        gbm = GeometricBrownianMotion(mu=0.05, sigma=0.2)
        
        # Valid states
        gbm.validate_state(np.array([100.0]))
        gbm.validate_state(np.array([[100.0], [200.0]]))
        
        # Invalid states
        with pytest.raises(ValidationError, match="State dimension .* doesn't match"):
            gbm.validate_state(np.array([100.0, 200.0]))
    
    def test_2d_state_validation(self):
        """Test 2D process state validation."""
        heston = create_standard_heston()
        
        # Valid states
        heston.validate_state(np.array([100.0, 0.04]))
        heston.validate_state(np.array([[100.0, 0.04], [200.0, 0.09]]))
        
        # Invalid states
        with pytest.raises(ValidationError, match="State dimension .* doesn't match"):
            heston.validate_state(np.array([100.0]))
    
    def test_invalid_state_dimensions(self):
        """Test invalid state array dimensions."""
        gbm = GeometricBrownianMotion(mu=0.05, sigma=0.2)
        
        with pytest.raises(ValidationError, match="State must be 1D or 2D array"):
            gbm.validate_state(np.array([[[100.0]]]))


class TestDiffusionComputation:
    """Test diffusion matrix computation from covariance."""
    
    def test_gbm_diffusion(self):
        """Test GBM diffusion computation."""
        gbm = GeometricBrownianMotion(mu=0.05, sigma=0.2)
        state = np.array([100.0])
        
        diffusion = gbm.diffusion(0.0, state)
        expected = np.array([[0.2 * 100.0]])
        assert_allclose(diffusion, expected)
    
    def test_heston_diffusion(self):
        """Test Heston diffusion computation."""
        heston = create_standard_heston()
        state = np.array([100.0, 0.04])
        
        diffusion = heston.diffusion(0.0, state)
        
        # Should be lower triangular from Cholesky decomposition
        assert diffusion.shape == (2, 2)
        # Check that diffusion @ diffusion.T equals covariance
        cov_reconstructed = diffusion @ diffusion.T
        cov_expected = heston.covariance(0.0, state)
        assert_allclose(cov_reconstructed, cov_expected, rtol=1e-10)