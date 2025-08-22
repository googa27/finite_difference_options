"""Tests for multi-dimensional stochastic processes."""
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from src.multidimensional_processes import (
    HestonModel, ThreeFactorModel, ProcessDimension,
    create_standard_heston, create_uncorrelated_three_factor
)
from src.exceptions import ValidationError


class TestHestonModel:
    """Test Heston stochastic volatility model."""
    
    def test_heston_initialization(self, heston_params):
        """Test Heston model creation and validation."""
        heston = HestonModel(**heston_params)
        
        assert heston.r == heston_params['r']
        assert heston.kappa == heston_params['kappa']
        assert heston.theta == heston_params['theta']
        assert heston.sigma_v == heston_params['sigma_v']
        assert heston.rho == heston_params['rho']
        assert heston.dimension == ProcessDimension.TWO_D
    
    def test_heston_parameter_validation(self, heston_params):
        """Test Heston parameter validation."""
        # Valid model
        heston = HestonModel(**heston_params)
        heston.validate_parameters()  # Should not raise
        
        # Invalid correlation
        with pytest.raises(ValidationError, match="Correlation.*between -1 and 1"):
            HestonModel(**{**heston_params, 'rho': -1.5})
        
        # Invalid kappa
        with pytest.raises(ValidationError, match="kappa.*positive"):
            HestonModel(**{**heston_params, 'kappa': -1.0})
        
        # Invalid theta
        with pytest.raises(ValidationError, match="theta.*positive"):
            HestonModel(**{**heston_params, 'theta': -0.01})
        
        # Invalid sigma_v
        with pytest.raises(ValidationError, match="sigma_v.*positive"):
            HestonModel(**{**heston_params, 'sigma_v': -0.1})
    
    def test_feller_condition_validation(self):
        """Test Feller condition validation."""
        # Valid Feller condition: 2κθ >= σ_v²
        valid_params = {'r': 0.05, 'kappa': 2.0, 'theta': 0.04, 'sigma_v': 0.3, 'rho': -0.7}
        HestonModel(**valid_params)  # Should not raise
        
        # Invalid Feller condition
        with pytest.raises(ValidationError, match="Feller condition violated"):
            HestonModel(r=0.05, kappa=1.0, theta=0.01, sigma_v=0.5, rho=-0.7)
    
    def test_heston_drift_single_state(self, heston_params, tolerance):
        """Test Heston drift vector computation for single state."""
        heston = HestonModel(**heston_params)
        
        # Test single point
        state = np.array([100.0, 0.04])  # S=100, V=0.04
        drift = heston.drift(state)
        
        expected_drift_s = heston.r * state[0]  # r*S
        expected_drift_v = heston.kappa * (heston.theta - state[1])  # κ(θ-V)
        
        assert_allclose(drift[0], expected_drift_s, **tolerance)
        assert_allclose(drift[1], expected_drift_v, **tolerance)
        assert drift.shape == (2,)
    
    def test_heston_drift_multiple_states(self, heston_params, tolerance):
        """Test Heston drift vector computation for multiple states."""
        heston = HestonModel(**heston_params)
        
        # Test array of states
        states = np.array([[100.0, 0.04], [120.0, 0.06]])
        drifts = heston.drift(states)
        
        assert drifts.shape == (2, 2)
        assert_allclose(drifts[0, 0], heston.r * 100.0, **tolerance)
        assert_allclose(drifts[1, 0], heston.r * 120.0, **tolerance)
        assert_allclose(drifts[0, 1], heston.kappa * (heston.theta - 0.04), **tolerance)
        assert_allclose(drifts[1, 1], heston.kappa * (heston.theta - 0.06), **tolerance)
    
    def test_heston_diffusion_single_state(self, heston_params, tolerance):
        """Test Heston diffusion matrix computation for single state."""
        heston = HestonModel(**heston_params)
        
        state = np.array([100.0, 0.04])
        diffusion = heston.diffusion_matrix(state)
        
        # Expected diffusion matrix structure
        sqrt_v = np.sqrt(state[1])
        expected = np.array([
            [state[0] * sqrt_v, 0.0],
            [heston.rho * heston.sigma_v * sqrt_v, 
             heston.sigma_v * sqrt_v * np.sqrt(1 - heston.rho**2)]
        ])
        
        assert_allclose(diffusion, expected, **tolerance)
        assert diffusion.shape == (2, 2)
    
    def test_heston_diffusion_multiple_states(self, heston_params, tolerance):
        """Test Heston diffusion matrix computation for multiple states."""
        heston = HestonModel(**heston_params)
        
        states = np.array([[100.0, 0.04], [120.0, 0.06]])
        diffusion = heston.diffusion_matrix(states)
        
        assert diffusion.shape == (2, 2, 2)
        
        # Check first state
        sqrt_v0 = np.sqrt(0.04)
        expected_00 = 100.0 * sqrt_v0
        assert_allclose(diffusion[0, 0, 0], expected_00, **tolerance)
        
        # Check second state
        sqrt_v1 = np.sqrt(0.06)
        expected_10 = 120.0 * sqrt_v1
        assert_allclose(diffusion[1, 0, 0], expected_10, **tolerance)
    
    def test_heston_negative_variance_handling(self, heston_params):
        """Test handling of negative variance in diffusion computation."""
        heston = HestonModel(**heston_params)
        
        # State with negative variance
        state = np.array([100.0, -0.01])
        diffusion = heston.diffusion_matrix(state)
        
        # Should handle negative variance gracefully (set to zero)
        assert np.all(np.isfinite(diffusion))
        assert diffusion[0, 0] == 0.0  # S * sqrt(max(V, 0)) = S * 0
    
    def test_create_standard_heston(self):
        """Test standard Heston model creation."""
        heston = create_standard_heston()
        
        assert heston.r == 0.05
        assert heston.kappa == 2.0
        assert heston.theta == 0.04
        assert heston.sigma_v == 0.3
        assert heston.rho == -0.7
        
        # Test with custom parameters
        custom_heston = create_standard_heston(r=0.03, rho=-0.5)
        assert custom_heston.r == 0.03
        assert custom_heston.rho == -0.5
        assert custom_heston.kappa == 2.0  # Default value


class TestThreeFactorModel:
    """Test three-factor stochastic model."""
    
    def test_three_factor_initialization(self):
        """Test three-factor model creation."""
        model = ThreeFactorModel(
            r=0.05, 
            mu=np.array([0.1, 0.08, 0.06]),
            sigma=np.array([0.2, 0.15, 0.1]),
            correlation_matrix=np.eye(3)
        )
        
        assert model.r == 0.05
        assert model.dimension == ProcessDimension.THREE_D
        assert_array_equal(model.mu, [0.1, 0.08, 0.06])
        assert_array_equal(model.sigma, [0.2, 0.15, 0.1])
        assert_array_equal(model.correlation_matrix, np.eye(3))
    
    def test_three_factor_parameter_validation(self):
        """Test three-factor parameter validation."""
        base_params = {
            'r': 0.05,
            'mu': np.array([0.1, 0.08, 0.06]),
            'sigma': np.array([0.2, 0.15, 0.1]),
            'correlation_matrix': np.eye(3)
        }
        
        # Valid model
        model = ThreeFactorModel(**base_params)
        model.validate_parameters()  # Should not raise
        
        # Wrong mu length
        with pytest.raises(ValidationError, match="mu must have length 3"):
            ThreeFactorModel(**{**base_params, 'mu': np.array([0.1, 0.08])})
        
        # Wrong sigma length
        with pytest.raises(ValidationError, match="sigma must have length 3"):
            ThreeFactorModel(**{**base_params, 'sigma': np.array([0.2, 0.15])})
        
        # Wrong correlation matrix shape
        with pytest.raises(ValidationError, match="correlation_matrix must be 3x3"):
            ThreeFactorModel(**{**base_params, 'correlation_matrix': np.eye(2)})
        
        # Negative volatility
        with pytest.raises(ValidationError, match="sigma.*positive"):
            ThreeFactorModel(**{**base_params, 'sigma': np.array([0.2, -0.15, 0.1])})
    
    def test_three_factor_correlation_validation(self):
        """Test correlation matrix validation."""
        base_params = {
            'r': 0.05,
            'mu': np.array([0.1, 0.08, 0.06]),
            'sigma': np.array([0.2, 0.15, 0.1])
        }
        
        # Non-symmetric correlation matrix
        with pytest.raises(ValidationError, match="symmetric"):
            non_symmetric = np.array([[1, 0.5, 0.3], [0.4, 1, 0.2], [0.3, 0.2, 1]])
            ThreeFactorModel(**{**base_params, 'correlation_matrix': non_symmetric})
        
        # Non-positive definite correlation matrix (rank deficient)
        with pytest.raises(ValidationError, match="positive definite"):
            bad_corr = np.array([[1.0, 1.0, 0.5], [1.0, 1.0, 0.5], [0.5, 0.5, 1.0]])
            ThreeFactorModel(**{**base_params, 'correlation_matrix': bad_corr})
        
        # Wrong diagonal elements
        with pytest.raises(ValidationError, match="diagonal elements must be 1"):
            wrong_diag = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]])
            ThreeFactorModel(**{**base_params, 'correlation_matrix': wrong_diag})
    
    def test_three_factor_drift_single_state(self, tolerance):
        """Test three-factor drift computation for single state."""
        model = ThreeFactorModel(
            r=0.05,
            mu=np.array([0.1, 0.08, 0.06]),
            sigma=np.array([0.2, 0.15, 0.1]),
            correlation_matrix=np.eye(3)
        )
        
        state = np.array([100.0, 80.0, 60.0])
        drift = model.drift(state)
        
        expected = model.mu * state
        assert_allclose(drift, expected, **tolerance)
        assert drift.shape == (3,)
    
    def test_three_factor_drift_multiple_states(self, tolerance):
        """Test three-factor drift computation for multiple states."""
        model = ThreeFactorModel(
            r=0.05,
            mu=np.array([0.1, 0.08, 0.06]),
            sigma=np.array([0.2, 0.15, 0.1]),
            correlation_matrix=np.eye(3)
        )
        
        states = np.array([[100.0, 80.0, 60.0], [110.0, 90.0, 70.0]])
        drifts = model.drift(states)
        
        expected = model.mu[np.newaxis, :] * states
        assert_allclose(drifts, expected, **tolerance)
        assert drifts.shape == (2, 3)
    
    def test_three_factor_diffusion_single_state(self, tolerance):
        """Test three-factor diffusion matrix for single state."""
        corr_matrix = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]])
        model = ThreeFactorModel(
            r=0.05,
            mu=np.array([0.1, 0.08, 0.06]),
            sigma=np.array([0.2, 0.15, 0.1]),
            correlation_matrix=corr_matrix
        )
        
        state = np.array([100.0, 80.0, 60.0])
        diffusion = model.diffusion_matrix(state)
        
        assert diffusion.shape == (3, 3)
        
        # Check that the diffusion matrix has correct structure
        # G = diag(σ * X) * L where L is Cholesky of correlation matrix
        chol = np.linalg.cholesky(corr_matrix)
        diagonal_vol = np.diag(model.sigma * state)
        expected = diagonal_vol @ chol.T
        
        assert_allclose(diffusion, expected, **tolerance)
    
    def test_three_factor_diffusion_multiple_states(self):
        """Test three-factor diffusion matrix for multiple states."""
        model = ThreeFactorModel(
            r=0.05,
            mu=np.array([0.1, 0.08, 0.06]),
            sigma=np.array([0.2, 0.15, 0.1]),
            correlation_matrix=np.eye(3)
        )
        
        states = np.array([[100.0, 80.0, 60.0], [110.0, 90.0, 70.0]])
        diffusion = model.diffusion_matrix(states)
        
        assert diffusion.shape == (2, 3, 3)
        
        # For uncorrelated case, diffusion should be diagonal
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    if j == k:
                        expected_val = model.sigma[j] * states[i, j]
                        assert diffusion[i, j, k] == expected_val
                    else:
                        assert diffusion[i, j, k] == 0.0
    
    def test_create_uncorrelated_three_factor(self):
        """Test uncorrelated three-factor model creation."""
        model = create_uncorrelated_three_factor()
        
        assert model.r == 0.05
        assert_array_equal(model.mu, [0.1, 0.08, 0.06])
        assert_array_equal(model.sigma, [0.2, 0.15, 0.1])
        assert_array_equal(model.correlation_matrix, np.eye(3))
        
        # Test with custom parameters
        custom_mu = np.array([0.12, 0.10, 0.08])
        custom_sigma = np.array([0.25, 0.20, 0.15])
        custom_model = create_uncorrelated_three_factor(
            r=0.03, mu=custom_mu, sigma=custom_sigma
        )
        
        assert custom_model.r == 0.03
        assert_array_equal(custom_model.mu, custom_mu)
        assert_array_equal(custom_model.sigma, custom_sigma)


class TestProcessDimension:
    """Test ProcessDimension enum."""
    
    def test_dimension_values(self):
        """Test dimension enum values."""
        assert ProcessDimension.TWO_D.value == 2
        assert ProcessDimension.THREE_D.value == 3
    
    def test_heston_dimension(self, heston_params):
        """Test Heston model dimension."""
        heston = HestonModel(**heston_params)
        assert heston.dimension == ProcessDimension.TWO_D
        assert heston.dimension.value == 2
    
    def test_three_factor_dimension(self):
        """Test three-factor model dimension."""
        model = create_uncorrelated_three_factor()
        assert model.dimension == ProcessDimension.THREE_D
        assert model.dimension.value == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
