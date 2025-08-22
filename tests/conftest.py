"""Shared test configuration and fixtures for the finite difference options library."""
import pytest
import numpy as np
from numpy.typing import NDArray

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
def standard_option_params() -> dict:
    """Standard option parameters for testing."""
    return {
        'strike': 100.0,
        'maturity': 1.0,
        'rate': 0.05,
        'sigma': 0.2
    }

@pytest.fixture
def heston_params() -> dict:
    """Standard Heston model parameters for testing."""
    return {
        'r': 0.05,
        'kappa': 2.0,
        'theta': 0.04,
        'sigma_v': 0.3,
        'rho': -0.7
    }

@pytest.fixture
def small_grid_2d() -> tuple:
    """Small 2D grid for testing."""
    s_grid = np.linspace(0.0, 200.0, 11)
    v_grid = np.linspace(0.0, 0.5, 6)
    return s_grid, v_grid

@pytest.fixture
def tolerance() -> dict:
    """Standard tolerances for numerical tests."""
    return {
        'rtol': 1e-10,
        'atol': 1e-12
    }
