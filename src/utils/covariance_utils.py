"""Utilities for covariance matrix operations.

This module provides common functionality for working with covariance matrices
across different stochastic processes.
"""
import numpy as np
from numpy.typing import NDArray
from typing import Tuple

from src.exceptions import ValidationError


def validate_covariance_matrix(matrix: NDArray[np.float64]) -> None:
    """Validate that a matrix is a valid covariance matrix.
    
    A valid covariance matrix must be:
    1. Square
    2. Symmetric
    3. Positive semi-definite
    
    Parameters
    ----------
    matrix : NDArray[np.float64]
        Matrix to validate.
        
    Raises
    ------
    ValidationError
        If matrix is not a valid covariance matrix.
    """
    if matrix.ndim != 2:
        raise ValidationError(f"Covariance matrix must be 2D, got {matrix.ndim}D")
    
    if matrix.shape[0] != matrix.shape[1]:
        raise ValidationError(
            f"Covariance matrix must be square, got shape {matrix.shape}"
        )
    
    # Check symmetry
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValidationError("Covariance matrix must be symmetric")
    
    # Check positive semi-definiteness
    eigenvals = np.linalg.eigvals(matrix)
    if np.any(eigenvals < -1e-10):  # Allow small numerical errors
        raise ValidationError(
            f"Covariance matrix must be positive semi-definite. "
            f"Minimum eigenvalue: {np.min(eigenvals):.2e}"
        )


def ensure_positive_definite(
    matrix: NDArray[np.float64],
    min_eigenvalue: float = 1e-10
) -> NDArray[np.float64]:
    """Ensure matrix is positive definite by regularization.
    
    Parameters
    ----------
    matrix : NDArray[np.float64]
        Input matrix.
    min_eigenvalue : float
        Minimum eigenvalue to enforce.
        
    Returns
    -------
    NDArray[np.float64]
        Regularized positive definite matrix.
    """
    eigenvals, eigenvecs = np.linalg.eigh(matrix)
    eigenvals = np.maximum(eigenvals, min_eigenvalue)
    return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T


def diffusion_to_covariance(
    diffusion: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Convert diffusion matrix to covariance matrix.
    
    Parameters
    ----------
    diffusion : NDArray[np.float64]
        Diffusion matrix σ.
        
    Returns
    -------
    NDArray[np.float64]
        Covariance matrix Σ = σσᵀ.
    """
    return diffusion @ diffusion.T


def correlation_to_covariance(
    volatilities: NDArray[np.float64],
    correlation: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Convert volatilities and correlation to covariance matrix.
    
    Parameters
    ----------
    volatilities : NDArray[np.float64]
        Volatility vector.
    correlation : NDArray[np.float64]
        Correlation matrix.
        
    Returns
    -------
    NDArray[np.float64]
        Covariance matrix.
    """
    vol_matrix = np.diag(volatilities)
    return vol_matrix @ correlation @ vol_matrix


def covariance_to_correlation(
    covariance: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract volatilities and correlation from covariance matrix.
    
    Parameters
    ----------
    covariance : NDArray[np.float64]
        Covariance matrix.
        
    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64]]
        (volatilities, correlation_matrix)
    """
    volatilities = np.sqrt(np.diag(covariance))
    
    # Avoid division by zero
    vol_inv = np.where(volatilities > 1e-12, 1.0 / volatilities, 0.0)
    vol_inv_matrix = np.diag(vol_inv)
    
    correlation = vol_inv_matrix @ covariance @ vol_inv_matrix
    
    # Ensure diagonal is exactly 1.0
    np.fill_diagonal(correlation, 1.0)
    
    return volatilities, correlation


def batch_covariance_computation(
    base_covariance: NDArray[np.float64],
    state_dependent_factors: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Compute covariance for multiple states efficiently.
    
    For state-dependent covariance of the form:
    Σ(x) = base_cov * state_factors
    
    Parameters
    ----------
    base_covariance : NDArray[np.float64]
        Base covariance matrix (d, d).
    state_dependent_factors : NDArray[np.float64]
        State-dependent scaling factors (n_states, d, d).
        
    Returns
    -------
    NDArray[np.float64]
        Batch covariance matrices (n_states, d, d).
    """
    return base_covariance[None, :, :] * state_dependent_factors


def cholesky_decomposition_safe(
    matrix: NDArray[np.float64],
    regularization: float = 1e-10
) -> NDArray[np.float64]:
    """Safe Cholesky decomposition with regularization.
    
    Parameters
    ----------
    matrix : NDArray[np.float64]
        Positive semi-definite matrix.
    regularization : float
        Regularization parameter.
        
    Returns
    -------
    NDArray[np.float64]
        Lower triangular Cholesky factor.
    """
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        # Add regularization and try again
        regularized = matrix + regularization * np.eye(matrix.shape[0])
        return np.linalg.cholesky(regularized)


def matrix_sqrt(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute matrix square root using eigendecomposition.
    
    Parameters
    ----------
    matrix : NDArray[np.float64]
        Positive semi-definite matrix.
        
    Returns
    -------
    NDArray[np.float64]
        Matrix square root.
    """
    eigenvals, eigenvecs = np.linalg.eigh(matrix)
    eigenvals = np.maximum(eigenvals, 0.0)  # Ensure non-negative
    sqrt_eigenvals = np.sqrt(eigenvals)
    return eigenvecs @ np.diag(sqrt_eigenvals) @ eigenvecs.T
