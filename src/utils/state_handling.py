"""Utilities for state validation and transformation.

This module provides common functionality for handling state vectors
across different stochastic processes.
"""
import numpy as np
from numpy.typing import NDArray
from typing import Union

from src.exceptions import ValidationError


def ensure_state_array(state: Union[float, NDArray[np.float64]]) -> NDArray[np.float64]:
    """Convert scalar or array-like to proper state array.
    
    Parameters
    ----------
    state : Union[float, NDArray[np.float64]]
        Input state (scalar or array).
        
    Returns
    -------
    NDArray[np.float64]
        Properly formatted state array.
    """
    if np.isscalar(state):
        return np.array([state])
    return np.asarray(state, dtype=np.float64)


def validate_state_dimensions(
    state: NDArray[np.float64],
    expected_dimension: int,
    process_name: str = "process"
) -> None:
    """Validate state vector dimensions.
    
    Parameters
    ----------
    state : NDArray[np.float64]
        State vector to validate.
    expected_dimension : int
        Expected state dimension.
    process_name : str
        Process name for error messages.
        
    Raises
    ------
    ValidationError
        If state dimensions don't match expected.
    """
    state = ensure_state_array(state)
    
    if state.ndim == 1:
        actual_dim = len(state)
    elif state.ndim == 2:
        actual_dim = state.shape[-1]
    else:
        raise ValidationError(
            f"State must be 1D or 2D array, got {state.ndim}D for {process_name}"
        )
    
    if actual_dim != expected_dimension:
        raise ValidationError(
            f"State dimension {actual_dim} doesn't match {process_name} "
            f"dimension {expected_dimension}"
        )


def validate_positive_state_components(
    state: NDArray[np.float64],
    component_indices: list,
    component_names: list = None,
    min_value: float = 1e-10
) -> NDArray[np.float64]:
    """Validate and enforce positivity of specific state components.
    
    Parameters
    ----------
    state : NDArray[np.float64]
        State vector.
    component_indices : list
        Indices of components that must be positive.
    component_names : list, optional
        Names of components for error messages.
    min_value : float
        Minimum value to enforce for numerical stability.
        
    Returns
    -------
    NDArray[np.float64]
        State with enforced positivity constraints.
        
    Raises
    ------
    ValidationError
        If components are negative and cannot be corrected.
    """
    state = ensure_state_array(state)
    result = state.copy()
    
    if component_names is None:
        component_names = [f"component_{i}" for i in component_indices]
    
    if state.ndim == 1:
        for idx, name in zip(component_indices, component_names):
            if state[idx] < 0:
                raise ValidationError(f"{name} must be non-negative, got {state[idx]}")
            result[idx] = max(state[idx], min_value)
    else:
        for idx, name in zip(component_indices, component_names):
            if np.any(state[:, idx] < 0):
                negative_mask = state[:, idx] < 0
                raise ValidationError(
                    f"{name} must be non-negative, found {np.sum(negative_mask)} "
                    f"negative values"
                )
            result[:, idx] = np.maximum(state[:, idx], min_value)
    
    return result


def create_state_matrix(*grids: NDArray[np.float64]) -> NDArray[np.float64]:
    """Create state matrix from spatial grids.
    
    Parameters
    ----------
    *grids : NDArray[np.float64]
        Spatial grids for each dimension.
        
    Returns
    -------
    NDArray[np.float64]
        State matrix with shape (n_points, n_dimensions).
    """
    if len(grids) == 1:
        # 1D case
        return grids[0].reshape(-1, 1)
    else:
        # Multi-dimensional case
        mesh_grids = np.meshgrid(*grids, indexing='ij')
        flat_grids = [grid.flatten() for grid in mesh_grids]
        return np.column_stack(flat_grids)


def reshape_to_grid(
    flat_result: NDArray[np.float64],
    grid_shape: tuple,
    result_dimension: int = 1
) -> NDArray[np.float64]:
    """Reshape flat result back to grid shape.
    
    Parameters
    ----------
    flat_result : NDArray[np.float64]
        Flattened result array.
    grid_shape : tuple
        Original grid shape.
    result_dimension : int
        Dimension of the result (1 for scalar, >1 for vector).
        
    Returns
    -------
    NDArray[np.float64]
        Reshaped result.
    """
    if result_dimension == 1:
        return flat_result.reshape(grid_shape)
    else:
        return flat_result.reshape(grid_shape + (result_dimension,))


def batch_state_evaluation(
    state_matrix: NDArray[np.float64],
    evaluation_func,
    batch_size: int = 1000
) -> NDArray[np.float64]:
    """Evaluate function on state matrix in batches for memory efficiency.
    
    Parameters
    ----------
    state_matrix : NDArray[np.float64]
        State matrix (n_points, n_dimensions).
    evaluation_func : callable
        Function to evaluate on each batch.
    batch_size : int
        Size of each batch.
        
    Returns
    -------
    NDArray[np.float64]
        Concatenated results from all batches.
    """
    n_points = state_matrix.shape[0]
    results = []
    
    for i in range(0, n_points, batch_size):
        end_idx = min(i + batch_size, n_points)
        batch = state_matrix[i:end_idx]
        batch_result = evaluation_func(batch)
        results.append(batch_result)
    
    return np.concatenate(results, axis=0)


def interpolate_state_values(
    state_points: NDArray[np.float64],
    values: NDArray[np.float64],
    query_points: NDArray[np.float64],
    method: str = 'linear'
) -> NDArray[np.float64]:
    """Interpolate values at query points.
    
    Parameters
    ----------
    state_points : NDArray[np.float64]
        Known state points.
    values : NDArray[np.float64]
        Values at known points.
    query_points : NDArray[np.float64]
        Points to interpolate at.
    method : str
        Interpolation method.
        
    Returns
    -------
    NDArray[np.float64]
        Interpolated values.
    """
    from scipy.interpolate import griddata
    
    return griddata(
        state_points, values, query_points, 
        method=method, fill_value=0.0
    )
