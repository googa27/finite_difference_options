"""Finite difference Greeks calculation.

This module provides functions to compute option Greeks using finite difference methods.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from ..utils.exceptions import ValidationError


class FiniteDifferenceGreeks:
    """Calculator for finite difference Greeks."""
    
    def delta(self, prices: NDArray[np.float64], s_grid: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute Delta (first derivative w.r.t. underlying price).
        
        Parameters
        ----------
        prices : NDArray[np.float64]
            Option prices on the grid.
        s_grid : NDArray[np.float64]
            Underlying asset price grid.
            
        Returns
        -------
        NDArray[np.float64]
            Delta values.
        """
        if len(prices.shape) < 1:
            raise ValidationError("Prices array must have at least 1 dimension")
            
        if len(s_grid) < 2:
            raise ValidationError("Asset price grid must have at least 2 points")
            
        ds = s_grid[1] - s_grid[0]
        
        # For multi-dimensional prices, compute gradient along asset dimension
        if len(prices.shape) == 1:
            # 1D case
            return np.gradient(prices, ds)
        elif len(prices.shape) == 2:
            # 2D case (time, asset)
            return np.gradient(prices, ds, axis=1)
        elif len(prices.shape) == 3:
            # 3D case (time, asset, volatility)
            return np.gradient(prices, ds, axis=1)
        else:
            raise ValidationError(f"Unsupported price array dimension: {len(prices.shape)}")
    
    def gamma(self, prices: NDArray[np.float64], s_grid: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute Gamma (second derivative w.r.t. underlying price).
        
        Parameters
        ----------
        prices : NDArray[np.float64]
            Option prices on the grid.
        s_grid : NDArray[np.float64]
            Underlying asset price grid.
            
        Returns
        -------
        NDArray[np.float64]
            Gamma values.
        """
        if len(prices.shape) < 1:
            raise ValidationError("Prices array must have at least 1 dimension")
            
        if len(s_grid) < 3:
            raise ValidationError("Asset price grid must have at least 3 points for second derivative")
            
        ds = s_grid[1] - s_grid[0]
        
        # For multi-dimensional prices, compute second derivative along asset dimension
        if len(prices.shape) == 1:
            # 1D case
            return np.gradient(np.gradient(prices, ds), ds)
        elif len(prices.shape) == 2:
            # 2D case (time, asset)
            first_deriv = np.gradient(prices, ds, axis=1)
            return np.gradient(first_deriv, ds, axis=1)
        elif len(prices.shape) == 3:
            # 3D case (time, asset, volatility)
            first_deriv = np.gradient(prices, ds, axis=1)
            return np.gradient(first_deriv, ds, axis=1)
        else:
            raise ValidationError(f"Unsupported price array dimension: {len(prices.shape)}")
    
    def theta(self, prices: NDArray[np.float64], t_grid: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute Theta (first derivative w.r.t. time).
        
        Parameters
        ----------
        prices : NDArray[np.float64]
            Option prices on the grid.
        t_grid : NDArray[np.float64]
            Time grid.
            
        Returns
        -------
        NDArray[np.float64]
            Theta values.
        """
        if len(prices.shape) < 1:
            raise ValidationError("Prices array must have at least 1 dimension")
            
        if len(t_grid) < 2:
            raise ValidationError("Time grid must have at least 2 points")
            
        dt = t_grid[1] - t_grid[0]
        
        # For multi-dimensional prices, compute gradient along time dimension
        if len(prices.shape) == 1:
            # 1D case - assume this is time dimension
            return np.gradient(prices, dt)
        elif len(prices.shape) == 2:
            # 2D case (time, asset)
            return np.gradient(prices, dt, axis=0)
        elif len(prices.shape) == 3:
            # 3D case (time, asset, volatility)
            return np.gradient(prices, dt, axis=0)
        else:
            raise ValidationError(f"Unsupported price array dimension: {len(prices.shape)}")