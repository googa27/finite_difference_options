"""Deprecated unified processes module.

This module provides backward compatibility for the old unified_processes.py.
All functionality has been moved to the new processes package.
"""
import warnings
from typing import *

# Issue deprecation warning
warnings.warn(
    "unified_processes.py is deprecated. Use 'from src.processes import ...' instead. "
    "This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from new location for backward compatibility
from src.processes import *

__all__ = [
    'StochasticProcess', 'AffineProcess', 'NonAffineProcess',
    'ProcessDimension', 'ProcessType',
    'GeometricBrownianMotion', 'OrnsteinUhlenbeck', 'CoxIngersollRoss', 'HestonModel',
    'ConstantElasticityVariance', 'SABRModel',
    'create_black_scholes_process', 'create_vasicek_process', 'create_cir_process',
    'create_standard_heston', 'create_cev_process', 'create_sabr_model'
]
