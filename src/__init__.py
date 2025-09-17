"""Unified finite difference options pricing framework.

This package provides a comprehensive framework for pricing financial options
using finite difference methods with support for multi-dimensional processes.
"""

# Core components
from .instruments.base import EuropeanCall, EuropeanPut, EuropeanOption
from .processes.base import StochasticProcess, AffineProcess, NonAffineProcess
from .processes.affine import (
    GeometricBrownianMotion,
    OrnsteinUhlenbeck,
    CoxIngersollRoss,
    HestonModel,
    create_black_scholes_process,
    create_vasicek_process,
    create_cir_process,
    create_standard_heston,
)
from .processes.nonaffine import (
    ConstantElasticityVariance,
    SABRModel,
    create_cev_process,
    create_sabr_model,
)
from .pricing.engines import (
    GridParameters,
    PricingEngine,
    PricingResult,
    UnifiedPricingEngine,
    create_default_pricing_engine,
    create_linear_grid,
    create_log_grid,
    create_unified_pricing_engine,
)
from .solvers.adi import ADISolver, create_adi_solver
from .validation import validate_positive, validate_non_negative, validate_probability

# Backward compatibility imports
from .pricing.instruments.base import UnifiedInstrument
from .pricing.instruments.options import (
    UnifiedEuropeanOption,
    UnifiedBasketOption,
    create_unified_european_call,
    create_unified_european_put,
    create_unified_basket_call,
)

# Utility functions
from .utils.process_validators import validate_weights_sum_to_one
from .utils.covariance_utils import matrix_sqrt
from .utils.state_handling import ensure_state_array, validate_state_dimensions

# Exceptions
from src.exceptions import (
    FiniteDifferenceError,
    ValidationError,
    GridError,
    ModelError,
    InstrumentError,
    PricingError,
    BoundaryConditionError,
    TimeSteppingError,
    ConvergenceError,
)

__all__ = [
    # Instruments
    "EuropeanCall",
    "EuropeanPut",
    "EuropeanOption",

    # Processes
    "StochasticProcess",
    "AffineProcess",
    "NonAffineProcess",
    "GeometricBrownianMotion",
    "OrnsteinUhlenbeck",
    "CoxIngersollRoss",
    "HestonModel",
    "ConstantElasticityVariance",
    "SABRModel",
    "create_black_scholes_process",
    "create_vasicek_process",
    "create_cir_process",
    "create_standard_heston",
    "create_cev_process",
    "create_sabr_model",

    # Pricing
    "GridParameters",
    "PricingEngine",
    "PricingResult",
    "UnifiedPricingEngine",
    "create_default_pricing_engine",
    "create_linear_grid",
    "create_log_grid",
    "create_unified_pricing_engine",

    # Backward compatibility
    "UnifiedInstrument",
    "UnifiedEuropeanOption",
    "UnifiedBasketOption",
    "create_unified_european_call",
    "create_unified_european_put",
    "create_unified_basket_call",

    # Solvers
    "ADISolver",
    "create_adi_solver",

    # Validation
    "validate_positive",
    "validate_non_negative",
    "validate_probability",
    "validate_weights_sum_to_one",

    # Utilities
    "matrix_sqrt",
    "ensure_state_array",
    "validate_state_dimensions",

    # Exceptions
    "FiniteDifferenceError",
    "ValidationError",
    "GridError",
    "ModelError",
    "InstrumentError",
    "PricingError",
    "BoundaryConditionError",
    "TimeSteppingError",
    "ConvergenceError",
]
