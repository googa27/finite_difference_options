"""Unified validation package.

This package contains validation functions for the unified pricing framework.
"""

from .benchmark_registry import (
    BenchmarkCase,
    BenchmarkRegistryError,
    BenchmarkRunResult,
    OracleSpec,
    TolerancePolicy,
    default_benchmark_registry,
    registry_as_dict,
    registry_by_id,
    run_registered_benchmark,
    validate_benchmark_registry,
    write_benchmark_result_json,
    write_registry_json,
)
from .validators import (
    validate_positive,
    validate_non_negative,
    validate_probability,
    validate_grid_parameters,
    validate_option_parameters,
    validate_model_parameters,
    validate_array,
    validate_spot_price,
)

__all__ = [
    "BenchmarkCase",
    "BenchmarkRegistryError",
    "BenchmarkRunResult",
    "OracleSpec",
    "TolerancePolicy",
    "default_benchmark_registry",
    "registry_as_dict",
    "registry_by_id",
    "run_registered_benchmark",
    "validate_benchmark_registry",
    "write_benchmark_result_json",
    "write_registry_json",
    "validate_positive",
    "validate_non_negative",
    "validate_probability",
    "validate_grid_parameters",
    "validate_option_parameters",
    "validate_model_parameters",
    "validate_array",
    "validate_spot_price",
]
