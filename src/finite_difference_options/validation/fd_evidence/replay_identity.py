"""Explicit replay identities for currently available numerical implementations."""

from __future__ import annotations

import os
import platform
from hashlib import sha256
from importlib import resources
from typing import Any

import numpy as np
import scipy

V1_SCHEMA = "finite-difference-options.fd-verification-evidence/v1"
V1_METHOD = "compiled_black_scholes_banded.v1"
SUPPORTED_CODE_VERSIONS = frozenset({"0.1.0"})


def v1_runtime_identity() -> dict[str, Any]:
    """Bind exact replay to runtime/build and the numerical implementation files.

    This contains no timing/random fields. Matching hashes establish identity
    against the executing package, not trust in an arbitrary external artifact.
    """
    package = resources.files("finite_difference_options")
    paths = (
        "solvers/_compiled_black_scholes.py",
        "solvers/_tridiagonal.py",
        "integrations/compiled_pde_black_scholes_route.py",
        "integrations/compiled_pde_adapter.py",
        "integrations/_compiled_pde_contracts.py",
        "integrations/_compiled_pde_validation.py",
        "validation/black_scholes_parity.py",
        "validation/fd_evidence/integrity.py",
        "validation/fd_verification.py",
        "validation/fd_evidence/grid_metrics.py",
        "validation/fd_evidence/carry_bounds.py",
        "validation/fd_evidence/manufactured.py",
        "validation/fd_evidence/perturbations.py",
        "validation/fd_evidence/replay_identity.py",
    )
    return {
        "method": V1_METHOD,
        "dtype": "float64",
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "libc": list(platform.libc_ver()),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numpy_build": np.show_config(mode="dicts"),
        "scipy_build": scipy.show_config(mode="dicts"),
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
        "implementation_sha256": {path: sha256(package.joinpath(path).read_bytes()).hexdigest() for path in paths},
    }
