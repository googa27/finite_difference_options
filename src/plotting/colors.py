"""Colormap utilities shared across backends and UI."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# Reasonable defaults
DEFAULT_SEQUENTIAL = "viridis"
DEFAULT_DIVERGING = "RdBu"


def map_matplotlib_to_plotly(name: str) -> str:
    mapping = {
        "viridis": "Viridis",
        "magma": "Magma",
        "plasma": "Plasma",
        "cividis": "Cividis",
        "inferno": "Inferno",
        "rdbu": "RdBu",
        "coolwarm": "RdBu",
    }
    return mapping.get(name.lower(), name)


def symmetric_bounds(arr: Optional[np.ndarray]) -> Tuple[Optional[float], Optional[float]]:
    if arr is None:
        return None, None
    m = float(np.nanmax(np.abs(arr)))
    return -m, m

