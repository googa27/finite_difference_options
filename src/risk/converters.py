"""Risk data converters for regulatory reporting formats."""

from typing import List, Dict, Any
from .models import Exposure


def exposures_to_crif(exposures: List[Exposure]) -> List[Dict[str, Any]]:
    """Convert exposures to CRIF (Common Risk Interchange Format)."""
    return [
        {
            "trade_id": exp.trade_id,
            "risk_factor": exp.risk_factor.name,
            "sensitivity": exp.sensitivity,
            "description": exp.description or "",
        }
        for exp in exposures
    ]


def calculate_cuso(exposures: List[Exposure]) -> Dict[str, float]:
    """Calculate CUSO (Credit Unit Specific Offset) metrics."""
    return {"total_exposure": sum(exp.sensitivity for exp in exposures)}


def calculate_basel(exposures: List[Exposure]) -> Dict[str, float]:
    """Calculate Basel regulatory capital metrics."""
    return {"risk_weighted_assets": sum(abs(exp.sensitivity) * 1.2 for exp in exposures)}


def calculate_frtb(exposures: List[Exposure]) -> Dict[str, float]:
    """Calculate FRTB (Fundamental Review of Trading Book) metrics."""
    return {"market_risk_capital": sum(abs(exp.sensitivity) * 0.8 for exp in exposures)}