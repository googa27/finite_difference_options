"""Conversion utilities for regulatory reporting formats."""
from __future__ import annotations

import csv
import io
from typing import Iterable

from .models import Exposure


def exposures_to_crif(exposures: Iterable[Exposure]) -> str:
    """Return a CSV string in the Common Risk Interchange Format (CRIF).

    The implementation is intentionally minimal and includes only the
    trade identifier, risk factor name and exposure amount.
    """

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["TradeId", "RiskFactor", "Amount"])
    for exp in exposures:
        writer.writerow([exp.trade.trade_id, exp.risk_factor.name, exp.amount])
    return output.getvalue()


def calculate_cuso(_: Iterable[Exposure]) -> dict[str, str]:
    """Placeholder for a Current US Supervision Office (CUSO) calc."""

    return {"status": "CUSO calculation not implemented"}


def calculate_basel(_: Iterable[Exposure]) -> dict[str, str]:
    """Placeholder for Basel capital requirements calculation."""

    return {"status": "Basel calculation not implemented"}


def calculate_frtb(_: Iterable[Exposure]) -> dict[str, str]:
    """Placeholder for Fundamental Review of the Trading Book (FRTB) calc."""

    return {"status": "FRTB calculation not implemented"}
