"""Data models for regulatory risk reporting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Trade:
    """Basic representation of a financial trade."""

    trade_id: str
    product_type: str
    notional: float
    currency: str
    description: Optional[str] = None


@dataclass
class RiskFactor:
    """Market risk factor affecting a trade."""

    name: str
    value: float
    description: Optional[str] = None


@dataclass
class Exposure:
    """Exposure of a trade to a risk factor."""

    trade: Trade
    risk_factor: RiskFactor
    amount: float
