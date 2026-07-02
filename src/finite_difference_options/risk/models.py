"""Data models for regulatory risk reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

CapabilityStatus = Literal[
    "production", "validated", "experimental", "scaffold", "unsupported"
]

REQUIRED_REGULATORY_CONTRACT_FIELDS: tuple[str, ...] = (
    "trade_id",
    "portfolio_id",
    "legal_entity_id",
    "source_system",
    "value_date",
    "reporting_currency",
    "amount_currency",
    "amount_unit",
    "risk_class",
    "risk_type",
    "qualifier",
    "bucket",
    "label_1",
    "label_2",
    "tenor",
    "curve",
    "jurisdiction",
    "standard_profile",
    "standard_version",
    "effective_date",
    "methodology_version",
    "input_contract_hash",
)


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


@dataclass(frozen=True)
class RegulatoryStandard:
    """Declared status for a regulatory/reporting route.

    The route is intentionally not executable unless an exact standard/profile,
    version, effective date, jurisdiction, licensing status, and complete input
    contract are supplied by a future implementation.
    """

    route: str
    standard_id: str
    display_name: str
    capability_status: CapabilityStatus
    reason: str
    profile: str = "not-selected"
    version: str = "not-selected"
    effective_date: str = "not-selected"
    jurisdiction: str = "not-selected"
    licensing_status: str = "not-evaluated"
    required_contract_fields: tuple[str, ...] = field(
        default_factory=lambda: REQUIRED_REGULATORY_CONTRACT_FIELDS
    )

    def to_problem_detail(self) -> dict[str, Any]:
        """Return RFC-7807-like machine-readable failure details."""

        return {
            "type": "https://googa27.github.io/finite_difference_options/problems/regulatory-standard-not-implemented",
            "title": "Regulatory/reporting route is not implemented for a declared standard",
            "http_status": 501,
            "route": self.route,
            "standard_id": self.standard_id,
            "display_name": self.display_name,
            "capability_status": self.capability_status,
            "profile": self.profile,
            "version": self.version,
            "effective_date": self.effective_date,
            "jurisdiction": self.jurisdiction,
            "licensing_status": self.licensing_status,
            "required_contract_fields": list(self.required_contract_fields),
            "reason": self.reason,
        }


class NotImplementedForStandard(RuntimeError):
    """Raised when a regulatory route lacks a conformance-backed standard."""

    def __init__(self, standard: RegulatoryStandard):
        self.standard = standard
        super().__init__(standard.reason)

    def to_problem_detail(self) -> dict[str, Any]:
        """Return serialisable problem details for API and tests."""

        return self.standard.to_problem_detail()
