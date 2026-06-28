"""Regulatory report strategy registry.

The public strategy names are kept for API compatibility, but every regulatory
route currently fails closed through :class:`NotImplementedForStandard` until a
versioned standard/profile and conformance suite are implemented.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import NoReturn

from src.risk.converters import REGULATORY_STANDARDS
from src.risk.models import Exposure, NotImplementedForStandard


class ReportStrategy(ABC):
    """Abstract base class for regulatory report generation strategies."""

    @abstractmethod
    def generate_report(self, exposures: Iterable[Exposure]) -> NoReturn:
        """Generate a regulatory report or fail closed before producing output."""

        _ = exposures
        raise NotImplementedError

class UnsupportedRegulatoryReportStrategy(ReportStrategy):
    """Fail-closed strategy for a named but unimplemented regulatory route."""

    def __init__(self, report_type: str):
        self.report_type = report_type

    def generate_report(self, exposures: Iterable[Exposure]) -> NoReturn:
        """Raise typed metadata instead of returning a partial/success payload."""

        _ = list(exposures)
        raise NotImplementedForStandard(REGULATORY_STANDARDS[self.report_type])


class CRIFReportStrategy(UnsupportedRegulatoryReportStrategy):
    """CRIF scaffold: disabled until an exact ISDA CRIF profile is implemented."""

    def __init__(self) -> None:
        super().__init__("crif")


class CUSOReportStrategy(UnsupportedRegulatoryReportStrategy):
    """CUSO route: unsupported until an authoritative specification exists."""

    def __init__(self) -> None:
        super().__init__("cuso")


class BaselReportStrategy(UnsupportedRegulatoryReportStrategy):
    """Basel route: disabled until a versioned Basel/FRTB subset is implemented."""

    def __init__(self) -> None:
        super().__init__("basel")


class FRTBReportStrategy(UnsupportedRegulatoryReportStrategy):
    """FRTB route: disabled until a versioned calculation subset is implemented."""

    def __init__(self) -> None:
        super().__init__("frtb")


class ReportFactory:
    """Factory for creating report strategy instances."""

    @staticmethod
    def get_strategy(report_type: str) -> ReportStrategy:
        """Return a named strategy or reject unknown report types."""

        strategies: dict[str, type[ReportStrategy]] = {
            "crif": CRIFReportStrategy,
            "cuso": CUSOReportStrategy,
            "basel": BaselReportStrategy,
            "frtb": FRTBReportStrategy,
        }
        try:
            return strategies[report_type]()
        except KeyError as exc:
            raise ValueError(f"Unknown report type: {report_type}") from exc
