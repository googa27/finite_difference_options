from abc import ABC, abstractmethod
import csv
import io
from typing import Iterable

from src.risk.models import Exposure


class ReportStrategy(ABC):
    """Abstract base class for regulatory report generation strategies."""

    @abstractmethod
    def generate_report(self, exposures: Iterable[Exposure]) -> dict:
        """Generate a regulatory report from a list of exposures."""
        pass


class CRIFReportStrategy(ReportStrategy):
    """Strategy for generating CRIF reports."""

    def generate_report(self, exposures: Iterable[Exposure]) -> dict:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["TradeId", "RiskFactor", "Amount"])
        for exp in exposures:
            writer.writerow([exp.trade.trade_id, exp.risk_factor.name, exp.amount])
        return {"crif": output.getvalue()}


class CUSOReportStrategy(ReportStrategy):
    """Strategy for generating CUSO reports (placeholder)."""

    def generate_report(self, exposures: Iterable[Exposure]) -> dict:
        return {"status": "CUSO calculation not implemented"}


class BaselReportStrategy(ReportStrategy):
    """Strategy for generating Basel reports (placeholder)."""

    def generate_report(self, exposures: Iterable[Exposure]) -> dict:
        return {"status": "Basel calculation not implemented"}


class FRTBReportStrategy(ReportStrategy):
    """Strategy for generating FRTB reports (placeholder)."""

    def generate_report(self, exposures: Iterable[Exposure]) -> dict:
        return {"status": "FRTB calculation not implemented"}


class ReportFactory:
    """Factory for creating ReportStrategy instances."""

    @staticmethod
    def get_strategy(report_type: str) -> ReportStrategy:
        if report_type == "crif":
            return CRIFReportStrategy()
        elif report_type == "cuso":
            return CUSOReportStrategy()
        elif report_type == "basel":
            return BaselReportStrategy()
        elif report_type == "frtb":
            return FRTBReportStrategy()
        else:
            raise ValueError(f"Unknown report type: {report_type}")
