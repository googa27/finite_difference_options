"""Fail-closed tests for regulatory-reporting placeholder routes."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.risk import Exposure, RiskFactor, Trade
from src.risk.converters import calculate_basel, calculate_cuso, calculate_frtb, exposures_to_crif
from src.risk.models import NotImplementedForStandard
from src.risk.reporting_strategies import ReportFactory


def _sample_exposures() -> list[Exposure]:
    return [
        Exposure(
            trade=Trade(trade_id="T-1", product_type="IRS", notional=1_000_000.0, currency="USD"),
            risk_factor=RiskFactor(name="USD-SOFR-5Y", value=0.042),
            amount=12_345.67,
        )
    ]


def _sample_payload() -> list[dict]:
    return [
        {
            "trade": {
                "trade_id": "T-1",
                "product_type": "IRS",
                "notional": 1_000_000.0,
                "currency": "USD",
            },
            "risk_factor": {"name": "USD-SOFR-5Y", "value": 0.042},
            "amount": 12_345.67,
        }
    ]


@pytest.mark.parametrize(
    ("name", "function"),
    [
        ("crif", exposures_to_crif),
        ("cuso", calculate_cuso),
        ("basel", calculate_basel),
        ("frtb", calculate_frtb),
    ],
)
def test_regulatory_converters_raise_typed_not_implemented_instead_of_placeholder_results(name, function) -> None:
    with pytest.raises(NotImplementedForStandard) as exc_info:
        function(_sample_exposures())

    detail = exc_info.value.to_problem_detail()
    assert detail["route"] == name
    assert detail["http_status"] == 501
    assert detail["capability_status"] in {"scaffold", "unsupported"}
    assert detail["standard_id"]
    assert detail["profile"] == "not-selected"
    assert detail["version"] == "not-selected"
    assert detail["effective_date"] == "not-selected"
    assert detail["jurisdiction"] == "not-selected"
    assert detail["licensing_status"] in {"not-evaluated", "no-authoritative-specification"}
    assert detail["required_contract_fields"]
    assert "placeholder" not in repr(detail).lower()


@pytest.mark.parametrize("report_type", ["crif", "cuso", "basel", "frtb"])
def test_report_strategies_fail_closed_instead_of_success_dictionaries(report_type: str) -> None:
    strategy = ReportFactory.get_strategy(report_type)

    with pytest.raises(NotImplementedForStandard) as exc_info:
        strategy.generate_report(_sample_exposures())

    detail = exc_info.value.to_problem_detail()
    assert detail["route"] == report_type
    assert "status" not in detail
    assert "crif" not in detail
    assert "risk_weighted_assets" not in detail
    assert "market_risk_capital" not in detail
    assert "total_exposure" not in detail


@pytest.mark.parametrize("report_type", ["crif", "cuso", "basel", "frtb"])
def test_report_api_routes_return_http_501_with_standard_metadata(report_type: str) -> None:
    client = TestClient(app)

    response = client.post(f"/reports/{report_type}", json=_sample_payload())

    assert response.status_code == 501
    body = response.json()
    detail = body["detail"]
    assert detail["route"] == report_type
    assert detail["http_status"] == 501
    assert detail["standard_id"]
    assert detail["capability_status"] in {"scaffold", "unsupported"}
    assert detail["profile"] == "not-selected"
    assert detail["version"] == "not-selected"
    assert detail["required_contract_fields"]
    assert "crif" not in body
    assert "status" not in body
