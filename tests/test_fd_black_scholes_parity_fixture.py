"""Public-synthetic Black--Scholes parity fixture tests."""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.contracts import SolverEvidence  # noqa: E402
from src.validation.black_scholes_parity import (  # noqa: E402
    BlackScholesParityCase,
    black_scholes_call_oracle,
    run_public_black_scholes_parity_fixture,
)


FIXTURE_PATH = pathlib.Path(__file__).resolve().parent / "fixtures" / "arxiv_lab_bs_oracle_v1.json"


def test_black_scholes_oracle_matches_known_public_synthetic_case() -> None:
    oracle = black_scholes_call_oracle(spot=1.0, strike=1.0, rate=0.05, sigma=0.2, maturity=1.0)

    assert abs(oracle - 0.1045058357) < 1e-10


def test_public_black_scholes_fixture_converges_with_evidence() -> None:
    report = run_public_black_scholes_parity_fixture()

    assert report.converged
    assert report.final_abs_error <= report.case.tolerance
    assert report.final_abs_error < report.observations[0].abs_error
    assert len(report.convergence_table()) == 3

    evidence = report.evidence
    assert isinstance(evidence, SolverEvidence)
    assert evidence.fixture_id == "public-synthetic.black-scholes-call.v0"
    assert evidence.route_id == "fd.black_scholes_1d.crank_nicolson"
    assert evidence.backend_id == "finite_difference_options.fd_backend.v0"
    assert len(evidence.config_hash) == 64
    assert evidence.code_version == "local-checkout"
    assert evidence.seed is None
    assert evidence.measure == "risk_neutral"
    assert evidence.numeraire == "money_market_account"
    assert evidence.units == {"underlying": "synthetic_currency", "time": "ACT/365F"}
    assert evidence.valuation_date == "2026-01-02"
    assert evidence.maturity_date == "2027-01-02"
    assert evidence.resource_controls == {
        "max_s_steps": 120,
        "max_t_steps": 200,
        "grid_levels": 3,
        "deterministic": "true",
    }
    assert any("right boundary" in item for item in evidence.boundary_assumptions)


def test_public_black_scholes_fixture_is_public_synthetic_and_deterministic() -> None:
    case = BlackScholesParityCase(code_version="test-head")
    first = run_public_black_scholes_parity_fixture(case=case)
    second = run_public_black_scholes_parity_fixture(case=case)

    assert first.evidence.as_dict() == second.evidence.as_dict()
    assert first.convergence_table() == second.convergence_table()
    assert "synthetic" in first.evidence.units["underlying"]
    assert first.evidence.seed is None


def test_public_black_scholes_fixture_emits_delta_gamma_reference_errors() -> None:
    report = run_public_black_scholes_parity_fixture()

    assert report.reference_delta > 0.0
    assert report.reference_delta <= 1.0
    assert report.reference_gamma > 0.0
    assert 0.0 <= report.delta <= 1.0
    assert report.gamma > 0.0

    assert report.errors["delta_abs"] <= 5e-2
    assert report.errors["gamma_abs"] <= 2e-2
    assert report.errors["price_abs"] <= report.case.tolerance
    assert report.errors["price_abs"] == report.final_abs_error
    assert report.no_arbitrage["value_bound_ok"]
    assert report.no_arbitrage["upper_bound_ok"]
    assert report.no_arbitrage["delta_lower_bound_ok"]
    assert report.no_arbitrage["delta_upper_bound_ok"]
    assert report.no_arbitrage["gamma_non_negative_ok"]


def test_arxiv_lab_payload_is_static_file_and_consumable() -> None:
    assert FIXTURE_PATH.exists(), "Fixture JSON expected under tests/fixtures."

    cached = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    report = run_public_black_scholes_parity_fixture()
    payload = report.as_dict()

    assert cached["schema_version"] == "arxiv-lab/fd-oracle-fixture/v0"
    assert cached["problem_spec"]["schema_version"] == "quant-problem-spec/v0"

    typed_boundary = cached["result_export"]["boundary"]["typed"]
    assert typed_boundary[0]["boundary_type"] == "second_derivative"
    assert typed_boundary[0]["expression"] == "d²V/dS²=0"
    assert typed_boundary[1]["boundary_type"] == "neumann"
    assert cached["problem_spec"]["mathematical_problem"]["boundary_conditions"] == {
        "S=0": "second_derivative_zero_gamma",
        "S=S_max": "neumann_delta_one",
    }
    assert cached["result_export"]["time_axis"]["direction"] == "decreasing"
    assert cached["problem_spec"]["solver_plan"]["time_controls"] == {"theta": 0.5}

    assert cached["result_export"]["no_arbitrage"]["value_bound_ok"] is True
    assert cached["result_export"]["no_arbitrage"]["upper_bound_ok"] is True

    assert cached["result_export"]["solution"]["price"] == payload["result_export"]["solution"]["price"]
    assert cached["result_export"]["solution"]["delta"] == payload["result_export"]["solution"]["delta"]
    assert cached["result_export"]["solution"]["gamma"] == payload["result_export"]["solution"]["gamma"]
