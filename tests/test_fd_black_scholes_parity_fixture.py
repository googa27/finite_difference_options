"""Public-synthetic Black--Scholes parity fixture tests."""
from __future__ import annotations

import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.contracts import SolverEvidence  # noqa: E402
from src.validation.black_scholes_parity import (  # noqa: E402
    BlackScholesParityCase,
    black_scholes_call_oracle,
    run_public_black_scholes_parity_fixture,
)


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


def test_fixture_is_public_synthetic_and_deterministic() -> None:
    case = BlackScholesParityCase(code_version="test-head")
    first = run_public_black_scholes_parity_fixture(case=case)
    second = run_public_black_scholes_parity_fixture(case=case)

    assert first.evidence.as_dict() == second.evidence.as_dict()
    assert first.convergence_table() == second.convergence_table()
    assert "synthetic" in first.evidence.units["underlying"]
    assert first.evidence.seed is None
