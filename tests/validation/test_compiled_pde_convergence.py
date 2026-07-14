"""Compiled-PDE convergence and perturbation evidence tests for issue #142."""

from __future__ import annotations

from finite_difference_options.validation.fd_verification import run_fd_bs_verification_benchmark


def test_compiled_pde_evidence_records_applied_boundary_schedule_and_residuals() -> None:
    bundle = run_fd_bs_verification_benchmark()
    finest = bundle["results"]["full_refinement"]["rows"][-1]
    schedule = finest["boundary_schedule_applied"]

    assert schedule[0] == {
        "step_index": 0,
        "tau": 0.0,
        "calendar_time": 1.0,
        "lower": 0.0,
        "upper": 2.0,
        "source": "terminal_payoff_boundary",
    }
    assert schedule[-1]["source"] == "compiled_boundary_expression"
    assert schedule[-1]["upper"] > schedule[0]["upper"]
    assert finest["algebraic_residual_linf"] <= 1.0e-8
    assert finest["algebraic_residual_l2"] <= 1.0e-8
    manufactured = bundle["results"]["manufactured_solution"]
    assert manufactured["rows"][-1]["pde_consistency_linf"] <= 1.0e-5
    assert manufactured["min_observed_pde_consistency_order"] >= 1.8


def test_compiled_pde_perturbed_sign_source_reaction_and_boundary_fail() -> None:
    bundle = run_fd_bs_verification_benchmark()
    perturbations = bundle["results"]["perturbations"]

    assert perturbations["baseline_passes"] is True
    cases = perturbations["cases"]
    assert set(cases) == {
        "operator_sign_flip",
        "reaction_sign_flip",
        "source_shift",
        "static_boundary",
    }
    assert cases["operator_sign_flip"]["residual_linf"] > 1.0e-2
    assert cases["reaction_sign_flip"]["residual_linf"] > 1.0e-2
    assert cases["source_shift"]["residual_linf"] > 1.0e-3
    assert cases["static_boundary"]["boundary_linf"] > 1.0e-3
    assert all(case["passes"] is False for case in cases.values())
