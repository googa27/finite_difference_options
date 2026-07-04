"""Cache and public FD solver-contract regressions for #55/#130."""

from __future__ import annotations

import pytest

from finite_difference_options.contracts import DEFAULT_FD_CAPABILITY_MANIFEST, UnsupportedRouteError
from finite_difference_options.integrations import released_fd_solver_contract, solve_public_quant_problem_spec
from finite_difference_options.solvers import BandedOperatorCache
from finite_difference_options.validation.black_scholes_parity import run_public_black_scholes_parity_fixture
from finite_difference_options.validation.pinares_fixed_price_proxy import (
    PINARES_FIXED_PRICE_PROXY_PROBLEM_ID,
    public_pinares_fixed_price_problem_spec,
    run_public_pinares_fixed_price_proxy_fixture,
)


def test_released_public_solver_contract_advertises_pinares_and_cache_support() -> None:
    contract = released_fd_solver_contract()
    manifest = contract.capability_manifest

    assert contract.schema_version == "finite-difference-options.public-fd-solver-contract/v0"
    assert contract.backend_id == DEFAULT_FD_CAPABILITY_MANIFEST.backend_id
    assert contract.contract_version == DEFAULT_FD_CAPABILITY_MANIFEST.contract_version
    assert PINARES_FIXED_PRICE_PROXY_PROBLEM_ID in contract.supported_problem_ids
    assert "public_synthetic" in contract.supported_privacy_classes
    assert "solve_public_quant_problem_spec" in contract.entry_points[0]
    assert manifest["feature_support"]["pinares_fixed_price_proxy"] == "validated"
    assert manifest["feature_support"]["released_public_solver_contract"] == "validated"
    assert manifest["feature_support"]["operator_factorization_cache"] == "validated"
    assert manifest["resource_controls"]["operator_factorization_cache"].startswith("enabled_for_public")


def test_public_pinares_solver_contract_executes_fixture_and_reuses_operator_cache() -> None:
    cache = BandedOperatorCache()
    payload = public_pinares_fixed_price_problem_spec()

    first = solve_public_quant_problem_spec(payload, operator_cache=cache)
    info_after_first = cache.info()
    second = solve_public_quant_problem_spec(payload, operator_cache=cache)
    info_after_second = cache.info()

    assert first.passed
    assert second.passed
    assert second.problem_id == PINARES_FIXED_PRICE_PROXY_PROBLEM_ID
    assert second.values == pytest.approx(first.values)
    assert info_after_first.misses == 3
    assert info_after_first.hits >= info_after_first.solves - 3
    assert info_after_second.misses == info_after_first.misses
    assert info_after_second.hits >= info_after_first.hits + 3
    assert second.diagnostics["operator_cache"]["reuse_count"] >= 3
    assert second.evidence["privacy_class"] == "public_synthetic"


def test_repeated_public_pinares_fixture_cache_preserves_report_values() -> None:
    cache = BandedOperatorCache()

    first = run_public_pinares_fixed_price_proxy_fixture(operator_cache=cache)
    info_after_first = cache.info()
    second = run_public_pinares_fixed_price_proxy_fixture(operator_cache=cache)
    info_after_second = cache.info()

    assert first.converged
    assert second.converged
    assert second.as_dict()["result_export"]["solution"] == pytest.approx(first.as_dict()["result_export"]["solution"])
    assert info_after_first.entries == 3
    assert info_after_first.misses == 3
    assert info_after_second.hits >= info_after_first.hits + 3
    assert second.evidence.resource_controls["operator_cache"]["reuse_count"] >= 3


def test_cached_black_scholes_route_matches_legacy_public_fixture_tolerance() -> None:
    legacy = run_public_black_scholes_parity_fixture(grid_levels=((40, 40), (80, 120)))
    cache = BandedOperatorCache()
    cached = run_public_black_scholes_parity_fixture(
        grid_levels=((40, 40), (80, 120)),
        operator_cache=cache,
    )

    assert cached.converged
    assert cached.price == pytest.approx(legacy.price, abs=legacy.case.tolerance)
    assert cached.oracle_price == legacy.oracle_price
    assert cache.info().misses == 2


def test_public_solver_contract_rejects_label_compatible_mutated_payload_before_solve() -> None:
    payload = public_pinares_fixed_price_problem_spec()
    payload["mathematical_problem"] = {
        **payload["mathematical_problem"],
        "terminal_payoff": {
            **payload["mathematical_problem"]["terminal_payoff"],
            "expression": "max(S - K, 0)",
        },
    }

    with pytest.raises(UnsupportedRouteError, match="exact validated public-synthetic"):
        solve_public_quant_problem_spec(payload, operator_cache=BandedOperatorCache())
