"""Cache and public FD solver-contract regressions for #55/#130."""

from __future__ import annotations

from typing import Any

import pytest

from finite_difference_options.contracts import DEFAULT_FD_CAPABILITY_MANIFEST, UnsupportedRouteError
from finite_difference_options.integrations import released_fd_solver_contract, solve_public_quant_problem_spec
from finite_difference_options.solvers import BandedOperatorCache
from finite_difference_options.validation.black_scholes_parity import (
    public_black_scholes_problem_spec,
    run_public_black_scholes_parity_fixture,
)
from finite_difference_options.validation.pinares_fixed_price_proxy import (
    PINARES_FIXED_PRICE_PROXY_PROBLEM_ID,
    public_pinares_fixed_price_problem_spec,
    run_public_pinares_fixed_price_proxy_fixture,
)


class _AlwaysEqual:
    def __eq__(self, _other: object) -> bool:
        return True


class _HidingDict(dict[str, object]):
    def keys(self) -> Any:
        return (key for key in super().keys() if key != "private_terms")

    def items(self) -> Any:
        return ((key, value) for key, value in super().items() if key != "private_terms")

    def __iter__(self) -> Any:
        return iter(self.keys())


def _apply_noncanonical_mutation(payload: dict[str, object], mutation: str) -> None:
    if mutation == "unknown_private_field":
        payload["private_terms"] = {"synthetic_probe": object()}
    elif mutation == "unknown_conventions":
        payload["conventions"] = {"measure": "PRIVATE_MEASURE", "numeraire": object()}
    elif mutation == "missing_problem_hash":
        del payload["problem_hash"]
    elif mutation == "boolean_strike":
        payload["financial_graph"]["instrument"]["strike"] = True  # type: ignore[index]
    elif mutation == "float_grid_level":
        payload["solver_plan"]["resource_controls"]["grid_levels"] = 3.0  # type: ignore[index]
    elif mutation == "nan_coefficient":
        payload["mathematical_problem"]["pde_operator_terms"][0]["coefficient"] = float("nan")  # type: ignore[index]
    elif mutation == "custom_equality":
        payload["financial_graph"]["instrument"]["kind"] = _AlwaysEqual()  # type: ignore[index]
    else:
        raise AssertionError(f"unknown test mutation {mutation}")


def test_released_public_solver_contract_advertises_pinares_and_cache_support() -> None:
    contract = released_fd_solver_contract()
    manifest = contract.capability_manifest

    assert contract.schema_version == "finite-difference-options.public-fd-solver-contract/v0"
    assert contract.backend_id == DEFAULT_FD_CAPABILITY_MANIFEST.backend_id
    assert contract.contract_version == DEFAULT_FD_CAPABILITY_MANIFEST.contract_version
    assert PINARES_FIXED_PRICE_PROXY_PROBLEM_ID in contract.supported_problem_ids
    assert "public_synthetic" in contract.supported_privacy_classes
    assert "solve_public_quant_problem_spec" in contract.entry_points[0]
    assert contract.entry_point_groups == ("haircut.solver_backends",)
    assert "haircut_engine.solver_backends" not in contract.entry_point_groups
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


@pytest.mark.parametrize(
    "mutation",
    (
        "unknown_private_field",
        "unknown_conventions",
        "missing_problem_hash",
        "boolean_strike",
        "float_grid_level",
        "nan_coefficient",
        "custom_equality",
    ),
)
def test_public_solver_contract_rejects_noncanonical_json_before_solve(mutation: str) -> None:
    payload = public_black_scholes_problem_spec()
    _apply_noncanonical_mutation(payload, mutation)

    with pytest.raises(UnsupportedRouteError, match="exact validated public-synthetic"):
        solve_public_quant_problem_spec(payload, operator_cache=BandedOperatorCache())


def test_public_pinares_contract_rejects_unknown_nonjson_field_before_solve() -> None:
    payload = public_pinares_fixed_price_problem_spec()
    payload["conventions"] = {"measure": "PRIVATE_MEASURE", "numeraire": object()}

    with pytest.raises(UnsupportedRouteError, match="exact validated public-synthetic"):
        solve_public_quant_problem_spec(payload, operator_cache=BandedOperatorCache())


def test_custom_mapping_cannot_hide_private_fields_from_public_solver() -> None:
    payload = _HidingDict(public_black_scholes_problem_spec())
    payload["private_terms"] = {"synthetic_probe": object()}
    assert dict.__contains__(payload, "private_terms")

    with pytest.raises(UnsupportedRouteError, match="exact validated public-synthetic"):
        solve_public_quant_problem_spec(payload, operator_cache=BandedOperatorCache())
