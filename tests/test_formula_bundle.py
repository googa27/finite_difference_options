# ruff: noqa: E501
from __future__ import annotations

import json


from finite_difference_options.contracts.formula_bundle import (
    finite_difference_formula_bundle,
    formula_bundle_json,
    validate_formula_bundle,
)


def _contains_forbidden_style_key(value: object) -> bool:
    if isinstance(value, dict):
        if {"color", "style", "style_token", "css"} & set(value):
            return True
        return any(_contains_forbidden_style_key(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_forbidden_style_key(child) for child in value)
    return False


def test_fd_formula_bundle_is_valid_neutral_formula_contract() -> None:
    payload = finite_difference_formula_bundle()
    assert payload["bundle_version"] == "formula_bundle.v1"
    assert payload["producer"] == "finite_difference_options"
    assert not validate_formula_bundle(payload)
    roles = {component["role"] for formula in payload["formulas"] for component in formula["components"]}
    assert {"diffusion", "drift", "boundary_condition", "residual"} <= roles
    assert not _contains_forbidden_style_key(payload)


def test_fd_formula_bundle_json_round_trips() -> None:
    payload = json.loads(formula_bundle_json())
    assert payload["bundle_hash"].startswith("sha256:")
    assert {formula["formula_id"] for formula in payload["formulas"]} == {
        "fd_theta_black_scholes_stencil",
        "fd_american_put_lcp",
    }


def test_fd_formula_bundle_validator_rejects_non_object_components() -> None:
    payload = finite_difference_formula_bundle()
    payload["formulas"][0]["components"].append("not-a-component")

    errors = validate_formula_bundle(payload)

    assert "formulas[0].components[4] must be an object" in errors
