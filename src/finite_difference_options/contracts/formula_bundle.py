# ruff: noqa: E501
from __future__ import annotations

import json
from hashlib import sha256
from typing import Any

FORMULA_BUNDLE_VERSION = "formula_bundle.v1"
ROLE_TAXONOMY_VERSION = "formula_role_taxonomy.v1"
DEFAULT_SOURCE_VINTAGE = "public-synthetic-contract-v1"


def stable_hash(payload: dict[str, Any]) -> str:
    return "sha256:" + sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def component(component_id: str, role: str, label: str, latex: str, **metadata: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"component_id": component_id, "role": role, "label": label, "latex": latex}
    if metadata:
        payload["metadata"] = metadata
    return payload


def source_ref(
    source_id: str, uri: str, *, source_type: str = "documentation", vintage: str = DEFAULT_SOURCE_VINTAGE
) -> dict[str, Any]:
    base = {
        "source_id": source_id,
        "source_type": source_type,
        "uri": uri,
        "privacy_class": "public-synthetic",
        "vintage": vintage,
    }
    base["source_hash"] = stable_hash(
        {"source_id": source_id, "source_type": source_type, "uri": uri, "vintage": vintage}
    )
    return base


def formula(
    formula_id: str,
    title: str,
    formula_kind: str,
    expression_latex: str,
    components: list[dict[str, Any]],
    *,
    method_id: str | None = None,
    formulation_kind: str | None = None,
    source_refs: list[dict[str, Any]] | None = None,
    annotation_intents: list[dict[str, Any]] | None = None,
    tags: list[str] | None = None,
    assumptions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "formula_id": formula_id,
        "title": title,
        "formula_kind": formula_kind,
        "expression_latex": expression_latex,
        "components": components,
        "source_refs": source_refs or [],
        "annotation_intents": annotation_intents or [],
        "assumptions": assumptions or {"privacy_class": "public-synthetic"},
        "tags": tags or [],
        "role_taxonomy_version": ROLE_TAXONOMY_VERSION,
    }
    if method_id is not None:
        payload["method_id"] = method_id
    if formulation_kind is not None:
        payload["formulation_kind"] = formulation_kind
    payload["formula_hash"] = stable_hash(payload)
    return payload


def bundle(
    bundle_id: str, producer: str, formulas: list[dict[str, Any]], source_refs: list[dict[str, Any]]
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "bundle_id": bundle_id,
        "bundle_version": FORMULA_BUNDLE_VERSION,
        "producer": producer,
        "formulas": formulas,
        "source_refs": source_refs,
        "tags": ["project-16", "public-synthetic", "formula_bundle.v1"],
    }
    payload["bundle_hash"] = stable_hash(payload)
    return payload


def validate_formula_bundle(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("bundle_version") != FORMULA_BUNDLE_VERSION:
        errors.append("unsupported bundle_version")
    formulas = payload.get("formulas")
    if not isinstance(formulas, list) or not formulas:
        errors.append("formulas must be a non-empty list")
        return errors
    ids: set[str] = set()
    for index, item in enumerate(formulas):
        if not isinstance(item, dict):
            errors.append(f"formulas[{index}] must be an object")
            continue
        formula_id = item.get("formula_id")
        if not isinstance(formula_id, str) or not formula_id:
            errors.append(f"formulas[{index}].formula_id is required")
        elif formula_id in ids:
            errors.append(f"duplicate formula_id {formula_id}")
        else:
            ids.add(formula_id)
        components = item.get("components")
        if not isinstance(components, list) or not components:
            errors.append(f"formulas[{index}].components must be a non-empty list")
        for component_index, component_item in enumerate(components if isinstance(components, list) else []):
            if not isinstance(component_item, dict):
                errors.append(f"formulas[{index}].components[{component_index}] must be an object")
                continue
            if not {"component_id", "role", "label", "latex"} <= set(component_item):
                errors.append(f"formulas[{index}] component missing required keys")
            if any(key in component_item for key in ("color", "style", "style_token", "css")):
                errors.append(f"formulas[{index}] component contains renderer styling")
    return errors


def finite_difference_formula_bundle() -> dict[str, Any]:
    refs = [source_ref("finite_difference_options.docs.capability_matrix", "docs/CAPABILITY_MATRIX.md")]
    formulas = [
        formula(
            "fd_theta_black_scholes_stencil",
            "Theta-scheme finite-difference Black-Scholes stencil",
            "finite_difference_stencil",
            r"\frac{V_i^{n+1}-V_i^n}{\Delta t}+\theta\mathcal{L}_h V_i^{n+1}+(1-\theta)\mathcal{L}_h V_i^n=0",
            [
                component("time_step", "estimator", "time-step quotient", r"\frac{V_i^{n+1}-V_i^n}{\Delta t}"),
                component(
                    "diffusion_stencil",
                    "diffusion",
                    "second-derivative stencil",
                    r"\frac{1}{2}\sigma^2 S_i^2 D_{SS}V_i",
                ),
                component("drift_stencil", "drift", "first-derivative stencil", r"rS_iD_SV_i"),
                component("boundary_closure", "boundary_condition", "truncated-domain boundary closure", r"B_hV=b_h"),
            ],
            method_id="finite_difference.theta.black_scholes.v1",
            formulation_kind="generator_pde_discretization",
            source_refs=refs,
            annotation_intents=[
                {"component_id": "diffusion_stencil", "kind": "underbrace", "label": "diffusion stencil"}
            ],
            tags=["finite_difference", "theta_scheme", "black_scholes"],
        ),
        formula(
            "fd_american_put_lcp",
            "American put finite-difference LCP residual",
            "finite_difference_lcp",
            r"\min\{V_i^n-\Phi_i, A_hV^n-b_h\}=0",
            [
                component("obstacle_gap", "obstacle", "payoff obstacle gap", r"V_i^n-\Phi_i"),
                component("residual", "residual", "discrete continuation residual", r"A_hV^n-b_h"),
                component("boundary_condition", "boundary_condition", "boundary rows", r"B_hV=b_h"),
            ],
            method_id="finite_difference.american_lcp.v1",
            formulation_kind="variational_inequality_discretization",
            source_refs=refs,
            tags=["finite_difference", "american_option", "lcp"],
        ),
    ]
    return bundle("finite_difference_options_formula_bundle_v1", "finite_difference_options", formulas, refs)


def formula_bundle_json() -> str:
    return json.dumps(finite_difference_formula_bundle(), indent=2, sort_keys=True)
