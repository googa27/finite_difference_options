"""Captured pre-refactor artifacts remain exact-replay inputs, not tolerance oracles."""

from __future__ import annotations

import gzip
import json
from hashlib import sha256
from pathlib import Path

import pytest

from finite_difference_options.integrations.compiled_pde_adapter import (
    packaged_compiled_black_scholes_fixture,
    solve_compiled_pde_payload,
)
from finite_difference_options.validation import fd_verification as verification

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _capture(name):
    provenance = json.loads((FIXTURES / "compiled_pde_authentic_v0_provenance.json").read_text())
    raw = gzip.decompress((FIXTURES / f"compiled_pde_authentic_v0_{name}.json.gz").read_bytes())
    assert sha256(raw).hexdigest() == provenance["sha256"][f"authentic-v0-{name}.json"]
    return json.loads(raw)


def test_authentic_prechange_v0_is_exactly_replayed_or_explicitly_refused():
    captured = _capture("evidence")
    fresh = verification.run_fd_bs_verification_benchmark()
    same_numerics = all(
        verification._canonicalize(captured[key]) == verification._canonicalize(fresh[key])
        for key in ("request", "config", "convention", "results")
    )
    if same_numerics:
        verification.validate_fd_bs_verification_bundle(captured)
    else:
        # Different BLAS/NumPy roundoff does not authorize accepting old numbers.
        with pytest.raises(verification.FDVerificationError, match="recomputed numerical truth"):
            verification.validate_fd_bs_verification_bundle(captured)
    assert captured["schema_version"].endswith("/v0")


def test_authentic_v0_solve_keeps_schema_and_public_identity():
    captured = _capture("solve")
    fresh = solve_compiled_pde_payload(packaged_compiled_black_scholes_fixture()).as_dict()
    for key in ("schema_version", "backend_id", "problem_id", "route"):
        assert verification._canonicalize(fresh[key]) == verification._canonicalize(captured[key])
    # Numerical equality across every field in the original capture runtime is a
    # separate release acceptance probe; CI across BLAS builds is not bitwise proof.
    assert fresh["status"] == captured["status"] == "passed"


def test_retained_v0_numerical_function_asts_match_c85b403():
    import ast
    import inspect

    from finite_difference_options.integrations import compiled_pde_black_scholes_route as dense

    provenance = json.loads((FIXTURES / "compiled_pde_authentic_v0_provenance.json").read_text())
    for name, expected in provenance["retained_v0_function_ast_sha256"].items():
        tree = ast.parse(inspect.getsource(getattr(dense, name)))
        actual = _ast_fingerprint(tree.body[0])
        assert actual == expected, name


def _ast_fingerprint(node):
    import ast

    def normalize(value):
        if value is Ellipsis:
            return {"literal": "Ellipsis"}
        if isinstance(value, ast.AST):
            return {
                "node": type(value).__name__,
                **{
                    name: normalize(field)
                    for name, field in ast.iter_fields(value)
                    if field is not None and field != []
                },
            }
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    return sha256(json.dumps(normalize(node), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
