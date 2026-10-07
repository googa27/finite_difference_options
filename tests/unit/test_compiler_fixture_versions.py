"""Exact registered compiler-version admission and pre-solve refusal controls."""

from __future__ import annotations

import copy
import hashlib
import json
from importlib import resources
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from finite_difference_options.integrations import compiled_pde_adapter as adapter

V0 = "pde_ir_symbolic_compiler.v0"
V1 = "pde_ir_symbolic_compiler.v1"
H0 = "sha256:970088e5dcb16535edfd230bfe992ea7eb68aede901c7b543682b39f1a5ac32e"
H1 = "sha256:b449647e7f8deea870b0e8fbe0cfd4355040a53b9853d69c17443f0b3a6d9cb2"
CANDIDATE_PATH: Path | None = None


def current_fixture() -> dict:
    resource = resources.files("finite_difference_options.validation.fixtures").joinpath(
        "compiled_pde_black_scholes_call_compiler_v1.json"
    )
    text = CANDIDATE_PATH.read_text() if CANDIDATE_PATH is not None else resource.read_text()
    return json.loads(text)


def rehash(payload: dict) -> None:
    compiled = payload["compiled_operator_result"]["compiled_operator"]
    unsigned = {key: value for key, value in compiled.items() if key != "compiled_hash"}
    data = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    compiled["compiled_hash"] = "sha256:" + hashlib.sha256(data.encode()).hexdigest()


class CompilerFixtureVersions(unittest.TestCase):
    def test_original_fixture_and_screen_are_unchanged(self) -> None:
        old = adapter.packaged_compiled_black_scholes_fixture()
        self.assertEqual(old["compiled_operator_result"]["compiled_operator"]["compiled_hash"], H0)
        self.assertTrue(adapter.screen_compiled_pde_payload(old).supported)
        self.assertEqual(adapter.EXPECTED_COMPILED_HASH, H0)

    def test_actual_current_record_is_admitted(self) -> None:
        payload = current_fixture()
        compiled = payload["compiled_operator_result"]["compiled_operator"]
        self.assertEqual(compiled["compiler_evidence"]["compiler_version"], V1)
        self.assertEqual(compiled["compiled_hash"], H1)
        old = adapter.packaged_compiled_black_scholes_fixture()
        for key in ("source_pde_ir", "solver_plan", "privacy_class", "problem_id", "schema_version"):
            self.assertEqual(payload[key], old[key])
        self.assertNotEqual(payload["artifact_manifest"], old["artifact_manifest"])
        result = adapter.screen_compiled_pde_payload(payload)
        self.assertTrue(result.supported, result.diagnostics)
        self.assertEqual(result.route["compiled_hash"], H1)

    def test_named_factory_preserves_legacy_selection_and_owns_its_json(self) -> None:
        factory = adapter.packaged_compiled_black_scholes_fixture_for_compiler
        self.assertEqual(factory(V0), adapter.packaged_compiled_black_scholes_fixture())
        current = factory(V1)
        self.assertEqual(current, current_fixture())
        current["source_pde_ir"]["privacy_class"] = "private"
        self.assertEqual(factory(V1), current_fixture())
        for version in ("", "v1", "pde_ir_symbolic_compiler.v2", True, [], {}):
            with self.subTest(version=version), self.assertRaises(adapter.CompiledPDEAdapterError):
                factory(version)

    def test_wrong_pairs_and_mutations_refuse_before_numerical_work(self) -> None:
        for version in (V0, V1):
            template = adapter.packaged_compiled_black_scholes_fixture() if version == V0 else current_fixture()
            variants = []
            for observed in (V1 if version == V0 else V0, "pde_ir_symbolic_compiler.v2", True, [], {}):
                p = copy.deepcopy(template)
                p["compiled_operator_result"]["compiled_operator"]["compiler_evidence"]["compiler_version"] = observed
                variants.append(("version-pair", p))
            p = copy.deepcopy(template)
            p["compiled_operator_result"]["compiled_operator"]["compiled_hash"] = H1 if version == V0 else H0
            variants.append(("hash-pair", p))
            for field in ("normalized", "expression_hash", "declared_result_unit"):
                p = copy.deepcopy(template)
                expression = p["compiled_operator_result"]["compiled_operator"]["expressions"][0]
                expression[field] = {"dimension": "dimensionless"} if field == "declared_result_unit" else "altered"
                rehash(p)
                variants.append(("rehashed-" + field, p))
            for name in ("private", "unknown-field", "numerics", "boundary", "manifest-provenance"):
                p = copy.deepcopy(template)
                if name == "private":
                    p["privacy_class"] = "private"
                elif name == "unknown-field":
                    p["unexpected"] = True
                elif name == "numerics":
                    p["solver_plan"]["grid_type"] = "nonuniform"
                elif name == "boundary":
                    p["source_pde_ir"]["boundary_conditions"][0]["kind"] = "neumann"
                else:
                    p["artifact_manifest"]["manifest_id"] = "invented"
                variants.append((name, p))
            for name, p in variants:
                with self.subTest(compiler=version, mutation=name):
                    with patch.object(adapter, "_run_compiled_black_scholes_route") as numerical:
                        self.assertFalse(adapter.screen_compiled_pde_payload(p).supported)
                        with self.assertRaises(adapter.CompiledPDEAdapterError):
                            adapter.solve_compiled_pde_payload(p)
                        numerical.assert_not_called()

    def test_current_record_runs_real_analytical_and_convergence_gates(self) -> None:
        old = adapter.solve_compiled_pde_payload(adapter.packaged_compiled_black_scholes_fixture())
        new = adapter.solve_compiled_pde_payload(current_fixture())
        self.assertTrue(old.passed)
        self.assertTrue(new.passed)
        self.assertEqual(new.values, old.values)
        self.assertEqual(new.diagnostics, old.diagnostics)
        self.assertEqual(new.evidence["compiled_hash"], H1)
        self.assertEqual(old.evidence["compiled_hash"], H0)
        for name, reference, tolerance in (
            ("price", "oracle_price", 5e-4),
            ("delta", "reference_delta", 1e-3),
            ("gamma", "reference_gamma", 8e-3),
        ):
            self.assertLessEqual(abs(new.values[name] - new.values[reference]), tolerance)


if __name__ == "__main__":
    if "--candidate-fixture" in sys.argv:
        index = sys.argv.index("--candidate-fixture")
        CANDIDATE_PATH = Path(sys.argv[index + 1])
        del sys.argv[index : index + 2]
    unittest.main()
