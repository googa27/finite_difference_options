from __future__ import annotations

import importlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_API_DOC = ROOT / "docs" / "PUBLIC_API.md"
THEORY_DOC = ROOT / "docs" / "THEORY.md"
MODULES = [
    "finite_difference_options",
    "finite_difference_options.boundary_conditions",
    "finite_difference_options.contracts",
    "finite_difference_options.greeks",
    "finite_difference_options.grids",
    "finite_difference_options.integrations",
    "finite_difference_options.pricing",
    "finite_difference_options.processes",
    "finite_difference_options.solvers",
    "finite_difference_options.validation",
]


def _documented_exports(text: str, module_name: str) -> list[str]:
    pattern = re.compile(
        rf"<!-- public-api:{re.escape(module_name)}:start -->(.*?)"
        rf"<!-- public-api:{re.escape(module_name)}:end -->",
        re.DOTALL,
    )
    match = pattern.search(text)
    assert (
        match is not None
    ), f"docs/PUBLIC_API.md is missing a synchronized section for {module_name}"
    return re.findall(r"^- `([^`]+)`$", match.group(1), flags=re.MULTILINE)


def test_public_api_doc_is_synchronized_to_runtime_all_exports() -> None:
    text = PUBLIC_API_DOC.read_text(encoding="utf-8")
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        runtime_exports = list(getattr(module, "__all__", []))
        documented_exports = _documented_exports(text, module_name)
        assert (
            documented_exports == runtime_exports
        ), f"{module_name} docs/PUBLIC_API.md exports drifted from __all__"
        missing_attrs = [name for name in runtime_exports if not hasattr(module, name)]
        assert (
            not missing_attrs
        ), f"{module_name}.__all__ contains missing attributes: {missing_attrs}"


def test_public_api_and_theory_docs_cover_fail_closed_math_contract() -> None:
    assert (
        PUBLIC_API_DOC.is_file()
    ), "docs/PUBLIC_API.md must document the public import surface"
    assert (
        THEORY_DOC.is_file()
    ), "docs/THEORY.md must be the canonical FD theory summary"
    public_text = PUBLIC_API_DOC.read_text(encoding="utf-8")
    theory_text = THEORY_DOC.read_text(encoding="utf-8")
    required_public_phrases = [
        "Package-root convenience exports are intentionally empty",
        "haircut.solver_backends",
        "Unsupported routes fail closed",
    ]
    required_theory_phrases = [
        "0.5 tr(a Hessian) + b dot grad - c value + source",
        "Boundary conditions are typed records",
        "A process dimension by itself is not sufficient to select ADI",
        "European PDE output must not be relabeled as American output",
    ]
    assert not [
        phrase for phrase in required_public_phrases if phrase not in public_text
    ]
    assert not [
        phrase for phrase in required_theory_phrases if phrase not in theory_text
    ]
