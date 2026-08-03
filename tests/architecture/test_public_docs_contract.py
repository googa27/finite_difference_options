from __future__ import annotations

import ast
import fnmatch
import importlib
import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_API_DOC = ROOT / "docs" / "PUBLIC_API.md"
THEORY_DOC = ROOT / "docs" / "THEORY.md"
ARCHITECTURE_CONTRACT = ROOT / "docs" / "ARCHITECTURE.yaml"
PYPROJECT = ROOT / "pyproject.toml"


def _init_declares_all(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ):
            return True
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "__all__":
            return True
    return False


def _public_facade_modules() -> tuple[str, ...]:
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    package_find = pyproject["tool"]["setuptools"]["packages"]["find"]
    source_roots = [ROOT / root for root in package_find["where"]]
    includes = tuple(package_find.get("include", ["*"]))
    excludes = tuple(package_find.get("exclude", []))
    modules: set[str] = set()
    for source_root in source_roots:
        for init_file in source_root.rglob("__init__.py"):
            module_name = init_file.parent.relative_to(source_root).as_posix().replace("/", ".")
            if not any(fnmatch.fnmatchcase(module_name, pattern) for pattern in includes):
                continue
            if any(fnmatch.fnmatchcase(module_name, pattern) for pattern in excludes):
                continue
            if _init_declares_all(init_file):
                modules.add(module_name)
    return tuple(sorted(modules))


def _documented_exports(text: str, module_name: str) -> list[str]:
    pattern = re.compile(
        rf"<!-- public-api:{re.escape(module_name)}:start -->(.*?)"
        rf"<!-- public-api:{re.escape(module_name)}:end -->",
        re.DOTALL,
    )
    match = pattern.search(text)
    assert match is not None, f"docs/PUBLIC_API.md is missing a synchronized section for {module_name}"
    return re.findall(r"^- `([^`]+)`$", match.group(1), flags=re.MULTILINE)


def test_public_api_doc_is_synchronized_to_runtime_all_exports() -> None:
    text = PUBLIC_API_DOC.read_text(encoding="utf-8")
    modules = _public_facade_modules()
    assert {
        "finite_difference_options.instruments",
        "finite_difference_options.exceptions",
        "finite_difference_options.models",
        "finite_difference_options.risk",
    }.issubset(modules)
    for module_name in modules:
        module = importlib.import_module(module_name)
        runtime_exports = list(getattr(module, "__all__", []))
        documented_exports = _documented_exports(text, module_name)
        assert documented_exports == runtime_exports, f"{module_name} docs/PUBLIC_API.md exports drifted from __all__"
        missing_attrs = [name for name in runtime_exports if not hasattr(module, name)]
        assert not missing_attrs, f"{module_name}.__all__ contains missing attributes: {missing_attrs}"


def test_architecture_yaml_tracks_public_api_theory_and_time_grid_contracts() -> None:
    contract = json.loads(ARCHITECTURE_CONTRACT.read_text(encoding="utf-8"))
    required_documents = set(contract["governance"]["required_documents"])
    assert {"docs/PUBLIC_API.md", "docs/THEORY.md"}.issubset(required_documents)
    public_contracts = contract["architecture"]["public_contracts"]
    assert public_contracts["public_api_manifest"]["document"] == "docs/PUBLIC_API.md"
    assert public_contracts["theory_contract"]["document"] == "docs/THEORY.md"
    assert public_contracts["time_grid"]["solver"] == "finite_difference_options.solvers.FiniteDifferenceSolver.solve"
    assert public_contracts["time_grid"]["monotonicity"] == "finite_strictly_increasing_1d_nodes"
    assert public_contracts["time_grid"]["nonuniform_intervals"] == "each_interval_used_as_step_dt"


def test_public_api_and_theory_docs_cover_fail_closed_math_contract() -> None:
    assert PUBLIC_API_DOC.is_file(), "docs/PUBLIC_API.md must document the public import surface"
    assert THEORY_DOC.is_file(), "docs/THEORY.md must be the canonical FD theory summary"
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
    assert not [phrase for phrase in required_public_phrases if phrase not in public_text]
    assert not [phrase for phrase in required_theory_phrases if phrase not in theory_text]
