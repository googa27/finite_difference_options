"""Executable architecture gates for the FD backend transition.

These checks intentionally ratchet the current transitional layout instead of
pretending the target ``finite_difference_options`` namespace already exists.
They implement the M0 gate from issue #60: architecture documentation exists,
new package growth is visible, and numerical core code cannot start depending
on app/UI/visualization stacks while the package migration proceeds.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.architecture

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
ARCHITECTURE_DOC = ROOT / "docs" / "ARCHITECTURE.md"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

# Transitional top-level packages present before the PEP 621/package-namespace
# migration in #51/#52. Adding a new one is an architecture event and must be
# reflected in docs/ARCHITECTURE.md before code lands.
TRANSITIONAL_SRC_PACKAGES = {
    "boundary_conditions",
    "exceptions",
    "greeks",
    "instruments",
    "models",
    "plotting",
    "pricing",
    "processes",
    "risk",
    "solvers",
    "utils",
    "validation",
}

NUMERICAL_CORE_PACKAGES = {
    "boundary_conditions",
    "exceptions",
    "greeks",
    "instruments",
    "models",
    "pricing",
    "processes",
    "solvers",
    "utils",
    "validation",
}

FORBIDDEN_CORE_IMPORTS = {
    "fastapi",
    "typer",
    "uvicorn",
    "streamlit",
    "matplotlib",
    "seaborn",
    "plotly",
}

REQUIRED_ARCHITECTURE_PHRASES = {
    "Target package topology",
    "Dependency direction",
    "Compatibility and deprecation policy",
    "Architecture fitness gates",
    "No module imports the distribution as `src`",
    "No new root package under `src/`",
    "deptry",
    "haircut-engine",
}


def _python_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts and "egg-info" not in path.parts
    )


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    return imports


def test_architecture_document_exists_and_covers_required_transition_rules() -> None:
    assert ARCHITECTURE_DOC.is_file(), "docs/ARCHITECTURE.md is the M0 architecture source of truth."
    text = ARCHITECTURE_DOC.read_text(encoding="utf-8")
    missing = sorted(phrase for phrase in REQUIRED_ARCHITECTURE_PHRASES if phrase not in text)
    assert not missing, "docs/ARCHITECTURE.md is missing required M0 transition language: " + ", ".join(missing)


def test_current_src_packages_are_declared_transition_baseline() -> None:
    actual = {
        path.name
        for path in SRC_ROOT.iterdir()
        if path.is_dir() and not path.name.startswith("__") and any(path.rglob("*.py"))
    }
    unexpected = actual - TRANSITIONAL_SRC_PACKAGES
    assert not unexpected, (
        "New top-level src packages must be added to docs/ARCHITECTURE.md and the architecture "
        f"baseline before code lands. Unexpected packages: {sorted(unexpected)}"
    )


def test_numerical_core_does_not_import_optional_app_or_visualization_stacks() -> None:
    violations: dict[str, list[str]] = {}
    for package in sorted(NUMERICAL_CORE_PACKAGES):
        package_root = SRC_ROOT / package
        if not package_root.exists():
            continue
        for path in _python_files(package_root):
            forbidden = sorted(_imports(path).intersection(FORBIDDEN_CORE_IMPORTS))
            if forbidden:
                violations[str(path.relative_to(ROOT))] = forbidden
    assert not violations, "Numerical core imported optional app/UI/visualization stacks: " + repr(violations)


def test_ci_exposes_architecture_gate() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    assert "tests/architecture" in workflow, "CI must run the architecture gate from tests/architecture."
