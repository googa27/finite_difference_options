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

FORBIDDEN_EXTERNAL_STACKS = {
    "fastapi",
    "typer",
    "uvicorn",
    "streamlit",
    "matplotlib",
    "seaborn",
    "plotly",
}

FORBIDDEN_INTERNAL_APP_PACKAGES = {
    "api",
    "cli",
    "plotting",
    "risk",
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


def _package_parts(path: Path) -> list[str]:
    relative_module = path.relative_to(SRC_ROOT).with_suffix("")
    parts = list(relative_module.parts)
    if parts and parts[-1] == "__init__":
        return parts[:-1]
    return parts[:-1]


def _resolve_import_from_base(path: Path, node: ast.ImportFrom) -> str:
    module_parts = node.module.split(".") if node.module else []
    if node.level == 0:
        return ".".join(module_parts)

    package_parts = _package_parts(path)
    keep = max(0, len(package_parts) - node.level + 1)
    return ".".join([*package_parts[:keep], *module_parts])


def _imports_from_tree(tree: ast.AST, path: Path) -> set[str]:
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_import_from_base(path, node)
            if base:
                imports.add(base)
            for alias in node.names:
                if alias.name == "*":
                    continue
                imports.add(f"{base}.{alias.name}" if base else alias.name)
    return imports


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return _imports_from_tree(tree, path)


def _without_src_prefix(import_name: str) -> str:
    return import_name.removeprefix("src.")


def _is_forbidden_core_import(import_name: str) -> bool:
    top_level = import_name.split(".")[0]
    if top_level in FORBIDDEN_EXTERNAL_STACKS:
        return True

    normalized = _without_src_prefix(import_name)
    return any(
        normalized == package or normalized.startswith(f"{package}.")
        for package in FORBIDDEN_INTERNAL_APP_PACKAGES
    )


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
    missing = TRANSITIONAL_SRC_PACKAGES - actual
    assert not unexpected, (
        "New top-level src packages must be added to docs/ARCHITECTURE.md and the architecture "
        f"baseline before code lands. Unexpected packages: {sorted(unexpected)}"
    )
    assert not missing, (
        "Removed top-level src packages must shrink docs/ARCHITECTURE.md and the architecture "
        f"baseline in the same PR. Missing packages: {sorted(missing)}"
    )


def test_import_parser_detects_src_prefixed_and_relative_app_imports() -> None:
    tree = ast.parse(
        "from src import plotting\n"
        "from src.plotting import plot_surface\n"
        "from ..plotting import Plotter\n"
        "from api import main\n"
        "from cli.main import app\n"
        "from src.risk.reporting_strategies import ReportFactory\n"
        "import matplotlib.pyplot\n"
    )
    imports = _imports_from_tree(tree, SRC_ROOT / "pricing" / "example.py")
    assert {
        "src.plotting",
        "src.plotting.plot_surface",
        "plotting",
        "plotting.Plotter",
        "api",
        "api.main",
        "cli.main",
        "cli.main.app",
        "src.risk.reporting_strategies",
        "src.risk.reporting_strategies.ReportFactory",
    } <= imports
    assert all(
        _is_forbidden_core_import(name)
        for name in [
            "src.plotting",
            "src.plotting.plot_surface",
            "plotting",
            "plotting.Plotter",
            "api",
            "api.main",
            "cli.main",
            "cli.main.app",
            "src.risk.reporting_strategies",
            "src.risk.reporting_strategies.ReportFactory",
            "matplotlib.pyplot",
        ]
    )


def test_numerical_core_does_not_import_optional_app_or_visualization_stacks() -> None:
    violations: dict[str, list[str]] = {}
    for package in sorted(NUMERICAL_CORE_PACKAGES):
        package_root = SRC_ROOT / package
        if not package_root.exists():
            continue
        for path in _python_files(package_root):
            forbidden = sorted(name for name in _imports(path) if _is_forbidden_core_import(name))
            if forbidden:
                violations[str(path.relative_to(ROOT))] = forbidden
    assert not violations, "Numerical core imported optional app/UI/visualization stacks: " + repr(violations)


def test_ci_exposes_architecture_gate() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    assert "tests/architecture" in workflow, "CI must run the architecture gate from tests/architecture."
