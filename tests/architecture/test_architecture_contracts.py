"""Executable architecture gates for the installable FD package topology."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.architecture

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
PACKAGE_ROOT = SRC_ROOT / "finite_difference_options"
ARCHITECTURE_DOC = ROOT / "docs" / "ARCHITECTURE.md"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
DISTRIBUTION_PACKAGE = "finite_difference_options"

EXPECTED_PACKAGE_BOUNDARIES = {
    "api",
    "boundary_conditions",
    "cli",
    "contracts",
    "exceptions",
    "greeks",
    "instruments",
    "integrations",
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
    "contracts",
    "exceptions",
    "greeks",
    "instruments",
    "integrations",
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
    "haircut-engine",
}


def test_architecture_contract_file_is_executable_source_of_truth() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_architecture_contract.py"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


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


def _without_distribution_prefix(import_name: str) -> str:
    if import_name == DISTRIBUTION_PACKAGE:
        return import_name
    return import_name.removeprefix(f"{DISTRIBUTION_PACKAGE}.")


def _is_forbidden_core_import(import_name: str) -> bool:
    if import_name == "src" or import_name.startswith("src."):
        return True

    top_level = import_name.split(".")[0]
    if top_level in FORBIDDEN_EXTERNAL_STACKS:
        return True

    normalized = _without_distribution_prefix(import_name)
    return any(
        normalized == package or normalized.startswith(f"{package}.")
        for package in FORBIDDEN_INTERNAL_APP_PACKAGES
    )


def test_architecture_document_exists_and_covers_required_package_rules() -> None:
    assert (
        ARCHITECTURE_DOC.is_file()
    ), "docs/ARCHITECTURE.md is the architecture source of truth."
    text = ARCHITECTURE_DOC.read_text(encoding="utf-8")
    missing = sorted(
        phrase for phrase in REQUIRED_ARCHITECTURE_PHRASES if phrase not in text
    )
    assert (
        not missing
    ), "docs/ARCHITECTURE.md is missing required package language: " + ", ".join(
        missing
    )


def test_current_distribution_package_boundaries_are_declared() -> None:
    actual = {
        path.name
        for path in PACKAGE_ROOT.iterdir()
        if path.is_dir() and not path.name.startswith("__") and any(path.rglob("*.py"))
    }
    unexpected = actual - EXPECTED_PACKAGE_BOUNDARIES
    missing = EXPECTED_PACKAGE_BOUNDARIES - actual
    assert not unexpected, (
        "New package boundaries must be added to docs/ARCHITECTURE.md and the architecture "
        f"baseline before code lands. Unexpected packages: {sorted(unexpected)}"
    )
    assert not missing, (
        "Removed package boundaries must shrink docs/ARCHITECTURE.md and the architecture "
        f"baseline in the same PR. Missing packages: {sorted(missing)}"
    )


def test_import_parser_detects_distribution_and_relative_app_imports() -> None:
    tree = ast.parse(
        "from finite_difference_options import plotting\n"
        "from finite_difference_options.plotting import plot_surface\n"
        "from ..plotting import Plotter\n"
        "from finite_difference_options.api import main\n"
        "from finite_difference_options.cli.main import app\n"
        "from finite_difference_options.risk.reporting_strategies import ReportFactory\n"
        "import matplotlib.pyplot\n"
        "import src.validation\n"
    )
    imports = _imports_from_tree(tree, PACKAGE_ROOT / "pricing" / "example.py")
    assert {
        "finite_difference_options",
        "finite_difference_options.plotting",
        "finite_difference_options.plotting.plot_surface",
        "finite_difference_options.plotting.Plotter",
        "finite_difference_options.api",
        "finite_difference_options.api.main",
        "finite_difference_options.cli.main",
        "finite_difference_options.cli.main.app",
        "finite_difference_options.risk.reporting_strategies",
        "finite_difference_options.risk.reporting_strategies.ReportFactory",
        "matplotlib.pyplot",
        "src",
        "src.validation",
    } <= imports
    assert all(
        _is_forbidden_core_import(name)
        for name in [
            "finite_difference_options.plotting",
            "finite_difference_options.plotting.plot_surface",
            "finite_difference_options.plotting.Plotter",
            "finite_difference_options.api",
            "finite_difference_options.api.main",
            "finite_difference_options.cli.main",
            "finite_difference_options.cli.main.app",
            "finite_difference_options.risk.reporting_strategies",
            "finite_difference_options.risk.reporting_strategies.ReportFactory",
            "matplotlib.pyplot",
            "src",
            "src.validation",
        ]
    )


def test_numerical_core_does_not_import_optional_app_or_visualization_stacks() -> None:
    violations: dict[str, list[str]] = {}
    for package in sorted(NUMERICAL_CORE_PACKAGES):
        package_root = PACKAGE_ROOT / package
        if not package_root.exists():
            continue
        for path in _python_files(package_root):
            forbidden = sorted(
                name for name in _imports(path) if _is_forbidden_core_import(name)
            )
            if forbidden:
                violations[str(path.relative_to(ROOT))] = forbidden
    assert (
        not violations
    ), "Numerical core imported optional app/UI/visualization stacks: " + repr(
        violations
    )


def test_base_package_import_surface_does_not_import_app_or_visualization_stacks() -> (
    None
):
    forbidden = sorted(
        name
        for name in _imports(PACKAGE_ROOT / "__init__.py")
        if _is_forbidden_core_import(name)
    )
    assert (
        not forbidden
    ), "Base package initializer imported app/UI/reporting stacks: " + repr(forbidden)


def test_no_public_code_or_tests_import_historical_src_package() -> None:
    violations: list[str] = []
    for root in [PACKAGE_ROOT, ROOT / "tests"]:
        for path in _python_files(root):
            historical = sorted(
                name
                for name in _imports(path)
                if name == "src" or name.startswith("src.")
            )
            if historical:
                violations.append(f"{path.relative_to(ROOT)}: {historical}")
    assert not violations, "Historical src imports remain: " + repr(violations)


def test_ci_exposes_packaging_and_architecture_gates() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    assert (
        "tests/architecture" in workflow
    ), "CI must run the architecture gate from tests/architecture."
    assert (
        "tests/test_packaging_contract.py" in workflow
    ), "CI must run the packaging contract."
    assert (
        "python -m build --sdist --wheel" in workflow
    ), "CI must build the sdist and wheel."
