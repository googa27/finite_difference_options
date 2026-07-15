"""Package metadata and installed-import contract for finite_difference_options."""

from __future__ import annotations

import ast
import importlib
import importlib.metadata as metadata
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.packaging

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "finite_difference_options"
DIST = "finite-difference-options"


def _python_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*.py") if "__pycache__" not in path.parts and "egg-info" not in path.parts
    )


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
                imports.add(node.module.split(".")[0])
    return imports


def _requires_dist() -> list[str]:
    return metadata.metadata(DIST).get_all("Requires-Dist") or []


def _has_extra_dependency(requires_dist: list[str], extra: str, dependency: str) -> bool:
    marker = f'extra == "{extra}"'
    return any(req.lower().startswith(dependency.lower()) and marker in req for req in requires_dist)


def test_distribution_metadata_declares_supported_runtime_contract() -> None:
    project = metadata.metadata(DIST)
    assert project["Name"] == DIST
    requires_python = project["Requires-Python"]
    assert ">=3.12" in requires_python
    assert "<3.13" in requires_python

    requires_dist = _requires_dist()
    for dependency in ["findiff", "numpy", "pydantic", "scipy"]:
        assert any(req.lower().startswith(dependency) for req in requires_dist)

    joined = "\n".join(requires_dist).lower()
    for optional in [
        "fastapi",
        "uvicorn",
        "typer",
        "streamlit",
        "matplotlib",
        "plotly",
        "seaborn",
    ]:
        assert f"{optional}" in joined


def test_core_metadata_keeps_application_stacks_optional() -> None:
    requires_dist = _requires_dist()
    unconditional = [req for req in requires_dist if "extra ==" not in req]
    assert not any(req.lower().startswith(("fastapi", "uvicorn", "typer", "streamlit")) for req in unconditional)
    assert not any(req.lower().startswith(("matplotlib", "plotly", "seaborn")) for req in unconditional)

    for extra, dependency in [
        ("api", "fastapi"),
        ("api", "uvicorn"),
        ("cli", "typer"),
        ("ui", "streamlit"),
        ("viz", "matplotlib"),
        ("viz", "plotly"),
        ("viz", "seaborn"),
        ("validation", "pytest"),
        ("build", "build"),
        ("audit", "pip-audit"),
    ]:
        assert _has_extra_dependency(requires_dist, extra, dependency)


def test_public_distribution_import_surface_is_real_package_namespace() -> None:
    package = importlib.import_module(PACKAGE)
    assert package.__name__ == PACKAGE
    for module_name in [
        "finite_difference_options.contracts",
        "finite_difference_options.integrations.haircut_backend",
        "finite_difference_options.validation.black_scholes_parity",
        "finite_difference_options.pricing",
        "finite_difference_options.pricing.boundary_conditions",
        "finite_difference_options.solvers",
    ]:
        importlib.import_module(module_name)

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.validation.black_scholes_parity")


def test_entry_points_advertise_cli_and_haircut_backend() -> None:
    console_scripts = metadata.entry_points(group="console_scripts")
    assert any(
        ep.name == "fd-options" and ep.value == "finite_difference_options.cli.main:app" for ep in console_scripts
    )

    solver_backends = metadata.entry_points(group="haircut.solver_backends")
    matches = [
        ep
        for ep in solver_backends
        if ep.name == "finite_difference_options"
        and ep.value == "finite_difference_options.integrations.haircut_backend:create_backend"
    ]
    assert len(matches) == 1
    assert not [
        ep
        for ep in metadata.entry_points(group="haircut_engine.solver_backends")
        if ep.name == "finite_difference_options"
    ]


def test_python_sources_do_not_mutate_sys_path_for_checkout_only_imports() -> None:
    violations: list[str] = []
    for root in [ROOT / "tests", ROOT / "src" / PACKAGE, ROOT / "scripts"]:
        for path in _python_files(root):
            text = path.read_text(encoding="utf-8")
            forbidden_path_mutations = ("sys.path." + "append", "sys.path." + "insert")
            if any(token in text for token in forbidden_path_mutations):
                violations.append(str(path.relative_to(ROOT)))
    assert not violations, "Checkout-only sys.path mutation remains: " + repr(violations)


def test_sources_and_tests_do_not_import_historical_src_package() -> None:
    violations: list[str] = []
    for root in [ROOT / "tests", ROOT / "src" / PACKAGE, ROOT / "scripts"]:
        for path in _python_files(root):
            historical = sorted(name for name in _imports(path) if name == "src" or name.startswith("src."))
            if historical:
                violations.append(f"{path.relative_to(ROOT)}: {historical}")
    assert not violations, "Historical src imports remain: " + repr(violations)


def test_wheel_contains_only_real_distribution_package(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist_dir)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheels = sorted(dist_dir.glob("finite_difference_options-*.whl"))
    assert len(wheels) == 1

    with zipfile.ZipFile(wheels[0]) as archive:
        names = set(archive.namelist())

    assert "finite_difference_options/__init__.py" in names
    assert "finite_difference_options/py.typed" in names
    assert any(name.startswith("finite_difference_options/contracts/") for name in names)
    assert any(name.startswith("finite_difference_options/integrations/") for name in names)
    assert "finite_difference_options/pricing/boundary_conditions/__init__.py" in names
    assert "finite_difference_options/validation/fixtures/compiled_pde_black_scholes_call_v0.json" in names
    assert not any(name == "src/__init__.py" or name.startswith("src/") for name in names)
    assert not any(name.startswith(("tests/", ".agent_workspace/", ".gemini_project/")) for name in names)
