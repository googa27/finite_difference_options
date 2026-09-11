"""The compiled DTO/validation/metric boundaries are enforced, not just listed."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

pytestmark = pytest.mark.architecture
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def contract_copy(tmp_path):
    spec = importlib.util.spec_from_file_location("fd_boundary_gate", ROOT / "scripts/check_architecture_contract.py")
    assert spec is not None and spec.loader is not None
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    for name in ("src", "docs", ".github"):
        shutil.copytree(ROOT / name, tmp_path / name, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "AGENTS.md", tmp_path / "AGENTS.md")
    return checker, tmp_path


@pytest.mark.parametrize(
    "module",
    [
        "integrations/_compiled_pde_contracts.py",
        "integrations/_compiled_pde_validation.py",
        "validation/fd_evidence/grid_metrics.py",
    ],
)
def test_extracted_module_is_required_by_contract(contract_copy, module):
    checker, root = contract_copy
    (root / "src/finite_difference_options" / module).unlink()
    errors = checker.validate_contract(root, root / "docs/architecture_contract.toml")
    assert any("canonical implementation path is missing" in error and module in error for error in errors), errors


@pytest.mark.parametrize(
    ("module", "statement"),
    [
        (
            "integrations/_compiled_pde_contracts.py",
            "from finite_difference_options.integrations import compiled_pde_adapter",
        ),
        ("integrations/_compiled_pde_contracts.py", "from . import compiled_pde_adapter"),
        ("integrations/_compiled_pde_contracts.py", "from .compiled_pde_adapter import screen_compiled_pde_payload"),
        (
            "integrations/_compiled_pde_validation.py",
            "import finite_difference_options.integrations.compiled_pde_adapter",
        ),
        ("integrations/_compiled_pde_validation.py", "from . import compiled_pde_adapter"),
        ("integrations/_compiled_pde_validation.py", "from ..validation import fd_verification"),
        ("validation/fd_evidence/grid_metrics.py", "from finite_difference_options.validation import fd_verification"),
        ("validation/fd_evidence/grid_metrics.py", "from .. import fd_verification"),
        ("validation/fd_evidence/grid_metrics.py", "from ..fd_verification import run_fd_bs_verification_benchmark"),
    ],
)
def test_extracted_modules_cannot_import_orchestration_backwards(contract_copy, module, statement):
    checker, root = contract_copy
    source = root / "src/finite_difference_options" / module
    with source.open("a") as stream:
        stream.write("\n" + statement + "\n")
    errors = checker.validate_contract(root, root / "docs/architecture_contract.toml")
    assert any("Import rule violation" in error and module in error for error in errors), errors
