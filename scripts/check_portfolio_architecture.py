#!/usr/bin/env python3
"""Validate the portfolio architecture contract without third-party dependencies.

`docs/ARCHITECTURE.yaml` is deliberately written in the JSON subset of YAML 1.2,
so the standard-library JSON parser is sufficient for the bootstrap gate. Richer
repository gates may add PyYAML or check-jsonschema; this checker never attempts
to reimplement a YAML parser.
"""

from __future__ import annotations

import ast
import json
import math
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs" / "ARCHITECTURE.yaml"
DEFAULT_MAX_ENTRIES = 10
DEFAULT_MAX_LINES = 500
REQUIRED_TOP_LEVEL = {
    "schema_version",
    "repository",
    "architecture",
    "source_layout",
    "limits",
    "libraries",
    "interfaces",
    "tests",
    "data",
    "governance",
    "exceptions",
}
REQUIRED_EXCEPTION_FIELDS = {
    "rule",
    "path",
    "reason",
    "owner",
    "risk",
    "accepted_ceiling",
    "refactoring_trigger",
}
REQUIRED_AI_POLICY = {
    "output_trust": "untrusted_until_human_review_and_executable_verification",
    "human_accountability": True,
    "change_scope": "small_reviewable_single_purpose_slices",
    "agent_generated_tests": "not_sufficient_as_sole_oracle",
    "dependency_changes": "human_approval_and_existence_maintenance_license_security_verification",
    "high_risk_review": "human_required",
    "least_privilege": "workspace_scoped_network_and_secret_access_requires_approval",
    "provenance": "record_agent_assistance_and_verification_evidence",
    "measurement": "objective_metrics_not_self_report",
}
REQUIRED_AI_METRICS = {
    "lead_time",
    "review_time",
    "ci_failure_rate",
    "revert_rate",
    "defect_escape_rate",
    "code_churn",
}
REQUIRED_HIERARCHY_POLICY = {
    "principle": "semantic_cohesion_and_low_coupling_not_equal_branch_size",
    "empty_runtime_directories": "forbidden_without_exact_exception",
    "init_modules": "facade_only_no_domain_implementation",
    "concentration": "review_trigger_not_rebalance_mandate",
    "minimum_branches": 3,
    "minimum_descendant_modules": 20,
    "include_direct_modules_as_branch": True,
    "new_or_worsened_unclassified_imbalance": "forbidden",
}
REQUIRED_STRUCTURAL_ROLES = {
    "namespace_package",
    "compatibility_facade",
    "generated_mount",
    "plugin_namespace",
    "adapter_namespace",
    "test_mirror",
    "package_data",
    "monorepo_boundary",
}
HIERARCHY_EXCEPTION_RULES = {
    "empty_runtime_directory",
    "hierarchy_imbalance",
    "init_module_implementation",
}
ALLOWED_INIT_FUNCTIONS = {"__getattr__", "__dir__"}
IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "build",
    "dist",
}
DEFAULT_METADATA = {
    "__init__.py",
    "README.md",
    "ARCHITECTURE.md",
    "ARCHITECTURE.yaml",
    "py.typed",
}


class ContractLoadError(ValueError):
    """Architecture contract could not be loaded."""

    @classmethod
    def missing(cls, relative_path: Path) -> ContractLoadError:
        return cls(f"missing {relative_path}")

    @classmethod
    def invalid_json_subset(cls, error: json.JSONDecodeError) -> ContractLoadError:
        return cls(
            "docs/ARCHITECTURE.yaml must remain in the JSON-compatible YAML 1.2 "
            f"subset for the dependency-free bootstrap checker: {error}"
        )


class ContractShapeError(TypeError):
    """Architecture contract has an invalid root shape."""

    def __init__(self) -> None:
        super().__init__("architecture contract root must be an object")


def ignored_name(name: str) -> bool:
    return name in IGNORED_DIRS or name.endswith(".egg-info")


def ignored_path(path: Path) -> bool:
    return any(ignored_name(part) for part in path.parts)


def load_contract() -> dict[str, Any]:
    try:
        payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractLoadError.missing(CONTRACT.relative_to(ROOT)) from exc
    except json.JSONDecodeError as exc:
        raise ContractLoadError.invalid_json_subset(exc) from exc
    if not isinstance(payload, dict):
        raise ContractShapeError
    return payload


def exception_map(
    contract: dict[str, Any], errors: list[str]
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for index, item in enumerate(contract.get("exceptions", [])):
        if not isinstance(item, dict):
            errors.append(f"exceptions[{index}] must be an object")
            continue
        missing = REQUIRED_EXCEPTION_FIELDS - set(item)
        if missing:
            errors.append(f"exceptions[{index}] missing metadata: {sorted(missing)}")
            continue
        key = (str(item["rule"]), str(item["path"]))
        if key[0] in HIERARCHY_EXCEPTION_RULES:
            role = item.get("structural_role")
            if role not in REQUIRED_STRUCTURAL_ROLES:
                errors.append(
                    f"exceptions[{index}].structural_role must be one of "
                    f"{sorted(REQUIRED_STRUCTURAL_ROLES)} for {key[0]}"
                )
            review_by = item.get("review_by")
            try:
                review_date = date.fromisoformat(str(review_by))
            except ValueError:
                errors.append(
                    f"exceptions[{index}].review_by must be an ISO date for {key[0]}"
                )
            else:
                if review_date < date.today():
                    errors.append(
                        f"exceptions[{index}] hierarchy review expired on {review_date}"
                    )
            evidence = item.get("evidence")
            if not isinstance(evidence, str) or not evidence.strip():
                errors.append(
                    f"exceptions[{index}].evidence is required for {key[0]}"
                )
        if key in result:
            errors.append(f"duplicate exception for {key[0]}:{key[1]}")
        result[key] = item
    return result


def require_exception(
    exceptions: dict[tuple[str, str], dict[str, Any]],
    rule: str,
    path: str,
    actual: int,
    errors: list[str],
) -> None:
    item = exceptions.get((rule, path))
    if item is None:
        errors.append(f"{rule} violation at {path}: {actual}; no documented exception")
        return
    ceiling = item.get("accepted_ceiling")
    if not isinstance(ceiling, int):
        errors.append(f"{rule} exception at {path} must have integer accepted_ceiling")
    elif actual > ceiling:
        errors.append(f"{rule} no-growth ratchet exceeded at {path}: {actual}>{ceiling}")


def runtime_dir(path: Path) -> bool:
    try:
        return any(
            candidate.suffix == ".py"
            for candidate in path.rglob("*.py")
            if not ignored_path(candidate)
        )
    except OSError:
        return False


def _init_contains_implementation(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return ["unparseable"]
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            names.append(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name not in ALLOWED_INIT_FUNCTIONS:
                names.append(node.name)
    return names


def _marker_only_package(path: Path) -> bool:
    init_path = path / "__init__.py"
    if not init_path.is_file():
        return False
    metadata = DEFAULT_METADATA | {".gitkeep"}
    for candidate in path.rglob("*"):
        if ignored_path(candidate) or not candidate.is_file() or candidate == init_path:
            continue
        if candidate.name not in metadata:
            return False
    try:
        tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return False
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                continue
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            return False
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Pass)):
            continue
        return False
    return True


def _concentration_trigger(branch_counts: list[int]) -> tuple[bool, float, float]:
    branch_count = len(branch_counts)
    total = sum(branch_counts)
    shares = [count / total for count in branch_counts]
    largest_share = max(shares)
    entropy = -sum(share * math.log(share) for share in shares)
    effective_branches = math.exp(entropy)
    effective_fraction = effective_branches / branch_count
    if branch_count == 3:
        triggered = largest_share >= 0.85 and (
            effective_branches <= 2.25 or effective_fraction <= 0.60
        )
    elif branch_count <= 5:
        triggered = largest_share >= 0.80 and (
            effective_branches <= 2.50 or effective_fraction <= 0.50
        )
    elif branch_count <= 7:
        triggered = largest_share >= 0.70 and effective_fraction <= 0.45
    else:
        triggered = largest_share >= 0.65 and effective_fraction <= 0.45
    return triggered, largest_share, effective_branches


def validate_hierarchy_policy(
    contract: dict[str, Any],
    exceptions: dict[tuple[str, str], dict[str, Any]],
    errors: list[str],
) -> None:
    layout = contract["source_layout"]
    policy = layout.get("hierarchy_policy")
    if not isinstance(policy, dict):
        errors.append("source_layout.hierarchy_policy must be an object")
        return
    for key, expected in REQUIRED_HIERARCHY_POLICY.items():
        if policy.get(key) != expected:
            errors.append(f"source_layout.hierarchy_policy.{key} must be {expected!r}")
    roles = policy.get("structural_role_exclusions")
    if not isinstance(roles, list) or set(roles) != REQUIRED_STRUCTURAL_ROLES:
        errors.append(
            "source_layout.hierarchy_policy.structural_role_exclusions must contain exactly "
            f"{sorted(REQUIRED_STRUCTURAL_ROLES)}"
        )
    evidence = policy.get("evidence")
    if not isinstance(evidence, list) or len(evidence) < 4:
        errors.append("source_layout.hierarchy_policy.evidence must contain at least four sources")
    else:
        for index, item in enumerate(evidence):
            if not isinstance(item, dict) or not str(item.get("source", "")).startswith("https://"):
                errors.append(
                    f"source_layout.hierarchy_policy.evidence[{index}] needs an HTTPS source"
                )
            if not isinstance(item, dict) or not str(item.get("finding", "")).strip():
                errors.append(
                    f"source_layout.hierarchy_policy.evidence[{index}] needs a finding"
                )
    if not layout.get("python_rules_applicable", True):
        return
    minimum_branches = int(policy.get("minimum_branches", 3))
    minimum_modules = int(policy.get("minimum_descendant_modules", 20))
    for rel_root in layout.get("python_source_roots", []):
        source_root = ROOT / rel_root
        if not source_root.is_dir():
            continue
        directories = [source_root, *sorted(path for path in source_root.rglob("*") if path.is_dir())]
        for current in directories:
            if ignored_path(current):
                continue
            rel_dir = current.relative_to(ROOT).as_posix()
            if _marker_only_package(current):
                require_exception(
                    exceptions, "empty_runtime_directory", rel_dir, 1, errors
                )
            init_path = current / "__init__.py"
            if init_path.is_file():
                names = _init_contains_implementation(init_path)
                if names:
                    rel_init = init_path.relative_to(ROOT).as_posix()
                    if ("init_module_implementation", rel_init) not in exceptions:
                        errors.append(
                            "init_module_implementation violation at "
                            f"{rel_init}: {sorted(names)}; move implementation to cohesive modules "
                            "and keep __init__.py as a facade"
                        )
            branches: list[tuple[str, int]] = []
            direct_modules = sum(
                path.suffix == ".py" and path.name != "__init__.py"
                for path in current.iterdir()
                if path.is_file()
            )
            if direct_modules:
                branches.append(("__direct_modules__", direct_modules))
            for child in sorted(path for path in current.iterdir() if path.is_dir()):
                if ignored_path(child) or child.name.startswith("."):
                    continue
                count = sum(
                    candidate.name != "__init__.py" and not ignored_path(candidate)
                    for candidate in child.rglob("*.py")
                )
                if count:
                    branches.append((child.name, count))
            counts = [count for _, count in branches]
            if len(counts) < minimum_branches or sum(counts) < minimum_modules:
                continue
            triggered, largest_share, effective_branches = _concentration_trigger(counts)
            if triggered and ("hierarchy_imbalance", rel_dir) not in exceptions:
                errors.append(
                    "hierarchy_imbalance review required at "
                    f"{rel_dir}: branches={branches}, largest_share={largest_share:.3f}, "
                    f"effective_branches={effective_branches:.2f}; semantic review is required, "
                    "not mechanical rebalancing"
                )


def validate_source(
    contract: dict[str, Any],
    exceptions: dict[tuple[str, str], dict[str, Any]],
    errors: list[str],
) -> None:
    layout = contract["source_layout"]
    if not layout.get("python_rules_applicable", True):
        return
    max_entries = int(contract["limits"]["max_immediate_runtime_entries"])
    max_lines = int(contract["limits"]["max_python_module_lines"])
    allowed_non_python = set(layout.get("allowed_non_python_files", []))
    metadata = DEFAULT_METADATA | set(layout.get("metadata_names", []))
    roots = [ROOT / path for path in layout.get("python_source_roots", [])]
    for source_root in roots:
        rel_root = source_root.relative_to(ROOT).as_posix()
        if not source_root.is_dir():
            errors.append(f"declared Python source root is missing: {rel_root}")
            continue
        for current, dirs, files in os.walk(source_root):
            dirs[:] = sorted(
                name for name in dirs if not ignored_name(name) and not name.startswith(".")
            )
            current_path = Path(current)
            rel_dir = current_path.relative_to(ROOT).as_posix()
            runtime_dirs = [name for name in dirs if runtime_dir(current_path / name)]
            runtime_files = [
                name for name in files if name.endswith(".py") and name != "__init__.py"
            ]
            count = len(runtime_dirs) + len(runtime_files)
            if count > max_entries:
                require_exception(exceptions, "source_fanout", rel_dir, count, errors)
            for filename in files:
                rel = (current_path / filename).relative_to(ROOT).as_posix()
                allowed = (
                    filename.endswith((".py", ".pyi"))
                    or filename in metadata
                    or rel in allowed_non_python
                )
                if not allowed:
                    require_exception(exceptions, "source_entry_type", rel, 1, errors)
        for module in sorted(source_root.rglob("*.py")):
            if ignored_path(module):
                continue
            try:
                lines = len(module.read_text(encoding="utf-8").splitlines())
            except UnicodeDecodeError:
                errors.append(f"Python module is not UTF-8 text: {module.relative_to(ROOT)}")
                continue
            if lines > max_lines:
                require_exception(
                    exceptions,
                    "python_module_max_lines",
                    module.relative_to(ROOT).as_posix(),
                    lines,
                    errors,
                )


def validate_repository(contract: dict[str, Any], errors: list[str]) -> None:
    repository = contract["repository"]
    for key in ("owner", "name", "profile", "status"):
        if not repository.get(key):
            errors.append(f"repository.{key} is required")


def validate_limits(contract: dict[str, Any], errors: list[str]) -> None:
    limits = contract["limits"]
    if limits.get("max_immediate_runtime_entries") != DEFAULT_MAX_ENTRIES:
        errors.append(
            "default max_immediate_runtime_entries must be 10; "
            "repo override belongs in a documented exception"
        )
    if limits.get("max_python_module_lines") != DEFAULT_MAX_LINES:
        errors.append(
            "default max_python_module_lines must be 500; "
            "repo override belongs in a documented exception"
        )


def validate_documents_and_tests(contract: dict[str, Any], errors: list[str]) -> None:
    for rel in contract["governance"].get("required_documents", []):
        path = ROOT / rel
        exists_with_content = path.is_file() and bool(
            path.read_text(encoding="utf-8", errors="ignore").strip()
        )
        if not exists_with_content:
            errors.append(f"required document missing or empty: {rel}")
    for suite in contract["tests"].get("required_suites", []):
        if not (ROOT / "tests" / suite).is_dir():
            errors.append(f"required test suite directory missing: tests/{suite}")


def validate_interfaces(contract: dict[str, Any], errors: list[str]) -> None:
    ai = contract["interfaces"].get("ai", {})
    human = contract["interfaces"].get("human", {})
    if ai.get("context_file") != "AGENTS.md":
        errors.append("interfaces.ai.context_file must be AGENTS.md")
    if not ai.get("interaction") or not ai.get("capability_discovery"):
        errors.append("AI interaction and capability discovery decisions are required")
    if not human.get("interaction") or not human.get("dunder_policy"):
        errors.append("human interaction and dunder policy decisions are required")


def validate_libraries_and_data(contract: dict[str, Any], errors: list[str]) -> None:
    libraries = contract["libraries"]
    if not libraries.get("selection_policy"):
        errors.append("maintained-library selection policy is required")
    if not isinstance(libraries.get("decisions"), list):
        errors.append("libraries.decisions must be a list")
    core = contract["data"].get("core_repositories", {})
    for name in ("PDP", "financial_problem_formulations", "ui_and_artifacts"):
        if name not in core:
            errors.append(f"data.core_repositories must decide {name} posture")


def validate_ai_assisted_development(contract: dict[str, Any], errors: list[str]) -> None:
    governance = contract.get("governance", {})
    policy = governance.get("ai_assisted_development")
    if not isinstance(policy, dict):
        errors.append("governance.ai_assisted_development must be an object")
        return
    for key, expected in REQUIRED_AI_POLICY.items():
        if policy.get(key) != expected:
            errors.append(
                f"governance.ai_assisted_development.{key} must be {expected!r}"
            )
    metrics = policy.get("metrics")
    if not isinstance(metrics, list) or set(metrics) != REQUIRED_AI_METRICS:
        errors.append(
            "governance.ai_assisted_development.metrics must contain exactly "
            f"{sorted(REQUIRED_AI_METRICS)}"
        )
    evidence = policy.get("evidence")
    if not isinstance(evidence, list) or len(evidence) < 4:
        errors.append(
            "governance.ai_assisted_development.evidence must contain at least four sources"
        )
        return
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            errors.append(
                f"governance.ai_assisted_development.evidence[{index}] must be an object"
            )
            continue
        source = item.get("source")
        finding = item.get("finding")
        if not isinstance(source, str) or not source.startswith("https://"):
            errors.append(
                f"governance.ai_assisted_development.evidence[{index}].source "
                "must be an HTTPS URL"
            )
        if not isinstance(finding, str) or not finding.strip():
            errors.append(
                f"governance.ai_assisted_development.evidence[{index}].finding is required"
            )


def validate_contract(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_TOP_LEVEL - set(contract)
    if missing:
        return [f"contract missing top-level keys: {sorted(missing)}"]
    exceptions = exception_map(contract, errors)
    validate_repository(contract, errors)
    validate_limits(contract, errors)
    validate_documents_and_tests(contract, errors)
    validate_interfaces(contract, errors)
    validate_libraries_and_data(contract, errors)
    validate_ai_assisted_development(contract, errors)
    validate_hierarchy_policy(contract, exceptions, errors)
    validate_source(contract, exceptions, errors)
    return errors


def write_line(message: str) -> None:
    sys.stdout.write(f"{message}\n")


def main() -> int:
    try:
        contract = load_contract()
    except (ContractLoadError, ContractShapeError) as exc:
        write_line(f"architecture contract FAILED\n- {exc}")
        return 1
    errors = validate_contract(contract)
    if errors:
        write_line("architecture contract FAILED")
        for error in errors:
            write_line(f"- {error}")
        return 1
    repository = contract["repository"]
    write_line(
        "architecture contract OK: "
        f"{repository['owner']}/{repository['name']} profile={repository['profile']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
