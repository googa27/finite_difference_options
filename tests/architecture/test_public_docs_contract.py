from __future__ import annotations

import ast
import builtins
import fnmatch
import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_API_DOC = ROOT / "docs" / "PUBLIC_API.md"
THEORY_DOC = ROOT / "docs" / "THEORY.md"
ARCHITECTURE_CONTRACT = ROOT / "docs" / "ARCHITECTURE.yaml"
PYPROJECT = ROOT / "pyproject.toml"


def _literal_string_sequence(node: ast.AST, constants: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    if isinstance(node, ast.List | ast.Tuple):
        values: list[str] = []
        for element in node.elts:
            if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
                raise AssertionError(f"unsupported non-literal __all__ element: {ast.dump(element)}")
            values.append(element.value)
        return tuple(values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _literal_string_sequence(node.left, constants) + _literal_string_sequence(node.right, constants)
    if isinstance(node, ast.Name) and node.id in constants:
        return constants[node.id]
    raise AssertionError(f"unsupported __all__ expression: {ast.dump(node)}")


def _name_targets(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, ast.Tuple | ast.List):
        return {name for element in target.elts for name in _name_targets(element)}
    return set()


def _facade_all_exports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    constants: dict[str, tuple[str, ...]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [name for target in node.targets for name in _name_targets(target)]
            if "__all__" in targets:
                return _literal_string_sequence(node.value, constants)
            for name in targets:
                try:
                    constants[name] = _literal_string_sequence(node.value, constants)
                except AssertionError:
                    constants.pop(name, None)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            if node.target.id == "__all__":
                return _literal_string_sequence(node.value, constants)
            try:
                constants[node.target.id] = _literal_string_sequence(node.value, constants)
            except AssertionError:
                constants.pop(node.target.id, None)
    raise AssertionError(f"{path} does not declare a statically parseable top-level __all__")


def _public_facade_init_files() -> dict[str, Path]:
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    package_find = pyproject["tool"]["setuptools"]["packages"]["find"]
    source_roots = [ROOT / root for root in package_find["where"]]
    includes = tuple(package_find.get("include", ["*"]))
    excludes = tuple(package_find.get("exclude", []))
    modules: dict[str, Path] = {}
    for source_root in source_roots:
        for init_file in source_root.rglob("__init__.py"):
            module_name = init_file.parent.relative_to(source_root).as_posix().replace("/", ".")
            if not any(fnmatch.fnmatchcase(module_name, pattern) for pattern in includes):
                continue
            if any(fnmatch.fnmatchcase(module_name, pattern) for pattern in excludes):
                continue
            try:
                _facade_all_exports(init_file)
            except AssertionError:
                continue
            modules[module_name] = init_file
    return dict(sorted(modules.items()))


def _public_facade_modules() -> tuple[str, ...]:
    return tuple(_public_facade_init_files())


def _documented_exports(text: str, module_name: str) -> list[str]:
    pattern = re.compile(
        rf"<!-- public-api:{re.escape(module_name)}:start -->(.*?)"
        rf"<!-- public-api:{re.escape(module_name)}:end -->",
        re.DOTALL,
    )
    match = pattern.search(text)
    assert match is not None, f"docs/PUBLIC_API.md is missing a synchronized section for {module_name}"
    return re.findall(r"^- `([^`]+)`$", match.group(1), flags=re.MULTILINE)


def _bound_names_in_statement(node: ast.stmt) -> set[str]:
    if isinstance(node, ast.Import):
        return {alias.asname or alias.name.split(".", 1)[0] for alias in node.names}
    if isinstance(node, ast.ImportFrom):
        return {alias.asname or alias.name for alias in node.names if alias.name != "*"}
    if isinstance(node, ast.Assign):
        return {name for target in node.targets for name in _name_targets(target)}
    if isinstance(node, ast.AnnAssign):
        return _name_targets(node.target)
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return {node.name}
    if isinstance(node, ast.Try):
        statements = [*node.body, *node.orelse, *node.finalbody]
        for handler in node.handlers:
            statements.extend(handler.body)
        return {name for statement in statements for name in _bound_names_in_statement(statement)}
    if isinstance(node, ast.If):
        return {name for statement in [*node.body, *node.orelse] for name in _bound_names_in_statement(statement)}
    return set()


def _top_level_bound_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {name for node in tree.body for name in _bound_names_in_statement(node)}


def _assert_public_api_doc_is_synchronized_to_static_all_exports() -> None:
    text = PUBLIC_API_DOC.read_text(encoding="utf-8")
    modules = _public_facade_init_files()
    assert {
        "finite_difference_options.instruments",
        "finite_difference_options.exceptions",
        "finite_difference_options.models",
        "finite_difference_options.risk",
    }.issubset(modules)
    for module_name, init_file in modules.items():
        static_exports = list(_facade_all_exports(init_file))
        documented_exports = _documented_exports(text, module_name)
        assert documented_exports == static_exports, f"{module_name} docs/PUBLIC_API.md exports drifted from __all__"
        missing_attrs = [name for name in static_exports if name not in _top_level_bound_names(init_file)]
        assert not missing_attrs, f"{module_name}.__all__ contains statically unbound names: {missing_attrs}"


def test_public_api_doc_is_synchronized_to_static_all_exports() -> None:
    _assert_public_api_doc_is_synchronized_to_static_all_exports()


def test_public_api_doc_sync_does_not_import_optional_viz_dependencies(monkeypatch) -> None:
    real_import = builtins.__import__
    blocked = {"matplotlib", "seaborn"}

    def reject_optional_viz_imports(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name.split(".", 1)[0] in blocked:
            raise ImportError(f"blocked optional dependency for docs sync test: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_optional_viz_imports)

    _assert_public_api_doc_is_synchronized_to_static_all_exports()


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
