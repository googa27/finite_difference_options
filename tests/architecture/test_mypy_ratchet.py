"""No-growth guard for legacy mypy suppressions."""

from __future__ import annotations

import configparser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {
    (
        "finite_difference_options.plotting.base",
        "finite_difference_options.plotting.config_manager",
        "finite_difference_options.processes.base",
        "finite_difference_options.processes.nonaffine",
        "finite_difference_options.processes.affine",
    ): {"no-untyped-def"},
    ("finite_difference_options.plotting",): {"misc", "assignment"},
    ("finite_difference_options.utils.state_handling",): {"assignment", "no-untyped-def"},
    ("finite_difference_options.boundary_conditions.builder",): {"assignment"},
    ("finite_difference_options.greeks.finite_difference",): {"call-overload", "assignment"},
    ("finite_difference_options.instruments.base",): {"no-untyped-def", "arg-type"},
    ("finite_difference_options.solvers.base",): {"arg-type", "no-untyped-def"},
    ("finite_difference_options.pricing.engines.unified",): {"union-attr"},
    ("finite_difference_options.pricing.engines.finite_difference",): {"assignment", "arg-type"},
    ("finite_difference_options.integrations.public_solver_contract",): {"assignment", "attr-defined"},
    ("finite_difference_options.pricing.workflows.option_pricer",): {
        "no-untyped-def",
        "override",
        "union-attr",
    },
    ("finite_difference_options.pricing.instruments.options",): {"assignment", "no-untyped-def"},
    ("finite_difference_options.api.main",): {"arg-type", "no-untyped-def"},
}


def test_legacy_mypy_suppressions_are_exact_and_exclude_haircut_adapter() -> None:
    config = configparser.ConfigParser()
    config.read(ROOT / "mypy.ini")
    assert dict(config["mypy"]) == {
        "python_version": "3.12",
        "mypy_path": "src",
        "disallow_untyped_defs": "True",
        "ignore_missing_imports": "True",
    }
    actual: dict[tuple[str, ...], set[str]] = {}
    for section in config.sections():
        if not section.startswith("mypy-"):
            continue
        modules = tuple(section.removeprefix("mypy-").split(","))
        codes = {item.strip() for item in config[section]["disable_error_code"].split(",")}
        actual[modules] = codes

    assert actual == EXPECTED
    assert not any(
        module.startswith("finite_difference_options.integrations.haircut_")
        for modules in actual
        for module in modules
    )
