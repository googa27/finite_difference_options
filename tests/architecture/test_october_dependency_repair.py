"""Keep the October audited optional environments on supported patched releases."""

from pathlib import Path
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[2]
# Full environment audits remain required: these are regression witnesses,
# not a substitute for the current advisory database.
PATCHED = {
    "certifi": ("2026.10.18", "2027"),
    "filelock": ("4.2.1", "5"),
    "gitpython": ("3.2.1", "4"),
    "msgpack": ("1.2.3", "2"),
    "pillow": ("13.0.0", "14"),
    "virtualenv": ("21.15.2", "22"),
    "packaging": ("26.3", "27"),
    "streamlit": ("1.66.1", "2"),
}


def _assert_safe(name: str, value: str) -> None:
    minimum, maximum = PATCHED[name]
    assert Version(minimum) <= Version(value) < Version(maximum), (name, value)


def test_both_declared_locks_reject_original_unsafe_versions() -> None:
    native = {p["name"]: p["version"] for p in tomllib.loads((ROOT / "uv.lock").read_text())["package"]}
    legacy = {}
    for line in (ROOT / "requirements-dev.lock.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        requirement = Requirement(line)
        name = canonicalize_name(requirement.name)
        if name not in PATCHED:
            continue
        assert name not in legacy
        assert requirement.marker is None and requirement.url is None and not requirement.extras
        specs = list(requirement.specifier)
        assert len(specs) == 1 and specs[0].operator == "==" and "*" not in specs[0].version
        legacy[name] = specs[0].version
    for name in PATCHED:
        assert name in native and name in legacy
        _assert_safe(name, native[name])
        _assert_safe(name, legacy[name])


def test_original_audit_findings_and_partial_virtualenv_fix_are_rejected() -> None:
    for name, value in {
        "certifi": "2026.6.17",
        "filelock": "3.29.7",
        "gitpython": "3.1.60",
        "msgpack": "1.2.1",
        "pillow": "12.3.0",
        "virtualenv": "21.14.9",
        "packaging": "26.2",
        "streamlit": "1.66.0",
    }.items():
        try:
            _assert_safe(name, value)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"unsafe or incompatible {name}=={value} admitted")


def test_existing_direct_profiles_keep_positive_safe_bounds() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    profiles = metadata["project"]["optional-dependencies"]
    for profile, names in (("dev", ("gitpython", "virtualenv", "streamlit")), ("ui", ("streamlit",))):
        requirements = {canonicalize_name(r.name): r for r in map(Requirement, profiles[profile])}
        for name in names:
            requirement = requirements[name]
            minimum, maximum = map(Version, PATCHED[name])
            assert requirement.marker is None and requirement.url is None
            assert any(s.operator in {">=", ">", "~="} and Version(s.version) >= minimum for s in requirement.specifier)
            assert any(s.operator == "<" and Version(s.version) <= maximum for s in requirement.specifier)
    # Preserve the canonical numerical dependency boundary exactly.
    assert metadata["project"]["requires-python"] == ">=3.12,<3.13"
    assert metadata["project"]["dependencies"] == [
        "findiff>=0.12,<1",
        "numpy>=2,<3",
        "pydantic>=2,<3",
        "scipy>=1.16,<2",
    ]
