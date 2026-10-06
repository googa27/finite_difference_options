"""Declared optional metadata and both locks exclude audited unsafe versions."""

from __future__ import annotations

from pathlib import Path
import re
import sys
import tomllib
import unittest

from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[2]
FLOORS = {
    "httpx2": "2.12.0",
    "httpcore2": "2.10.0",
    "pip": "26.2.0",
    "urllib3": "2.8.0",
    "virtualenv": "21.14.4",
    "anyio": "4.14.2",
}
UNSAFE = {
    "httpx2": "2.9.1",
    "httpcore2": "2.9.1",
    "pip": "26.1.2",
    "urllib3": "2.7.0",
    "virtualenv": "21.14.3",
    "anyio": "4.14.1",
}
DIRECT_DEV_FLOORS = tuple(name for name in FLOORS if name != "httpcore2")


def legacy_pinned_versions(text: str, name: str) -> list[str]:
    """Read pinned version tokens after joining requirements continuation lines."""
    logical_lines = re.sub(r"\\\r?\n", "", text)
    return re.findall(r"^" + re.escape(name) + r"==([^\s\\]+)", logical_lines, re.MULTILINE)


class DeclaredSecurityFloors(unittest.TestCase):
    def assert_declared_floor(self, requirement: Requirement, name: str) -> None:
        """Require an inclusive safe floor, beyond excluding one bad release."""
        self.assertIsNone(requirement.marker, f"{requirement} must govern every supported environment")
        floor = Version(FLOORS[name])
        # Version membership checks satisfaction of all requirement specifiers.
        self.assertIn(floor, requirement.specifier)
        lower_bounds = [Version(spec.version) for spec in requirement.specifier if spec.operator in {">=", ">", "~="}]
        self.assertTrue(
            any(bound >= floor for bound in lower_bounds),
            f"{requirement} does not establish the security floor {floor}",
        )
        self.assertNotIn(Version(UNSAFE[name]), requirement.specifier)

    def test_profile_floors_cannot_be_disabled_by_environment_markers(self) -> None:
        for name, floor in FLOORS.items():
            for marker in ('python_version < "3"', 'sys_platform == "win32"'):
                with self.subTest(package=name, marker=marker):
                    marked = Requirement(f"{name}>={floor}; {marker}")
                    self.assertIn(Version(floor), marked.specifier)
                    with self.assertRaises(AssertionError):
                        self.assert_declared_floor(marked, name)

    def test_direct_development_mirror_enforces_every_declared_floor(self) -> None:
        text = (ROOT / "requirements-dev.txt").read_text()
        logical_lines = re.sub(r"\\\r?\n", "", text)
        requirements: dict[str, list[Requirement]] = {}
        for line in logical_lines.splitlines():
            line = line.partition("#")[0].strip()
            if not line or line == "-r requirements.txt":
                continue
            requirement = Requirement(line)
            requirements.setdefault(requirement.name, []).append(requirement)
        for name in DIRECT_DEV_FLOORS:
            with self.subTest(package=name):
                entries = requirements.get(name, [])
                self.assertEqual(len(entries), 1, f"{name} must have one direct mirrored floor")
                self.assert_declared_floor(entries[0], name)

    def test_known_bad_exclusions_cannot_replace_complete_security_floors(self) -> None:
        for name, floor in FLOORS.items():
            major = Version(floor).major
            with self.subTest(package=name, requirement="weakened lower bound"):
                weakened = Requirement(f"{name}>={major},!={UNSAFE[name]},<{major + 1}")
                # Both old endpoint assertions pass for this insecure range.
                self.assertIn(Version(floor), weakened.specifier)
                self.assertNotIn(Version(UNSAFE[name]), weakened.specifier)
                with self.assertRaises(AssertionError):
                    self.assert_declared_floor(weakened, name)
            with self.subTest(package=name, requirement="inclusive lower bound"):
                self.assert_declared_floor(Requirement(f"{name}>={floor},<{major + 1}"), name)
            with self.subTest(package=name, requirement="compatible lower bound"):
                self.assert_declared_floor(Requirement(f"{name}~={floor}"), name)

    def test_httpx2_metadata_excludes_unsafe_validation_and_dev_versions(self) -> None:
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
        for profile in ("validation", "dev"):
            req = next(
                Requirement(x)
                for x in metadata["project"]["optional-dependencies"][profile]
                if Requirement(x).name == "httpx2"
            )
            with self.subTest(profile=profile):
                self.assert_declared_floor(req, "httpx2")

    def test_declared_dev_and_audit_tools_exclude_unsafe_versions(self) -> None:
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
        for profile in ("dev", "audit"):
            requirements = {
                Requirement(x).name: Requirement(x) for x in metadata["project"]["optional-dependencies"][profile]
            }
            for name in ("pip", "urllib3"):
                with self.subTest(profile=profile, package=name):
                    self.assertIn(name, requirements)
                    self.assert_declared_floor(requirements[name], name)
        requirements = {
            Requirement(x).name: Requirement(x) for x in metadata["project"]["optional-dependencies"]["dev"]
        }
        self.assertIn("virtualenv", requirements)
        self.assert_declared_floor(requirements["virtualenv"], "virtualenv")

    def test_http_facing_anyio_metadata_excludes_original_legacy_unsafe_version(self) -> None:
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
        for profile in ("api", "validation", "dev"):
            requirements = {
                Requirement(x).name: Requirement(x) for x in metadata["project"]["optional-dependencies"][profile]
            }
            with self.subTest(profile=profile):
                self.assertIn("anyio", requirements)
                self.assert_declared_floor(requirements["anyio"], "anyio")

    def test_legacy_pins_support_requirement_continuations(self) -> None:
        for name, floor in FLOORS.items():
            with self.subTest(package=name, continuation="following hash"):
                text = f"{name}=={floor} \\\n    --hash=sha256:fixture\n"
                self.assertEqual(legacy_pinned_versions(text, name), [floor])
            with self.subTest(package=name, continuation="split version token"):
                first, rest = floor.split(".", 1)
                text = f"{name}=={first}.\\\n{rest}\n"
                self.assertEqual(legacy_pinned_versions(text, name), [floor])

    def test_native_uv_lock_excludes_audited_unsafe_packages(self) -> None:
        packages = {x["name"]: x["version"] for x in tomllib.loads((ROOT / "uv.lock").read_text())["package"]}
        for name, floor in FLOORS.items():
            with self.subTest(package=name):
                self.assertIn(name, packages)
                self.assertGreaterEqual(Version(packages[name]), Version(floor))

    def test_legacy_audit_lock_excludes_audited_unsafe_packages(self) -> None:
        text = (ROOT / "requirements-dev.lock.txt").read_text()
        for name, floor in FLOORS.items():
            with self.subTest(package=name):
                versions = legacy_pinned_versions(text, name)
                self.assertEqual(len(versions), 1)
                self.assertGreaterEqual(Version(versions[0]), Version(floor))

    def test_core_keeps_installer_transport_and_virtualenv_tools_optional(self) -> None:
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
        core = {Requirement(x).name for x in metadata["project"]["dependencies"]}
        self.assertFalse(core & set(FLOORS))


if __name__ == "__main__":
    if "--source-root" in sys.argv:
        index = sys.argv.index("--source-root")
        ROOT = Path(sys.argv[index + 1])
        del sys.argv[index : index + 2]
    unittest.main()
