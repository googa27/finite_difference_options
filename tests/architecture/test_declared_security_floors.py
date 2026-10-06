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


class DeclaredSecurityFloors(unittest.TestCase):
    def test_httpx2_metadata_excludes_unsafe_validation_and_dev_versions(self) -> None:
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
        for profile in ("validation", "dev"):
            req = next(
                Requirement(x)
                for x in metadata["project"]["optional-dependencies"][profile]
                if Requirement(x).name == "httpx2"
            )
            with self.subTest(profile=profile):
                self.assertNotIn(Version(UNSAFE["httpx2"]), req.specifier)
                self.assertIn(Version(FLOORS["httpx2"]), req.specifier)

    def test_declared_dev_and_audit_tools_exclude_unsafe_versions(self) -> None:
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
        for profile in ("dev", "audit"):
            requirements = {
                Requirement(x).name: Requirement(x) for x in metadata["project"]["optional-dependencies"][profile]
            }
            for name in ("pip", "urllib3"):
                with self.subTest(profile=profile, package=name):
                    self.assertIn(name, requirements)
                    self.assertNotIn(Version(UNSAFE[name]), requirements[name].specifier)
                    self.assertIn(Version(FLOORS[name]), requirements[name].specifier)
        requirements = {
            Requirement(x).name: Requirement(x) for x in metadata["project"]["optional-dependencies"]["dev"]
        }
        self.assertIn("virtualenv", requirements)
        self.assertNotIn(Version(UNSAFE["virtualenv"]), requirements["virtualenv"].specifier)
        self.assertIn(Version(FLOORS["virtualenv"]), requirements["virtualenv"].specifier)

    def test_http_facing_anyio_metadata_excludes_original_legacy_unsafe_version(self) -> None:
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
        for profile in ("api", "validation", "dev"):
            requirements = {
                Requirement(x).name: Requirement(x) for x in metadata["project"]["optional-dependencies"][profile]
            }
            with self.subTest(profile=profile):
                self.assertIn("anyio", requirements)
                self.assertNotIn(Version(UNSAFE["anyio"]), requirements["anyio"].specifier)
                self.assertIn(Version(FLOORS["anyio"]), requirements["anyio"].specifier)

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
                versions = re.findall(r"^" + re.escape(name) + r"==([^\s\\]+)", text, re.MULTILINE)
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
