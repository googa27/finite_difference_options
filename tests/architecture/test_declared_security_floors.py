"""Declared optional metadata and both locks exclude audited unsafe versions."""

from __future__ import annotations

from pathlib import Path
import re
import sys
import tomllib
import unittest

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
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
CEILINGS = {"httpx2": "3", "httpcore2": "3", "pip": "27", "urllib3": "3", "virtualenv": "22", "anyio": "5"}
UNSAFE = {
    "httpx2": "2.9.1",
    "httpcore2": "2.9.1",
    "pip": "26.1.2",
    "urllib3": "2.7.0",
    "virtualenv": "21.14.3",
    "anyio": "4.14.1",
}
DIRECT_DEV_FLOORS = tuple(name for name in FLOORS if name != "httpcore2")


def join_requirement_continuations(text: str) -> str:
    """Match pip joining: remove escaped newlines, preserve all other whitespace.

    Version tokens must not contain whitespace, including indentation introduced
    inside a split token. Indented hash/option continuations remain supported.
    """
    return re.sub(r"\\\r?\n", "", text)


def legacy_pinned_versions(text: str, name: str) -> list[str]:
    """Read complete governed plain exact pins, never a version prefix.

    Continuations and trailing hash options/comments are normalized first.
    Conditional markers, extras, direct URLs and non-exact pins are outside the
    committed plain-freeze contract and fail closed rather than disappear.
    """
    versions: list[str] = []
    for line in join_requirement_continuations(text).splitlines():
        line = re.split(r"\s+#", line, maxsplit=1)[0].strip()
        token = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", line)
        if token is None or token.group(1).lower() != name.lower():
            continue
        line = re.sub(r"\s+--hash=\S+", "", line)
        try:
            requirement = Requirement(line)
        except InvalidRequirement as exc:
            raise AssertionError(f"invalid governed legacy requirement {line!r}") from exc
        assert requirement.marker is None, f"{requirement} must not disable a governed legacy pin"
        assert not requirement.extras and requirement.url is None, f"{requirement} must be a plain exact pin"
        specifiers = list(requirement.specifier)
        assert len(specifiers) == 1, f"{requirement} must have one exact version"
        specifier = specifiers[0]
        assert specifier.operator == "==" and "*" not in specifier.version, f"{requirement} must pin one version"
        versions.append(str(Version(specifier.version)))
    return versions


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
        ceiling = Version(CEILINGS[name])
        upper_bounds: list[tuple[Version, bool]] = []
        for specifier in requirement.specifier:
            if specifier.operator in {"<", "<="}:
                upper_bounds.append((Version(specifier.version), specifier.operator == "<"))
            elif specifier.operator == "~=":
                # PEP440 compatible release >=V.N, ==V.* has an exclusive
                # successor of the release prefix, preserving any epoch.
                version = Version(specifier.version)
                prefix = list(version.release[:-1])
                prefix[-1] += 1
                upper = ".".join(str(part) for part in prefix)
                if version.epoch:
                    upper = f"{version.epoch}!{upper}"
                upper_bounds.append((Version(upper), True))
        self.assertTrue(
            any(
                Version(bound.base_version) < ceiling or (exclusive and bound == ceiling)
                for bound, exclusive in upper_bounds
            ),
            f"{requirement} must retain the exclusive compatibility ceiling {ceiling}",
        )
        self.assertNotIn(Version(UNSAFE[name]), requirement.specifier)

    def assert_pinned_interval(self, value: str, name: str) -> None:
        version = Version(value)
        interval = SpecifierSet(f">={FLOORS[name]},<{CEILINGS[name]}")
        self.assertTrue(interval.contains(version, prereleases=True), f"{name}=={value} leaves {interval}")

    def test_governed_plain_pin_admission_rejects_unbounded_and_malformed_inputs(self) -> None:
        for name, floor in FLOORS.items():
            first, rest = floor.split(".", 1)
            malformed_split = name + "==" + first + "." + "\\" + "\n    " + rest
            for text in (
                f"{name}>={floor}",
                f"{name}=={Version(floor).major}.*",
                f"{name}[fixture]=={floor}",
                f"{name} @ https://example.invalid/fixture.whl",
                malformed_split,
            ):
                with self.subTest(package=name, rejected=text):
                    with self.assertRaises(AssertionError):
                        legacy_pinned_versions(text, name)
            with self.subTest(package=name, valid="comment"):
                self.assertEqual(legacy_pinned_versions(f"  {name} == {floor} # fixture", name), [floor])
            for value in (UNSAFE[name], CEILINGS[name], CEILINGS[name] + ".0a1", CEILINGS[name] + ".dev1"):
                with self.subTest(package=name, incompatible_pin=value):
                    with self.assertRaises(AssertionError):
                        self.assert_pinned_interval(value, name)

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
        logical_lines = join_requirement_continuations(text)
        requirements: dict[str, list[Requirement]] = {}
        for line in logical_lines.splitlines():
            line = line.partition("#")[0].strip()
            if not line or line == "-r requirements.txt":
                continue
            try:
                requirement = Requirement(line)
            except InvalidRequirement as exc:
                self.fail(f"unsupported direct development requirement {line!r}: {exc}")
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

    def test_governed_legacy_pins_cannot_be_disabled_or_partially_parsed(self) -> None:
        for name, floor in FLOORS.items():
            for marker in ('python_version < "3"', 'sys_platform == "win32"'):
                with self.subTest(package=name, marker=marker):
                    text = f"{name}=={floor}; {marker}"
                    self.assertIsNotNone(Requirement(text).marker)
                    with self.assertRaises(AssertionError):
                        legacy_pinned_versions(text, name)

    def test_declared_security_ranges_cannot_lose_compatibility_ceilings(self) -> None:
        for name, floor in FLOORS.items():
            ceiling = Version(floor).major + 1
            for suffix in ("", f",<{ceiling + 1}", f",!={ceiling}", f",<{ceiling}.0a1", f",<={ceiling}.dev1"):
                with self.subTest(package=name, weakening=suffix):
                    requirement = Requirement(f"{name}>={floor}{suffix}")
                    self.assertIn(Version(floor), requirement.specifier)
                    with self.assertRaises(AssertionError):
                        self.assert_declared_floor(requirement, name)
            with self.subTest(package=name, strengthening="inclusive safe ceiling"):
                self.assert_declared_floor(Requirement(f"{name}>={floor},<={floor}"), name)

    def test_native_uv_lock_excludes_audited_unsafe_packages(self) -> None:
        packages = {x["name"]: x["version"] for x in tomllib.loads((ROOT / "uv.lock").read_text())["package"]}
        for name in FLOORS:
            with self.subTest(package=name):
                self.assertIn(name, packages)
                self.assert_pinned_interval(packages[name], name)

    def test_legacy_audit_lock_excludes_audited_unsafe_packages(self) -> None:
        text = (ROOT / "requirements-dev.lock.txt").read_text()
        for name in FLOORS:
            with self.subTest(package=name):
                versions = legacy_pinned_versions(text, name)
                self.assertEqual(len(versions), 1)
                self.assert_pinned_interval(versions[0], name)

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
