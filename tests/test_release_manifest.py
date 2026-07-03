from __future__ import annotations

import importlib.util
import json
import tarfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SCRIPT = REPO_ROOT / "scripts" / "write_release_manifest.py"

spec = importlib.util.spec_from_file_location("write_release_manifest", MANIFEST_SCRIPT)
assert spec is not None and spec.loader is not None
write_release_manifest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(write_release_manifest)
build_manifest = write_release_manifest.build_manifest


def test_release_manifest_records_artifact_and_governance_hashes(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel = dist / "finite_difference_options-0.1.0-py3-none-any.whl"
    sdist = dist / "finite_difference_options-0.1.0.tar.gz"

    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("finite_difference_options/__init__.py", "")
    with tarfile.open(sdist, "w:gz") as archive:
        payload = tmp_path / "PKG-INFO"
        payload.write_text("Name: finite-difference-options\n", encoding="utf-8")
        archive.add(payload, arcname="finite_difference_options-0.1.0/PKG-INFO")

    manifest = build_manifest(dist)
    payload = json.loads(json.dumps(manifest))

    assert payload["schema_version"] == "finite-difference-options.release-manifest.v0"
    assert payload["source"]["repository"] == "googa27/finite_difference_options"
    assert {artifact["path"] for artifact in payload["artifacts"]} == {
        "dist/finite_difference_options-0.1.0-py3-none-any.whl",
        "dist/finite_difference_options-0.1.0.tar.gz",
    }
    assert all(len(artifact["sha256"]) == 64 for artifact in payload["artifacts"])
    assert "docs/CAPABILITY_MATRIX.md" in {
        item["path"] for item in payload["governance_inputs"]
    }
    assert payload["capability_evidence"]["benchmark_registry_fixture"] == (
        "tests/fixtures/fd_benchmark_registry_v1.json"
    )
