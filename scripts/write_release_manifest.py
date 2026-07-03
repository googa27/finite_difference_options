#!/usr/bin/env python3
"""Write a reproducible release-manifest skeleton for built artifacts.

The manifest intentionally records only public repository metadata and hashes of
checked-in/build artifacts. It does not read secrets or environment variables
that could contain credentials.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_LOCKFILES = (
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-dev.lock.txt",
    "docs/CAPABILITY_MATRIX.md",
    "tests/fixtures/fd_benchmark_registry_v1.json",
)

PYTHON_DISTRIBUTION_SUFFIXES = (".whl", ".tar.gz")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    base = relative_to or ROOT
    return {
        "path": path.relative_to(base).as_posix() if path.is_relative_to(base) else path.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def build_manifest(dist: Path, lockfiles: tuple[str, ...] = DEFAULT_LOCKFILES) -> dict[str, Any]:
    artifact_paths = sorted(
        path
        for path in dist.glob("*")
        if path.is_file() and path.name.endswith(PYTHON_DISTRIBUTION_SUFFIXES)
    )
    if not artifact_paths:
        raise SystemExit(f"No distribution artifacts found under {dist}")

    lockfile_records = []
    for relative in lockfiles:
        path = ROOT / relative
        if path.exists():
            lockfile_records.append(_file_record(path))

    return {
        "schema_version": "finite-difference-options.release-manifest.v0",
        "generated_at_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        "source": {
            "repository": os.environ.get("GITHUB_REPOSITORY", "googa27/finite_difference_options"),
            "sha": os.environ.get("GITHUB_SHA", "local"),
            "ref": os.environ.get("GITHUB_REF_NAME", "local"),
            "workflow": os.environ.get("GITHUB_WORKFLOW", "local"),
            "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
        },
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "artifacts": [_file_record(path, relative_to=dist.parent) for path in artifact_paths],
        "governance_inputs": lockfile_records,
        "capability_evidence": {
            "capability_matrix": "docs/CAPABILITY_MATRIX.md",
            "benchmark_registry_fixture": "tests/fixtures/fd_benchmark_registry_v1.json",
            "ci_policy": "docs/CI_POLICY.md",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=ROOT / "dist")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_manifest(args.dist)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
