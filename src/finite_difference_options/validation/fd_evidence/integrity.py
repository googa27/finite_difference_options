"""Hashing helpers for FD verification evidence bundles."""

from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256
from typing import Any, cast

HASH_KEYS = (
    "request_hash",
    "config_hash",
    "provenance_hash",
    "convention_hash",
    "result_hash",
    "status_hash",
    "evidence_hash",
)


def hashes_for_bundle(bundle: Mapping[str, Any]) -> dict[str, str]:
    """Return hashes binding request, config, provenance, conventions, results, and status."""

    evidence = cast(Mapping[str, Any], bundle.get("evidence", {}))
    status = evidence.get("status")
    return {
        "request_hash": hash_payload(bundle.get("request")),
        "config_hash": hash_payload(bundle.get("config")),
        "provenance_hash": hash_payload(bundle.get("provenance")),
        "convention_hash": hash_payload(bundle.get("convention")),
        "result_hash": hash_payload(bundle.get("results")),
        "status_hash": hash_payload(status),
        "evidence_hash": hash_payload(
            {
                "schema_version": bundle.get("schema_version"),
                "benchmark_id": bundle.get("benchmark_id"),
                "request": bundle.get("request"),
                "config": bundle.get("config"),
                "provenance": bundle.get("provenance"),
                "convention": bundle.get("convention"),
                "results": bundle.get("results"),
                "status": status,
            }
        ),
    }


def hash_payload(payload: object) -> str:
    """Return a deterministic SHA-256 tag for canonical JSON payloads."""

    return "sha256:" + sha256(canonicalize(payload).encode("utf-8")).hexdigest()


def canonicalize(payload: object) -> str:
    """Return strict canonical JSON for deterministic evidence comparison."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


__all__ = ["HASH_KEYS", "canonicalize", "hash_payload", "hashes_for_bundle"]
