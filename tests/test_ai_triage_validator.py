"""Tests for deterministic AI issue-triage validation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "scripts" / "validate_ai_triage_output.py"
_spec = importlib.util.spec_from_file_location("validate_ai_triage_output", VALIDATOR_PATH)
assert _spec is not None and _spec.loader is not None
_validator = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _validator
_spec.loader.exec_module(_validator)
TriageValidationError = _validator.TriageValidationError
validate_triage_output = _validator.validate_triage_output

ALLOWED = {
    "bug",
    "enhancement",
    "type:task",
    "priority:P1",
    "status/needs-triage",
    "workstream:architecture",
}


def test_single_output_normalizes_fenced_json_and_binds_event_issue() -> None:
    result = validate_triage_output(
        '```json\n{"labels_to_set":["bug","type:task"],"explanation":"Reproducible failure"}\n```',
        mode="single",
        allowed_labels=ALLOWED,
        candidate_numbers={123},
        default_issue_number=123,
    )

    assert result == {
        "entries": [
            {
                "issue_number": 123,
                "labels_to_add": ["bug", "type:task"],
                "explanation": "Reproducible failure",
            }
        ],
        "entry_count": 1,
    }


def test_unknown_label_rejects_without_mutation() -> None:
    with pytest.raises(TriageValidationError, match="not in repository allowlist"):
        validate_triage_output(
            '{"labels_to_set":["admin"]}',
            mode="single",
            allowed_labels=ALLOWED,
            candidate_numbers={123},
            default_issue_number=123,
        )


def test_extra_properties_are_rejected() -> None:
    with pytest.raises(TriageValidationError, match="unknown triage entry properties"):
        validate_triage_output(
            '{"labels_to_set":["bug"],"shell":"gh issue edit 1"}',
            mode="single",
            allowed_labels=ALLOWED,
            candidate_numbers={123},
            default_issue_number=123,
        )


def test_batch_output_must_target_trusted_candidate_numbers() -> None:
    raw = json.dumps(
        [
            {"issue_number": 123, "labels_to_set": ["bug"]},
            {"issue_number": 999, "labels_to_set": ["enhancement"]},
        ]
    )

    with pytest.raises(TriageValidationError, match="not in trusted candidate set"):
        validate_triage_output(
            raw,
            mode="batch",
            allowed_labels=ALLOWED,
            candidate_numbers={123, 124},
        )


def test_batch_duplicate_issue_numbers_are_rejected() -> None:
    raw = json.dumps(
        [
            {"issue_number": 123, "labels_to_set": ["bug"]},
            {"issue_number": 123, "labels_to_set": ["enhancement"]},
        ]
    )

    with pytest.raises(TriageValidationError, match="duplicate triage entry"):
        validate_triage_output(
            raw,
            mode="batch",
            allowed_labels=ALLOWED,
            candidate_numbers={123},
        )


def test_high_impact_labels_require_human_approval() -> None:
    with pytest.raises(TriageValidationError, match="protected label"):
        validate_triage_output(
            '{"labels_to_set":["priority:P0"]}',
            mode="single",
            allowed_labels={*ALLOWED, "priority:P0"},
            candidate_numbers={123},
            default_issue_number=123,
        )


def test_high_impact_label_variants_are_protected_case_insensitively() -> None:
    with pytest.raises(TriageValidationError, match="protected label"):
        validate_triage_output(
            '{"labels_to_set":["priority/p0"]}',
            mode="single",
            allowed_labels={*ALLOWED, "priority/p0"},
            candidate_numbers={123},
            default_issue_number=123,
        )


def test_triage_marker_cannot_be_the_only_ai_applied_label() -> None:
    with pytest.raises(TriageValidationError, match="protected label"):
        validate_triage_output(
            '{"labels_to_set":["status/needs-triage"]}',
            mode="single",
            allowed_labels=ALLOWED,
            candidate_numbers={123},
            default_issue_number=123,
        )


def test_output_size_limit_is_enforced() -> None:
    with pytest.raises(TriageValidationError, match="limit is 10 bytes"):
        validate_triage_output(
            '{"labels_to_set":[]}',
            mode="single",
            allowed_labels=ALLOWED,
            candidate_numbers={123},
            default_issue_number=123,
            max_bytes=10,
        )
