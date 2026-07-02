"""Validate advisory AI issue-triage output before GitHub mutations.

The AI step sees untrusted issue content and returns untrusted text.  This
script is the deterministic authority boundary: it accepts only a small closed
JSON schema, checks every label against the repository allowlist, binds every
issue number to a trusted candidate set, rejects high-impact protected labels,
and emits normalized JSON for the apply step.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
DEFAULT_PROTECTED_LABEL_RE = r"^(priority:P0|security|type:security|status/needs-triage)$"


class TriageValidationError(ValueError):
    """Raised when AI triage output is not safe to apply."""


@dataclass(frozen=True)
class NormalizedEntry:
    """Validated label decision for one issue."""

    issue_number: int
    labels_to_add: tuple[str, ...]
    explanation: str


def _strip_code_fence(raw: str) -> str:
    text = raw.strip()
    match = _CODE_FENCE_RE.match(text)
    if match:
        return match.group(1).strip()
    return text


def _load_json(raw: str, *, max_bytes: int) -> Any:
    encoded_len = len(raw.encode("utf-8"))
    if encoded_len > max_bytes:
        raise TriageValidationError(f"model output is {encoded_len} bytes; limit is {max_bytes} bytes")
    try:
        return json.loads(_strip_code_fence(raw))
    except json.JSONDecodeError as exc:
        raise TriageValidationError(f"model output is not valid JSON: {exc.msg}") from exc


def _parse_allowed_labels(value: str) -> set[str]:
    value = value.strip()
    if not value:
        return set()
    if value.startswith("["):
        loaded = json.loads(value)
        if not isinstance(loaded, list) or not all(isinstance(x, str) for x in loaded):
            raise TriageValidationError("allowed labels JSON must be a string array")
        return set(loaded)
    return {part.strip() for part in value.split(",") if part.strip()}


def _parse_candidates(value: str) -> set[int]:
    value = value.strip()
    if not value:
        return set()
    if value.startswith("["):
        loaded = json.loads(value)
        if not isinstance(loaded, list):
            raise TriageValidationError("candidate issue numbers JSON must be an array")
        values = loaded
    else:
        values = [part.strip() for part in value.split(",") if part.strip()]
    candidates: set[int] = set()
    for item in values:
        if isinstance(item, dict):
            item = item.get("number")
        if isinstance(item, bool) or not isinstance(item, int | str):
            raise TriageValidationError(f"invalid candidate issue number: {item!r}")
        try:
            number = int(item)
        except ValueError as exc:
            raise TriageValidationError(f"invalid candidate issue number: {item!r}") from exc
        if number <= 0:
            raise TriageValidationError(f"invalid candidate issue number: {number}")
        candidates.add(number)
    return candidates


def _string_field(entry: dict[str, Any], key: str, *, max_length: int) -> str:
    value = entry.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TriageValidationError(f"{key} must be a string")
    value = value.strip()
    if len(value) > max_length:
        raise TriageValidationError(f"{key} exceeds {max_length} characters")
    return value


def _entry_number(entry: dict[str, Any], *, default_issue_number: int | None) -> int:
    number = entry.get("issue_number", default_issue_number)
    if isinstance(number, bool) or not isinstance(number, int | str):
        raise TriageValidationError("issue_number must be an integer")
    try:
        parsed = int(number)
    except ValueError as exc:
        raise TriageValidationError("issue_number must be an integer") from exc
    if parsed <= 0:
        raise TriageValidationError("issue_number must be positive")
    return parsed


def _entry_labels(
    entry: dict[str, Any],
    *,
    allowed_labels: set[str],
    max_labels: int,
    protected_label_re: re.Pattern[str],
) -> tuple[str, ...]:
    raw_labels = entry.get("labels_to_set", entry.get("labels_to_add"))
    if raw_labels is None:
        raise TriageValidationError("entry must contain labels_to_set or labels_to_add")
    if not isinstance(raw_labels, list):
        raise TriageValidationError("labels field must be an array")
    if len(raw_labels) > max_labels:
        raise TriageValidationError(f"too many labels: {len(raw_labels)} > {max_labels}")

    labels: list[str] = []
    seen: set[str] = set()
    for label in raw_labels:
        if not isinstance(label, str):
            raise TriageValidationError("labels must be strings")
        normalized = label.strip()
        if not normalized:
            raise TriageValidationError("labels must not be empty")
        if normalized not in allowed_labels:
            raise TriageValidationError(f"label is not in repository allowlist: {normalized}")
        if protected_label_re.search(normalized):
            raise TriageValidationError(
                f"protected label requires human approval and cannot be AI-applied: {normalized}"
            )
        if normalized not in seen:
            seen.add(normalized)
            labels.append(normalized)
    return tuple(labels)


def _validate_entry(
    raw_entry: Any,
    *,
    allowed_labels: set[str],
    candidate_numbers: set[int],
    default_issue_number: int | None,
    max_labels: int,
    max_explanation_chars: int,
    protected_label_re: re.Pattern[str],
) -> NormalizedEntry:
    if not isinstance(raw_entry, dict):
        raise TriageValidationError("each triage entry must be an object")
    allowed_keys = {"issue_number", "labels_to_set", "labels_to_add", "explanation"}
    extra = set(raw_entry) - allowed_keys
    if extra:
        raise TriageValidationError(f"unknown triage entry properties: {sorted(extra)}")

    issue_number = _entry_number(raw_entry, default_issue_number=default_issue_number)
    if candidate_numbers and issue_number not in candidate_numbers:
        raise TriageValidationError(
            f"issue #{issue_number} is not in trusted candidate set {sorted(candidate_numbers)}"
        )
    labels = _entry_labels(
        raw_entry,
        allowed_labels=allowed_labels,
        max_labels=max_labels,
        protected_label_re=protected_label_re,
    )
    explanation = _string_field(raw_entry, "explanation", max_length=max_explanation_chars)
    return NormalizedEntry(issue_number, labels, explanation)


def validate_triage_output(
    raw_output: str,
    *,
    mode: str,
    allowed_labels: set[str],
    candidate_numbers: set[int],
    default_issue_number: int | None = None,
    max_bytes: int = 20_000,
    max_entries: int = 10,
    max_labels: int = 6,
    max_explanation_chars: int = 500,
    protected_label_regex: str = DEFAULT_PROTECTED_LABEL_RE,
) -> dict[str, Any]:
    """Return normalized safe triage JSON or raise ``TriageValidationError``."""

    if mode not in {"single", "batch"}:
        raise TriageValidationError("mode must be single or batch")
    if not allowed_labels:
        raise TriageValidationError("allowed label set is empty")

    parsed = _load_json(raw_output, max_bytes=max_bytes)
    entries_raw = [parsed] if mode == "single" else parsed
    if not isinstance(entries_raw, list):
        raise TriageValidationError("batch triage output must be a JSON array")
    if len(entries_raw) > max_entries:
        raise TriageValidationError(f"too many triage entries: {len(entries_raw)} > {max_entries}")
    if mode == "single" and len(entries_raw) != 1:
        raise TriageValidationError("single triage output must contain exactly one object")

    protected_label_re = re.compile(protected_label_regex)
    entries: list[NormalizedEntry] = []
    seen_numbers: set[int] = set()
    for raw_entry in entries_raw:
        entry = _validate_entry(
            raw_entry,
            allowed_labels=allowed_labels,
            candidate_numbers=candidate_numbers,
            default_issue_number=default_issue_number,
            max_labels=max_labels,
            max_explanation_chars=max_explanation_chars,
            protected_label_re=protected_label_re,
        )
        if entry.issue_number in seen_numbers:
            raise TriageValidationError(f"duplicate triage entry for issue #{entry.issue_number}")
        seen_numbers.add(entry.issue_number)
        entries.append(entry)

    return {
        "entries": [
            {
                "issue_number": entry.issue_number,
                "labels_to_add": list(entry.labels_to_add),
                "explanation": entry.explanation,
            }
            for entry in entries
        ],
        "entry_count": len(entries),
    }


def _read_arg_or_env(value: str | None, env_name: str | None) -> str:
    if env_name:
        return os.environ.get(env_name, "")
    return value or ""


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["single", "batch"], required=True)
    parser.add_argument("--raw", default=None)
    parser.add_argument("--raw-env", default=None)
    parser.add_argument("--allowed-labels", default=None)
    parser.add_argument("--allowed-labels-env", default=None)
    parser.add_argument("--candidate-issues", default=None)
    parser.add_argument("--candidate-issues-env", default=None)
    parser.add_argument("--issue-number", type=int, default=None)
    parser.add_argument("--max-bytes", type=int, default=20_000)
    parser.add_argument("--max-entries", type=int, default=10)
    parser.add_argument("--max-labels", type=int, default=6)
    parser.add_argument("--max-explanation-chars", type=int, default=500)
    parser.add_argument("--protected-label-regex", default=DEFAULT_PROTECTED_LABEL_RE)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        raw_output = _read_arg_or_env(args.raw, args.raw_env)
        allowed_labels = _parse_allowed_labels(_read_arg_or_env(args.allowed_labels, args.allowed_labels_env))
        candidate_numbers = _parse_candidates(_read_arg_or_env(args.candidate_issues, args.candidate_issues_env))
        normalized = validate_triage_output(
            raw_output,
            mode=args.mode,
            allowed_labels=allowed_labels,
            candidate_numbers=candidate_numbers,
            default_issue_number=args.issue_number,
            max_bytes=args.max_bytes,
            max_entries=args.max_entries,
            max_labels=args.max_labels,
            max_explanation_chars=args.max_explanation_chars,
            protected_label_regex=args.protected_label_regex,
        )
    except (json.JSONDecodeError, re.error, TriageValidationError) as exc:
        print(f"AI triage output rejected: {exc}", file=sys.stderr)
        return 2

    output = json.dumps(normalized, sort_keys=True)
    if args.out:
        args.out.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main(sys.argv[1:]))
