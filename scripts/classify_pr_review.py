"""Bound advisory PR review data without granting model mutation authority.

No provider, credential or billing probes occur here. A bounded result is model
output, not proof of provider identity, account eligibility or review quality.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

REPOSITORY = "googa27/finite_difference_options"
MODEL = "gemini-2.5-flash"
MAX_OUTPUT = 131072
MAX_CONTEXT = 65536
_SHA = re.compile(r"[0-9a-f]{40}")
_HUNK = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class ReviewRefusal(ValueError):
    """Untrusted review data does not satisfy the bounded contract."""


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReviewRefusal("duplicate JSON key")
        result[key] = value
    return result


def _json(raw: str) -> Any:
    try:
        return json.loads(raw, object_pairs_hook=_pairs, parse_constant=lambda _: _refuse())
    except (ValueError, RecursionError) as exc:
        raise ReviewRefusal("invalid JSON") from exc


def _refuse() -> None:
    raise ReviewRefusal("non-finite JSON")


def _text(value: Any, limit: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.encode("utf-8")) > limit:
        raise ReviewRefusal("invalid bounded text")
    if any(ord(char) < 32 and char not in "\n\t" for char in value) or "\x7f" in value:
        raise ReviewRefusal("control character")
    return value


def _integer(value: Any, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ReviewRefusal("invalid integer")
    return value


def _sha(value: Any) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ReviewRefusal("invalid commit identity")
    return value


def configure(mode: str, model: str, *, event_name: str = "", requested: str = "") -> dict[str, Any]:
    """Refuse implicit models, aliases, mutable CLI versions and unknown modes."""
    if event_name != "workflow_dispatch" or requested != "true" or mode in {"", "disabled"}:
        return {"status": "disabled", "run_provider": False, "model": ""}
    if mode != "free-tier-operator-asserted" or model != MODEL:
        return {"status": "configuration_refused", "run_provider": False, "model": ""}
    return {"status": "ready", "run_provider": True, "model": MODEL}


def _path(value: Any) -> str:
    value = _text(value, 300)
    if (
        any(ord(char) < 32 for char in value)
        or value.startswith("/")
        or "\\" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ReviewRefusal("invalid changed path")
    return value


def _added_lines(patch: str, additions: int, deletions: int) -> list[int]:
    """Reject absent/truncated/malformed patches rather than claim full review."""
    added: list[int] = []
    removed = 0
    old_left = new_left = 0
    line_number = 0
    old_end = new_end = 0
    has_hunk = last_was_data = False
    for line in patch.splitlines():
        match = _HUNK.match(line)
        if match:
            if old_left or new_left:
                raise ReviewRefusal("incomplete diff hunk")
            old_start, old_count = int(match[1]), int(match[2] or 1)
            new_start, new_count = int(match[3]), int(match[4] or 1)
            if (old_count and old_start == 0) or (new_count and new_start == 0) or not (old_count or new_count):
                raise ReviewRefusal("invalid nonempty diff range")
            # Zero-count ranges name the preceding line; nonempty ranges name
            # their first line. Normalize both to consumed-line boundaries.
            old_before = old_start - (1 if old_count else 0)
            new_before = new_start - (1 if new_count else 0)
            if has_hunk and (old_before < old_end or new_before < new_end):
                raise ReviewRefusal("overlapping or descending diff ranges")
            if old_before - old_end != new_before - new_end:
                raise ReviewRefusal("inconsistent unchanged diff spans")
            old_end, new_end = old_before + old_count, new_before + new_count
            old_left, new_left = old_count, new_count
            line_number = new_start
            has_hunk, last_was_data = True, False
        elif line.startswith("+"):
            added.append(line_number)
            line_number += 1
            new_left -= 1
        elif line.startswith("-"):
            removed += 1
            old_left -= 1
        elif line.startswith(" "):
            line_number += 1
            old_left -= 1
            new_left -= 1
        elif line == r"\ No newline at end of file":
            if not last_was_data:
                raise ReviewRefusal("orphan or duplicate newline marker")
        else:
            raise ReviewRefusal("invalid diff hunk")
        if not match:
            last_was_data = line.startswith(("+", "-", " "))
        if old_left < 0 or new_left < 0:
            raise ReviewRefusal("invalid diff line counts")
    if not has_hunk or old_left or new_left or len(added) != additions or removed != deletions:
        raise ReviewRefusal("incomplete diff")
    return added


def prepare_context(raw: dict[str, Any]) -> dict[str, Any]:
    """Use native API context; no PR source checkout or execution is needed."""
    if raw.get("repository") != REPOSITORY or raw.get("state") != "open" or raw.get("repository_public") is not True:
        raise ReviewRefusal("wrong repository or closed PR")
    number = _integer(raw.get("pr_number"))
    head, base = _sha(raw.get("head_sha")), _sha(raw.get("base_sha"))
    if raw.get("event_head") not in {None, "", head}:
        raise ReviewRefusal("event head moved")
    files = raw.get("files")
    if not isinstance(files, list) or not 1 <= len(files) <= 30 or _integer(raw.get("changed_files")) != len(files):
        raise ReviewRefusal("incomplete or oversized file list")
    bounded = []
    seen = set()
    for item in files:
        path = _path(item.get("filename"))
        if path in seen:
            raise ReviewRefusal("duplicate file")
        seen.add(path)
        patch = _text(item.get("patch"), 16000)
        lines = _added_lines(patch, _integer(item.get("additions"), 0), _integer(item.get("deletions"), 0))
        bounded.append({"path": path, "added_lines": lines, "patch": patch})
    context = {"repository": REPOSITORY, "pr_number": number, "head_sha": head, "base_sha": base, "files": bounded}
    if len(json.dumps(context).encode("utf-8")) > MAX_CONTEXT:
        raise ReviewRefusal("oversized context")
    return context


def _validate_review(value: Any, context: dict[str, Any]) -> dict[str, Any]:
    keys = {"schema_version", "repository", "pr_number", "head_sha", "summary", "findings"}
    if not isinstance(value, dict) or set(value) != keys:
        raise ReviewRefusal("unknown or missing review fields")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ReviewRefusal("unknown schema")
    if (
        value["repository"] != context["repository"]
        or _integer(value["pr_number"]) != context["pr_number"]
        or _sha(value["head_sha"]) != context["head_sha"]
    ):
        raise ReviewRefusal("wrong review identity")
    _text(value["summary"], 2000)
    findings = value["findings"]
    if not isinstance(findings, list) or len(findings) > 20:
        raise ReviewRefusal("too many findings")
    paths = {item["path"]: item["added_lines"] for item in context["files"]}
    for finding in findings:
        if not isinstance(finding, dict) or set(finding) != {"path", "line", "severity", "message"}:
            raise ReviewRefusal("invalid finding schema")
        path = _path(finding["path"])
        if path not in paths or _integer(finding["line"]) not in paths[path]:
            raise ReviewRefusal("finding outside changed lines")
        if finding["severity"] not in {"low", "medium", "high"}:
            raise ReviewRefusal("unknown severity")
        _text(finding["message"], 2000)
    return value


def classify_review(
    context: dict[str, Any], *, stdout: str, stderr: str, outcome: str, current_head: str, model: str
) -> dict[str, Any]:
    """Classify captured files, including a successful wrapper with errors."""
    result: dict[str, Any] = {"provider_verified": False, "github_mutations": 0}
    status = "invalid_output"
    try:
        if current_head != context["head_sha"]:
            status = "stale_head"
        elif len(stdout.encode("utf-8")) > MAX_OUTPUT or len(stderr.encode("utf-8")) > MAX_OUTPUT:
            status = "oversized_output"
        elif stderr.strip():
            status = "service_error"
        elif outcome != "success":
            status = "action_failed"
        elif not stdout.strip():
            status = "empty_output"
        else:
            envelope = _json(stdout)
            if not isinstance(envelope, dict):
                raise ReviewRefusal("CLI envelope must be an object")
            if "error" in envelope:
                status = "service_error"
            elif not envelope.get("response"):
                status = "empty_output"
            else:
                response = _text(envelope["response"], MAX_OUTPUT)
                stats = envelope.get("stats", {})
                models = stats.get("models", {})
                if model != MODEL or set(models) != {model}:
                    raise ReviewRefusal("missing or substituted model")
                api = models[model]["api"]
                _integer(api["totalRequests"])
                if type(api["totalErrors"]) is not int or api["totalErrors"] != 0:
                    raise ReviewRefusal("model request errors")
                calls = stats["tools"]["totalCalls"]
                if type(calls) is not int or calls != 0:
                    raise ReviewRefusal("unexpected tool calls")
                result["review"] = _validate_review(_json(response), context)
                status = "bounded_review"
    except (ReviewRefusal, KeyError, TypeError, AttributeError, UnicodeError):
        status = "invalid_output"
    result["status"] = status
    return result


def _read(path: Path, maximum: int = MAX_OUTPUT) -> str:
    with path.open("rb") as handle:
        raw = handle.read(maximum + 1)
    if len(raw) > maximum:
        raise ReviewRefusal("oversized file")
    return raw.decode("utf-8")


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Write fixed status summaries and JSON artifacts, never model Markdown."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("configure", "prepare", "classify"))
    args = parser.parse_args()
    directory = Path(".advisory")
    plan_path = directory / "plan.json"
    if args.command == "configure":
        result = configure(
            os.environ.get("REVIEW_MODE", ""),
            os.environ.get("REVIEW_MODEL", ""),
            event_name=os.environ.get("GITHUB_EVENT_NAME", ""),
            requested=os.environ.get("REVIEW_REQUESTED", ""),
        )
        _write(plan_path, result)
        with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as handle:
            handle.write(f"run_provider={str(result['run_provider']).lower()}\nmodel={result['model']}\n")
        return
    plan = _json(_read(plan_path))
    if args.command == "prepare":
        ready = False
        try:
            if not plan["run_provider"]:
                raise ReviewRefusal("disabled configuration")
            context = prepare_context(_json(_read(directory / "raw-context.json")))
            _write(directory / "context.json", context)
            ready = True
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            plan["status"] = "context_refused"
            _write(plan_path, plan)
        with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as handle:
            handle.write(f"ready={str(ready).lower()}\n")
            if ready:
                handle.write("payload=" + json.dumps(context) + "\n")
        return
    result = {"status": plan["status"], "provider_verified": False, "github_mutations": 0}
    if plan["status"] == "ready":
        try:
            context = _json(_read(directory / "context.json"))
            current = _json(_read(directory / "current.json"))
            current_head = current["head_sha"] if current["state"] == "open" else ""
            result = classify_review(
                context,
                stdout=_read(Path("gemini-artifacts/stdout.log")),
                stderr=_read(Path("gemini-artifacts/stderr.log")),
                outcome=os.environ.get("REVIEW_OUTCOME", ""),
                current_head=current_head,
                model=plan["model"],
            )
        except (OSError, ValueError, KeyError, TypeError):
            result["status"] = "capture_refused"
    _write(directory / "report.json", result)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as handle:
            handle.write(
                f"### Advisory PR review\n\nStatus: `{result['status']}`. "
                "Review-service events, not Python/Node/package validation failures. "
                "No GitHub mutations; no provider/account/quota attestation. "
                "CI remains the blocking source of truth.\n"
            )


if __name__ == "__main__":
    main()
