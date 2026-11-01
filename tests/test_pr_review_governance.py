"""Advisory review has no mutation authority and cannot accept a green error."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/classify_pr_review.py"
HEAD = "1" * 40
BASE = "2" * 40
REPO = "googa27/finite_difference_options"


def load_classifier():
    assert SCRIPT.exists(), "The review needs a deterministic output classifier"
    spec = importlib.util.spec_from_file_location("pr_review", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def context():
    return {
        "repository": REPO,
        "pr_number": 186,
        "head_sha": HEAD,
        "base_sha": BASE,
        "files": [{"path": "src/example.py", "added_lines": [2, 3]}],
    }


def review():
    return {
        "schema_version": 1,
        "repository": REPO,
        "pr_number": 186,
        "head_sha": HEAD,
        "summary": "Checked the bounded patch.",
        "findings": [
            {
                "path": "src/example.py",
                "line": 2,
                "severity": "medium",
                "message": "The new division needs a zero guard.",
            }
        ],
    }


def envelope(value=None):
    return json.dumps(
        {
            "response": json.dumps(review() if value is None else value),
            "stats": {
                "models": {"gemini-2.5-flash": {"api": {"totalRequests": 1, "totalErrors": 0}}},
                "tools": {"totalCalls": 0},
            },
        }
    )


def classify(stdout=None, stderr="", outcome="success", current=HEAD):
    return load_classifier().classify_review(
        context(),
        stdout=envelope() if stdout is None else stdout,
        stderr=stderr,
        outcome=outcome,
        current_head=current,
        model="gemini-2.5-flash",
    )


def test_existing_job_must_not_grant_model_github_mutation_or_oidc():
    workflow = yaml.safe_load((ROOT / ".github/workflows/gemini-pr-review.yml").read_text())
    job = workflow["jobs"]["review-pr"]
    assert job["permissions"] == {"contents": "read", "pull-requests": "read"}
    steps = job["steps"]
    action = next(step for step in steps if "run-gemini-cli@" in step.get("uses", ""))
    assert action["env"]["GITHUB_TOKEN"] == ""
    assert "generate_token" not in [step.get("id") for step in steps]
    assert action["with"]["gemini_model"] == "${{ steps.configure.outputs.model }}"
    assert action["if"] == (
        "github.event_name == 'workflow_dispatch' && "
        "inputs.run_provider == true && steps.prepare.outputs.ready == 'true'"
    )
    settings = json.loads(action["with"]["settings"])
    assert settings["tools"]["core"] == []
    assert settings["mcp"]["allowed"] == []
    assert settings["billing"]["overageStrategy"] == "never"
    assert settings["experimental"]["enableAgents"] is False
    assert settings["experimental"]["dynamicModelConfiguration"] is False


@pytest.mark.parametrize(
    "mode,model,status,run",
    [
        ("", "", "disabled", False),
        ("disabled", "gemini-2.5-flash", "disabled", False),
        ("enabled", "gemini-2.5-flash", "configuration_refused", False),
        ("free-tier-operator-asserted", "", "configuration_refused", False),
        ("free-tier-operator-asserted", "auto", "configuration_refused", False),
        ("free-tier-operator-asserted", "gemini-3.8-flash", "configuration_refused", False),
        ("free-tier-operator-asserted", "gemini-2.5-flash", "ready", True),
    ],
)
def test_default_and_unknown_config_never_run_provider(mode, model, status, run):
    result = load_classifier().configure(mode, model, event_name="workflow_dispatch", requested="true")
    assert result["status"] == status
    assert result["run_provider"] is run


def test_green_wrapper_model_not_found_is_not_review():
    result = classify(stdout='{"error":{"type":"ModelNotFoundError","code":404}}')
    assert result["status"] == "service_error"
    assert "review" not in result


@pytest.mark.parametrize(
    "stderr",
    [
        "ModelNotFoundError: models/gemini-2.5-flash",
        "RESOURCE_EXHAUSTED",
        "UNAUTHENTICATED",
        "ServiceUnavailable: 503",
        "warning: unknown service output",
    ],
)
def test_any_captured_service_stderr_prevents_completed_review(stderr):
    assert classify(stderr=stderr)["status"] == "service_error"


@pytest.mark.parametrize(
    "stdout,status",
    [
        ("", "empty_output"),
        ("   ", "empty_output"),
        ("Not JSON", "invalid_output"),
        ("{}", "empty_output"),
        ('{"response":""}', "empty_output"),
        ("x" * 131073, "oversized_output"),
        ('{"response":"first","response":"second"}', "invalid_output"),
    ],
)
def test_empty_malformed_duplicate_and_oversized_output_cannot_earn_review(stdout, status):
    assert classify(stdout=stdout)["status"] == status


def test_valid_bounded_review_is_artifact_data_and_not_provider_attestation():
    result = classify()
    assert result["status"] == "bounded_review"
    assert result["review"] == review()
    assert result["provider_verified"] is False
    assert result["github_mutations"] == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("head_sha", "3" * 40),
        ("pr_number", True),
        ("pr_number", 185),
        ("repository", "attacker/repo"),
        ("summary", ""),
        ("unknown", "ignored?"),
    ],
)
def test_response_cannot_move_identity_or_extend_schema(field, value):
    value_review = review()
    value_review[field] = value
    assert classify(stdout=envelope(value_review))["status"] == "invalid_output"


@pytest.mark.parametrize(
    "field,value",
    [
        ("path", "../private"),
        ("path", "unchanged.py"),
        ("line", 1),
        ("line", True),
        ("severity", "urgent"),
        ("message", ""),
        ("message", "\x1b[31mcommand"),
        ("extra", "do write"),
    ],
)
def test_findings_must_point_to_changed_lines_and_stay_bounded(field, value):
    value_review = review()
    value_review["findings"][0][field] = value
    assert classify(stdout=envelope(value_review))["status"] == "invalid_output"


def test_moved_head_or_failed_action_cannot_earn_review():
    assert classify(current="4" * 40)["status"] == "stale_head"
    assert classify(outcome="failure")["status"] == "action_failed"


@pytest.mark.parametrize(
    "models,tools",
    [
        ({}, 0),
        ({"gemini-2.5-pro": {}}, 0),
        ({"gemini-2.5-flash": {"api": {"totalRequests": 1, "totalErrors": 1}}}, 0),
        ({"gemini-2.5-flash": {"api": {"totalRequests": 1, "totalErrors": 0}}}, 1),
    ],
)
def test_missing_model_evidence_substitution_errors_or_tools_refuse(models, tools):
    value = json.loads(envelope())
    value["stats"] = {"models": models, "tools": {"totalCalls": tools}}
    assert classify(stdout=json.dumps(value))["status"] == "invalid_output"


def raw_context():
    return {
        "repository": REPO,
        "repository_public": True,
        "state": "open",
        "pr_number": 186,
        "head_sha": HEAD,
        "base_sha": BASE,
        "event_head": HEAD,
        "changed_files": 1,
        "files": [
            {
                "filename": "src/example.py",
                "additions": 2,
                "deletions": 1,
                "patch": "@@ -1,2 +1,3 @@\n unchanged\n-old\n+new\n+second",
            }
        ],
    }


def test_complete_native_patch_derives_exact_added_line_allowlist():
    prepared = load_classifier().prepare_context(raw_context())
    assert prepared["files"][0]["added_lines"] == [2, 3]


@pytest.mark.parametrize(
    "field,value",
    [
        ("state", "closed"),
        ("repository_public", False),
        ("repository", "other/repo"),
        ("event_head", "3" * 40),
        ("pr_number", True),
        ("changed_files", 2),
        ("changed_files", True),
    ],
)
def test_native_context_refuses_wrong_identity_or_partial_file_list(field, value):
    raw = raw_context()
    raw[field] = value
    module = load_classifier()
    with pytest.raises(module.ReviewRefusal):
        module.prepare_context(raw)


@pytest.mark.parametrize(
    "field,value",
    [
        ("filename", "../escape.py"),
        ("filename", "line\nbreak.py"),
        ("patch", "@@ -1,2 +1,3 @@\n unchanged\n-old\n+new"),
        ("patch", "@@ -1 +1 @@\n-old\n+new"),
        ("patch", None),
        ("additions", True),
    ],
)
def test_missing_truncated_binary_or_unsafe_patch_cannot_be_reviewed(field, value):
    module = load_classifier()
    raw = raw_context()
    raw["files"][0][field] = value
    with pytest.raises(module.ReviewRefusal):
        module.prepare_context(raw)


def test_cli_disabled_path_has_no_provider_or_credential_dependency(tmp_path):
    import os
    import subprocess

    output = tmp_path / "outputs"
    summary = tmp_path / "summary"
    env = {"PATH": os.environ["PATH"], "GITHUB_OUTPUT": str(output), "GITHUB_STEP_SUMMARY": str(summary)}
    for command in ("configure", "classify"):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), command], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10
        )
        assert completed.returncode == 0, completed.stderr
    assert output.read_text() == "run_provider=false\nmodel=\n"
    assert json.loads((tmp_path / ".advisory/report.json").read_text()) == {
        "status": "disabled",
        "provider_verified": False,
        "github_mutations": 0,
    }
    assert "Status: `disabled`" in summary.read_text()
    assert not (tmp_path / "gemini-artifacts").exists()


def test_cli_captured_successful_model_error_is_preserved_as_service_status(tmp_path):
    import os
    import subprocess

    directory = tmp_path / ".advisory"
    directory.mkdir()
    (directory / "plan.json").write_text(json.dumps({"status": "ready", "model": "gemini-2.5-flash"}))
    (directory / "context.json").write_text(json.dumps(context()))
    (directory / "current.json").write_text(json.dumps({"state": "open", "head_sha": HEAD}))
    logs = tmp_path / "gemini-artifacts"
    logs.mkdir()
    raw_error = '{"error":{"type":"ModelNotFoundError","code":404,"message":"unavailable"}}'
    (logs / "stdout.log").write_text(raw_error)
    (logs / "stderr.log").write_text("")
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "classify"],
        cwd=tmp_path,
        env={"PATH": os.environ["PATH"], "REVIEW_OUTCOME": "success"},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads((directory / "report.json").read_text())["status"] == "service_error"
    assert (logs / "stdout.log").read_text() == raw_error


def test_native_context_workflow_uses_immutable_diff_request(tmp_path):
    """Execute the real workflow adapter with a read-only native API double."""
    import subprocess
    import shutil

    workflow = yaml.safe_load((ROOT / ".github/workflows/gemini-pr-review.yml").read_text())
    code = next(step for step in workflow["jobs"]["review-pr"]["steps"] if step.get("id") == "context")["with"][
        "script"
    ]
    fixture = raw_context()
    harness = """
const assert = require('node:assert/strict');
const input = JSON.parse(process.env.FIXTURE);
const context = {repo: {owner:'googa27',repo:'finite_difference_options'},
  payload:{pull_request:{number:186,head:{sha:input.head_sha}}}};
let calls = 0;
const github = {rest: {pulls: {get: async args => {
  assert.deepEqual(args,{...context.repo,pull_number:186});
  return {data: {state:'open',base:{repo:{full_name:input.repository,private:false},sha:input.base_sha},
    head:{sha:input.head_sha},changed_files:1}};
}},repos:{compareCommitsWithBasehead: async args => {
  assert.deepEqual(args,{...context.repo,basehead:'2222222222222222222222222222222222222222...1111111111111111111111111111111111111111',per_page:100});
  calls++; return {data:{files:input.files}};
}}}};
(async () => { BODY; assert.equal(calls,1); })().catch(error=>{console.error(error);process.exitCode=1;});
""".replace("BODY", code)
    completed = subprocess.run(
        [shutil.which("node"), "-e", harness],
        cwd=tmp_path,
        env={"FIXTURE": json.dumps(fixture)},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    captured = json.loads((tmp_path / ".advisory/raw-context.json").read_text())
    assert captured["head_sha"] == HEAD
    assert captured["base_sha"] == BASE
    assert captured["files"] == fixture["files"]


@pytest.mark.parametrize(
    "event,requested",
    [
        ("pull_request", "true"),
        ("issue_comment", "true"),
        ("pull_request_review_comment", "true"),
        ("pull_request_review", "true"),
        ("push", "true"),
        ("schedule", "true"),
        ("", "true"),
        ("workflow_dispatch", "false"),
        ("workflow_dispatch", ""),
        ("workflow_dispatch", "True"),
        ("workflow_dispatch", "1"),
    ],
)
def test_exact_operator_variables_never_enable_nonmanual_or_unrequested_provider(tmp_path, event, requested):
    import os
    import subprocess

    output = tmp_path / "outputs"
    env = {
        "PATH": os.environ["PATH"],
        "GITHUB_OUTPUT": str(output),
        "GITHUB_EVENT_NAME": event,
        "REVIEW_REQUESTED": requested,
        "REVIEW_MODE": "free-tier-operator-asserted",
        "REVIEW_MODEL": "gemini-2.5-flash",
    }
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "configure"], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10
    )
    assert completed.returncode == 0, completed.stderr
    assert output.read_text() == "run_provider=false\nmodel=\n"
    assert json.loads((tmp_path / ".advisory/plan.json").read_text())["status"] == "disabled"


def test_workflow_dispatch_has_explicit_false_default_and_provider_event_guard():
    workflow = yaml.safe_load((ROOT / ".github/workflows/gemini-pr-review.yml").read_text())
    events = workflow.get("on", workflow.get(True))
    opt_in = events["workflow_dispatch"]["inputs"]["run_provider"]
    assert opt_in["type"] == "boolean"
    assert opt_in["default"] is False
    steps = workflow["jobs"]["review-pr"]["steps"]
    action = next(step for step in steps if "run-gemini-cli@" in step.get("uses", ""))
    assert action["if"] == (
        "github.event_name == 'workflow_dispatch' && "
        "inputs.run_provider == true && steps.prepare.outputs.ready == 'true'"
    )


def test_only_manual_true_exact_variables_and_valid_public_native_context_are_eligible(tmp_path):
    import os
    import subprocess

    output = tmp_path / "outputs"
    env = {
        "PATH": os.environ["PATH"],
        "GITHUB_OUTPUT": str(output),
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "REVIEW_REQUESTED": "true",
        "REVIEW_MODE": "free-tier-operator-asserted",
        "REVIEW_MODEL": "gemini-2.5-flash",
    }
    configure_result = subprocess.run(
        [sys.executable, str(SCRIPT), "configure"], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10
    )
    assert configure_result.returncode == 0, configure_result.stderr
    assert output.read_text() == "run_provider=true\nmodel=gemini-2.5-flash\n"
    (tmp_path / ".advisory/raw-context.json").write_text(json.dumps(raw_context()))
    prepared = subprocess.run(
        [sys.executable, str(SCRIPT), "prepare"], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10
    )
    assert prepared.returncode == 0, prepared.stderr
    assert "ready=true\n" in output.read_text()
    bound = json.loads((tmp_path / ".advisory/context.json").read_text())
    assert bound["head_sha"] == HEAD
    assert bound["files"][0]["added_lines"] == [2, 3]
    assert not (tmp_path / "gemini-artifacts").exists()


def test_action_prompt_receives_prepared_context_without_recursive_shell_expansion(tmp_path):
    import os
    import subprocess

    workflow = yaml.safe_load((ROOT / ".github/workflows/gemini-pr-review.yml").read_text())
    action = next(step for step in workflow["jobs"]["review-pr"]["steps"] if "run-gemini-cli@" in step.get("uses", ""))
    payload = json.dumps(context())
    prompt = action["with"]["prompt"].replace("${{ steps.prepare.outputs.payload }}", payload)
    # The exact pinned action forwards --prompt "${PROMPT}"; Bash never expands
    # variable references inside that value. A literal ${REVIEW_CONTEXT} is a bug.
    result = subprocess.run(
        ["bash", "-c", 'printf "%s" "${PROMPT}"'],
        cwd=tmp_path,
        env={"PATH": os.environ["PATH"], "PROMPT": prompt, "REVIEW_CONTEXT": payload},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert payload in result.stdout
    assert "${REVIEW_CONTEXT}" not in result.stdout
