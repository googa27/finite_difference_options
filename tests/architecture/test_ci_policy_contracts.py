from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_third_party_actions_are_pinned_to_full_commit_shas() -> None:
    workflows = sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml"))
    assert workflows

    unpinned: list[str] = []
    uses_re = re.compile(r"^\s*(?:-\s*)?uses:\s*['\"]?([^'\"\s]+)")
    for workflow_path in workflows:
        for line_number, line in enumerate(
            workflow_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            match = uses_re.search(line)
            if not match:
                continue
            target = match.group(1)
            if target.startswith(("./", "docker://")):
                continue
            if "@" not in target:
                unpinned.append(f"{workflow_path.relative_to(REPO_ROOT)}:{line_number}:{target}")
                continue
            ref = target.rsplit("@", maxsplit=1)[1]
            if not re.fullmatch(r"[0-9a-f]{40}", ref):
                unpinned.append(f"{workflow_path.relative_to(REPO_ROOT)}:{line_number}:{target}")

    assert unpinned == []


def test_blocking_ci_has_actionable_python_and_stable_suite_contract() -> None:
    workflow = _read(".github/workflows/ci.yml")

    assert "Package / Python" in workflow
    assert "python -m build --sdist --wheel" in workflow
    assert "python -m twine check dist/*" in workflow
    assert "Clean wheel import smoke outside checkout" in workflow
    assert "Static smoke gate" in workflow
    assert "python -m compileall -q src tests scripts" in workflow
    assert "ruff check . --select E9,F63,F7,F82" in workflow
    assert "mypy --ignore-missing-imports" in workflow
    assert "Architecture and packaging contracts" in workflow
    assert "pytest -q tests/architecture tests/test_packaging_contract.py --no-cov" in workflow
    assert "Stable regression suite" in workflow
    assert "pytest -q --cov=finite_difference_options" in workflow
    assert "Optional profile /" in workflow
    assert "python -m pip install -r requirements-dev.lock.txt" in workflow
    assert "python -m pip check" in workflow
    assert "python -m pip_audit --progress-spinner=off --skip-editable" in workflow
    assert "cyclonedx-py environment --of JSON -o sbom.json" in workflow
    assert "Generate release manifest" in workflow
    assert "scripts/write_release_manifest.py --dist dist --output dist/release-manifest.json" in workflow
    assert "retention-days: 14" in workflow


def test_ci_jobs_declare_bounded_runtime_and_artifact_retention() -> None:
    workflow = _read(".github/workflows/ci.yml")

    for job in ("package", "test", "optional-profiles", "audit", "node"):
        section = workflow.split(f"  {job}:\n", maxsplit=1)[1]
        next_job = re.search(r"\n  [a-zA-Z0-9_-]+:\n", section)
        job_section = section[: next_job.start()] if next_job else section
        assert "timeout-minutes:" in job_section, job

    assert workflow.count("retention-days: 14") >= 4


def test_node_profile_never_references_a_missing_lockfile_cache_path() -> None:
    workflow = _read(".github/workflows/ci.yml")

    assert "lockfile=true" in workflow
    assert "Install dependencies with lockfile" in workflow
    assert "Install dependencies without lockfile" in workflow
    assert "cache-dependency-path: nextjs-client/package-lock.json" not in workflow
    assert "Run production build" in workflow
    assert "npm run build" in workflow
    assert "npm audit --audit-level=high" in workflow
    assert "npm sbom --json > npm-sbom.json" in workflow


def test_gemini_review_and_issue_triage_failures_are_advisory() -> None:
    pr_review = _read(".github/workflows/gemini-pr-review.yml")
    issue_triage = _read(".github/workflows/gemini-issue-automated-triage.yml")
    scheduled_triage = _read(".github/workflows/gemini-issue-scheduled-triage.yml")

    assert "continue-on-error: true" in pr_review
    assert "Classify unavailable Gemini PR review as advisory" in pr_review
    assert "review-service events, not as Python/Node/package validation failures" in pr_review
    assert "Post PR review failure comment" not in pr_review

    for workflow in (issue_triage, scheduled_triage):
        assert "continue-on-error: true" in workflow
        assert "labels left unchanged" in workflow
        assert "Post Issue Analysis Failure Comment" not in workflow


def test_issue_triage_ai_output_is_schema_validated_before_additive_label_writes() -> None:
    automated = _read(".github/workflows/gemini-issue-automated-triage.yml")
    scheduled = _read(".github/workflows/gemini-issue-scheduled-triage.yml")

    for workflow in (automated, scheduled):
        assert "run-gemini-cli@v0" not in workflow
        assert "google-github-actions/run-gemini-cli@f77273f4c914e4bf38440cf36a0369cb64a37489" in workflow
        assert "gemini_cli_version: '0.40.0-preview.3'" in workflow
        assert "statuses: 'write'" not in workflow
        assert "issues: 'read'" in workflow
        assert "issues: 'write'" in workflow
        assert "GITHUB_TOKEN: ''" in workflow
        assert "validate_ai_triage_output.py" in workflow
        assert "github.rest.issues.addLabels" in workflow
        assert "github.rest.issues.removeLabel" in workflow
        assert "TRIAGE_LABEL_EXISTS === 'true' && labels.length > 0" in workflow
        assert "github.rest.issues.setLabels" not in workflow
        assert "Raw labels JSON" not in workflow
        assert "protected labels that require deterministic handling" in workflow
        assert "status/needs-triage" in workflow


def test_review_and_cli_gemini_actions_are_pinned_to_reviewed_sha() -> None:
    pr_review = _read(".github/workflows/gemini-pr-review.yml")
    cli = _read(".github/workflows/gemini-cli.yml")

    for workflow in (pr_review, cli):
        assert "google-github-actions/run-gemini-cli@v0" not in workflow
        assert "google-github-actions/run-gemini-cli@f77273f4c914e4bf38440cf36a0369cb64a37489" in workflow


def test_scheduled_issue_triage_rotates_candidate_batches() -> None:
    scheduled = _read(".github/workflows/gemini-issue-scheduled-triage.yml")

    assert "sortedCandidates" in scheduled
    assert "GITHUB_RUN_NUMBER" in scheduled
    assert "rotatedCandidates" in scheduled
    assert ".slice(0, 5)" in scheduled


def test_issue_triage_apply_job_is_separate_from_ai_credentials() -> None:
    automated = _read(".github/workflows/gemini-issue-automated-triage.yml")
    scheduled = _read(".github/workflows/gemini-issue-scheduled-triage.yml")

    for workflow in (automated, scheduled):
        assert "analyze-" in workflow
        assert "apply-labels:" in workflow
        apply_section = workflow.split("  apply-labels:", maxsplit=1)[1]
        assert "gemini_api_key" not in apply_section
        assert "use_vertex_ai" not in apply_section
        assert "gcp_workload_identity_provider" not in apply_section
        assert "google-github-actions/run-gemini-cli" not in apply_section


def test_ci_policy_is_documented_from_agent_contract_and_architecture() -> None:
    policy = _read("docs/CI_POLICY.md")
    agents = _read("AGENTS.md")
    architecture = _read("docs/ARCHITECTURE.md")

    assert "# CI policy" in policy
    assert "Blocking signals" in policy
    assert "Advisory automated-review signals" in policy
    assert "docs/CI_POLICY.md" in agents
    assert "docs/CI_POLICY.md" in architecture
