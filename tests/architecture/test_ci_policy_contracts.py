from pathlib import Path
import re
import tomllib

from packaging.requirements import Requirement

REPO_ROOT = Path(__file__).resolve().parents[2]

NODE24_OFFICIAL_ACTION_PINS = {
    "actions/checkout": "3d3c42e5aac5ba805825da76410c181273ba90b1",  # v7.0.1
    "actions/setup-python": "5fda3b95a4ea91299a34e894583c3862153e4b97",  # v7.0.0
    "actions/setup-node": "820762786026740c76f36085b0efc47a31fe5020",  # v7.0.0
    "actions/upload-artifact": "043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",  # v7.0.1
}


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _requirement_names(requirements: list[str]) -> set[str]:
    return {
        Requirement(requirement).name.lower()
        for requirement in requirements
        if requirement and not requirement.startswith(("#", "-r "))
    }


def test_third_party_actions_are_pinned_to_full_commit_shas() -> None:
    workflows = sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml"))
    assert workflows

    unpinned: list[str] = []
    uses_re = re.compile(r"^\s*(?:-\s*)?uses:\s*['\"]?([^'\"\s]+)")
    for workflow_path in workflows:
        for line_number, line in enumerate(workflow_path.read_text(encoding="utf-8").splitlines(), start=1):
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


def test_official_actions_use_reviewed_node24_commits() -> None:
    workflows = sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml"))
    action_pin_re = re.compile(r"(actions/(?:checkout|setup-python|setup-node|upload-artifact))@([0-9a-f]{40})")
    seen: set[str] = set()

    for workflow_path in workflows:
        for line_number, line in enumerate(workflow_path.read_text(encoding="utf-8").splitlines(), start=1):
            match = action_pin_re.search(line)
            if match is None:
                continue
            action, actual_sha = match.groups()
            seen.add(action)
            assert actual_sha == NODE24_OFFICIAL_ACTION_PINS[action], (
                f"{workflow_path}:{line_number}:{action}@{actual_sha}"
            )

    assert seen == set(NODE24_OFFICIAL_ACTION_PINS)


def test_blocking_ci_has_actionable_python_and_stable_suite_contract() -> None:
    workflow = _read(".github/workflows/ci.yml")

    assert "Package / Python" in workflow
    assert "python -m build --sdist --wheel" in workflow
    assert "python -m twine check dist/*" in workflow
    assert "Clean wheel import smoke outside checkout" in workflow
    assert "Static smoke gate" in workflow
    assert "python -m compileall -q src tests scripts" in workflow
    assert "ruff check . --select E9,F63,F7,F82" in workflow
    assert "ruff format --check ." in workflow
    assert "mypy --ignore-missing-imports" in workflow
    assert "Architecture and packaging contracts" in workflow
    assert "pytest -q tests/architecture tests/test_packaging_contract.py --no-cov" in workflow
    assert "Stable regression suite" in workflow
    assert "pytest -q --cov=finite_difference_options" in workflow
    assert "Optional profile /" in workflow
    assert "python -m pip install -r requirements-dev.lock.txt" in workflow
    assert "python -m pip check" in workflow
    assert workflow.count("python -m pip_audit --progress-spinner=off --skip-editable") >= 2
    audit_sequence = [
        "python -m pip install -e '.[dev]'",
        "python -m pip check",
        "python -m pip_audit --progress-spinner=off --skip-editable",
        "python -m pip install -r requirements-dev.lock.txt",
        "python -m pip check",
        "python -m pip_audit --progress-spinner=off --skip-editable",
    ]
    cursor = 0
    for command in audit_sequence:
        cursor = workflow.find(command, cursor)
        assert cursor >= 0, f"missing or out-of-order audit command: {command}"
        cursor += len(command)
    assert "cyclonedx-py environment --of JSON -o sbom.json" in workflow
    assert "Generate release manifest" in workflow
    assert "scripts/write_release_manifest.py --dist dist --output dist/release-manifest.json" in workflow
    assert "retention-days: 14" in workflow


def test_formatter_has_one_exactly_pinned_owner_across_local_and_ci_surfaces() -> None:
    pyproject = tomllib.loads(_read("pyproject.toml"))
    dev_dependencies = pyproject["project"]["optional-dependencies"]["dev"]
    pre_commit = _read(".pre-commit-config.yaml")
    requirements = _read("requirements-dev.txt")
    locked_requirements = _read("requirements-dev.lock.txt")
    qwen = _read("QWEN.md")
    project_context = _read("docs/PROJECT_CONTEXT.md")

    assert "ruff==0.15.21" in dev_dependencies
    assert not any(dependency.startswith("black") for dependency in dev_dependencies)
    assert "ruff==0.15.21" in requirements.splitlines()
    assert "ruff==0.15.21" in locked_requirements.splitlines()
    assert not any(line.startswith(("black==", "pytokens==")) for line in locked_requirements.splitlines())
    assert "entry: ruff format --check --force-exclude" in pre_commit
    assert "entry: ruff check --force-exclude" in pre_commit
    assert "entry: black" not in pre_commit
    assert "Ruff formatting and linting" in qwen
    assert "Black formatting" not in qwen
    assert "**Dev**: pytest, mypy, ruff" in project_context
    assert "**Dev**: pytest, mypy, ruff, black" not in project_context


def test_testclient_dependency_uses_httpx2_without_legacy_httpx() -> None:
    pyproject = tomllib.loads(_read("pyproject.toml"))
    extras = pyproject["project"]["optional-dependencies"]
    requirements = _read("requirements-dev.txt").splitlines()
    locked_requirements = _read("requirements-dev.lock.txt").splitlines()

    for profile in ("validation", "dev"):
        assert "httpx2>=2,<3" in extras[profile]
        assert "httpx2" in _requirement_names(extras[profile])
        assert "httpx" not in _requirement_names(extras[profile])
        assert "httpcore" not in _requirement_names(extras[profile])

    assert "finite-difference-options[api]" in extras["validation"]

    assert "httpx2>=2,<3" in requirements
    assert "httpx2" in _requirement_names(requirements)
    assert "httpx" not in _requirement_names(requirements)
    assert "httpcore" not in _requirement_names(requirements)
    assert any(line.startswith("httpx2==") for line in locked_requirements)
    assert "httpx2" in _requirement_names(locked_requirements)
    assert "httpx" not in _requirement_names(locked_requirements)
    assert "httpcore" not in _requirement_names(locked_requirements)
    assert "httpx" not in {package["name"].lower() for package in tomllib.loads(_read("uv.lock"))["package"]}
    assert "httpcore" not in {package["name"].lower() for package in tomllib.loads(_read("uv.lock"))["package"]}


def test_security_critical_dev_pins_are_coherent_across_locks() -> None:
    pyproject = tomllib.loads(_read("pyproject.toml"))
    uv_lock = tomllib.loads(_read("uv.lock"))
    dev_dependencies = pyproject["project"]["optional-dependencies"]["dev"]
    requirements = _read("requirements-dev.txt").splitlines()
    locked_requirements = _read("requirements-dev.lock.txt").splitlines()
    uv_versions = {package["name"].lower(): package["version"] for package in uv_lock["package"]}

    expected_direct = {
        "cryptography>=50,<51",
        "GitPython>=3.1.60,<4",
    }
    assert expected_direct <= set(dev_dependencies)
    assert expected_direct <= set(requirements)
    assert "cryptography==50.0.0" in locked_requirements
    assert "GitPython==3.1.60" in locked_requirements
    assert uv_versions["cryptography"] == "50.0.0"
    assert uv_versions["gitpython"] == "3.1.60"


def test_build_tool_avoids_yanked_release_across_locks() -> None:
    pyproject = tomllib.loads(_read("pyproject.toml"))
    uv_lock = tomllib.loads(_read("uv.lock"))
    extras = pyproject["project"]["optional-dependencies"]
    requirements = _read("requirements-dev.txt").splitlines()
    locked_requirements = _read("requirements-dev.lock.txt").splitlines()
    uv_versions = {package["name"].lower(): package["version"] for package in uv_lock["package"]}

    for profile in ("build", "dev"):
        assert "build>=1,<1.5.1" in extras[profile]
    assert "build>=1,<1.5.1" in requirements
    assert "build==1.5.0" in locked_requirements
    assert uv_versions["build"] == "1.5.0"


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
    assert "npm sbom --sbom-format cyclonedx --json > npm-sbom.json" in workflow


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
