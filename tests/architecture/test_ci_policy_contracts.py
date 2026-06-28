from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_blocking_ci_has_actionable_python_and_stable_suite_contract() -> None:
    workflow = _read(".github/workflows/ci.yml")

    assert "Static smoke gate" in workflow
    assert "python -m compileall -q api src tests" in workflow
    assert "ruff check . --select E9,F63,F7,F82" in workflow
    assert "Architecture fitness gate" in workflow
    assert "pytest -q tests/architecture" in workflow
    assert "Stable regression suite" in workflow
    assert "PYTHONPATH=. pytest -q" in workflow
    assert "tests/test_api_schema_contracts.py" in workflow
    assert "tests/test_api_route_validation.py" in workflow
    assert "tests/test_api_interpolation_diagnostics.py" in workflow
    assert "tests/test_unified_pricing_engine.py" in workflow


def test_node_profile_never_references_a_missing_lockfile_cache_path() -> None:
    workflow = _read(".github/workflows/ci.yml")

    assert "lockfile=true" in workflow
    assert "Install dependencies with lockfile" in workflow
    assert "Install dependencies without lockfile" in workflow
    assert "cache-dependency-path: nextjs-client/package-lock.json" not in workflow


def test_gemini_review_and_issue_triage_failures_are_advisory() -> None:
    pr_review = _read(".github/workflows/gemini-pr-review.yml")
    issue_triage = _read(".github/workflows/gemini-issue-automated-triage.yml")

    assert "continue-on-error: true" in pr_review
    assert "Classify unavailable Gemini PR review as advisory" in pr_review
    assert (
        "review-service events, not as Python/Node/package validation failures"
        in pr_review
    )
    assert "Post PR review failure comment" not in pr_review

    assert "continue-on-error: true" in issue_triage
    assert "Classify unavailable Gemini issue triage as advisory" in issue_triage
    assert "labels left unchanged" in issue_triage
    assert "Post Issue Analysis Failure Comment" not in issue_triage


def test_ci_policy_is_documented_from_agent_contract_and_architecture() -> None:
    policy = _read("docs/CI_POLICY.md")
    agents = _read("AGENTS.md")
    architecture = _read("docs/ARCHITECTURE.md")

    assert "# CI policy" in policy
    assert "Blocking signals" in policy
    assert "Advisory automated-review signals" in policy
    assert "docs/CI_POLICY.md" in agents
    assert "docs/CI_POLICY.md" in architecture
