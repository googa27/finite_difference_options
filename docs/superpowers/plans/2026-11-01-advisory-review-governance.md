# Advisory review governance implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Close #186’s model-selection/classification defect without enabling a paid or unverified provider path.

**Architecture:** Keep numerical package ownership fixed. Bound advisory automation through explicit default-disabled configuration and deterministic result classification. Preserve patched action identity and existing triage authority.

**Tech Stack:** Existing GitHub Actions, Python standard-library JSON validation, pytest/PyYAML static controls.

**Spec:** ../specs/2026-11-01-advisory-review-governance-design.md

## Global constraints

Base4ac7f5a77425b4b3de6a5b1532f38a1f48f638da/treebbdc55d62c3473aad2bbb9dd706d5b3e2a42bdf7. No numerical/runtime/lock/triage changes. No provider calls, credentials inspection, billing, settings/protection change or remote write before coordinator lease.

## Review focus

Unavailable provider with a green wrapper; empty/invalid/oversized response; PR/head identity drift; incomplete or forged native review evidence; untrusted output reaching write authority. Each needs a meaningful negative control. Configuration that merely asserts free eligibility cannot prove it.

### Task1: Bind actual action boundary and meaningful failure controls

**Files:** .github/workflows/gemini-pr-review.yml; new focused classifier under scripts/; new tests under tests/unit and tests/architecture; canonical docs/ARCHITECTURE.yaml, PRD.md, ARCHITECTURE.md, CI_POLICY.md and AGENTS.md.

- [ ] Read the complete pinned action and current workflow/current issue plus the required canonical documents. Determine exact minimal response contract and authority boundary; retain a concise interface note before edits.
- [ ] Add executable RED fixtures proving the current default/outer-outcome logic mishandles the actual ModelNotFoundError case and invalid versus valid bounded results. Verify expected failure before changes.
- [ ] Implement the smallest cohesive explicit-default-disabled configuration and deterministic classifier/workflow boundary. No implicit model/CLI fallback or model-controlled write authority; preserve all unrelated controls.
- [ ] Run focused behavior and workflow negative mutations, full current architecture/static/security gates, and the unchanged project command; preserve every failure by name. No new dependency installation should be needed.
- [ ] Commit a clean exact-tree checkpoint, bind complete diff and logs, and obtain separate independent review before any publication.

### Task2: Ordinary integration and actual-default acceptance (separate lease)

- [ ] Coordinator inspects frozen candidate/independent findings and freshly leases main/head before native publication.
- [ ] Publish an ordinary PR with exact tree and truthful local/API mapping. Verify genuine current checks and classify disabled/unavailable provider accurately; no paid rerun.
- [ ] After guarded normal merge, export resulting default, replay bounded classification/static/source-free controls and preserve native job evidence. Close #186 only with the explicit provider-eligibility limitation and real substitute-review evidence, without a provider-pass claim.

The reviewed local manual-only successor preserves the original candidate and
requires workflow_dispatch plus a dedicated boolean run_provider input defaulting
to false before any provider eligibility. Both exact operator variables, pinned
no-fallback CLI and trusted native bounded public context remain mandatory.
Automatic events and absent/false manual requests emit disabled artifacts only.
No provider dispatch is part of local testing or publication authorization.
