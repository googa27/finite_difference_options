# CI policy

Audit baseline: 2026-07-03
Owning issues: [#69](https://github.com/googa27/finite_difference_options/issues/69), [#51](https://github.com/googa27/finite_difference_options/issues/51), [#67](https://github.com/googa27/finite_difference_options/issues/67)

## Blocking signals

The `CI` workflow is the blocking repository validation signal for pull requests and pushes.

### Package job

The package job builds the release artifacts from PEP 621 metadata on Python 3.12:

1. `python -m build --sdist --wheel`;
2. `python -m twine check dist/*`;
3. clean-wheel import smoke outside the repository checkout;
4. release manifest generation with artifact, lockfile, capability-matrix and benchmark-registry hashes;
5. upload of the sdist/wheel/manifest artifacts with bounded retention.

The clean-wheel smoke imports only core/contract/validation/backend modules, proving that the wheel does not depend on repository-relative `src.*` imports.

### Test job

The test job installs the development profile with `python -m pip install -e '.[dev]'` and runs:

1. static smoke gate:
   - `python -m compileall -q src tests scripts`;
   - `ruff check . --select E9,F63,F7,F82`;
   - mypy on contract-critical modules with ignore-missing-imports and follow-imports=silent;
2. architecture, documentation, and packaging contracts:
   - `python scripts/check_architecture_contract.py`;
   - `python scripts/check_markdown_links.py`;
   - `pytest -q tests/architecture tests/test_packaging_contract.py --no-cov`;
3. stable regression suite:
   - `pytest -q --cov=finite_difference_options --cov-report=xml --junitxml=junit.xml`;
4. derivative validation artifact:
   - `python scripts/run_greek_derivative_validation.py --mode pr --output artifacts/fd-greek-derivative-validation.json`.

The stable suite is now the whole deterministic Python suite. Known broader performance/application maturity work is still tracked in GitHub issues, but package/import regressions are blocking. The derivative artifact is uploaded with coverage/JUnit artifacts so PRs expose Delta/Gamma convergence, strike-alignment and Rannacher-stability evidence instead of relying only on console logs.

### Scheduled derivative validation

The `Derivative validation broad matrix` workflow runs weekly and on demand. It executes `scripts/run_greek_derivative_validation.py --mode broad` and uploads `fd-greek-derivative-validation-broad.json`; failures are blocking for that workflow and indicate that the derivative maturity claim needs review before release.

### Optional-profile job

The optional-profile job builds a wheel and installs it in clean environments for these extras:

- `api`;
- `cli`;
- `ui`;
- `viz`;
- `validation`.

Each profile imports its advertised optional surface from the installed wheel. This keeps FastAPI, Typer, Streamlit, Matplotlib/Plotly/Seaborn, and test tooling out of core metadata while still proving the extras resolve.

### Audit/SBOM job

The audit job installs the development profile, applies `requirements-dev.lock.txt`, runs `python -m pip check`, runs `python -m pip_audit --progress-spinner=off --skip-editable`, and emits a CycloneDX JSON SBOM.

`requirements-dev.lock.txt` is the pinned reproducible development/audit environment. `pyproject.toml` remains the package metadata source of truth and intentionally keeps compatible runtime ranges.

### Node job

The Node job is a presence-gated application-profile smoke:

- if `nextjs-client/package.json` is absent, the job records that the Next.js client is absent and skips install/test/lint/build/audit;
- if `nextjs-client/package-lock.json` is present, it uses `npm ci`;
- if the package exists without a lockfile, it uses `npm install --no-audit --no-fund` and emits an explicit log message;
- when a maintained frontend exists, the job runs tests, lint, production build, npm audit, and emits an npm SBOM artifact from the committed lockfile.

If the frontend becomes a maintained deliverable, a successor issue must add a committed lockfile, explicit `test`, `lint`, and `build` scripts, and separate dependency/audit evidence. Until then, frontend checks are scoped to the optional application profile, not the numerical core package.

## Supply-chain/runtime controls

All third-party GitHub Actions are pinned to reviewed full commit SHAs. Mutable tags such as `@v4` or `@v0` may appear only in explanatory comments, not as executable refs. The CI workflow declares top-level read-only contents permission, workflow concurrency/cancellation, per-job timeouts, and explicit artifact retention.

Release-manifest generation is intentionally lightweight and deterministic: `scripts/write_release_manifest.py` hashes built distributions plus governance inputs (`pyproject.toml`, requirements/lock files, capability matrix, and benchmark-registry fixture). This links source commit, artifact hashes, CI run metadata, and production route evidence without reading secrets.

## Advisory automated-review signals

Gemini PR review, issue triage, and conversational CLI workflows are advisory automation. They may add useful feedback, but quota, authentication, configuration, external-service, or tool failures are classified as review-service events rather than repository-validation failures.

The workflows therefore:

- do not post generic failure comments for service failures;
- leave labels unchanged when advisory issue triage cannot produce trustworthy JSON;
- write a warning and step summary that the failure is advisory;
- keep `CI` as the source of truth for blocking Python/Node/package validation.

Issue #61 hardens issue triage's authority boundary:

- the Gemini issue analysis job has no GitHub issue-write permission and receives an empty `GITHUB_TOKEN`;
- the label-application job has no Gemini/GCP/model credentials and receives only bounded job outputs;
- `google-github-actions/run-gemini-cli` is pinned to the reviewed patched `v0.1.22` commit and installs Gemini CLI `0.40.0-preview.3`;
- deterministic validation in `scripts/validate_ai_triage_output.py` enforces a closed JSON schema, bounded output size, trusted candidate issue numbers, trusted repository label allowlists, deduplication, and case-insensitive protected/control-label rejection for P0/security/triage-marker variants;
- label writes are additive (`addLabels`) and only remove `status/needs-triage` after a valid decision, so unrelated human labels are preserved;
- scheduled triage is chunked to at most five candidate issues per run and rotates the candidate window by `GITHUB_RUN_NUMBER` so low-confidence early candidates cannot permanently starve later issues.

If Gemini returns concrete repository findings, those findings still require normal engineering treatment: reproduce, fix, test, and document the response in the PR.

## Future hardening not closed by #51

Issue #51 establishes package metadata, clean-wheel smoke, optional-profile smoke, lock/audit/SBOM evidence, and package topology gates. Successor issues continue to own:

- minimum-supported versus latest-compatible numerical dependency matrices beyond Python 3.12;
- full repository lint/type debt repayment beyond the smoke gates;
- benchmark-regression budgets and performance artifacts;
- maintained frontend dependency and vulnerability policy if the frontend is promoted from optional example to deliverable.

The architecture contract gate (`python scripts/check_architecture_contract.py`) must remain in blocking CI beside `pytest -q tests/architecture tests/test_packaging_contract.py --no-cov`; `docs/architecture_contract.toml` is the reviewed topology source of truth.
