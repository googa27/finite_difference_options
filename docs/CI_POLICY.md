# CI policy

Audit baseline: 2026-06-26
Owning issue: [#69](https://github.com/googa27/finite_difference_options/issues/69)

## Blocking signals

The `CI` workflow is the blocking repository validation signal for pull requests and pushes.

### Python job

The Python job uses Python 3.12 and reports failures by command group:

1. install dependencies from `requirements.txt` and `requirements-dev.txt`;
2. static smoke gate:
   - `python -m compileall -q src tests`;
   - `ruff check . --select E9,F63,F7,F82`;
3. architecture fitness gate:
   - `pytest -q tests/architecture`;
4. stable regression suite:
   - boundary conditions;
   - callable bond;
   - FD backend capability manifest;
   - public-synthetic Black-Scholes parity/evidence fixture;
   - finite-difference Greeks;
   - Greeks;
   - multidimensional adapter coefficients;
   - option pricer/result/instrument tests;
   - PDE pricer;
   - plot API/backends;
   - pricing-engine import smoke;
   - unified pricing engine;
   - unified processes.

The stable suite is intentionally narrower than the whole repository. Known broader numerical, packaging, typing, performance, and application debt is tracked in GitHub issues and must not make unrelated documentation or architecture PRs permanently red.

### Node job

The Node job is a presence-gated application-profile smoke:

- if `nextjs-client/package.json` is absent, the job records that the Next.js client is absent and skips install/test/lint;
- if `nextjs-client/package-lock.json` is present, it uses `npm ci`;
- if the package exists without a lockfile, it uses `npm install --no-audit --no-fund` and emits an explicit log message instead of passing a missing lock path to `actions/setup-node`.

If the frontend becomes a maintained deliverable, a successor issue must add a committed lockfile, explicit `test` and `lint` scripts, and separate dependency/audit evidence. Until then, frontend checks are scoped to the optional application profile, not the numerical core package.

## Advisory automated-review signals

Gemini PR review, issue triage, and conversational CLI workflows are advisory automation. They may add useful feedback, but quota, authentication, configuration, external-service, or tool failures are classified as review-service events rather than repository-validation failures.

The workflows therefore:

- do not post generic failure comments for service failures;
- leave labels unchanged when advisory issue triage cannot produce trustworthy JSON;
- write a warning and step summary that the failure is advisory;
- keep `CI` as the source of truth for blocking Python/Node/package validation.

If Gemini returns concrete repository findings, those findings still require normal engineering treatment: reproduce, fix, test, and document the response in the PR.

## Future hardening not closed by #69

Issue #69 restores a trustworthy blocking baseline. It does not claim to finish all release hardening. Successor issues continue to own:

- installable package/wheel testing outside the checkout;
- minimum-supported versus latest-compatible numerical profiles;
- full repository lint/type/test debt repayment;
- action pinning, SBOM, audit, and benchmark profiles;
- maintained frontend dependency and vulnerability policy if the frontend is promoted from optional example to deliverable.


The architecture contract gate (`python scripts/check_architecture_contract.py`) must remain in blocking CI beside `pytest -q tests/architecture`; `docs/architecture_contract.toml` is the reviewed topology source of truth.
