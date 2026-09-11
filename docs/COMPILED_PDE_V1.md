# Versioned compiled Black–Scholes numerics

Issues [#170](https://github.com/googa27/finite_difference_options/issues/170) and [#176](https://github.com/googa27/finite_difference_options/issues/176) introduce an explicit v1 route. Existing `solve_compiled_pde_payload(payload)`, `run_fd_bs_verification_benchmark()`, both existing CLI routes and the original writer keep v0. Their captured numerical bytes remain the same under the capture runtime. The public DTO identities and v0 schema strings are unchanged.

## Selection and consumer boundary

```python
from finite_difference_options.integrations import (
    packaged_compiled_black_scholes_fixture,
    solve_compiled_pde_payload_v1,
)
from finite_difference_options.validation.fd_verification import (
    run_fd_bs_verification_benchmark_v1,
    validate_fd_bs_verification_bundle,
    write_fd_bs_verification_json_v1,
)

result = solve_compiled_pde_payload_v1(packaged_compiled_black_scholes_fixture())
evidence = run_fd_bs_verification_benchmark_v1()
validate_fd_bs_verification_bundle(evidence)
assert result.passed and evidence["evidence"]["status"] == "passed"
```

The v1 solve schema is `finite-difference-options.compiled-pde-solve-result/v1`; the verification schema is `finite-difference-options.fd-verification-evidence/v1`, with benchmark identity `FD-BS-001-V1`. Screening uses the same exact public-synthetic `pde_ir.v0` fixture and recomputed source/compiler hashes. This does not enable arbitrary products, arbitrary coefficients, private data, new boundary types or a general symbolic PDE executor. FPF remains an optional producer through its public compiler and serialized contract; FD imports no FPF implementation. The pinned UI/artifacts consumer supports v0. V1 is an explicit Python opt-in and has no UI support claim.

## Mathematical and numerical contract

Time-to-maturity is `tau = T - t`; the physical coordinate is spot `S`. The constant-coefficient European call satisfies

\[
V_\tau = L V = \tfrac12\sigma^2 S^2 V_{SS} + (r-q)S V_S-rV,
\qquad V(S,0)=\max(S-K,0).
\]

On the finite domain `[0,Smax]`, the lower Dirichlet value is zero; the upper approximation is `max(Smax exp(-q tau) - K exp(-r tau), 0)`. This inherited truncation policy is an asymptotic boundary, not the exact finite-domain option value. Boundary values are evaluated at every actual time node; fixed identity boundary rows can be reused. No source term, early exercise or Rannacher smoothing is added.

For adjacent spacings `a=S[i]-S[i-1]`, `b=S[i+1]-S[i]`, the first-derivative weights are `[-b/(a(a+b)), (b-a)/(ab), a/(b(a+b))]`; second-derivative weights are `[2/(a(a+b)), -2/(ab), 2/(b(a+b))]`. Combining them with diffusion, drift and reaction gives the three row-aligned diagonals. This reproduces constants, linear and quadratic polynomials exactly. Second-order spatial consistency applies to uniform or appropriately smoothly graded grids; arbitrary irregular spacing does not establish second-order convergence.

Each interval uses

\[
(I-\theta\Delta\tau L)V^{n+1}
= V^n+(1-\theta)\Delta\tau L V^n
\]

with new boundary values imposed on the right-hand side. The public fixture uses `theta=1/2`. The private kernel preserves the theta family for parity testing, including explicit and implicit endpoints. Crank–Nicolson's smooth-solution temporal order does not imply monotonicity or unconditional positivity for arbitrary central-difference drift grids or a nonsmooth payoff. The existing temporal refinement, manufactured consistency, algebraic/boundary residuals, Greek tolerances and sign/source/reaction/boundary negative controls remain the acceptance criteria; none is relaxed. See [Crank and Nicolson's original paper](https://www.cambridge.org/core/journals/mathematical-proceedings-of-the-cambridge-philosophical-society/article/abs/practical-method-for-numerical-evaluation-of-solutions-of-partial-differential-equations-of-the-heatconduction-type/B3230893A53384D418228AB39D41A451).

## Maintained kernel, reuse and bounds

The existing private `TridiagonalLU` adapter owns [SciPy dgttrf](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lapack.dgttrf.html) and [dgttrs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lapack.dgttrs.html). [LAPACK](https://netlib.org/lapack/explore-html/d6/d46/group__gttrf_ga8d1e46216e6c861c89bd4328b8be52a1.html) supplies partial pivoting and the second superdiagonal. No new dependency or custom LU implementation is introduced. SciPy is an actively developed [BSD-licensed project](https://scipy.org/about/). A general sparse solve or `solve_banded` on every step would discard the useful factor reuse for this narrow invariant operator.

The cache is local to one invocation: grid, coefficients, theta, dtype and boundary row types are fixed. Its key is the exact binary64 interval, with no rounding/coalescing; nominally uniform `linspace` intervals can have several distinct keys. A subsequent call always assembles fresh state. Three-diagonal application avoids dense assembly in the solver. Operator storage is `O(N)`; factor storage is `O(U N)` for `U` distinct intervals; retained solution history is `O(N M)`. Validation deliberately retains an independent dense matrix for residual/reference checks. Reported owned-array bytes exclude Python objects, temporary/native allocations and overall process RSS.

V1 pointwise diagnostics use discounted carry bounds:

\[
\max(S e^{-qT}-K e^{-rT},0)\le C\le S e^{-qT},
\quad 0\le\Delta\le e^{-qT},\quad\Gamma\ge0.
\]

These assume deterministic proportional carry and fixed BSM parameters for Delta; they are necessary single-point checks, not a full volatility-surface arbitrage proof. The price normalization is consistent with [Gatheral–Jacquier, section 2](https://arxiv.org/html/1204.0646v4). Negative yields may produce a valid call value or Delta above the unadjusted spot/one bounds. Tests include analytic counterexamples to the old helper plus negative controls for all five bounds. V0's old diagnostic output is retained for exact replay; this correction does not retroactively authenticate old generalized-carry claims.

## Replay policy and migration

The schema selects the implementation. Currently only package provenance `code_version="0.1.0"` has a retained, verified implementation. A fabricated older version such as `0.0.1`, an unknown schema, wrong distribution, malformed/NaN evidence or unavailable runtime is refused with `FDVerificationError` before numerical work. Earlier acceptance of arbitrary nonempty version strings was a provenance defect; it is intentionally removed. No version is silently mapped to whichever solver happens to be current.

V1 binds the method, float64 dtype, Python/NumPy/SciPy versions, platform/build metadata, declared thread environment and hashes of the executing numerical/validation implementation files. Matching metadata is a prerequisite; exact recomputation remains decisive. It is not proof of all hardware state or authenticity of an external sender. Changing these identities requires the corresponding implementation/environment or newly generated evidence; there is no tolerance-based replay migration.

The checker independently recomputes hashes and results, and compares the complete v1 envelope. Rehashing a false price, cache count or status cannot turn it into evidence. The v0 path retains its old arithmetic and canonical numerical comparisons. Captured c85b403 artifacts and provenance live under `tests/fixtures/compiled_pde_authentic_v0_*`; different BLAS roundoff must produce explicit refusal, never rewritten historical bytes. Same-runtime byte equality is separately checked in the release packet. Consumers should store their original artifact, request the appropriate named implementation and treat unavailable replay as a refusal. Package/schema identifiers and dependency pins alone do not establish cross-platform bitwise portability.

## Runnable composition and performance evidence

With separately verified FPF and FD wheels installed, run `python scripts/example_compiled_pde_v1_fpf.py --out /tmp/fpf-fd-v1.json`. It calls FPF's public fixture/compiler, passes the actual serialized output through FD screening and v1 solve, recomputes the v1 verification bundle, and records both distributions and the narrow evidence limits. It makes no provider/network calls.

Run `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python scripts/benchmark_compiled_pde_versions.py --out /tmp/fd-versions.json`. The runner compares complete retained-dense and banded kernels on identical 121-by-200 and 401-by-400 node grids plus the whole evidence pipeline. It records first-call and alternating warm-process samples, exact input/grid hashes, cache counts, owned-array storage limits, NumPy/SciPy build details and errors against the same analytic value. Each invocation starts with a fresh local cache. Scientific evidence and timing are separate; no speedup is a portable performance guarantee.

The recorded 2026-09-11 run uses Python 3.12.3, NumPy 2.5.3 and SciPy 1.18.1 with all four listed thread variables set to one. These are medians of five alternating warm-process runs; each local LU cache starts empty. Full-history parity uses `rtol=atol=3e-12`, with maximum absolute differences 6.22e-15 and 2.45e-14 in the two grids. Raw samples, full inputs, actual cache counts and memory limits are in [the benchmark receipt](benchmarks/compiled_pde_versions_20260911.json).

| Workload | Dense v0 seconds | Banded v1 seconds | Measured speedup |
|---|---:|---:|---:|
| Complete kernel, 121 space × 200 time nodes | 0.034288 | 0.007505 | 4.57× |
| Complete kernel, 401 space × 400 time nodes | 0.971711 | 0.017057 | 56.97× |
| Complete verification evidence | 0.392753 | 0.108122 | 3.63× |
