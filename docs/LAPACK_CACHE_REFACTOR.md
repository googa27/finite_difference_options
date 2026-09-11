# Cached tridiagonal factorization and solve decision

Date: 2026-09-11. Baseline: `0bc300ea161093ab90a4b209e9326956f22e46f5`.
Fixture classification: public-synthetic. No market observations are used.

## Mathematical statement and preserved behavior

For transformed time tau = T - t, the one-dimensional Black-Scholes operator is
`L V = 0.5 sigma^2 S^2 V_SS + (r - q) S V_S - r V`. The existing nonuniform
central spatial stencil and theta step remain unchanged:
`(I - theta dt L) V_next = (I + (1 - theta) dt L) V_previous`.
The initial tau row is the explicitly requested vanilla payoff; the last row is
valuation. Dirichlet identity rows and the existing time-dependent call/put
boundary values remain unchanged. This patch changes the linear solver only.

`BandedOperatorCache`, `CachedBlackScholesFiniteDifferenceSolver` and
`OperatorCacheInfo` retain their public interfaces. Cache identity still contains
grid bytes/shape, float64 dtype, rate, dividend yield, volatility, theta and dt.
RHS and boundary values are never cached. Factor and RHS-coefficient arrays are
owned read-only snapshots; every solve copies the supplied RHS before permitting
LAPACK to overwrite its work buffer. Vector and multi-column output rank is
preserved, including noncontiguous input views.

## Maintained-library decision

| Capability | Selected implementation | Alternatives | Evidence and boundary |
|---|---|---|---|
| Factor a real tridiagonal matrix once | SciPy `dgttrf` | Python Thomas algorithm; generic sparse LU | [SciPy API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lapack.dgttrf.html), [LAPACK algorithm and status contract](https://www.netlib.org/lapack/double/dgttrf.f) |
| Solve repeatedly for one or more RHS | SciPy `dgttrs` | Python substitution loops; `solve_banded` refactorization for every RHS | [SciPy API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lapack.dgttrs.html), [LAPACK multiple-RHS contract](https://www.netlib.org/lapack/double/dgttrs.f) |
| Numerical dependency and licensing | Existing core `scipy>=1.16,<2` | Add a numerical dependency or maintain custom native code | [SciPy LAPACK wrappers](https://docs.scipy.org/doc/scipy/reference/linalg.lapack.html), [SciPy license](https://github.com/scipy/scipy/blob/main/LICENSE.txt) |

The private adapter is `solvers/_tridiagonal.py`. LAPACK owns elimination,
partial pivoting and substitution. Custom code owns row-aligned FD diagonal
conversion (`lower[1:]`, `upper[:-1]`), cache memory ownership, shape/finiteness
checks and typed failure translation. No dependency or public export is added.
Tridiagonal factorization remains O(n), each k-column solve O(nk), and factor
storage O(n); this is an implementation improvement, not an asymptotic claim.

## Failure behavior and limits

- Negative factorization status or nonzero solve status raises `ValidationError`.
- Exactly singular pivots and final U pivots with magnitude at most `1e-14`
  are refused. The absolute floor is retained from the previous implementation;
  it is not a condition-number estimate or a scale-invariant regularization rule.
- Partial pivoting can solve a nonsingular zero-diagonal system that unpivoted
  Thomas previously rejected. This is an intentional numerical robustness
  improvement, exercised against an independent dense oracle.
- Non-finite coefficients/factors/RHS/solutions and mismatched RHS shapes fail
  with typed errors. Failed construction does not populate the cache.
- The internal SciPy adapter requires at least three rows because the wrapper
  rejects smaller inputs in the tested environment. The public Black-Scholes
  solver already requires at least five grid points; its supported domain is
  unchanged.
- Empty `(n, 0)` RHS batches return an owned empty array before a native call.
  Independent review reproduced native memory corruption with this wrapper
  input, despite an apparently correct returned shape. The subprocess regression
  checks successful interpreter exit as well as shape.

## Measured performance

The [machine-readable benchmark](benchmarks/cached_lapack_20260911.json) records
raw samples, versions, CPU/platform, NumPy and SciPy build metadata, thread
environment, dtype, matrix/grid identities and reuse counts. The baseline source
snapshot is byte-identical to the stated Git revision. Final measurements time
the actual adapter, including validation, factor ownership and RHS copies.

For 101/501/2001-node matrices and one/eight RHS columns, median factorization
speedups were about 2.9–27.1 times, and repeated solve speedups about 5.8–48.9
times. Both paths agreed with `scipy.linalg.solve_banded` at
`rtol=atol=1e-12`. Timings are local CPU observations with five sample groups;
they are not universal guarantees or CI pass thresholds.

The representative complete solver benchmark used 400 time steps, sigma 0.2,
rate 0.04, dividend yield 0.01, maturity 1, strike 100 and spot domain [0, 400].
Seven samples alternated baseline/current ordering in the same process. Median
complete-solve speedups were **3.27 times at 101 nodes** and **9.73 times at
501 nodes**. Full solution matrices agreed within `rtol=atol=3e-12`; this
tolerance allows the maximum absolute difference of approximately `3.13e-12`
at 501 nodes. Both implementations recorded the same one miss, one cache entry,
3,199 hits and 3,200 solves after the timed and verification runs.

The benchmark harness and byte-pinned baseline snapshot are retained in the
portfolio review workspace as `reports/local/benchmark_fd_lapack.py` and
`reports/local/_fd_black_scholes_before.py`. The initial prototype benchmark is
historical evidence only; its raw-wrapper factorization timings omit some final
adapter validation and must not be substituted for the final numbers here.

## Verification

`tests/unit/test_cached_lapack_system.py` uses independently assembled dense
uniform-grid theta matrices, vector/multi-RHS solves, analytic call/put prices,
cache invalidation/reuse, singular and non-finite failures, immutable factors,
pivot-required systems and a native-process empty-batch regression. Existing
public solver/Pinares/Black-Scholes parity tests remain required, followed by
the full stable suite, architecture/typing/format gates, derivative evidence
and clean-wheel validation. This refactor does not raise a route's maturity or
claim a new PDE, boundary, stochastic-process, obstacle or pricing capability.
