# Cached LAPACK benchmark inputs

`fixtures/black_scholes_0bc300e.py` is an unmodified historical baseline from this same MIT-licensed repository, commit `0bc300ea161093ab90a4b209e9326956f22e46f5`, path `src/finite_difference_options/solvers/black_scholes.py`. It is benchmark evidence, not an installed implementation or maintained solver API. Retaining it makes a downloaded checkout sufficient to reproduce comparisons without external workspace files or Git history.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/benchmark_cached_lapack.py --output benchmark.json
```

Install the current repository's development profile first. Run `--quick` for a small deterministic metadata/parity smoke; quick timings are not performance evidence. The complete benchmark records actual source hashes, grid/coefficient construction, dtype, boundaries, exact nonzeros/storage, separate cold/warm solver samples, an independently instrumented Python allocation peak and cumulative Linux process RSS. Native allocation coverage is explicitly limited. No timing threshold is part of CI; compare distributions under controlled workloads before extrapolating.
