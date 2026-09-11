"""Reproduce pivoted LAPACK speed, parity, memory and cache-state measurements.\n\nRun against an explicitly installed checkout; no source or package is modified.\n"""

from __future__ import annotations

import argparse
import dataclasses
import tracemalloc
import resource
import hashlib
import importlib.util
import sys
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import solve_banded

from finite_difference_options.solvers import CachedBlackScholesFiniteDifferenceSolver
from finite_difference_options.solvers._tridiagonal import TridiagonalLU
from finite_difference_options.solvers.black_scholes import _black_scholes_tridiagonal_operator

baseline_path = Path(__file__).resolve().parents[1] / "benchmarks/fixtures/black_scholes_0bc300e.py"
if (
    hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    != "7176deef84b8548b1e9987595b10f741d1e5329d06e7335be9e6c2baf60c23e6"
):
    raise ValueError("Historical benchmark baseline no longer matches its pinned Git source")
spec = importlib.util.spec_from_file_location("fd_reference_black_scholes", baseline_path)
assert spec is not None and spec.loader is not None
baseline = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = baseline
spec.loader.exec_module(baseline)
_factor_tridiagonal = baseline._factor_tridiagonal
_solve_factored_tridiagonal = baseline._solve_factored_tridiagonal


def factor(lower, diagonal, upper):
    return TridiagonalLU.factor(lower, diagonal, upper)


def solve(factors, rhs):
    return factors.solve(rhs)


def timed(fn, *, calls=20, repeats=5):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(calls):
            fn()
        samples.append((time.perf_counter_ns() - start) / calls / 1e9)
    return {"median_seconds": statistics.median(samples), "samples_seconds": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--quick", action="store_true", help="Small deterministic contract smoke, not performance evidence"
    )
    args = parser.parse_args()
    calls, repeats = (2, 2) if args.quick else (20, 5)
    time_steps = 20 if args.quick else 400

    def measure(fn):
        return timed(fn, calls=calls, repeats=repeats)

    rows = []
    for n in (21, 51) if args.quick else (101, 501, 2001):
        grid = np.linspace(0.0, 400.0, n)
        low, diagonal, upper = _black_scholes_tridiagonal_operator(
            grid, risk_free_rate=0.04, dividend_yield=0.01, volatility=0.2
        )
        low = -0.005 * low
        diagonal = 1 - 0.005 * diagonal
        upper = -0.005 * upper
        low[0] = low[-1] = upper[0] = upper[-1] = 0
        diagonal[0] = diagonal[-1] = 1
        bands = np.zeros((3, n))
        bands[0, 1:] = upper[:-1]
        bands[1] = diagonal
        bands[2, :-1] = low[1:]
        old = _factor_tridiagonal(low, diagonal, upper)
        new = factor(low, diagonal, upper)
        old_factor = measure(lambda: _factor_tridiagonal(low, diagonal, upper))
        new_factor = measure(lambda: factor(low, diagonal, upper))
        for nrhs in (1, 8):
            rhs = np.random.default_rng(774).normal(size=(n, nrhs))
            if nrhs == 1:
                rhs = rhs[:, 0]
            oracle = solve_banded((1, 1), bands, rhs)
            old_value = _solve_factored_tridiagonal(low, *old, rhs)
            new_value = solve(new, rhs)
            np.testing.assert_allclose(old_value, oracle, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(new_value, oracle, rtol=1e-12, atol=1e-12)
            old_solve = measure(lambda: _solve_factored_tridiagonal(low, *old, rhs))
            new_solve = measure(lambda: solve(new, rhs))
            row = {
                "nodes": n,
                "rhs_columns": nrhs,
                "dtype": "float64",
                "theta": 0.5,
                "dt": 0.01,
                "grid_sha256": hashlib.sha256(grid.tobytes()).hexdigest(),
                "old_factor": old_factor,
                "lapack_factor": new_factor,
                "old_solve": old_solve,
                "lapack_solve": new_solve,
                "solve_speedup": old_solve["median_seconds"] / new_solve["median_seconds"],
                "factor_speedup": old_factor["median_seconds"] / new_factor["median_seconds"],
                "max_abs_difference": float(np.max(np.abs(old_value - new_value))),
                "max_abs_oracle_error": float(np.max(np.abs(new_value - oracle))),
                "factor_reuse": calls * repeats,
                "calls_per_sample": calls,
                "repeats": repeats,
            }
            tracemalloc.start()
            solve(new, rhs)
            _, python_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            row.update(
                matrix_nonzeros=int(np.count_nonzero(bands)),
                matrix_band_storage_bytes=bands.nbytes,
                rhs_bytes=rhs.nbytes,
                factor_storage_bytes=sum(getattr(new, field.name).nbytes for field in dataclasses.fields(new)),
                measured_python_solve_peak_bytes=python_peak,
                cache_state="factor_once_solve_many; factor setup excluded from solve timing",
                rhs_seed=774,
                grid={"kind": "linspace", "start": 0.0, "stop": 400.0, "nodes": n},
                coefficients={"risk_free_rate": 0.04, "dividend_yield": 0.01, "volatility": 0.2},
                boundary_rows="identity",
                parity_rtol=1e-12,
                parity_atol=1e-12,
            )
            rows.append(row)
            print(
                n,
                nrhs,
                "solve_speedup",
                round(row["solve_speedup"], 2),
                "factor_speedup",
                round(row["factor_speedup"], 2),
                flush=True,
            )
    end_to_end = []
    for n in (21,) if args.quick else (101, 501):
        params = dict(
            spot_grid=np.linspace(0, 400, n),
            time_grid=np.linspace(0, 1, time_steps + 1),
            strike=100.0,
            risk_free_rate=0.04,
            dividend_yield=0.01,
            volatility=0.2,
            option_type="call",
        )
        old_solver = baseline.CachedBlackScholesFiniteDifferenceSolver()
        new_solver = CachedBlackScholesFiniteDifferenceSolver()
        old_times = []
        new_times = []
        for repeat in range(3 if args.quick else 7):
            order = (
                ((old_solver, old_times), (new_solver, new_times))
                if repeat % 2 == 0
                else ((new_solver, new_times), (old_solver, old_times))
            )
            for solver, samples in order:
                start = time.perf_counter_ns()
                values = solver.solve_european(**params)
                samples.append((time.perf_counter_ns() - start) / 1e9)
        old_value = old_solver.solve_european(**params)
        new_value = new_solver.solve_european(**params)
        np.testing.assert_allclose(new_value, old_value, rtol=3e-12, atol=3e-12)
        end_to_end.append(
            {
                "nodes": n,
                "time_steps": time_steps,
                "old_samples_seconds": old_times,
                "new_samples_seconds": new_times,
                "old_median_seconds": statistics.median(old_times),
                "new_median_seconds": statistics.median(new_times),
                "speedup": statistics.median(old_times) / statistics.median(new_times),
                "max_abs_solution_difference": float(np.max(np.abs(new_value - old_value))),
                "price_at_100_before": float(np.interp(100, params["spot_grid"], old_value[-1])),
                "price_at_100_after": float(np.interp(100, params["spot_grid"], new_value[-1])),
                "cache_before": old_solver.cache.info().as_dict(),
                "cache_after": new_solver.cache.info().as_dict(),
            }
        )
        end_to_end[-1].update(
            grid={"kind": "linspace", "start": 0.0, "stop": 400.0, "nodes": n},
            parameters={k: v for k, v in params.items() if k not in ("spot_grid", "time_grid")},
            time_grid={"kind": "linspace", "start": 0.0, "stop": 1.0, "steps": time_steps},
            theta=0.5,
            dtype="float64",
            cold_samples={"before": old_times[0], "after": new_times[0]},
            warm_samples={"before": old_times[1:], "after": new_times[1:]},
            warm_speedup=statistics.median(old_times[1:]) / statistics.median(new_times[1:]),
            solution_storage_bytes=new_value.nbytes,
            parity_rtol=3e-12,
            parity_atol=3e-12,
        )
        print("end_to_end", n, "speedup", end_to_end[-1]["speedup"], flush=True)
    report = {
        "schema": "fd-cached-lapack-benchmark.v1",
        "quick": args.quick,
        "classification": "public-synthetic",
        "process_peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "memory_notes": "Array bytes are exact logical storage. Tracemalloc is a separately instrumented solve and excludes untracked native allocations. RSS is cumulative whole-process Linux peak, not per-case allocation.",
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "candidate_source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(sys.modules[TridiagonalLU.__module__].__file__),
                Path(sys.modules[CachedBlackScholesFiniteDifferenceSolver.__module__].__file__),
            ]
        },
        "baseline_commit": "0bc300ea161093ab90a4b209e9326956f22e46f5",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy_build": np.show_config(mode="dicts"),
        "scipy_build": scipy.show_config(mode="dicts"),
        "thread_environment": {
            k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "notes": [
            "Local CPU microbenchmark; no timing threshold added to unit tests.",
            "Candidate includes finite RHS/solution checks and preserves input arrays.",
            "Factor-once solve-many; samples run in the same process; other machine workloads may affect timing.",
        ],
        "cases": rows,
        "end_to_end": end_to_end,
        "baseline_source_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
