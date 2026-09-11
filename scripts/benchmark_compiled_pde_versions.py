"""Compare retained dense v0 and opt-in banded v1 on identical compiled problems.

Run with BLAS/OpenMP threads set before Python starts. No source, environment or
external service is mutated; only the explicit JSON output is written.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import scipy

from finite_difference_options.integrations.compiled_pde_black_scholes_route import _solve_compiled_black_scholes_grid
from finite_difference_options.solvers._compiled_black_scholes import solve_compiled_black_scholes_grid_v1
from finite_difference_options.validation.black_scholes_parity import black_scholes_call_oracle
from finite_difference_options.validation.fd_evidence.replay_identity import v1_runtime_identity
from finite_difference_options.validation.fd_verification import (
    run_fd_bs_verification_benchmark,
    run_fd_bs_verification_benchmark_v1,
)


def _measure(function):
    start = time.perf_counter_ns()
    value = function()
    return value, (time.perf_counter_ns() - start) / 1e9


def _paired(old, new, repetitions):
    old_value, old_cold = _measure(old)
    new_value, new_cold = _measure(new)
    samples = {"v0": [], "v1": []}
    for repeat in range(repetitions):
        order = (("v0", old), ("v1", new)) if repeat % 2 == 0 else (("v1", new), ("v0", old))
        for name, function in order:
            _, elapsed = _measure(function)
            samples[name].append(elapsed)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    return (
        old_value,
        new_value,
        {
            "first_call_seconds": {"v0": old_cold, "v1": new_cold},
            "alternating_warm_seconds": samples,
            "median_seconds": medians,
            "speedup_v0_over_v1": medians["v0"] / medians["v1"],
            "cache_state": "fresh local factor cache on every invocation; exact-dt reuse occurs only within each solve",
        },
    )


def _grid_case(nodes, time_nodes, repetitions):
    inputs = dict(
        spot_grid=np.linspace(0.0, 3.0, nodes),
        time_grid=np.linspace(0.0, 1.0, time_nodes),
        strike=1.0,
        risk_free_rate=0.05,
        dividend_yield=0.0,
        volatility=0.2,
        theta=0.5,
    )
    old, new, timing = _paired(
        lambda: _solve_compiled_black_scholes_grid(**inputs),
        lambda: solve_compiled_black_scholes_grid_v1(**inputs),
        repetitions,
    )
    np.testing.assert_allclose(new[0], old[0], rtol=3e-12, atol=3e-12)
    assert new[1] == old[1]
    oracle = black_scholes_call_oracle(1.0, 1.0, 0.05, 0.2, 1.0)
    config = {name: value.tolist() if isinstance(value, np.ndarray) else value for name, value in inputs.items()}
    return {
        "inputs": config,
        "input_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "dimensions": {"space": nodes, "time_nodes": time_nodes, "steps": time_nodes - 1},
        "dtype": "float64",
        "device": "CPU",
        "timing_scope": (
            "complete kernel: validation, assembly, per-step factor/solve, "
            "boundary schedules, retained history and diagnostics"
        ),
        "timings": timing,
        "max_abs_all_time_slice_difference": float(np.max(np.abs(new[0] - old[0]))),
        "allclose_acceptance": {"rtol": 3e-12, "atol": 3e-12},
        "price_abs_errors": {
            "v0": abs(float(np.interp(1.0, inputs["spot_grid"], old[0][-1])) - oracle),
            "v1": abs(float(np.interp(1.0, inputs["spot_grid"], new[0][-1])) - oracle),
        },
        "v1_operator": new[2],
        "known_owned_arrays": {
            "v0_operator_bytes": nodes * nodes * 8,
            "v0_identity_bytes": nodes * nodes * 8,
            "v0_step_lhs_bytes": nodes * nodes * 8,
            "v1_operator_bytes": new[2]["operator_storage_bytes"],
            "v1_retained_factor_bytes": new[2]["factor_storage_bytes"],
            "both_history_bytes": nodes * time_nodes * 8,
            "limits": (
                "not peak RSS; excludes Python objects, boundary records, temporary expressions and native workspace"
            ),
        },
    }


def benchmark(repetitions):
    grid_cases = [_grid_case(nodes, times, repetitions) for nodes, times in ((121, 200), (401, 400))]
    old, new, evidence_timing = _paired(
        run_fd_bs_verification_benchmark,
        run_fd_bs_verification_benchmark_v1,
        repetitions,
    )
    assert old["evidence"]["status"] == new["evidence"]["status"] == "passed"
    return {
        "schema": "fd.compiled-version-performance/v1",
        "repetitions": repetitions,
        "execution_order": "v0 then v1 first calls; alternating order thereafter in one process",
        "runtime": v1_runtime_identity(),
        "hardware": {"processor": platform.processor(), "machine": platform.machine(), "logical_cpus": os.cpu_count()},
        "dense_reference_source_sha256": hashlib.sha256(
            inspect.getsource(_solve_compiled_black_scholes_grid).encode()
        ).hexdigest(),
        "numpy_build": np.show_config(mode="dicts"),
        "scipy_build": scipy.show_config(mode="dicts"),
        "grid_cases": grid_cases,
        "whole_evidence": {
            "timing_scope": (
                "same full refinement/oracle/residual/manufactured/negative-control pipeline, "
                "excluding duplicate validator run and JSON/file writing"
            ),
            "timings": evidence_timing,
            "v0_config": old["config"],
            "v1_config": new["config"],
            "v0_finest": old["results"]["full_refinement"]["rows"][-1],
            "v1_finest": new["results"]["full_refinement"]["rows"][-1],
            "both_status": "passed",
            "limits": "validation retains dense matrix residual references; scientific thresholds are unchanged",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    if args.repetitions < 3:
        parser.error("at least three repetitions are required")
    result = benchmark(args.repetitions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
