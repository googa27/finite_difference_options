"""A fresh checkout must reproduce benchmark inputs and numerical parity."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_small_benchmark_retains_reproducible_inputs_and_honest_measurement_scope(tmp_path):
    output = tmp_path / "benchmark.json"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/benchmark_cached_lapack.py"), "--quick", "--output", str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        env=os.environ | {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["quick"] is True
    assert report["classification"] == "public-synthetic"
    assert report["baseline_commit"] == "0bc300ea161093ab90a4b209e9326956f22e46f5"
    assert report["baseline_source_sha256"] == "7176deef84b8548b1e9987595b10f741d1e5329d06e7335be9e6c2baf60c23e6"
    assert len(report["candidate_source_sha256"]) == 2
    assert "not per-case" in report["memory_notes"]
    for row in report["cases"]:
        assert row["matrix_nonzeros"] == 3 * row["nodes"] - 4
        assert row["grid"] == {"kind": "linspace", "start": 0.0, "stop": 400.0, "nodes": row["nodes"]}
        assert row["factor_storage_bytes"] > 0
        assert row["measured_python_solve_peak_bytes"] > 0
        assert row["rhs_seed"] == 774
    for row in report["end_to_end"]:
        assert len(row["warm_samples"]["before"]) == len(row["old_samples_seconds"]) - 1
        assert row["cold_samples"]["after"] == row["new_samples_seconds"][0]
        assert row["parameters"]["strike"] == 100.0
        assert row["time_grid"]["steps"] == row["time_steps"]
