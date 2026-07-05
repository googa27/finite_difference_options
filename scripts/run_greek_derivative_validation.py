#!/usr/bin/env python3
"""Run the FD Greek derivative validation gate and write a JSON artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from finite_difference_options.validation.greek_derivative_gates import (
    run_greek_derivative_validation,
    write_greek_derivative_validation_artifact,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("pr", "broad"),
        default="pr",
        help="Validation matrix size. 'pr' is deterministic and CI-friendly; 'broad' adds stress cases.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/fd-greek-derivative-validation.json",
        help="Output JSON artifact path.",
    )
    args = parser.parse_args()

    report = run_greek_derivative_validation(mode=args.mode)
    write_greek_derivative_validation_artifact(Path(args.output), report)
    print(
        json.dumps(
            {
                "benchmark_id": report.benchmark_id,
                "mode": report.mode,
                "passed": report.passed,
                "metrics": report.metrics,
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
