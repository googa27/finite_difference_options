"""Compose installed public FPF compilation and explicit FD v1 APIs offline.

Requires separately verified FPF and FD wheels in the same environment. This is
one exact public-synthetic European-call seam, not arbitrary PDE or UI coverage.
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path


def run() -> dict:
    from financial_problem_formulations import black_scholes_call_pde_ir_fixture
    from financial_problem_formulations.algebra.pde_ir.compiler import compile_pde_ir_payload

    from finite_difference_options.integrations import (
        packaged_compiled_black_scholes_fixture,
        screen_compiled_pde_payload,
        solve_compiled_pde_payload_v1,
    )
    from finite_difference_options.validation.fd_verification import (
        run_fd_bs_verification_benchmark_v1,
        validate_fd_bs_verification_bundle,
    )

    source = black_scholes_call_pde_ir_fixture().to_dict()
    compiled = compile_pde_ir_payload(source).to_dict()
    payload = packaged_compiled_black_scholes_fixture()
    payload["source_pde_ir"] = source
    payload["compiled_operator_result"] = compiled
    screening = screen_compiled_pde_payload(payload)
    result = solve_compiled_pde_payload_v1(payload)
    evidence = run_fd_bs_verification_benchmark_v1()
    validate_fd_bs_verification_bundle(evidence)
    if not result.passed:
        raise RuntimeError("compiled v1 solve did not pass its numerical gates")
    return {
        "schema": "fpf-fd.compiled-v1-interoperability/v1",
        "producer_distribution": {"financial-problem-formulations": version("financial-problem-formulations")},
        "consumer_distribution": {"finite-difference-options": version("finite-difference-options")},
        "producer_calls": ["black_scholes_call_pde_ir_fixture", "algebra.pde_ir.compiler.compile_pde_ir_payload"],
        "classification": "public_synthetic",
        "interoperability": "actual public compiler output screened and solved by explicit FD v1",
        "limits": (
            "one exact fixture; no arbitrary PDE, private data, UI v1, "
            "cross-runtime bitwise or full-surface arbitrage claim"
        ),
        "screening": screening.as_dict(),
        "solve": result.as_dict(),
        "verification": evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    packet = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(packet, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
