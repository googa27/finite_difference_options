"""Regenerate the public-synthetic arXiv-Lab Black--Scholes FD fixture.

This maintainer script keeps fixture regeneration behind a command-line entry
point. Downstream consumers should read the JSON artifact directly rather than
import finite_difference_options internals.
"""

from __future__ import annotations

from pathlib import Path

from finite_difference_options.validation.black_scholes_parity import (
    export_public_black_scholes_fixture_json,
)

DEFAULT_FIXTURE_PATH = Path("tests/fixtures/arxiv_lab_bs_oracle_v1.json")


def main() -> None:
    """Regenerate the checked-in public-synthetic fixture JSON."""

    output = export_public_black_scholes_fixture_json(path=DEFAULT_FIXTURE_PATH)
    print(output)


if __name__ == "__main__":
    main()
