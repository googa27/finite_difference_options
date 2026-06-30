"""Regenerate the public-synthetic arXiv-Lab Black--Scholes FD fixture.

This maintainer script keeps the README regeneration command out of the
transitional ``src.*`` import surface. Downstream consumers should read the
JSON artifact directly rather than import finite_difference_options internals.
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.validation.black_scholes_parity import (  # noqa: E402
    export_public_black_scholes_fixture_json,
)


DEFAULT_FIXTURE_PATH = Path("tests/fixtures/arxiv_lab_bs_oracle_v1.json")


def main() -> None:
    """Regenerate the checked-in public-synthetic fixture JSON."""

    output = export_public_black_scholes_fixture_json(path=DEFAULT_FIXTURE_PATH)
    print(output)


if __name__ == "__main__":
    main()
