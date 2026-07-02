"""Regenerate the public-synthetic Pinares FD fixed-price proxy fixture."""

from __future__ import annotations

import json
from pathlib import Path

from finite_difference_options.validation.pinares_fixed_price_proxy import (
    export_public_pinares_fixed_price_proxy_fixture_json,
    public_pinares_fixed_price_problem_spec,
)

DEFAULT_RESULT_FIXTURE_PATH = Path("tests/fixtures/pinares_fd_fixed_price_proxy_v1.json")
DEFAULT_QPS_FIXTURE_PATH = Path("tests/fixtures/quant_problem_specs/pinares_fixed_price_proxy.json")


def main() -> None:
    """Regenerate checked-in public-synthetic Pinares fixture JSON files."""

    result_output = export_public_pinares_fixed_price_proxy_fixture_json(path=DEFAULT_RESULT_FIXTURE_PATH)
    DEFAULT_QPS_FIXTURE_PATH.write_text(
        json.dumps(public_pinares_fixed_price_problem_spec(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(result_output)
    print(DEFAULT_QPS_FIXTURE_PATH)


if __name__ == "__main__":
    main()
