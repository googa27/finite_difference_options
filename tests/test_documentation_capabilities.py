"""Documentation maturity and README example safety tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.processes.affine import create_black_scholes_process, create_standard_heston
from src.pricing import create_log_grid, create_unified_european_call, create_unified_pricing_engine

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
CAPABILITY_MATRIX = ROOT / "docs" / "CAPABILITY_MATRIX.md"


def test_readme_has_live_links_and_no_placeholder_badges() -> None:
    text = README.read_text(encoding="utf-8")

    assert "PLACEHOLDER" not in text
    assert "github.com/googa27/finite_difference_options/actions/workflows/ci.yml" in text
    assert "docs/CAPABILITY_MATRIX.md" in text


def test_capability_matrix_is_authoritative_and_cites_evidence_ids() -> None:
    text = CAPABILITY_MATRIX.read_text(encoding="utf-8")

    assert "# FD capability and maturity matrix" in text
    for status in ["validated", "experimental", "unsupported"]:
        assert f"`{status}`" in text
    for evidence_id in [
        "BS-CALL-PARITY-V0",
        "RANNACHER-GAMMA-V0",
        "QPS-VANILLA-CALL-V0",
        "FACTOR-ROLE-COMPAT-V0",
    ]:
        assert evidence_id in text
    assert "Heston stochastic volatility" in text
    assert "basket option" in text
    assert "two-leg spreads" in text
    assert "unsupported" in text


def test_readme_black_scholes_quickstart_executes() -> None:
    process = create_black_scholes_process(mu=0.05, sigma=0.2)
    option = create_unified_european_call(strike=100.0, maturity=0.25)
    engine = create_unified_pricing_engine(process)
    grid = create_log_grid(s_min=50.0, s_max=150.0, n_points=101, center=100.0)
    times = np.linspace(0.0, option.maturity, 12)

    prices = engine.price_option(option, grid, time_grid=times)

    assert prices.shape == (len(times), len(grid))
    assert np.all(np.isfinite(prices))


def test_readme_heston_example_is_not_a_basket_proxy() -> None:
    text = README.read_text(encoding="utf-8")

    heston_section = text.split("### Heston stochastic-volatility smoke example", maxsplit=1)[1].split(
        "### Unsupported basket payoff route", maxsplit=1
    )[0]
    assert "create_unified_basket_call" not in heston_section

    process = create_standard_heston(r=0.03, kappa=1.8, theta=0.05, sigma=0.35, rho=-0.35)
    option = create_unified_european_call(strike=100.0, maturity=0.25)
    engine = create_unified_pricing_engine(process)
    s_grid = create_log_grid(40.0, 220.0, 17, center=100.0)
    v_grid = np.linspace(0.01, 0.30, 8)
    times = np.linspace(0.0, option.maturity, 10)

    prices = engine.price_option(option, s_grid, v_grid, time_grid=times)

    assert prices.shape == (len(times), len(s_grid), len(v_grid))
    assert np.all(np.isfinite(prices))
