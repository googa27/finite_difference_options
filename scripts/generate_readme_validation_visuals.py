"""Generate README validation visuals from deterministic FD fixtures.

The figures intentionally use only checked-in public-synthetic fixtures and
executable validation runners. They do not ingest private Pinares data or invent
convergence rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from finite_difference_options.validation.black_scholes_parity import (
    run_public_black_scholes_parity_fixture,
)
from finite_difference_options.validation.pinares_fixed_price_proxy import (
    run_public_pinares_fixed_price_proxy_fixture,
)

BG = "#0d1117"
PANEL = "#161b22"
GRID = "#30363d"
TEXT = "#e6edf3"
MUTED = "#8b949e"
CYAN = "#2dd4ff"
MAGENTA = "#ff4db8"
GREEN = "#3fb950"
ORANGE = "#f0883e"
PURPLE = "#a371f7"


def _dark_pixel_ratio(path: Path) -> float:
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image, dtype=np.uint8)
    luminance = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    return float(np.mean(luminance < 64.0))


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.grid(True, color=GRID, alpha=0.45, linewidth=0.8)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.14, facecolor=BG)
    plt.close(fig)


def _plot_black_scholes(path: Path) -> dict[str, Any]:
    report = run_public_black_scholes_parity_fixture()
    rows = list(report.convergence_table())
    s_steps = np.array([row["s_steps"] for row in rows], dtype=float)
    errors = np.array([row["abs_error"] for row in rows], dtype=float)
    prices = np.array([row["price"] for row in rows], dtype=float)
    oracle = float(report.oracle_price)

    fig, (ax_price, ax_error) = plt.subplots(1, 2, figsize=(13.2, 5.8), facecolor=BG)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.23, top=0.78, wspace=0.18)
    fig.suptitle(
        "Finite-difference Black-Scholes validation fixture",
        color=TEXT,
        fontsize=15,
        fontweight="bold",
        y=0.95,
    )

    _style_axes(ax_price)
    ax_price.plot(s_steps, prices, color=PURPLE, marker="o", linewidth=2.4, label="FD numerical")
    ax_price.axhline(oracle, color=CYAN, linestyle="--", linewidth=2.2, label="analytical oracle")
    ax_price.set_xlabel("spot grid nodes")
    ax_price.set_ylabel("call value at S=K=1")
    ax_price.set_title("Value converges to analytical reference")
    ax_price.text(s_steps[-1], prices[-1], "  FD", color=PURPLE, va="center", fontsize=9)
    ax_price.text(s_steps[0], oracle, "  oracle", color=CYAN, va="bottom", fontsize=9)

    _style_axes(ax_error)
    ax_error.semilogy(s_steps, errors, color=MAGENTA, marker="o", linewidth=2.4)
    ax_error.axhline(report.case.tolerance, color=GREEN, linestyle="--", linewidth=2.0)
    ax_error.fill_between(s_steps, report.case.tolerance, errors.max() * 1.15, color=ORANGE, alpha=0.08)
    ax_error.set_xlabel("spot grid nodes")
    ax_error.set_ylabel("absolute price error")
    ax_error.set_title("Finest grid passes fixture tolerance")
    ax_error.text(s_steps[-1], errors[-1], f"  {errors[-1]:.2e}", color=MAGENTA, va="center", fontsize=9)
    ax_error.text(s_steps[0], report.case.tolerance, "  tolerance", color=GREEN, va="bottom", fontsize=9)

    caption = (
        "Source: run_public_black_scholes_parity_fixture(), "
        "fixture public-synthetic.black-scholes-call.v0; route fd.black_scholes_1d.crank_nicolson."
    )
    fig.text(0.02, 0.055, caption, color=MUTED, fontsize=7.5)
    _save(fig, path)

    return {
        "fixture_id": report.case.fixture_id,
        "route_id": report.case.route_id,
        "converged": report.converged,
        "oracle_price": oracle,
        "final_price": report.price,
        "final_abs_error": report.final_abs_error,
        "tolerance": report.case.tolerance,
        "delta_abs_error": report.errors["delta_abs"],
        "gamma_abs_error": report.errors["gamma_abs"],
        "rows": rows,
    }


def _plot_pinares(path: Path) -> dict[str, Any]:
    report = run_public_pinares_fixed_price_proxy_fixture()
    rows = list(report.convergence_table())
    s_steps = np.array([row["s_steps"] for row in rows], dtype=float)
    errors = np.array([row["abs_error_uf"] for row in rows], dtype=float)
    prices = np.array([row["price_uf"] for row in rows], dtype=float)
    oracle = float(report.oracle_price_uf)

    fig, (ax_price, ax_error) = plt.subplots(1, 2, figsize=(13.2, 5.8), facecolor=BG)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.23, top=0.78, wspace=0.18)
    fig.suptitle(
        "Pinares fixed-price proxy: public-synthetic FD evidence",
        color=TEXT,
        fontsize=15,
        fontweight="bold",
        y=0.95,
    )

    _style_axes(ax_price)
    ax_price.plot(s_steps, prices, color=PURPLE, marker="o", linewidth=2.4)
    ax_price.axhline(oracle, color=CYAN, linestyle="--", linewidth=2.2)
    ax_price.set_xlabel("spot grid nodes")
    ax_price.set_ylabel("survival-scaled value (UF)")
    ax_price.set_title("Proxy price against analytical survival-scaled oracle")
    ax_price.text(s_steps[-1], prices[-1], "  FD proxy", color=PURPLE, va="center", fontsize=9)
    ax_price.text(s_steps[0], oracle, "  oracle", color=CYAN, va="bottom", fontsize=9)

    _style_axes(ax_error)
    ax_error.semilogy(s_steps, errors, color=MAGENTA, marker="o", linewidth=2.4)
    ax_error.axhline(report.case.price_abs_tolerance_uf, color=GREEN, linestyle="--", linewidth=2.0)
    ax_error.fill_between(
        s_steps, report.case.price_abs_tolerance_uf, max(errors.max(), report.case.price_abs_tolerance_uf) * 1.25,
        color=ORANGE, alpha=0.08
    )
    ax_error.set_xlabel("spot grid nodes")
    ax_error.set_ylabel("absolute price error (UF)")
    ax_error.set_title("Finest grid inside 1.0 UF budget")
    ax_error.text(s_steps[-1], errors[-1], f"  {errors[-1]:.3f} UF", color=MAGENTA, va="center", fontsize=9)
    ax_error.text(s_steps[0], report.case.price_abs_tolerance_uf, "  budget", color=GREEN, va="bottom", fontsize=9)

    caption = (
        "Source: run_public_pinares_fixed_price_proxy_fixture(); public-synthetic Q* fixed-price proxy only. "
        "Full ROFR/family-contract/legal/tax/HJB routes intentionally fail closed."
    )
    fig.text(0.02, 0.055, caption, color=MUTED, fontsize=7.5)
    _save(fig, path)

    return {
        "fixture_id": report.case.fixture_id,
        "route_id": report.case.route_id,
        "converged": report.converged,
        "oracle_price_uf": oracle,
        "final_price_uf": report.price_uf,
        "final_abs_error_uf": report.final_abs_error_uf,
        "price_abs_tolerance_uf": report.case.price_abs_tolerance_uf,
        "delta_abs_error": report.errors["delta_abs"],
        "gamma_abs_error": report.errors["gamma_abs"],
        "rows": rows,
        "omission_note": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="docs/images")
    parser.add_argument(
        "--report",
        default="docs/images/readme_visual_provenance.json",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    assets = {
        "black_scholes": output_dir / "fd_black_scholes_convergence.png",
        "pinares_proxy": output_dir / "fd_pinares_proxy_convergence.png",
    }
    report = {
        "repository": "finite_difference_options",
        "generator": "scripts/generate_readme_validation_visuals.py",
        "palette": {
            "background": BG,
            "panel": PANEL,
            "analytical_reference": CYAN,
            "numerical_error": MAGENTA,
            "validated_pass": GREEN,
            "boundary_stability_caveat": ORANGE,
            "secondary_route": PURPLE,
        },
        "assets": {},
        "evidence": {
            "black_scholes": _plot_black_scholes(assets["black_scholes"]),
            "pinares_proxy": _plot_pinares(assets["pinares_proxy"]),
        },
    }
    for key, path in assets.items():
        image = Image.open(path)
        report["assets"][key] = {
            "path": str(path),
            "dpi": image.info.get("dpi"),
            "size_px": image.size,
            "dark_pixel_ratio": _dark_pixel_ratio(path),
            "visual_qa": "pass" if _dark_pixel_ratio(path) >= 0.65 else "review",
        }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"assets": {k: str(v) for k, v in assets.items()}, "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
