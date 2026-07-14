"""Typer-based command line interface for option pricing and plotting.

The CLI is intentionally small and primarily serves smoke-test and manual
exploration workflows. It wires together process construction, pricer execution,
optionally computes Greeks, and can render heatmap/surface plots.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import typer

from finite_difference_options.pricing import OptionPricer
from finite_difference_options.plotting.base import MatplotlibSeabornPlotter
from finite_difference_options.processes.affine import GeometricBrownianMotion
from finite_difference_options.instruments.base import EuropeanCall, EuropeanPut
from finite_difference_options.integrations.compiled_pde_adapter import (
    CompiledPDEAdapterError,
    load_compiled_pde_json,
    screen_compiled_pde_payload,
    solve_compiled_pde_payload,
)
from finite_difference_options.plotting.config_manager import PlottingConfigManager

app = typer.Typer(help="Command-line tools for option pricing.")
qps_app = typer.Typer(
    help="Screen and solve public-synthetic QuantProblemSpec/compiled PDE fixtures."
)
app.add_typer(qps_app, name="qps")
_QPS_PAYLOAD_ARGUMENT = typer.Argument(..., exists=True, dir_okay=False, readable=True)
_QPS_RESULT_OUT_OPTION = typer.Option(..., "--out", help="Destination for deterministic result JSON.")
_QPS_EVIDENCE_OUT_OPTION = typer.Option(..., "--evidence", help="Destination for deterministic evidence JSON.")


def _dump_json(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dump_json(payload), encoding="utf-8")


@qps_app.command("screen")
def qps_screen(
    payload: Path = _QPS_PAYLOAD_ARGUMENT,
    json_output: bool = typer.Option(False, "--json", help="Emit deterministic JSON."),
) -> None:
    """Screen a public-synthetic compiled PDE fixture before numerical work."""

    try:
        screen = screen_compiled_pde_payload(load_compiled_pde_json(payload))
    except CompiledPDEAdapterError as exc:
        error_payload = {
            "supported": False,
            "diagnostics": [asdict(item) for item in exc.diagnostics],
            "route": {},
        }
        typer.echo(_dump_json(error_payload) if json_output else "unsupported")
        raise typer.Exit(1) from exc
    screen_payload = screen.as_dict()
    typer.echo(
        _dump_json(screen_payload)
        if json_output
        else ("supported" if screen.supported else "unsupported")
    )
    if not screen.supported:
        raise typer.Exit(1)


@qps_app.command("solve")
def qps_solve(
    payload: Path = _QPS_PAYLOAD_ARGUMENT,
    out: Path = _QPS_RESULT_OUT_OPTION,
    evidence: Path = _QPS_EVIDENCE_OUT_OPTION,
) -> None:
    """Solve the exact public-synthetic compiled PDE fixture."""

    try:
        result = solve_compiled_pde_payload(load_compiled_pde_json(payload))
    except CompiledPDEAdapterError as exc:
        typer.echo(
            _dump_json(
                {
                    "status": "unsupported",
                    "diagnostics": [asdict(item) for item in exc.diagnostics],
                }
            )
        )
        raise typer.Exit(1) from exc
    result_payload = result.as_dict()
    _write_json(out, result_payload)
    _write_json(evidence, result.evidence)
    typer.echo(_dump_json(result_payload))


@app.command()
def price(
    option_type: str = typer.Option("Call", help="Option type: 'Call' or 'Put'."),
    strike: float = typer.Option(1.0, help="Strike price."),
    maturity: float = typer.Option(1.0, help="Time to maturity in years."),
    s0: float = typer.Option(1.0, help="Initial asset price."),
    rate: float = typer.Option(0.05, help="Risk-free interest rate."),
    sigma: float = typer.Option(0.2, help="Volatility of the underlying asset."),
    s_max: float = typer.Option(3.0, help="Maximum asset price in grid."),
    s_steps: int = typer.Option(100, help="Number of asset price steps."),
    t_steps: int = typer.Option(100, help="Number of time steps."),
    greeks: bool = typer.Option(False, help="Also compute Delta, Gamma and Theta."),
) -> None:
    """Compute Black--Scholes option price (and optional Greeks) on an asset grid.

    The command prints the interpolated spot price at ``s0``.

    Notes
    -----
    Greeks, when requested, are extracted from the model grid result using the
    same index as the price output.
    """
    model = GeometricBrownianMotion(mu=rate, sigma=sigma)
    option_cls = EuropeanCall if option_type == "Call" else EuropeanPut
    instrument = option_cls(strike=strike, maturity=maturity, model=model)
    pricer = OptionPricer(instrument=instrument)
    result = pricer.compute_grid(
        s_max=s_max,
        s_steps=s_steps,
        t_steps=t_steps,
        return_greeks=greeks,
    )
    s = result.s
    values = result.values
    s_idx = int(np.searchsorted(s, s0))
    price_at_s0 = float(values[-1, s_idx])
    typer.echo(f"Price: {price_at_s0}")
    if (
        greeks
        and result.delta is not None
        and result.gamma is not None
        and result.theta is not None
    ):
        typer.echo(f"Delta: {float(result.delta[-1, s_idx])}")
        typer.echo(f"Gamma: {float(result.gamma[-1, s_idx])}")
        typer.echo(f"Theta: {float(result.theta[-1, s_idx])}")


@app.command()
def plot(
    option_type: str = typer.Option("Call", help="Option type: 'Call' or 'Put'."),
    strike: float = typer.Option(1.0, help="Strike price."),
    maturity: float = typer.Option(1.0, help="Time to maturity in years."),
    rate: float = typer.Option(0.05, help="Risk-free interest rate."),
    sigma: float = typer.Option(0.2, help="Volatility of the underlying asset."),
    s_max: float = typer.Option(3.0, help="Maximum asset price in grid."),
    s_steps: int = typer.Option(100, help="Number of asset price steps."),
    t_steps: int = typer.Option(100, help="Number of time steps."),
    kind: str = typer.Option("heatmap", help="Plot type: 'heatmap' or 'surface'."),
    output: Optional[Path] = typer.Option(  # noqa: B008
        None, help="Optional path to save the plot."
    ),
) -> None:
    """Render option value grid as heatmap or 3D surface.

    Output target: if ``--output`` is given, saves image to the path and returns
    without interactive display.
    """
    model = GeometricBrownianMotion(mu=rate, sigma=sigma)
    option_cls = EuropeanCall if option_type == "Call" else EuropeanPut
    instrument = option_cls(strike=strike, maturity=maturity, model=model)
    pricer = OptionPricer(instrument=instrument)
    res = pricer.compute_grid(
        s_max=s_max,
        s_steps=s_steps,
        t_steps=t_steps,
    )
    plotter = MatplotlibSeabornPlotter()
    plot_config_manager = PlottingConfigManager()
    if kind == "surface":
        opts = plot_config_manager.surface()
        fig = plotter.surface(res.values, res.s, res.t, opts=opts)
    else:
        opts = plot_config_manager.heatmap()
        fig = plotter.heatmap(res.values, res.s, res.t, opts=opts)
    if output:
        fig.savefig(output)
    else:
        fig.show()


if __name__ == "__main__":
    app()
