"""Typer-based command line interface for option pricing and plotting."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import typer

from src.option_pricer import OptionPricer
from src.plotting.base import MatplotlibSeabornPlotter, PlotOptions
from src.models import GeometricBrownianMotion
from src.options import EuropeanCall, EuropeanPut

app = typer.Typer(help="Command-line tools for option pricing.")


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
    """Compute option price at ``s0`` and optionally Greeks."""
    model = GeometricBrownianMotion(rate=rate, sigma=sigma)
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
    if greeks and result.delta is not None and result.gamma is not None and result.theta is not None:
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
    kind: str = typer.Option(
        "heatmap", help="Plot type: 'heatmap' or 'surface'."
    ),
    output: Optional[Path] = typer.Option(  # noqa: B008
        None, help="Optional path to save the plot."),
) -> None:
    """Render option value grid as a heatmap or surface plot."""
    model = GeometricBrownianMotion(rate=rate, sigma=sigma)
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
