"""Typer-based command line interface for option pricing and plotting."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import typer

from src.option_pricer import OptionPricer
from src.plotter import MatplotlibSeabornPlotter

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
    pricer = OptionPricer(rate=rate, sigma=sigma)
    result = pricer.compute_grid(
        strike=strike,
        maturity=maturity,
        option_type=option_type,
        s_max=s_max,
        s_steps=s_steps,
        t_steps=t_steps,
        return_greeks=greeks,
    )
    s = result[0]
    values = result[2]
    s_idx = int(np.searchsorted(s, s0))
    price_at_s0 = float(values[-1, s_idx])
    typer.echo(f"Price: {price_at_s0}")
    if greeks:
        delta, gamma, theta = result[3], result[4], result[5]
        typer.echo(f"Delta: {float(delta[-1, s_idx])}")
        typer.echo(f"Gamma: {float(gamma[-1, s_idx])}")
        typer.echo(f"Theta: {float(theta[-1, s_idx])}")


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
    pricer = OptionPricer(rate=rate, sigma=sigma)
    s, t, values = pricer.compute_grid(
        strike=strike,
        maturity=maturity,
        option_type=option_type,
        s_max=s_max,
        s_steps=s_steps,
        t_steps=t_steps,
    )
    plotter = MatplotlibSeabornPlotter()
    fig = (
        plotter.surface(values, s, t)
        if kind == "surface"
        else plotter.heatmap(values, s, t)
    )
    if output:
        fig.savefig(output)
    else:
        fig.show()


if __name__ == "__main__":
    app()
