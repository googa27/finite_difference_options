"""FastAPI service for option pricing and Greek calculations."""
from __future__ import annotations

from typing import Optional

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from src.option_pricer import OptionPricer

app = FastAPI(title="Finite Difference Option Pricing")


class OptionRequest(BaseModel):
    """Input parameters for option pricing."""

    option_type: str
    strike: float
    maturity: float
    s0: float
    rate: float
    sigma: float
    s_max: Optional[float] = None
    s_steps: int = 100
    t_steps: int = 100


class PriceResponse(BaseModel):
    """Option price at the initial asset price."""

    price: float


class GreeksResponse(BaseModel):
    """Greeks at the initial asset price."""

    delta: float
    gamma: float
    theta: float


@app.post("/price", response_model=PriceResponse)
def price(request: OptionRequest) -> PriceResponse:
    """Return option price for the given parameters."""
    s_max = request.s_max or request.s0 * 3
    pricer = OptionPricer(rate=request.rate, sigma=request.sigma)
    s, _, values = pricer.compute_grid(
        strike=request.strike,
        maturity=request.maturity,
        option_type=request.option_type,
        s_max=s_max,
        s_steps=request.s_steps,
        t_steps=request.t_steps,
    )
    s_idx = int(np.searchsorted(s, request.s0))
    return PriceResponse(price=float(values[-1, s_idx]))


@app.post("/greeks", response_model=GreeksResponse)
def greeks(request: OptionRequest) -> GreeksResponse:
    """Return Delta, Gamma and Theta for the given parameters."""
    s_max = request.s_max or request.s0 * 3
    pricer = OptionPricer(rate=request.rate, sigma=request.sigma)
    s, _, values, delta, gamma, theta = pricer.compute_grid(
        strike=request.strike,
        maturity=request.maturity,
        option_type=request.option_type,
        s_max=s_max,
        s_steps=request.s_steps,
        t_steps=request.t_steps,
        return_greeks=True,
    )
    s_idx = int(np.searchsorted(s, request.s0))
    return GreeksResponse(
        delta=float(delta[-1, s_idx]),
        gamma=float(gamma[-1, s_idx]),
        theta=float(theta[-1, s_idx]),
    )
