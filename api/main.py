"""FastAPI service for option pricing and Greek calculations."""

from __future__ import annotations

from typing import Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.option_pricer import OptionPricer
from src.risk import (
    Exposure,
    RiskFactor,
    Trade,
    calculate_basel,
    calculate_cuso,
    calculate_frtb,
    exposures_to_crif,
)

app = FastAPI(title="Finite Difference Option Pricing")

# Allow browser applications from the specified domain to access the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-domain.com"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    res = pricer.compute_grid(
        strike=request.strike,
        maturity=request.maturity,
        option_type=request.option_type,
        s_max=s_max,
        s_steps=request.s_steps,
        t_steps=request.t_steps,
    )
    s_idx = int(np.searchsorted(res.s, request.s0))
    return PriceResponse(price=float(res.values[-1, s_idx]))


@app.post("/greeks", response_model=GreeksResponse)
def greeks(request: OptionRequest) -> GreeksResponse:
    """Return Delta, Gamma and Theta for the given parameters."""
    s_max = request.s_max or request.s0 * 3
    pricer = OptionPricer(rate=request.rate, sigma=request.sigma)
    res = pricer.compute_grid(
        strike=request.strike,
        maturity=request.maturity,
        option_type=request.option_type,
        s_max=s_max,
        s_steps=request.s_steps,
        t_steps=request.t_steps,
        return_greeks=True,
    )
    s_idx = int(np.searchsorted(res.s, request.s0))
    return GreeksResponse(
        delta=float(res.delta[-1, s_idx]),
        gamma=float(res.gamma[-1, s_idx]),
        theta=float(res.theta[-1, s_idx]),
    )


class TradeModel(BaseModel):
    """Trade description used in regulatory reports."""

    trade_id: str
    product_type: str
    notional: float
    currency: str
    description: Optional[str] = None


class RiskFactorModel(BaseModel):
    """Risk factor affecting trades."""

    name: str
    value: float
    description: Optional[str] = None


class ExposureModel(BaseModel):
    """Exposure of a trade to a risk factor."""

    trade: TradeModel
    risk_factor: RiskFactorModel
    amount: float


class CRIFResponse(BaseModel):
    """CSV output representing the CRIF report."""

    crif: str


class PlaceholderResponse(BaseModel):
    """Generic placeholder response."""

    status: str


def _convert_exposures(models: list[ExposureModel]) -> list[Exposure]:
    """Convert API models to dataclass exposures."""

    return [
        Exposure(
            trade=Trade(**m.trade.dict()),
            risk_factor=RiskFactor(**m.risk_factor.dict()),
            amount=m.amount,
        )
        for m in models
    ]


@app.post("/reports/crif", response_model=CRIFResponse)
def crif_report(exposures: list[ExposureModel]) -> CRIFResponse:
    """Generate a CRIF report for the provided exposures."""

    crif = exposures_to_crif(_convert_exposures(exposures))
    return CRIFResponse(crif=crif)


@app.post("/reports/cuso", response_model=PlaceholderResponse)
def cuso_report(exposures: list[ExposureModel]) -> PlaceholderResponse:
    """Placeholder CUSO report endpoint."""

    data = calculate_cuso(_convert_exposures(exposures))
    return PlaceholderResponse(**data)


@app.post("/reports/basel", response_model=PlaceholderResponse)
def basel_report(exposures: list[ExposureModel]) -> PlaceholderResponse:
    """Placeholder Basel report endpoint."""

    data = calculate_basel(_convert_exposures(exposures))
    return PlaceholderResponse(**data)


@app.post("/reports/frtb", response_model=PlaceholderResponse)
def frtb_report(exposures: list[ExposureModel]) -> PlaceholderResponse:
    """Placeholder FRTB report endpoint."""

    data = calculate_frtb(_convert_exposures(exposures))
    return PlaceholderResponse(**data)
