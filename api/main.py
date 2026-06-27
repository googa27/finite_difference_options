"""FastAPI service for option pricing and reporting examples.

Endpoints are intentionally minimal and are designed as reference API shapes for
front-end integrations and smoke testing.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.pricing import OptionPricer
from src.risk import (
    Exposure,
    RiskFactor,
    Trade,
)
from src.risk.reporting_strategies import ReportFactory
from src.processes.affine import GeometricBrownianMotion
from src.instruments.base import EuropeanCall, EuropeanPut

app = FastAPI(title="Finite Difference Option Pricing")

# Allow browser applications from the specified domain to access the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class OptionRequest(BaseModel):
    """Request payload for option pricing endpoints.

    Units follow model conventions:

    - rates in annual decimal units,
    - maturities in years,
    - ``s_steps`` and ``t_steps`` are integer grid resolutions.
    """

    option_type: str = "Call"
    strike: float
    maturity: float
    rate: float
    sigma: float
    s_max: Optional[float] = None
    s_steps: int = 100
    t_steps: int = 100


class PriceResponse(BaseModel):
    """Response containing dense price grid for a requested option.

    The full surface is returned to simplify downstream plotting and debugging.
    """

    s: list[float]
    t: list[float]
    values: list[list[float]]


class GreeksResponse(BaseModel):
    """Scalar Greeks sampled at spot for backward-compatible API consumers."""

    delta: float
    gamma: float
    theta: float


class FullPDEResponse(BaseModel):
    """Full PDE grid and Greeks for visualisation.

    All arrays are in nested list form to keep FastAPI/JSON serialization simple.
    """
    
    s: list[float]
    t: list[float]
    prices: list[list[float]]
    delta: list[list[float]]
    gamma: list[list[float]]
    theta: list[list[float]]


@app.post("/price", response_model=PriceResponse)
def price(request: OptionRequest) -> PriceResponse:
    """Return full PDE-generated value grid for the requested option.

    The endpoint uses default grid settings when ``s_max`` is omitted.
    """
    s_max = request.s_max or request.strike * 3
    model = GeometricBrownianMotion(mu=request.rate, sigma=request.sigma)
    option_cls = EuropeanCall if request.option_type == "Call" else EuropeanPut
    instrument = option_cls(strike=request.strike, maturity=request.maturity, model=model)
    pricer = OptionPricer(instrument=instrument)
    res = pricer.compute_grid(
        s_max=s_max,
        s_steps=request.s_steps,
        t_steps=request.t_steps,
    )
    return PriceResponse(
        s=res.s.tolist(),
        t=res.t.tolist(),
        values=res.values.tolist(),
    )


@app.post("/greeks", response_model=GreeksResponse)
def greeks(request: OptionRequest) -> GreeksResponse:
    """Return scalar Greeks at spot for the requested option contract."""
    s_max = request.s_max or request.strike * 3
    model = GeometricBrownianMotion(mu=request.rate, sigma=request.sigma)
    option_cls = EuropeanCall if request.option_type == "Call" else EuropeanPut
    instrument = option_cls(strike=request.strike, maturity=request.maturity, model=model)
    pricer = OptionPricer(instrument=instrument)
    res = pricer.compute_grid(
        s_max=s_max,
        s_steps=request.s_steps,
        t_steps=request.t_steps,
        return_greeks=True,
    )
    s_idx = int(np.searchsorted(res.s, request.strike))
    return GreeksResponse(
        delta=float(res.delta[-1, s_idx]),
        gamma=float(res.gamma[-1, s_idx]),
        theta=float(res.theta[-1, s_idx]),
    )


@app.post("/pde_solution", response_model=FullPDEResponse)
def pde_solution(request: OptionRequest) -> FullPDEResponse:
    """Return complete solution and Greeks grids.

    This is the most visualization-friendly API endpoint for frontend charts.
    """
    s_max = request.s_max or request.strike * 3
    model = GeometricBrownianMotion(mu=request.rate, sigma=request.sigma)
    option_cls = EuropeanCall if request.option_type == "Call" else EuropeanPut
    instrument = option_cls(strike=request.strike, maturity=request.maturity, model=model)
    pricer = OptionPricer(instrument=instrument)
    res = pricer.compute_grid(
        s_max=s_max,
        s_steps=request.s_steps,
        t_steps=request.t_steps,
        return_greeks=True,
    )
    return FullPDEResponse(
        s=res.s.tolist(),
        t=res.t.tolist(),
        prices=res.values.tolist(),
        delta=res.delta.tolist(),
        gamma=res.gamma.tolist(),
        theta=res.theta.tolist(),
    )


class TradeModel(BaseModel):
    """Trade description used by the lightweight reporting endpoints."""

    trade_id: str
    product_type: str
    notional: float
    currency: str
    description: Optional[str] = None


class RiskFactorModel(BaseModel):
    """Single risk factor used to drive report exposures."""

    name: str
    value: float
    description: Optional[str] = None


class ExposureModel(BaseModel):
    """Association of a trade and risk factor with an exposure amount."""

    trade: TradeModel
    risk_factor: RiskFactorModel
    amount: float


class CRIFResponse(BaseModel):
    """String payload for CRIF report output."""

    crif: str


class PlaceholderResponse(BaseModel):
    """Generic response wrapper for non-production reporting stubs."""

    status: str


def _convert_exposures(models: list[ExposureModel]) -> list[Exposure]:
    """Convert API request payloads into internal report domain objects."""

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
    """Generate a CRIF report for the provided exposures.

    Inputs are converted from API models to internal domain objects before
    invoking the configured report strategy.
    """

    strategy = ReportFactory.get_strategy("crif")
    crif = strategy.generate_report(_convert_exposures(exposures))
    return CRIFResponse(**crif)


@app.post("/reports/cuso", response_model=PlaceholderResponse)
def cuso_report(exposures: list[ExposureModel]) -> PlaceholderResponse:
    """Return placeholder CUSO report from the active strategy adapter."""

    strategy = ReportFactory.get_strategy("cuso")
    data = strategy.generate_report(_convert_exposures(exposures))
    return PlaceholderResponse(**data)


@app.post("/reports/basel", response_model=PlaceholderResponse)
def basel_report(exposures: list[ExposureModel]) -> PlaceholderResponse:
    """Return placeholder Basel report from the active strategy adapter."""

    strategy = ReportFactory.get_strategy("basel")
    data = strategy.generate_report(_convert_exposures(exposures))
    return PlaceholderResponse(**data)


@app.post("/reports/frtb", response_model=PlaceholderResponse)
def frtb_report(exposures: list[ExposureModel]) -> PlaceholderResponse:
    """Placeholder FRTB report endpoint."""

    strategy = ReportFactory.get_strategy("frtb")
    data = strategy.generate_report(_convert_exposures(exposures))
    return PlaceholderResponse(**data)