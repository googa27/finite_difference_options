"""FastAPI service for option pricing and reporting examples.

Endpoints are intentionally minimal and are designed as reference API shapes for
front-end integrations and smoke testing.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.pricing import OptionPricer
from src.risk import (
    Exposure,
    NotImplementedForStandard,
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


class OptionType(str, Enum):
    """Supported vanilla option payoffs for the reference API."""

    CALL = "Call"
    PUT = "Put"


class OptionRequest(BaseModel):
    """Validated request payload for option pricing endpoints.

    Units follow model conventions:

    - rates in annual decimal units,
    - maturities in years,
    - ``spot`` is the requested market state,
    - ``s_steps`` and ``t_steps`` are integer grid resolutions.
    """

    model_config = ConfigDict(extra="forbid")

    option_type: OptionType = OptionType.CALL
    spot: float = Field(..., gt=0.0, allow_inf_nan=False)
    strike: float = Field(..., gt=0.0, allow_inf_nan=False)
    maturity: float = Field(..., gt=0.0, allow_inf_nan=False)
    rate: float = Field(..., ge=-1.0, le=1.0, allow_inf_nan=False)
    sigma: float = Field(..., gt=0.0, le=5.0, allow_inf_nan=False)
    s_max: Optional[float] = Field(default=None, gt=0.0, allow_inf_nan=False)
    s_steps: int = Field(default=100, ge=5, le=5_000)
    t_steps: int = Field(default=100, ge=3, le=5_000)
    include_full_grid: bool = False
    max_output_nodes: int = Field(default=50_000, ge=1, le=50_000)

    @model_validator(mode="after")
    def validate_api_budget_and_domain(self) -> "OptionRequest":
        """Reject impossible or unbounded requests before numerical work."""

        s_max = self.resolved_s_max
        if self.spot > s_max:
            raise ValueError("spot must lie inside the spatial grid [0, s_max]")
        if self.strike > s_max:
            raise ValueError("strike must lie inside the spatial grid [0, s_max]")
        node_count = self.s_steps * self.t_steps
        if node_count > self.max_output_nodes:
            raise ValueError(f"request exceeds node budget: {node_count} > {self.max_output_nodes}")
        return self

    @property
    def resolved_s_max(self) -> float:
        """Return explicit or default spatial upper bound."""

        return float(self.s_max if self.s_max is not None else max(self.spot, self.strike) * 3.0)


class PriceGrid(BaseModel):
    """Bounded grid payload returned only when explicitly requested."""

    s: list[float]
    t: list[float]
    values: list[list[float]]


class PriceResponse(BaseModel):
    """Scalar option price response with optional bounded grid payload."""

    schema_version: str = "fd-api-v1"
    option_type: OptionType
    spot: float
    price: float
    grid: PriceGrid | None = None


class GreeksResponse(BaseModel):
    """Scalar Greeks sampled at the explicitly requested spot."""

    schema_version: str = "fd-api-v1"
    option_type: OptionType
    spot: float
    delta: float
    gamma: float
    theta: float


class FullPDEResponse(BaseModel):
    """Explicitly requested complete solution and Greeks grids."""

    schema_version: str = "fd-api-v1"
    option_type: OptionType
    spot: float
    s: list[float]
    t: list[float]
    prices: list[list[float]]
    delta: list[list[float]]
    gamma: list[list[float]]
    theta: list[list[float]]


def _option_class(option_type: OptionType) -> type[EuropeanCall] | type[EuropeanPut]:
    return EuropeanCall if option_type == OptionType.CALL else EuropeanPut


def _compute_grid(request: OptionRequest, *, return_greeks: bool = False):
    model = GeometricBrownianMotion(mu=request.rate, sigma=request.sigma)
    instrument = _option_class(request.option_type)(
        strike=request.strike,
        maturity=request.maturity,
        model=model,
    )
    pricer = OptionPricer(instrument=instrument)
    return pricer.compute_grid(
        s_max=request.resolved_s_max,
        s_steps=request.s_steps,
        t_steps=request.t_steps,
        return_greeks=return_greeks,
    )


def _interp_at_spot(values: np.ndarray, grid: np.ndarray, spot: float) -> float:
    return float(np.interp(spot, grid, values))


def _grid_payload(res) -> PriceGrid:
    return PriceGrid(s=res.s.tolist(), t=res.t.tolist(), values=res.values.tolist())


@app.post("/price", response_model=PriceResponse)
def price(request: OptionRequest) -> PriceResponse:
    """Return requested-spot price and optional bounded grid for an option."""

    res = _compute_grid(request)
    return PriceResponse(
        option_type=request.option_type,
        spot=request.spot,
        price=_interp_at_spot(res.values[-1], res.s, request.spot),
        grid=_grid_payload(res) if request.include_full_grid else None,
    )


@app.post("/greeks", response_model=GreeksResponse)
def greeks(request: OptionRequest) -> GreeksResponse:
    """Return scalar Greeks at the explicitly requested spot."""

    res = _compute_grid(request, return_greeks=True)
    if res.delta is None or res.gamma is None or res.theta is None:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail="Greek grid was not computed")
    return GreeksResponse(
        option_type=request.option_type,
        spot=request.spot,
        delta=_interp_at_spot(res.delta[-1], res.s, request.spot),
        gamma=_interp_at_spot(res.gamma[-1], res.s, request.spot),
        theta=_interp_at_spot(res.theta[-1], res.s, request.spot),
    )


@app.post("/pde_solution", response_model=FullPDEResponse)
def pde_solution(request: OptionRequest) -> FullPDEResponse:
    """Return complete solution and Greeks grids only after explicit opt-in."""

    if not request.include_full_grid:
        raise HTTPException(
            status_code=400,
            detail="include_full_grid must be true to return the full PDE grid",
        )
    res = _compute_grid(request, return_greeks=True)
    if res.delta is None or res.gamma is None or res.theta is None:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail="Greek grid was not computed")
    return FullPDEResponse(
        option_type=request.option_type,
        spot=request.spot,
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


class RegulatoryNotImplementedResponse(BaseModel):
    """Machine-readable failure for disabled regulatory/reporting routes."""

    detail: dict


def _raise_regulatory_not_implemented(error: NotImplementedForStandard) -> None:
    """Convert internal regulatory failures to HTTP 501 problem details."""

    detail = error.to_problem_detail()
    raise HTTPException(status_code=detail["http_status"], detail=detail) from error


def _convert_exposures(models: list[ExposureModel]) -> list[Exposure]:
    """Convert API request payloads into internal report domain objects."""

    return [
        Exposure(
            trade=Trade(**m.trade.model_dump()),
            risk_factor=RiskFactor(**m.risk_factor.model_dump()),
            amount=m.amount,
        )
        for m in models
    ]


@app.post("/reports/crif", response_model=RegulatoryNotImplementedResponse)
def crif_report(exposures: list[ExposureModel]) -> None:
    """Fail closed until an exact ISDA CRIF profile and conformance suite exist."""

    strategy = ReportFactory.get_strategy("crif")
    try:
        strategy.generate_report(_convert_exposures(exposures))
    except NotImplementedForStandard as exc:
        _raise_regulatory_not_implemented(exc)


@app.post("/reports/cuso", response_model=RegulatoryNotImplementedResponse)
def cuso_report(exposures: list[ExposureModel]) -> None:
    """Fail closed until an authoritative CUSO specification exists."""

    strategy = ReportFactory.get_strategy("cuso")
    try:
        strategy.generate_report(_convert_exposures(exposures))
    except NotImplementedForStandard as exc:
        _raise_regulatory_not_implemented(exc)


@app.post("/reports/basel", response_model=RegulatoryNotImplementedResponse)
def basel_report(exposures: list[ExposureModel]) -> None:
    """Fail closed until a versioned Basel market-risk subset is implemented."""

    strategy = ReportFactory.get_strategy("basel")
    try:
        strategy.generate_report(_convert_exposures(exposures))
    except NotImplementedForStandard as exc:
        _raise_regulatory_not_implemented(exc)


@app.post("/reports/frtb", response_model=RegulatoryNotImplementedResponse)
def frtb_report(exposures: list[ExposureModel]) -> None:
    """Fail closed until a versioned FRTB calculation subset is implemented."""

    strategy = ReportFactory.get_strategy("frtb")
    try:
        strategy.generate_report(_convert_exposures(exposures))
    except NotImplementedForStandard as exc:
        _raise_regulatory_not_implemented(exc)