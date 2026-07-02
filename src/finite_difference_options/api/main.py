"""FastAPI service for option pricing and reporting examples.

Endpoints are intentionally minimal and are designed as reference API shapes for
front-end integrations and smoke testing.
"""

from __future__ import annotations

import os
import time
from enum import Enum
from hmac import compare_digest
from threading import BoundedSemaphore, Lock
from time import monotonic as _monotonic
from typing import Any, Literal, Optional
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from finite_difference_options.instruments.base import EuropeanCall, EuropeanPut
from finite_difference_options.pricing import OptionPricer
from finite_difference_options.processes.affine import GeometricBrownianMotion
from finite_difference_options.risk import (
    Exposure,
    NotImplementedForStandard,
    RiskFactor,
    Trade,
)
from finite_difference_options.risk.reporting_strategies import ReportFactory

API_SCHEMA_VERSION: Literal["fd-api-v1"] = "fd-api-v1"

app = FastAPI(title="Finite Difference Option Pricing")


LOCAL_DEV_CORS_ORIGINS = ("http://localhost:5173",)


def _env_bool(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be a boolean string: one of 1/0, true/false, yes/no, on/off"
    )


def _env_int(name: str, *, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _env_float(name: str, *, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _env_csv(name: str) -> tuple[str, ...]:
    raw = os.environ.get(name, "")
    return tuple(item.strip() for item in raw.split(",") if item.strip())


class DeploymentSecurityPolicy(BaseModel):
    """Restrictive default deployment posture for the experimental API service."""

    model_config = ConfigDict(frozen=True)

    policy_name: str = "local_demo_security"
    allowed_origins: tuple[str, ...] = ()
    local_dev_cors: bool = False
    auth_required: bool = False
    api_key: Optional[str] = Field(default=None, repr=False, exclude=True)
    rate_limit_requests: int = Field(default=120, ge=0)
    rate_limit_window_seconds: float = Field(default=60.0, gt=0.0)
    protected_path_prefixes: tuple[str, ...] = (
        "/price",
        "/greeks",
        "/pde_solution",
        "/reports/",
    )

    @classmethod
    def from_env(cls) -> "DeploymentSecurityPolicy":
        """Build policy from environment without requiring secrets in tests."""

        return cls(
            allowed_origins=_env_csv("FDO_API_CORS_ORIGINS"),
            local_dev_cors=_env_bool("FDO_API_ENABLE_LOCAL_DEV_CORS"),
            auth_required=_env_bool("FDO_API_AUTH_REQUIRED"),
            api_key=os.environ.get("FDO_API_KEY") or None,
            rate_limit_requests=_env_int("FDO_API_RATE_LIMIT_REQUESTS", default=120),
            rate_limit_window_seconds=_env_float(
                "FDO_API_RATE_LIMIT_WINDOW_SECONDS", default=60.0
            ),
        )


DEFAULT_API_SECURITY_POLICY = DeploymentSecurityPolicy.from_env()


def _cors_origins(policy: DeploymentSecurityPolicy) -> list[str]:
    origins = set(policy.allowed_origins)
    if policy.local_dev_cors:
        origins.update(LOCAL_DEV_CORS_ORIGINS)
    return sorted(origins)


def _configure_cors(fastapi_app: FastAPI, policy: DeploymentSecurityPolicy) -> None:
    """Install CORS middleware with restrictive defaults and opt-in local dev."""

    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(policy),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )


class OptionType(str, Enum):
    """Supported vanilla option payoffs for the reference API."""

    CALL = "Call"
    PUT = "Put"


class PricingModel(str, Enum):
    """Recognized pricing model contracts at the API boundary."""

    BLACK_SCHOLES = "black_scholes"
    HESTON = "heston"
    LOCAL_VOLATILITY = "local_volatility"


class ProcessType(str, Enum):
    """Recognized stochastic-process contracts at the API boundary."""

    GEOMETRIC_BROWNIAN_MOTION = "geometric_brownian_motion"
    HESTON = "heston"
    SABR = "sabr"


class PayoffFamily(str, Enum):
    """Recognized payoff families at the API boundary."""

    VANILLA_EUROPEAN = "vanilla_european"
    BASKET = "basket"
    SPREAD = "spread"
    DIGITAL = "digital"


class ExerciseStyle(str, Enum):
    """Recognized exercise styles at the API boundary."""

    EUROPEAN = "european"
    AMERICAN = "american"


class FactorRole(str, Enum):
    """Recognized state-factor roles for payoff compatibility checks."""

    TRADABLE_SPOT = "tradable_spot"
    VARIANCE = "variance"
    VOLATILITY = "volatility"
    SHORT_RATE = "short_rate"
    AUXILIARY_STATE = "auxiliary_state"


class APIResourcePolicy(BaseModel):
    """Deployable local/demo runtime safety policy for numerical API routes."""

    policy_name: str = "local_demo"
    max_state_dimensions: int = 1
    max_s_steps: int = 5_000
    max_t_steps: int = 5_000
    max_compute_nodes: int = 50_000
    max_output_nodes: int = 50_000
    max_response_bytes: int = 8_000_000
    bytes_per_float: int = 8
    default_timeout_seconds: float = 2.0
    max_timeout_seconds: float = 5.0
    max_concurrent_solves: int = 1


DEFAULT_API_RESOURCE_POLICY = APIResourcePolicy()
_SOLVE_SEMAPHORE = BoundedSemaphore(DEFAULT_API_RESOURCE_POLICY.max_concurrent_solves)


class ResourceBudgetMetadata(BaseModel):
    """Budget metadata attached to API responses and pre-solve checks."""

    policy_name: str
    state_dimensions: int
    compute_nodes: int
    max_compute_nodes: int
    output_nodes: int
    max_output_nodes: int
    estimated_response_bytes: int
    max_response_bytes: int
    timeout_seconds: float
    max_concurrent_solves: int


class OptionRequest(BaseModel):
    """Validated request payload for option pricing endpoints.

    Units follow model conventions:

    - rates in annual decimal units,
    - maturities in years,
    - ``spot`` is the requested market state,
    - ``s_steps`` and ``t_steps`` are integer grid resolutions.
    """

    model_config = ConfigDict(extra="forbid")

    model: PricingModel = PricingModel.BLACK_SCHOLES
    process: ProcessType = ProcessType.GEOMETRIC_BROWNIAN_MOTION
    payoff_family: PayoffFamily = PayoffFamily.VANILLA_EUROPEAN
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN
    underlying_factor_role: FactorRole = FactorRole.TRADABLE_SPOT
    state_dimensions: int = Field(default=1, ge=1, le=8)
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
    max_response_bytes: int = Field(default=8_000_000, ge=1, le=8_000_000)
    timeout_seconds: Optional[float] = Field(default=None, gt=0.0, le=5.0)
    correlation: Optional[float] = Field(
        default=None, ge=-1.0, le=1.0, allow_inf_nan=False
    )
    variance: Optional[float] = Field(default=None, ge=0.0, allow_inf_nan=False)
    long_run_variance: Optional[float] = Field(
        default=None, ge=0.0, allow_inf_nan=False
    )
    mean_reversion: Optional[float] = Field(default=None, gt=0.0, allow_inf_nan=False)
    vol_of_vol: Optional[float] = Field(default=None, ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_api_budget_domain_and_model_fields(self) -> "OptionRequest":
        """Reject impossible, unbounded, or mismatched requests before numerical work."""

        policy = DEFAULT_API_RESOURCE_POLICY
        s_max = self.resolved_s_max
        if self.spot > s_max:
            raise ValueError("spot must lie inside the spatial grid [0, s_max]")
        if self.strike > s_max:
            raise ValueError("strike must lie inside the spatial grid [0, s_max]")
        if self.state_dimensions > policy.max_state_dimensions:
            raise ValueError(
                "state dimension budget exceeded: "
                f"{self.state_dimensions} > {policy.max_state_dimensions}"
            )
        if self.s_steps > policy.max_s_steps:
            raise ValueError(
                f"spatial step budget exceeded: {self.s_steps} > {policy.max_s_steps}"
            )
        if self.t_steps > policy.max_t_steps:
            raise ValueError(
                f"time step budget exceeded: {self.t_steps} > {policy.max_t_steps}"
            )
        node_count = self.state_dimensions * self.s_steps * self.t_steps
        node_budget = policy.max_compute_nodes
        if node_count > node_budget:
            raise ValueError(
                f"request exceeds node budget: {node_count} > {node_budget}"
            )
        if self.model == PricingModel.BLACK_SCHOLES:
            extra_model_fields = [
                name
                for name in (
                    "correlation",
                    "variance",
                    "long_run_variance",
                    "mean_reversion",
                    "vol_of_vol",
                )
                if getattr(self, name) is not None
            ]
            if extra_model_fields:
                raise ValueError(
                    "black_scholes route does not accept model-specific fields: "
                    + ", ".join(extra_model_fields)
                )
        return self

    @property
    def resolved_s_max(self) -> float:
        """Return explicit or default spatial upper bound."""

        return float(
            self.s_max if self.s_max is not None else max(self.spot, self.strike) * 3.0
        )


class RouteWarning(BaseModel):
    """Machine-readable non-blocking route warning."""

    code: str
    message: str
    severity: Literal["info", "warning"] = "warning"


class UnitMetadata(BaseModel):
    """Unit conventions for the public API schema."""

    price: str = "same currency as underlying; examples use dimensionless inputs"
    spot: str = "underlying price level"
    strike: str = "underlying price level"
    rate: str = "annual_decimal"
    volatility: str = "annual_decimal"
    time: str = "years"
    greek_delta: str = "price per spot unit"
    greek_gamma: str = "price per squared spot unit"
    greek_theta: str = "price per year"


class SolverMetadata(BaseModel):
    """Numerical route metadata exposed without claiming production maturity."""

    engine: str
    model: str
    process: str
    payoff_family: str
    exercise_style: str
    underlying_factor_role: str
    scheme: str
    s_steps: int
    t_steps: int
    grid_nodes: int
    output_grid_included: bool
    requested_outputs: list[str]


class ConvergenceDiagnostics(BaseModel):
    """Schema placeholder for convergence evidence attached to API results."""

    status: Literal["not_assessed", "not_applicable"] = "not_assessed"
    message: str
    residual_norm: float | None = None
    benchmark_error: float | None = None


class SamplingDiagnostics(BaseModel):
    """Requested-state sampling metadata for scalar API outputs."""

    requested_spot: float
    reference_grid: Literal["asset_price"] = "asset_price"
    method: Literal["grid_node", "linear_interpolation"]
    lower_index: int
    upper_index: int
    lower_spot: float
    upper_spot: float
    interpolation_weight: float
    bounded_policy: Literal["reject_outside_grid"] = "reject_outside_grid"
    extrapolated: bool = False


class GridSample(BaseModel):
    """Internal value plus diagnostics for requested-state grid sampling."""

    value: float
    sampling: SamplingDiagnostics


class ResponseMetadata(BaseModel):
    """Stable metadata envelope shared by success and error responses."""

    route: str
    route_maturity: Literal["experimental", "scaffold", "unsupported"]
    warnings: list[RouteWarning]
    units: UnitMetadata = Field(default_factory=UnitMetadata)
    solver: SolverMetadata | None = None
    convergence: ConvergenceDiagnostics | None = None
    sampling: SamplingDiagnostics | None = None
    resource_budget: ResourceBudgetMetadata | None = None


class APIResponseBase(BaseModel):
    """Common top-level stable fields for every public API response."""

    schema_version: Literal["fd-api-v1"] = API_SCHEMA_VERSION
    request_id: str
    run_id: str
    metadata: ResponseMetadata


class ErrorBody(BaseModel):
    """Machine-readable error summary."""

    code: str
    message: str
    http_status: int
    route: str
    source: str = "api"


class ErrorResponse(APIResponseBase):
    """Stable error envelope for validation, bad-request, and unsupported routes."""

    error: ErrorBody
    detail: dict[str, Any] | list[dict[str, Any]] | str | None = None


class PriceGrid(BaseModel):
    """Bounded grid payload returned only when explicitly requested."""

    s: list[float]
    t: list[float]
    values: list[list[float]]


class PriceResponse(APIResponseBase):
    """Scalar option price response with optional bounded grid payload."""

    option_type: OptionType
    spot: float
    price: float
    grid: PriceGrid | None = None


class GreeksResponse(APIResponseBase):
    """Scalar Greeks sampled at the explicitly requested spot."""

    option_type: OptionType
    spot: float
    delta: float
    gamma: float
    theta: float


class FullPDEResponse(APIResponseBase):
    """Explicitly requested complete solution and Greeks grids."""

    option_type: OptionType
    spot: float
    s: list[float]
    t: list[float]
    prices: list[list[float]]
    delta: list[list[float]]
    gamma: list[list[float]]
    theta: list[list[float]]


API_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorResponse},
    401: {"model": ErrorResponse},
    403: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    429: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
    501: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
    504: {"model": ErrorResponse},
}
UNSUPPORTED_ROUTE_RESPONSES: dict[int | str, dict[str, Any]] = {
    401: {"model": ErrorResponse},
    403: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    429: {"model": ErrorResponse},
    501: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
}


def _new_run_id() -> str:
    """Return an opaque run identifier for a single route evaluation."""

    return f"fd-run-{uuid4().hex}"


def _request_id(http_request: Request) -> str:
    """Propagate a caller-supplied request ID or create an opaque one."""

    return http_request.headers.get("x-request-id") or f"fd-req-{uuid4().hex}"


def _route_warning(route: str, *, route_maturity: str) -> RouteWarning:
    """Return the standard maturity warning for a public route."""

    if route_maturity == "unsupported":
        return RouteWarning(
            code="unsupported_route",
            message=f"{route} is intentionally fail-closed until its contract is implemented.",
        )
    return RouteWarning(
        code="experimental_api",
        message=(
            f"{route} is an experimental finite-difference service contract; "
            "results require independent numerical validation before production use."
        ),
    )


def _solver_metadata(
    request: OptionRequest,
    *,
    include_grid: bool,
    requested_outputs: list[str],
) -> SolverMetadata:
    """Build stable metadata for the current finite-difference route."""

    return SolverMetadata(
        engine="OptionPricer",
        model=request.model.value,
        process=request.process.value,
        payoff_family=request.payoff_family.value,
        exercise_style=request.exercise_style.value,
        underlying_factor_role=request.underlying_factor_role.value,
        scheme="finite_difference_black_scholes_reference",
        s_steps=request.s_steps,
        t_steps=request.t_steps,
        grid_nodes=request.s_steps * request.t_steps,
        output_grid_included=include_grid,
        requested_outputs=requested_outputs,
    )


def _convergence_metadata() -> ConvergenceDiagnostics:
    """Return explicit non-claim convergence metadata for experimental routes."""

    return ConvergenceDiagnostics(
        status="not_assessed",
        message=(
            "This endpoint returns a single-grid experimental solve; convergence, "
            "oracle parity, and production model-risk evidence are tracked by separate gates."
        ),
    )


def _success_metadata(
    route: str,
    request: OptionRequest,
    *,
    include_grid: bool,
    requested_outputs: list[str],
    sampling: SamplingDiagnostics | None = None,
    resource_budget: ResourceBudgetMetadata | None = None,
) -> ResponseMetadata:
    """Build the success metadata envelope required by the v1 API schema."""

    return ResponseMetadata(
        route=route,
        route_maturity="experimental",
        warnings=[_route_warning(route, route_maturity="experimental")],
        solver=_solver_metadata(
            request,
            include_grid=include_grid,
            requested_outputs=requested_outputs,
        ),
        convergence=_convergence_metadata(),
        sampling=sampling,
        resource_budget=resource_budget,
    )


def _error_metadata(route: str, *, code: str) -> ResponseMetadata:
    """Build the error metadata envelope without invoking numerical work."""

    route_maturity: Literal["experimental", "unsupported"] = (
        "unsupported" if code == "unsupported_route" else "experimental"
    )
    return ResponseMetadata(
        route=route,
        route_maturity=route_maturity,
        warnings=[_route_warning(route, route_maturity=route_maturity)],
        convergence=ConvergenceDiagnostics(
            status="not_applicable",
            message="No numerical solve was executed for this error response.",
        ),
    )


def _response_identity(http_request: Request) -> dict[str, str]:
    """Return common response identifiers."""

    return {"request_id": _request_id(http_request), "run_id": _new_run_id()}


def _error_code(status_code: int, detail: Any) -> str:
    """Classify HTTP failures into stable public error codes."""

    if isinstance(detail, dict) and isinstance(detail.get("code"), str):
        return str(detail["code"])
    if status_code == 501:
        return "unsupported_route"
    if isinstance(detail, dict) and detail.get("capability_status") in {
        "scaffold",
        "unsupported",
    }:
        return "unsupported_route"
    if status_code == 422:
        return "validation_error"
    if status_code == 400:
        return "bad_request"
    if status_code >= 500:
        return "internal_error"
    return "http_error"


def _error_message(code: str, detail: Any) -> str:
    """Return a concise human-readable message for the stable error envelope."""

    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        for key in ("message", "detail", "reason"):
            if isinstance(detail.get(key), str):
                return detail[key]
    if code == "validation_error":
        return "Request validation failed"
    if code == "unsupported_route":
        return "Route is unsupported or not implemented for the declared contract"
    if code == "internal_error":
        return "Internal server error"
    return "Request failed"


def _error_response(
    http_request: Request,
    *,
    status_code: int,
    detail: Any,
    source: str,
) -> JSONResponse:
    """Serialize all public API errors through the stable v1 envelope."""

    route = http_request.url.path
    code = _error_code(status_code, detail)
    body = ErrorResponse(
        **_response_identity(http_request),
        metadata=_error_metadata(route, code=code),
        error=ErrorBody(
            code=code,
            message=_error_message(code, detail),
            http_status=status_code,
            route=route,
            source=source,
        ),
        detail=detail,
    )
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


def _validation_error_details(exc: RequestValidationError) -> list[dict[str, Any]]:
    """Return validation errors without non-JSON-serializable exception objects."""

    details: list[dict[str, Any]] = []
    for error in exc.errors():
        clean_error = {key: value for key, value in error.items() if key != "ctx"}
        ctx = error.get("ctx")
        if isinstance(ctx, dict):
            clean_context = {key: str(value) for key, value in ctx.items()}
            if clean_context:
                clean_error["ctx"] = clean_context
        details.append(clean_error)
    return details


@app.exception_handler(RequestValidationError)
def _validation_exception_handler(
    http_request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return Pydantic/FastAPI validation failures as stable v1 error responses."""

    return _error_response(
        http_request,
        status_code=422,
        detail=_validation_error_details(exc),
        source="request_validation",
    )


@app.exception_handler(HTTPException)
def _http_exception_handler(http_request: Request, exc: HTTPException) -> JSONResponse:
    """Return explicit HTTP failures as stable v1 error responses."""

    return _error_response(
        http_request,
        status_code=exc.status_code,
        detail=exc.detail,
        source="http_exception",
    )


class _ProcessLocalRateLimiter:
    """Small in-process fixed-window limiter for local/demo API deployments."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._hits: dict[tuple[str, str], list[float]] = {}

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()

    def check(
        self,
        key: tuple[str, str],
        *,
        max_requests: int,
        window_seconds: float,
    ) -> tuple[bool, int, float]:
        if max_requests <= 0:
            return True, 0, 0.0
        now = time.monotonic()
        cutoff = now - window_seconds
        with self._lock:
            hits = [hit for hit in self._hits.get(key, []) if hit > cutoff]
            if len(hits) >= max_requests:
                retry_after = max(0.0, window_seconds - (now - hits[0]))
                self._hits[key] = hits
                return False, len(hits), retry_after
            hits.append(now)
            self._hits[key] = hits
            return True, len(hits), 0.0


_RATE_LIMITER = _ProcessLocalRateLimiter()


def _is_protected_path(path: str, policy: DeploymentSecurityPolicy) -> bool:
    for prefix in policy.protected_path_prefixes:
        if prefix.endswith("/"):
            if path.startswith(prefix):
                return True
        elif path == prefix or path.startswith(f"{prefix}/"):
            return True
    return False


def _extract_api_key(http_request: Request) -> str | None:
    authorization = http_request.headers.get("authorization")
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token.strip():
            return token.strip()
    api_key = http_request.headers.get("x-api-key")
    return api_key.strip() if api_key and api_key.strip() else None


def _constant_time_key_equal(presented_key: str, configured_key: str) -> bool:
    """Compare API keys without leaking timing and without ASCII-only TypeError."""

    return compare_digest(
        presented_key.encode("utf-8"),
        configured_key.encode("utf-8"),
    )


def _auth_failure(
    http_request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
) -> JSONResponse:
    return _error_response(
        http_request,
        status_code=status_code,
        detail={
            "code": code,
            "message": message,
            "capability_status": "deployment_hardening",
        },
        source="deployment_security",
    )


def _enforce_authentication(
    http_request: Request, policy: DeploymentSecurityPolicy
) -> JSONResponse | None:
    if not policy.auth_required:
        return None
    if not policy.api_key:
        return _auth_failure(
            http_request,
            status_code=503,
            code="auth_misconfigured",
            message="API authentication is required but no API key is configured",
        )
    presented_key = _extract_api_key(http_request)
    if presented_key is None:
        return _auth_failure(
            http_request,
            status_code=401,
            code="auth_required",
            message="API authentication is required for this route",
        )
    if not _constant_time_key_equal(presented_key, policy.api_key):
        return _auth_failure(
            http_request,
            status_code=403,
            code="auth_invalid",
            message="API authentication credentials were rejected",
        )
    return None


def _rate_limit_key(http_request: Request) -> tuple[str, str]:
    client_host = http_request.client.host if http_request.client else "unknown"
    return (http_request.url.path, client_host)


def _enforce_rate_limit(
    http_request: Request, policy: DeploymentSecurityPolicy
) -> JSONResponse | None:
    allowed, observed, retry_after = _RATE_LIMITER.check(
        _rate_limit_key(http_request),
        max_requests=policy.rate_limit_requests,
        window_seconds=policy.rate_limit_window_seconds,
    )
    if allowed:
        return None
    return _error_response(
        http_request,
        status_code=429,
        detail={
            "code": "rate_limit_exceeded",
            "message": "process-local API rate limit exceeded",
            "route": http_request.url.path,
            "resource": {
                "limit": "rate_limit_requests",
                "observed": observed,
                "allowed": policy.rate_limit_requests,
                "window_seconds": policy.rate_limit_window_seconds,
                "retry_after_seconds": retry_after,
                "scope": "process_local_demo",
            },
        },
        source="deployment_security",
    )


async def _apply_deployment_security(http_request: Request, call_next):
    """Apply fail-closed auth and process-local rate limits to API routes."""

    if http_request.method.upper() == "OPTIONS":
        return await call_next(http_request)
    policy = DEFAULT_API_SECURITY_POLICY
    if not _is_protected_path(http_request.url.path, policy):
        return await call_next(http_request)
    rate_response = _enforce_rate_limit(http_request, policy)
    if rate_response is not None:
        return rate_response
    auth_response = _enforce_authentication(http_request, policy)
    if auth_response is not None:
        return auth_response
    return await call_next(http_request)


@app.middleware("http")
async def _deployment_security_middleware(http_request: Request, call_next):
    return await _apply_deployment_security(http_request, call_next)


# Register CORS after the security middleware so it remains outermost and adds
# browser-readable headers to auth/rate-limit error envelopes for allowed origins.
_configure_cors(app, DEFAULT_API_SECURITY_POLICY)


@app.exception_handler(Exception)
def _unhandled_exception_handler(http_request: Request, exc: Exception) -> JSONResponse:
    """Return unexpected failures as redacted stable v1 error responses."""

    return _error_response(
        http_request,
        status_code=500,
        detail={"exception_type": type(exc).__name__},
        source="unhandled_exception",
    )


def _option_class(option_type: OptionType) -> type[EuropeanCall] | type[EuropeanPut]:
    return EuropeanCall if option_type == OptionType.CALL else EuropeanPut


def _contract_dict(request: OptionRequest) -> dict[str, str]:
    """Return the model/payoff/process contract fields as JSON-safe strings."""

    return {
        "model": request.model.value,
        "process": request.process.value,
        "payoff_family": request.payoff_family.value,
        "exercise_style": request.exercise_style.value,
        "underlying_factor_role": request.underlying_factor_role.value,
    }


def _supported_contract() -> dict[str, str]:
    """Return the only numerical route contract currently implemented."""

    return {
        "model": PricingModel.BLACK_SCHOLES.value,
        "process": ProcessType.GEOMETRIC_BROWNIAN_MOTION.value,
        "payoff_family": PayoffFamily.VANILLA_EUROPEAN.value,
        "exercise_style": ExerciseStyle.EUROPEAN.value,
        "underlying_factor_role": FactorRole.TRADABLE_SPOT.value,
    }


def _unsupported_contract_reason(request: OptionRequest) -> str | None:
    """Explain why a recognized contract is unsupported, or None if supported."""

    if request.model != PricingModel.BLACK_SCHOLES:
        return f"model {request.model.value} is not enabled"
    if request.process != ProcessType.GEOMETRIC_BROWNIAN_MOTION:
        return (
            f"process {request.process.value} is incompatible with model black_scholes"
        )
    if request.payoff_family != PayoffFamily.VANILLA_EUROPEAN:
        return f"payoff family {request.payoff_family.value} is incompatible with model black_scholes"
    if request.exercise_style != ExerciseStyle.EUROPEAN:
        return f"exercise style {request.exercise_style.value} is incompatible with model black_scholes"
    if request.underlying_factor_role != FactorRole.TRADABLE_SPOT:
        return (
            f"factor role {request.underlying_factor_role.value} is incompatible "
            "with one-dimensional tradable-spot Black-Scholes"
        )
    return None


def _ensure_supported_contract(request: OptionRequest, *, route: str) -> None:
    """Fail closed for recognized but currently unimplemented route contracts."""

    reason = _unsupported_contract_reason(request)
    if reason is None:
        return
    raise HTTPException(
        status_code=501,
        detail={
            "capability_status": "unsupported",
            "reason": reason,
            "route": route,
            "requested_contract": _contract_dict(request),
            "supported_contract": _supported_contract(),
        },
    )


def _timeout_seconds(request: OptionRequest) -> float:
    """Return request timeout clamped by the local API resource policy."""

    policy = DEFAULT_API_RESOURCE_POLICY
    requested = request.timeout_seconds or policy.default_timeout_seconds
    return min(float(requested), policy.max_timeout_seconds)


def _output_nodes(route: str, request: OptionRequest) -> int:
    """Estimate serialized numerical output nodes for a route before solving."""

    grid_nodes = request.s_steps * request.t_steps
    if route == "/price":
        return (
            1 + request.s_steps + request.t_steps + grid_nodes
            if request.include_full_grid
            else 1
        )
    if route == "/greeks":
        return 3
    if route == "/pde_solution":
        return (
            request.s_steps + request.t_steps + 4 * grid_nodes
            if request.include_full_grid
            else 0
        )
    return 0


def _resource_budget(route: str, request: OptionRequest) -> ResourceBudgetMetadata:
    """Compute and enforce route resource budgets before numerical allocation."""

    policy = DEFAULT_API_RESOURCE_POLICY
    compute_nodes = request.state_dimensions * request.s_steps * request.t_steps
    max_compute_nodes = policy.max_compute_nodes
    output_nodes = _output_nodes(route, request)
    max_output_nodes = min(policy.max_output_nodes, request.max_output_nodes)
    estimated_response_bytes = output_nodes * policy.bytes_per_float
    max_response_bytes = min(policy.max_response_bytes, request.max_response_bytes)
    budget = ResourceBudgetMetadata(
        policy_name=policy.policy_name,
        state_dimensions=request.state_dimensions,
        compute_nodes=compute_nodes,
        max_compute_nodes=max_compute_nodes,
        output_nodes=output_nodes,
        max_output_nodes=max_output_nodes,
        estimated_response_bytes=estimated_response_bytes,
        max_response_bytes=max_response_bytes,
        timeout_seconds=_timeout_seconds(request),
        max_concurrent_solves=policy.max_concurrent_solves,
    )
    if compute_nodes > max_compute_nodes:
        _raise_resource_limit(
            route,
            budget,
            limit="max_compute_nodes",
            observed=compute_nodes,
            allowed=max_compute_nodes,
        )
    if output_nodes > max_output_nodes:
        _raise_resource_limit(
            route,
            budget,
            limit="max_output_nodes",
            observed=output_nodes,
            allowed=max_output_nodes,
        )
    if estimated_response_bytes > max_response_bytes:
        _raise_resource_limit(
            route,
            budget,
            limit="max_response_bytes",
            observed=estimated_response_bytes,
            allowed=max_response_bytes,
        )
    return budget


def _raise_resource_limit(
    route: str,
    budget: ResourceBudgetMetadata,
    *,
    limit: str,
    observed: int,
    allowed: int,
) -> None:
    raise HTTPException(
        status_code=422,
        detail={
            "code": "resource_limit_exceeded",
            "message": f"request exceeds {limit}: {observed} > {allowed}",
            "route": route,
            "resource": {
                "limit": limit,
                "observed": observed,
                "allowed": allowed,
                **budget.model_dump(mode="json"),
            },
        },
    )


class _ResourceLease:
    """Non-blocking local concurrency slot plus cooperative deadline checks."""

    def __init__(self, route: str, budget: ResourceBudgetMetadata) -> None:
        self.route = route
        self.budget = budget
        self.started_at = _monotonic()
        self.deadline = self.started_at + budget.timeout_seconds
        self.acquired = False

    def __enter__(self) -> "_ResourceLease":
        self.acquired = _SOLVE_SEMAPHORE.acquire(blocking=False)
        if not self.acquired:
            raise HTTPException(
                status_code=429,
                detail={
                    "code": "concurrency_limit_exceeded",
                    "message": "no local/demo API solve concurrency slot is available",
                    "route": self.route,
                    "resource": {
                        "limit": "max_concurrent_solves",
                        **self.budget.model_dump(mode="json"),
                    },
                },
            )
        try:
            self.check_deadline(stage="before_solve")
        except BaseException:
            _SOLVE_SEMAPHORE.release()
            self.acquired = False
            raise
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.acquired:
            _SOLVE_SEMAPHORE.release()
            self.acquired = False

    def check_deadline(self, *, stage: str) -> None:
        now = _monotonic()
        if now < self.deadline:
            return
        elapsed = max(0.0, now - self.started_at)
        raise HTTPException(
            status_code=504,
            detail={
                "code": "request_timeout",
                "message": f"request exceeded cooperative timeout at {stage}",
                "route": self.route,
                "stage": stage,
                "timeout_seconds": self.budget.timeout_seconds,
                "elapsed_seconds": elapsed,
                "resource": self.budget.model_dump(mode="json"),
            },
        )


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


def _sample_grid_at_spot(
    values: np.ndarray, grid: np.ndarray, spot: float
) -> GridSample:
    """Sample a one-dimensional solution grid at the requested spot with diagnostics."""

    values_array = np.asarray(values, dtype=float)
    grid_array = np.asarray(grid, dtype=float)
    if values_array.ndim != 1 or grid_array.ndim != 1:
        raise HTTPException(
            status_code=500,
            detail="requested-state sampling expects one-dimensional grid arrays",
        )
    if len(values_array) != len(grid_array):
        raise HTTPException(
            status_code=500,
            detail="requested-state sampling grid/value dimensions do not match",
        )
    if len(grid_array) == 0:
        raise HTTPException(
            status_code=500,
            detail="requested-state sampling cannot use an empty grid",
        )

    requested_spot = float(spot)
    grid_min = float(grid_array[0])
    grid_max = float(grid_array[-1])
    if requested_spot < grid_min or requested_spot > grid_max:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "requested spot is outside computed grid bounds",
                "bounded_policy": "reject_outside_grid",
                "requested_spot": requested_spot,
                "grid_min": grid_min,
                "grid_max": grid_max,
            },
        )

    upper_index = int(np.searchsorted(grid_array, requested_spot, side="left"))
    if upper_index < len(grid_array) and np.isclose(
        grid_array[upper_index], requested_spot, rtol=0.0, atol=1e-12
    ):
        value = float(values_array[upper_index])
        diagnostics = SamplingDiagnostics(
            requested_spot=requested_spot,
            method="grid_node",
            lower_index=upper_index,
            upper_index=upper_index,
            lower_spot=float(grid_array[upper_index]),
            upper_spot=float(grid_array[upper_index]),
            interpolation_weight=0.0,
        )
        return GridSample(value=value, sampling=diagnostics)

    if upper_index <= 0 or upper_index >= len(
        grid_array
    ):  # pragma: no cover - defensive
        raise HTTPException(
            status_code=400,
            detail={
                "message": "requested spot is outside computed interpolation brackets",
                "bounded_policy": "reject_outside_grid",
                "requested_spot": requested_spot,
                "grid_min": grid_min,
                "grid_max": grid_max,
            },
        )

    lower_index = upper_index - 1
    lower_spot = float(grid_array[lower_index])
    upper_spot = float(grid_array[upper_index])
    bracket_width = upper_spot - lower_spot
    if bracket_width <= 0.0:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500,
            detail="requested-state sampling requires a strictly increasing grid",
        )
    weight = (requested_spot - lower_spot) / bracket_width
    value = float(
        (1.0 - weight) * values_array[lower_index] + weight * values_array[upper_index]
    )
    diagnostics = SamplingDiagnostics(
        requested_spot=requested_spot,
        method="linear_interpolation",
        lower_index=lower_index,
        upper_index=upper_index,
        lower_spot=lower_spot,
        upper_spot=upper_spot,
        interpolation_weight=float(weight),
    )
    return GridSample(value=value, sampling=diagnostics)


def _grid_payload(res) -> PriceGrid:
    return PriceGrid(s=res.s.tolist(), t=res.t.tolist(), values=res.values.tolist())


@app.post("/price", response_model=PriceResponse, responses=API_ERROR_RESPONSES)
def price(request: OptionRequest, http_request: Request) -> PriceResponse:
    """Return requested-spot price and optional bounded grid for an option."""

    route = "/price"
    _ensure_supported_contract(request, route=route)
    budget = _resource_budget(route, request)
    with _ResourceLease(route, budget) as lease:
        res = _compute_grid(request)
        lease.check_deadline(stage="after_solve")
        price_sample = _sample_grid_at_spot(res.values[-1], res.s, request.spot)
        lease.check_deadline(stage="after_sampling")
    return PriceResponse(
        **_response_identity(http_request),
        metadata=_success_metadata(
            route,
            request,
            include_grid=request.include_full_grid,
            requested_outputs=["price"],
            sampling=price_sample.sampling,
            resource_budget=budget,
        ),
        option_type=request.option_type,
        spot=request.spot,
        price=price_sample.value,
        grid=_grid_payload(res) if request.include_full_grid else None,
    )


@app.post("/greeks", response_model=GreeksResponse, responses=API_ERROR_RESPONSES)
def greeks(request: OptionRequest, http_request: Request) -> GreeksResponse:
    """Return scalar Greeks at the explicitly requested spot."""

    route = "/greeks"
    _ensure_supported_contract(request, route=route)
    budget = _resource_budget(route, request)
    with _ResourceLease(route, budget) as lease:
        res = _compute_grid(request, return_greeks=True)
        lease.check_deadline(stage="after_solve")
        if (
            res.delta is None or res.gamma is None or res.theta is None
        ):  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail="Greek grid was not computed")
        delta_sample = _sample_grid_at_spot(res.delta[-1], res.s, request.spot)
        gamma_sample = _sample_grid_at_spot(res.gamma[-1], res.s, request.spot)
        theta_sample = _sample_grid_at_spot(res.theta[-1], res.s, request.spot)
        lease.check_deadline(stage="after_sampling")
    return GreeksResponse(
        **_response_identity(http_request),
        metadata=_success_metadata(
            route,
            request,
            include_grid=False,
            requested_outputs=["delta", "gamma", "theta"],
            sampling=delta_sample.sampling,
            resource_budget=budget,
        ),
        option_type=request.option_type,
        spot=request.spot,
        delta=delta_sample.value,
        gamma=gamma_sample.value,
        theta=theta_sample.value,
    )


@app.post(
    "/pde_solution", response_model=FullPDEResponse, responses=API_ERROR_RESPONSES
)
def pde_solution(request: OptionRequest, http_request: Request) -> FullPDEResponse:
    """Return complete solution and Greeks grids only after explicit opt-in."""

    route = "/pde_solution"
    _ensure_supported_contract(request, route=route)
    if not request.include_full_grid:
        raise HTTPException(
            status_code=400,
            detail="include_full_grid must be true to return the full PDE grid",
        )
    budget = _resource_budget(route, request)
    with _ResourceLease(route, budget) as lease:
        res = _compute_grid(request, return_greeks=True)
        lease.check_deadline(stage="after_solve")
        if (
            res.delta is None or res.gamma is None or res.theta is None
        ):  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail="Greek grid was not computed")
        lease.check_deadline(stage="after_grid_assembly")
    return FullPDEResponse(
        **_response_identity(http_request),
        metadata=_success_metadata(
            route,
            request,
            include_grid=True,
            requested_outputs=["price_grid", "delta_grid", "gamma_grid", "theta_grid"],
            resource_budget=budget,
        ),
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


@app.post(
    "/reports/crif",
    response_model=ErrorResponse,
    status_code=501,
    responses=UNSUPPORTED_ROUTE_RESPONSES,
)
def crif_report(exposures: list[ExposureModel]) -> None:
    """Fail closed until an exact ISDA CRIF profile and conformance suite exist."""

    strategy = ReportFactory.get_strategy("crif")
    try:
        strategy.generate_report(_convert_exposures(exposures))
    except NotImplementedForStandard as exc:
        _raise_regulatory_not_implemented(exc)


@app.post(
    "/reports/cuso",
    response_model=ErrorResponse,
    status_code=501,
    responses=UNSUPPORTED_ROUTE_RESPONSES,
)
def cuso_report(exposures: list[ExposureModel]) -> None:
    """Fail closed until an authoritative CUSO specification exists."""

    strategy = ReportFactory.get_strategy("cuso")
    try:
        strategy.generate_report(_convert_exposures(exposures))
    except NotImplementedForStandard as exc:
        _raise_regulatory_not_implemented(exc)


@app.post(
    "/reports/basel",
    response_model=ErrorResponse,
    status_code=501,
    responses=UNSUPPORTED_ROUTE_RESPONSES,
)
def basel_report(exposures: list[ExposureModel]) -> None:
    """Fail closed until a versioned Basel market-risk subset is implemented."""

    strategy = ReportFactory.get_strategy("basel")
    try:
        strategy.generate_report(_convert_exposures(exposures))
    except NotImplementedForStandard as exc:
        _raise_regulatory_not_implemented(exc)


@app.post(
    "/reports/frtb",
    response_model=ErrorResponse,
    status_code=501,
    responses=UNSUPPORTED_ROUTE_RESPONSES,
)
def frtb_report(exposures: list[ExposureModel]) -> None:
    """Fail closed until a versioned FRTB calculation subset is implemented."""

    strategy = ReportFactory.get_strategy("frtb")
    try:
        strategy.generate_report(_convert_exposures(exposures))
    except NotImplementedForStandard as exc:
        _raise_regulatory_not_implemented(exc)
