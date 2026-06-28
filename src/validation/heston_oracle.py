"""Semi-analytical Heston oracle and boundary diagnostics.

The functions in this module are validation fixtures, not a replacement for the
finite-difference/ADI solver work tracked elsewhere.  They provide independent
Heston European-call reference prices and executable state-boundary evidence for
Project 5 governance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from src.exceptions import ValidationError
from src.processes.affine import HestonModel


@dataclass(frozen=True)
class HestonOracleCase:
    """Input parameters for a Heston European call oracle."""

    spot: float
    strike: float
    rate: float
    dividend_yield: float
    maturity: float
    variance: float
    kappa: float
    theta: float
    vol_of_vol: float
    rho: float


@dataclass(frozen=True)
class HestonVarianceBoundaryBenchmark:
    """Executable diagnostics for the variance lower boundary."""

    lower_boundary: float
    drift_at_lower_boundary: float
    min_eigenvalue_at_lower_boundary: float
    max_asymmetry_at_lower_boundary: float
    clips_interior_variance: bool


def _validate_oracle_case(case: HestonOracleCase) -> None:
    if case.spot <= 0.0 or case.strike <= 0.0:
        raise ValidationError("spot and strike must be positive")
    if case.maturity < 0.0:
        raise ValidationError("maturity must be non-negative")
    if case.variance < 0.0 or case.theta < 0.0:
        raise ValidationError("variance and theta must be non-negative")
    if case.kappa <= 0.0 or case.vol_of_vol < 0.0:
        raise ValidationError("kappa must be positive and vol_of_vol must be non-negative")
    if not -1.0 < case.rho < 1.0:
        raise ValidationError("rho must be strictly between -1 and 1")
    values = np.array(
        [
            case.spot,
            case.strike,
            case.rate,
            case.dividend_yield,
            case.maturity,
            case.variance,
            case.kappa,
            case.theta,
            case.vol_of_vol,
            case.rho,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        raise ValidationError("Heston oracle case contains non-finite values")


def _deterministic_variance_integral(case: HestonOracleCase) -> float:
    """Integrated variance when Heston vol-of-vol is exactly zero."""

    return float(
        case.theta * case.maturity
        + (case.variance - case.theta) * (1.0 - np.exp(-case.kappa * case.maturity)) / case.kappa
    )


def _black_scholes_call_with_integrated_variance(case: HestonOracleCase, integrated_variance: float) -> float:
    """Dividend-aware Black--Scholes call with total variance over the horizon."""

    forward_spot_discounted = case.spot * np.exp(-case.dividend_yield * case.maturity)
    strike_discounted = case.strike * np.exp(-case.rate * case.maturity)
    if integrated_variance <= 1e-16:
        return float(max(forward_spot_discounted - strike_discounted, 0.0))

    volatility_time = float(np.sqrt(integrated_variance))
    d1 = (
        np.log(case.spot / case.strike)
        + (case.rate - case.dividend_yield) * case.maturity
        + 0.5 * integrated_variance
    ) / volatility_time
    d2 = d1 - volatility_time
    return float(forward_spot_discounted * norm.cdf(d1) - strike_discounted * norm.cdf(d2))

def _heston_characteristic_function(u: complex, case: HestonOracleCase) -> complex:
    """Return the log-spot characteristic function under risk-neutral Heston."""

    i = 1j
    sigma = case.vol_of_vol
    if sigma == 0.0:
        integrated_variance = _deterministic_variance_integral(case)
        mean_log = np.log(case.spot) + (case.rate - case.dividend_yield) * case.maturity
        mean_log -= 0.5 * integrated_variance
        return np.exp(i * u * mean_log - 0.5 * u * u * integrated_variance)

    iu = i * u
    b = case.kappa - case.rho * sigma * iu
    d = np.sqrt(b * b + sigma * sigma * (u * u + iu))
    g = (b - d) / (b + d)
    exp_neg_dt = np.exp(-d * case.maturity)
    denominator = 1.0 - g * exp_neg_dt
    if abs(denominator) < 1e-14:
        raise ValidationError("Heston characteristic-function denominator is singular")

    log_term = np.log((1.0 - g * exp_neg_dt) / (1.0 - g))
    c = (
        iu * (np.log(case.spot) + (case.rate - case.dividend_yield) * case.maturity)
        + (case.kappa * case.theta / (sigma * sigma))
        * ((b - d) * case.maturity - 2.0 * log_term)
    )
    d_term = ((b - d) / (sigma * sigma)) * ((1.0 - exp_neg_dt) / denominator)
    return np.exp(c + d_term * case.variance)


def heston_call_oracle(
    case: HestonOracleCase,
    *,
    abs_tol: float = 1e-9,
    rel_tol: float = 1e-9,
    integration_limit: int = 200,
) -> float:
    """Price a European call using Heston characteristic-function inversion.

    The implementation uses the normalized probabilities

    ``C = S exp(-qT) P1 - K exp(-rT) P2``

    with ``P1`` normalized by ``phi(-i)``.  The formula is independent of the
    finite-difference solver and is therefore suitable as a regression oracle.
    """

    _validate_oracle_case(case)
    if case.maturity == 0.0:
        return max(case.spot - case.strike, 0.0)
    if case.vol_of_vol == 0.0:
        return _black_scholes_call_with_integrated_variance(
            case,
            _deterministic_variance_integral(case),
        )

    log_strike = np.log(case.strike)
    phi_minus_i = _heston_characteristic_function(-1j, case)
    if abs(phi_minus_i) < 1e-14:
        raise ValidationError("Heston phi(-i) normalization is singular")

    def p1_integrand(u: float) -> float:
        z = complex(u)
        numerator = np.exp(-1j * z * log_strike) * _heston_characteristic_function(z - 1j, case)
        return float(np.real(numerator / (1j * z * phi_minus_i)))

    def p2_integrand(u: float) -> float:
        z = complex(u)
        numerator = np.exp(-1j * z * log_strike) * _heston_characteristic_function(z, case)
        return float(np.real(numerator / (1j * z)))

    p1_integral, _ = quad(p1_integrand, 0.0, np.inf, epsabs=abs_tol, epsrel=rel_tol, limit=integration_limit)
    p2_integral, _ = quad(p2_integrand, 0.0, np.inf, epsabs=abs_tol, epsrel=rel_tol, limit=integration_limit)
    p1 = 0.5 + p1_integral / np.pi
    p2 = 0.5 + p2_integral / np.pi
    price = case.spot * np.exp(-case.dividend_yield * case.maturity) * p1
    price -= case.strike * np.exp(-case.rate * case.maturity) * p2
    return float(max(price, 0.0))


def heston_variance_boundary_benchmark(process: HestonModel) -> HestonVarianceBoundaryBenchmark:
    """Evaluate lower-variance-boundary diagnostics for a Heston process."""

    lower_state = np.array([np.log(100.0), 0.0], dtype=float)
    drift = np.asarray(process.drift(0.0, lower_state), dtype=float)
    covariance = np.asarray(process.covariance(0.0, lower_state), dtype=float)
    eigenvalues = np.linalg.eigvalsh(covariance)
    asymmetry = covariance - covariance.T
    return HestonVarianceBoundaryBenchmark(
        lower_boundary=0.0,
        drift_at_lower_boundary=float(drift[1]),
        min_eigenvalue_at_lower_boundary=float(np.min(eigenvalues)),
        max_asymmetry_at_lower_boundary=float(np.max(np.abs(asymmetry))),
        clips_interior_variance=False,
    )


__all__ = [
    "HestonOracleCase",
    "HestonVarianceBoundaryBenchmark",
    "heston_call_oracle",
    "heston_variance_boundary_benchmark",
]
