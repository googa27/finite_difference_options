"""Risk data converters for regulatory reporting formats.

These functions are retained as compatibility entry points, but all named
regulatory routes fail closed until an exact standard/profile/version and a
complete conformance suite exist. They deliberately do not emit toy multipliers,
partial CRIF-like rows, or success dictionaries.
"""
from __future__ import annotations

from typing import NoReturn

from .models import Exposure, NotImplementedForStandard, RegulatoryStandard

REGULATORY_STANDARDS: dict[str, RegulatoryStandard] = {
    "crif": RegulatoryStandard(
        route="crif",
        standard_id="ISDA-CRIF-UNSPECIFIED",
        display_name="ISDA Common Risk Interchange Format",
        capability_status="scaffold",
        licensing_status="not-evaluated",
        reason=(
            "CRIF export is disabled because no exact ISDA CRIF profile/version, licensing status, "
            "field order, casing policy, extension-field policy, or golden conformance fixtures are configured."
        ),
    ),
    "cuso": RegulatoryStandard(
        route="cuso",
        standard_id="CUSO-UNSPECIFIED-NO-AUTHORITATIVE-SPEC",
        display_name="CUSO regulatory/reporting route",
        capability_status="unsupported",
        licensing_status="no-authoritative-specification",
        reason=(
            "CUSO is disabled because no authoritative specification, owner, jurisdiction, or field contract has "
            "been identified. The route cannot infer a regulatory definition from the acronym."
        ),
    ),
    "basel": RegulatoryStandard(
        route="basel",
        standard_id="BCBS-MAR21-UNSPECIFIED",
        display_name="Basel Framework market-risk standardised approach",
        capability_status="scaffold",
        licensing_status="not-evaluated",
        reason=(
            "Basel/FRTB calculation is disabled because no exact framework chapter/version/effective date, "
            "risk-class subset, risk weights, correlations, rounding policy, or reconciliation fixture is configured."
        ),
    ),
    "frtb": RegulatoryStandard(
        route="frtb",
        standard_id="BCBS-FRTB-UNSPECIFIED",
        display_name="Fundamental Review of the Trading Book capital calculation",
        capability_status="scaffold",
        licensing_status="not-evaluated",
        reason=(
            "FRTB calculation is disabled because no exact standard/version/effective date, risk-class subset, "
            "risk weights, correlations, scenario rules, or reconciliation fixture is configured."
        ),
    ),
}


def _raise_not_implemented(route: str) -> NoReturn:
    raise NotImplementedForStandard(REGULATORY_STANDARDS[route])


def exposures_to_crif(exposures: list[Exposure]) -> NoReturn:
    """Fail closed instead of emitting partial CRIF-like records."""

    _ = exposures
    _raise_not_implemented("crif")


def calculate_cuso(exposures: list[Exposure]) -> NoReturn:
    """Fail closed because no authoritative CUSO standard is configured."""

    _ = exposures
    _raise_not_implemented("cuso")


def calculate_basel(exposures: list[Exposure]) -> NoReturn:
    """Fail closed instead of returning scalar placeholder Basel capital."""

    _ = exposures
    _raise_not_implemented("basel")


def calculate_frtb(exposures: list[Exposure]) -> NoReturn:
    """Fail closed instead of returning scalar placeholder FRTB capital."""

    _ = exposures
    _raise_not_implemented("frtb")
