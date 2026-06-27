"""Domain-neutral finite-difference contracts and capability manifests."""

from .backend_capabilities import (
    CapabilityStatus,
    DEFAULT_FD_CAPABILITY_MANIFEST,
    FDCapabilityManifest,
    FDRouteRequest,
    UnsupportedReason,
    UnsupportedRouteDiagnostic,
    UnsupportedRouteError,
    diagnose_unsupported_route,
    ensure_route_supported,
)

__all__ = [
    "CapabilityStatus",
    "DEFAULT_FD_CAPABILITY_MANIFEST",
    "FDCapabilityManifest",
    "FDRouteRequest",
    "UnsupportedReason",
    "UnsupportedRouteDiagnostic",
    "UnsupportedRouteError",
    "diagnose_unsupported_route",
    "ensure_route_supported",
]
