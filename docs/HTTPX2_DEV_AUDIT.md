# Development HTTP client audit refresh

Current CI on 2026-09-11 found six advisories after the reproducible development
lock restored `httpx2`/`httpcore2` 2.9.1. The earlier fresh-resolution audit passed;
the pinned-lock audit correctly caught the stale versions. Issue #172 tracks it.

Keep the existing optional/test-client dependency and update both coupled pins
to 2.12.0. This does not migrate the core numerical API or canonical `httpx`
consumers. HTTPX2 is the existing Pydantic-maintained BSD-3-Clause distribution,
not an alias for `httpx`; its published metadata requires `httpcore2==2.12.0`.
See the [official package](https://pypi.org/project/httpx2/2.12.0/) and
[source](https://github.com/pydantic/httpx2).

Validate the complete locked environment and its audit, the optional API profile,
packaging and remote CI. Keep the audit/SBOM workflow enabled and preserve the
failed run as baseline evidence; an editable project audit skip is not a claim
that an unpublished local distribution was vulnerability-scanned.
