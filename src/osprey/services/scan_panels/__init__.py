"""Scan panels: a FastAPI sidecar serving the operator-facing scan authoring,
results, and health panel bundles alongside a thin read-proxy onto the
Bluesky bridge.

Runs in a separate process from OSPREY's own venv, reachable over HTTP. See
``docs/source/how-to/scan-plans.rst`` (Phase 6) for the full architecture.
"""
