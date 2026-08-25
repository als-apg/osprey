"""Bluesky web sidecar: a FastAPI app serving the operator-facing plan authoring,
results, and health panel bundles alongside a thin read-proxy onto the
Bluesky bridge.

Runs in a separate process from OSPREY's own venv, reachable over HTTP. See
``docs/source/how-to/bluesky/write-plans.rst`` for the full architecture.
"""
