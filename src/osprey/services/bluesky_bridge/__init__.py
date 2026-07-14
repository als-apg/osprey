"""Bluesky bridge: a FastAPI service owning the Bluesky RunEngine + ophyd-async devices.

Runs in a separate container from OSPREY's own venv; the ``scan`` MCP server
talks to it over HTTP. See ``docs/source/how-to/scan-plans.rst`` (Phase 3) for
the full architecture.
"""
