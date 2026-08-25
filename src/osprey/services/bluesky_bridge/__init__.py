"""Bluesky bridge: a FastAPI service owning the Bluesky RunEngine + ophyd-async devices.

Runs in a separate container from OSPREY's own venv; the ``bluesky`` MCP server
talks to it over HTTP. See ``docs/source/how-to/bluesky/write-plans.rst`` for
the full architecture.
"""
