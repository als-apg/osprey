"""OSPREY Bluesky MCP Server — thin HTTP client for the facility Bluesky bridge.

Translates agent tool calls into HTTP requests against the facility-side
Bluesky bridge (a FastAPI service wrapping Bluesky's RunEngine, ophyd devices,
and a Tiled data store). This package makes no bluesky/ophyd/tiled imports;
see ``osprey.services.bluesky_bridge`` for the facility-side implementation.
"""
