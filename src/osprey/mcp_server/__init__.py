"""OSPREY MCP Server package.

Contains independent FastMCP servers:

- ``control_system`` — channel_read, channel_write, archiver_read
- ``workspace`` — artifacts, screen capture, data context

Shared utilities are split across focused modules:

- ``errors`` — structured error envelope (``make_error``)
- ``startup`` — server lifecycle (``startup_timer``, ``run_mcp_server``, etc.)
- ``http`` — HTTP/IPC helpers (``post_json``, ``gallery_url``, etc.)
- ``session`` — session metadata collection (``gather_session_metadata``)

Cross-layer config/workspace utilities live in ``osprey.utils.workspace``.
"""
