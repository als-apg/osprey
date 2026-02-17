"""OSPREY MCP Server package.

Contains three independent FastMCP servers:

- ``control_system`` — channel_read, channel_write, archiver_read
- ``python_executor`` — python_execute
- ``workspace`` — memory, artifacts, screen capture, data context

Shared utilities live in ``common.py``.
"""
