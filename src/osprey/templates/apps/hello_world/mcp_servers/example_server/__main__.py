"""Stdio entry point: ``python -m example_server``."""

from __future__ import annotations

from .server import mcp

if __name__ == "__main__":
    mcp.run()
