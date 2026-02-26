"""Web Debug UI for Osprey Event Streaming.

This package provides a web-based debug interface for monitoring Osprey events
in real-time. It uses FastAPI with WebSocket for streaming typed events to
a browser-based UI.

Features:
    - Real-time event streaming via WebSocket
    - Color-coded event timeline
    - Filter by event type and component
    - Collapsible event details
    - Session management for multiple clients

Usage:
    # Start the web debug UI server
    osprey web --port 8080

    # Or programmatically
    from osprey.interfaces.web import run_server
    run_server(port=8080)
"""

from .server import create_app, run_server

__all__ = [
    "create_app",
    "run_server",
]
