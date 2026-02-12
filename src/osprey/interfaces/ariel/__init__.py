"""ARIEL Web Interface.

This module provides a web-based interface for ARIEL (Agentic Retrieval Interface
for Electronic Logbooks), built with FastAPI.

Example usage:
    # Programmatic
    from osprey.interfaces.ariel import create_app, run_web

    app = create_app("config.yml")  # For ASGI servers
    run_web(port=8085)  # CLI entry point

    # CLI
    osprey ariel web --port 8085
"""

from osprey.interfaces.ariel.app import create_app, run_web

__all__ = ["create_app", "run_web"]
