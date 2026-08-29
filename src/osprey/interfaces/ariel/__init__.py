"""ARIEL Web Interface.

This module provides a web-based interface for ARIEL (Agentic Retrieval Interface
for Electronic Logbooks), built with FastAPI.

The panel's port is not a fixed number. Every OSPREY host port is the
deployment's ``deployment.port_base`` plus the fixed offset its family holds in
:data:`osprey.port_layout.LAYOUT`, so a deployment that moved its base moved
this panel with it. ``run_web`` defaults to that offset at the layout's
*default* base — right only for a caller with no config to resolve a base from
— and ``osprey ariel web`` hands it an already-resolved *port*, derived from
this deployment's base rather than from the default one.

Example usage:
    # Programmatic
    from osprey.interfaces.ariel import create_app, run_web

    app = create_app("config.yml")  # For ASGI servers
    run_web()                       # Serve on the layout's default port

    # CLI
    osprey ariel web
"""

from osprey.interfaces.ariel.app import create_app, run_web

__all__ = ["create_app", "run_web"]
