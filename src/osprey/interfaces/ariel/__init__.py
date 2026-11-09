"""ARIEL Web Interface.

This module provides a web-based interface for ARIEL (Agentic Retrieval Interface
for Electronic Logbooks), built with FastAPI. It runs inside the web terminal,
whose launcher builds the app from :func:`create_app`.

The panel's port is not a fixed number. Every OSPREY host port is the
deployment's ``deployment.port_base`` plus the fixed offset its family holds in
:data:`osprey.port_layout.LAYOUT`, so a deployment that moved its base moved
this panel with it.
"""

from osprey.interfaces.ariel.app import create_app

__all__ = ["create_app"]
