"""OSPREY Artifact Gallery.

Local web gallery for interactive plots, tables, and outputs produced by
Claude during analysis sessions.

The gallery's port is not a fixed number. Every OSPREY host port is the
deployment's ``deployment.port_base`` plus the fixed offset its family holds in
:data:`osprey.port_layout.LAYOUT`, so a deployment that moved its base moved
the gallery with it. ``run_server`` defaults to that offset at the layout's
*default* base — right only for a caller with no config to resolve a base from
— and ``osprey artifacts web`` hands it an already-resolved *port*, derived from
this deployment's base rather than from the default one.

Example usage:
    from osprey.interfaces.artifacts import create_app, run_server

    app = create_app()   # For ASGI servers
    run_server()         # Direct launch, on the layout's default port
"""

from osprey.interfaces.artifacts.app import create_app, run_server

__all__ = ["create_app", "run_server"]
