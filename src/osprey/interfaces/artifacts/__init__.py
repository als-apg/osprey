"""OSPREY Artifact Gallery.

Local web gallery for interactive plots, tables, and outputs produced by
Claude during analysis sessions.

Example usage:
    from osprey.interfaces.artifacts import create_app, run_server

    app = create_app()       # For ASGI servers
    run_server(port=8086)    # Direct launch
"""

from osprey.interfaces.artifacts.app import create_app, run_server

__all__ = ["create_app", "run_server"]
