"""OSPREY Artifact Gallery.

Local web gallery for interactive plots, tables, and outputs produced by
Claude during analysis sessions. It runs inside the web terminal, whose launcher
builds the app from :func:`create_app`.

The gallery's port is not a fixed number. Every OSPREY host port is the
deployment's ``deployment.port_base`` plus the fixed offset its family holds in
:data:`osprey.port_layout.LAYOUT`, so a deployment that moved its base moved
the gallery with it.
"""

from osprey.interfaces.artifacts.app import create_app

__all__ = ["create_app"]
