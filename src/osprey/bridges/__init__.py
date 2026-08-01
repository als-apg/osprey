"""OSPREY dispatch bridges.

Bridges connect an external messaging channel (chat, email, ...) to the OSPREY
dispatcher/worker pair. Each channel ships its own adapter package alongside the
channel-agnostic engine in :mod:`osprey.bridges.core`; an adapter contributes only
its own wire format and platform I/O.

Subpackages:
    core — channel-agnostic bridge engine (stores, dispatch client, pipeline, runtime)
"""
