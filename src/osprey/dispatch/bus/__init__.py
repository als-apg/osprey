"""The dispatch worker bus: workers connect to the dispatcher, not the other way round.

The dispatcher binds one ZeroMQ ``ROUTER`` socket and every dispatch worker
connects a ``DEALER`` to it, so no worker opens a listening port. The
dispatcher is the broker: it learns which workers are alive from their
heartbeats, sends each dispatch to the least-loaded one, and routes every
run-scoped call to the worker that owns the run.

Modules:

- :mod:`~osprey.dispatch.bus.protocol` — the wire format, shared by both ends.
- :mod:`~osprey.dispatch.bus.keys` — CURVE key material and the authenticator
  that names a peer by its key.
- :mod:`~osprey.dispatch.bus.broker` — the dispatcher's end.
- :mod:`~osprey.dispatch.bus.client` — the dispatcher's worker calls over the
  broker, raising the same exceptions as the HTTP client.
- :mod:`~osprey.dispatch.bus.link` — the worker's end.

Every module here depends on pyzmq and the standard library only. The worker
imports the link from here, and nothing here imports the worker: the
dispatcher is a separately deployed image.

Deliberately empty of re-exports, so importing the package costs nothing.
"""
