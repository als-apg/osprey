"""Osprey's unified audit ledger — one record shape, one writer, every surface.

Deliberately empty of re-exports. Consumers import the module they need
(``osprey.audit.envelope`` for the record shape, ``osprey.audit.writer`` for
the append) so that importing the package costs nothing: the envelope is a
stdlib-only leaf that the MCP middleware, the HTTP layer and the hooks all
depend on, and a package ``__init__`` that pulled the writer in would hand
every one of them the writer's imports as well.
"""
