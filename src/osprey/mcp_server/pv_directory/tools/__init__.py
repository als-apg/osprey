"""PV directory MCP tool modules.

A module that registers unconditionally decorates its tools with ``@mcp.tool()``.
A gated module declares ``PREDICATE`` (a key of :data:`ToolContext.predicates`)
and exposes ``register(server, ctx)``, which ``create_server`` calls only when
that key holds.
"""
