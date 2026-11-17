"""PV directory MCP tool modules.

A module that registers unconditionally decorates its tools with ``@mcp.tool()``.
A gated module exposes ``register(server)``; its gate is the ``optional_tools``
entry of ``FRAMEWORK_SERVERS["pv-directory"]`` keyed by the module's name, which
``register_tools`` reads and looks up in :data:`ToolContext.predicates`.
"""
