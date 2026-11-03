"""Which agent tools write: the one list every write gate degrades to.

:data:`WRITE_GATED_TOOLS` is the floor under every write gate the framework
ships. A render names the deployment's own write tools in ``hook_config.json``;
when that file cannot be read, the headless agent runner
(:mod:`osprey.agent_runner.write_tools`), the MCP audit middleware
(:mod:`osprey.mcp_server.audit_middleware`) and the hooks' shared helpers
(``osprey_hook_log.py``) all fall back to these names. The hooks ship as
copied-in template files run by a bare ``python3``, so they carry a literal of
the same list, and ``tests/registry/test_mixed_floor_driftguard.py`` pins every
copy against ``registry.mcp.framework_write_tools()``.

Whether a person is asked before a write is not read here: that is the
execution verdict's ``position`` (``asked`` or ``allowed``), the one answer
every surface shares.

A leaf module: the standard library only, so a renderer can import it without
the web stack.
"""

from __future__ import annotations

#: The MCP tools the writes check gates, by their full names.
WRITE_GATED_TOOLS: tuple[str, ...] = (
    "mcp__bluesky__queue_add",
    "mcp__bluesky__queue_start",
    "mcp__controls__channel_write",
    "mcp__phoebus__phoebus_drive",
    "mcp__python__execute",
    "mcp__python__execute_file",
    "mcp__python__pyaml_measure",
)
