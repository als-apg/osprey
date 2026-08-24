"""MCP tool: get_schema — the main-agent graph server's tool, re-registered here.

Reading the store's vocabulary is the same job in both paradigms: the labels,
relationship types and sampled property names come from the same corpus through
the same context, and the bookkeeping this tool hides is bookkeeping either way.
So the channel finder registers that server's callable rather than restating it.

The re-registration lives in its own module, named for the tool, because the
tool-vocabulary tests walk this package with ``pkgutil`` and import what they
find. A tool folded into ``server.py`` would be invisible to that walk.
"""

from osprey.mcp_server.graph.tools.get_schema import get_schema

from ..server import mcp

mcp.tool()(get_schema)
