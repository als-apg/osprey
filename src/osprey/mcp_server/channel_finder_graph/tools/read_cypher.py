"""MCP tool: read_cypher — the main-agent graph server's tool, re-registered here.

The graph channel finder runs the same read path as the main-agent graph server:
the same query gate, the same read transaction, the same row cap and timeout,
the same error envelopes. So it registers that server's callable rather than
restating it — one implementation, one docstring, one set of remedies, reachable
under both server names.

The re-registration lives in its own module, named for the tool, because the
tool-vocabulary tests walk this package with ``pkgutil`` and import what they
find. A tool folded into ``server.py`` would be invisible to that walk.
"""

from osprey.mcp_server.graph.tools.read_cypher import read_cypher

from ..server import mcp

mcp.tool()(read_cypher)
