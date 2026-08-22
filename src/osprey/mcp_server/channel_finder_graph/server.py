"""Graph Channel Finder MCP Server.

The fourth channel-finder paradigm (``channel_finder.pipeline_mode: graph``).
Where the other three paradigms search a channel database, this one answers
channel questions by writing read-only Cypher against the facility knowledge
graph in the ``graphdb`` service — the search strategy is the agent's, not a
resolution API's.

It ships under the ``channel-finder`` server name so every consumer that
selects the paradigm by config key — the registry, the rendered agent's tool
allowlist, the benchmark harness — reaches it unchanged.

Startup is the lean sequence of ``osprey.mcp_server.graph.server``, not the
shared channel-finder one: the graph store is this server's only backing
resource, so it primes the config builder and initialises the graph server
context and stops there. It must never call
``initialize_workspace_singletons()`` — this server saves no artifacts, and
arming the artifact listeners here would attribute a gallery process's work to
it.

Usage:
    python -m osprey.mcp_server.channel_finder_graph
"""

import logging

from fastmcp import FastMCP

from osprey.mcp_server.errors import make_error  # noqa: F401  (re-exported for tools)

logger = logging.getLogger("osprey.mcp_server.channel_finder_graph")

# ---------------------------------------------------------------------------
# FastMCP server instance -- imported by every tool module
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "channel-finder",
    instructions=(
        "Find control-system channel addresses by searching the facility's knowledge "
        "graph with read-only Cypher. Start with example_queries or get_schema to learn "
        "which labels, properties and predicates the store actually carries, then write "
        "the query the question needs and run it with read_cypher. Results are bounded "
        "by a row cap and a query timeout, so a truncated answer means narrow the query "
        "rather than retry it unchanged."
    ),
)


def _import_tools() -> None:
    """Import the tool modules so their ``@mcp.tool()`` decorators run.

    Named rather than inlined so the startup sequence can be exercised on its
    own: importing ``read_cypher`` and ``get_schema`` pulls in the main-agent
    ``graph`` server module they are re-registered from.
    """
    from osprey.mcp_server.channel_finder_graph.tools import (  # noqa: F401
        capabilities,
        example_queries,
        get_schema,
        read_cypher,
    )


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------
def create_server() -> FastMCP:
    """Initialize the graph server context and import tool modules, then return the server."""
    from osprey.mcp_server.graph.server_context import initialize_server_context
    from osprey.mcp_server.startup import prime_config_builder, startup_timer

    prime_config_builder()

    with startup_timer("server_context"):
        initialize_server_context()

    with startup_timer("tool_imports"):
        _import_tools()

    logger.info("Channel Finder Graph MCP server initialised with all tools registered")
    return mcp
