"""MCP tool: capabilities — report what the graph server offers.

PROMPT-PROVIDER: this tool's response is a static manifest read by the agent
before it plans a graph query. It deliberately does not dial the store, so a
successful response says nothing about whether the graph is reachable or seeded.
"""

import json
import logging

from fastmcp.exceptions import ToolError

from osprey.deployment.graphdb_service import GRAPHDB_SEED_COMMAND
from osprey.mcp_server.graph.server import make_error, mcp
from osprey.mcp_server.graph.server_context import (
    QUERY_MAX_ROWS_CONFIG_KEY,
    QUERY_TIMEOUT_CONFIG_KEY,
)

logger = logging.getLogger("osprey.mcp_server.graph.tools.capabilities")

_SERVER_NAME = "graph"

_DESCRIPTION = (
    "Read-only Cypher search over the facility knowledge graph — a "
    "NARAD-convention RDF corpus loaded into the graphdb service."
)

_TOOLS: tuple[dict[str, str], ...] = (
    {
        "name": "read_cypher",
        "purpose": "Run one read-only Cypher query and return the matching rows.",
    },
    {
        "name": "get_schema",
        "purpose": "List the node labels and relationship types present in the graph.",
    },
    {
        "name": "example_queries",
        "purpose": "Return curated, runnable Cypher examples for the common question shapes.",
    },
    {
        "name": "capabilities",
        "purpose": "Report this manifest: server description, tool list, and operating notes.",
    },
)

_NOTES: tuple[str, ...] = (
    "Read-only: writes, schema changes and administrative procedures are refused "
    "before the query is sent to the store.",
    "Start with example_queries or get_schema, then adapt an example rather than "
    "inventing a query shape.",
    f"Every result is bounded. The row cap comes from {QUERY_MAX_ROWS_CONFIG_KEY} "
    f"and the per-query deadline from {QUERY_TIMEOUT_CONFIG_KEY}; a truncated "
    "result means narrow the query.",
    f"An empty graph is a seeding gap, not a query bug — run `{GRAPHDB_SEED_COMMAND}` "
    "to load the corpus.",
)


@mcp.tool()
def capabilities() -> str:
    """Report the graph server's tools and operating limits.

    Returns the server name, a one-line description of what the graph holds,
    the tool manifest (each with a one-line purpose), and the operating notes:
    the read-only posture, the recommended order of use, the config keys that
    bound a result, and the command that seeds an empty graph.

    Does NOT require graph connectivity, so this is *not* a health check: a
    successful response says nothing about whether the store is reachable or
    seeded. Run ``read_cypher`` for that.

    Returns:
        JSON object with ``server``, ``description``, ``tools`` (list of
        ``{name, purpose}``) and ``notes`` (list of strings).
    """
    try:
        return json.dumps(
            {
                "server": _SERVER_NAME,
                "description": _DESCRIPTION,
                "tools": [dict(tool) for tool in _TOOLS],
                "notes": list(_NOTES),
            }
        )

    except ToolError:
        raise
    except Exception as exc:
        logger.exception("capabilities failed")
        return make_error(
            "internal_error",
            f"Failed to report graph capabilities: {exc}",
            ["Check the MCP server logs for details."],
        )
