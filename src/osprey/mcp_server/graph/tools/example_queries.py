"""MCP tool: example_queries — the curated, runnable Cypher starting points.

PROMPT-PROVIDER: this tool's response is a static catalogue read by the agent
before it writes a query. Like ``capabilities`` it deliberately does not dial the
store, so it answers while the graph is down and a successful response says
nothing about whether the corpus is reachable or seeded.
"""

import json
import logging

from fastmcp.exceptions import ToolError

from osprey.mcp_server.graph.server import make_error, mcp
from osprey.mcp_server.graph.tools.examples_data import EXAMPLE_QUERIES

logger = logging.getLogger("osprey.mcp_server.graph.tools.example_queries")

_NOTES: tuple[str, ...] = (
    "Each example carries its parameters separately from its Cypher. Pass the "
    "chosen parameter dict to read_cypher as `params` — never paste values into "
    "the query text.",
    "Pick the parameter set matching the corpus this deployment seeded: `als` "
    "for the shipped ALS corpus, `demo` for the generated demo machine. The "
    "Cypher is the same either way; only the values differ.",
    "Every example already ends in a LIMIT, so an example that comes back "
    "truncated was truncated by the server's row cap, not by the query.",
    "Results are bounded by services.graphdb.query_max_rows; a truncated "
    "result means narrow the query rather than retry it unchanged.",
)


@mcp.tool()
def example_queries() -> str:
    """Return curated, runnable Cypher examples for the common graph questions.

    Read this before writing Cypher against the facility knowledge graph. The
    examples are written against the real shape of the corpus — node labels,
    relationship types and property spellings that actually exist — so adapting
    one is reliable where inventing a query from scratch is guesswork. Together
    they cover:

    * device rollups by class, both concrete and aggregated up the ontology;
    * walking one section of the machine in beamline order;
    * every PV bound to a named device, split by read/write direction;
    * the class hierarchy, as single edges and as whole inheritance chains;
    * a read/write summary across all channel bindings;
    * which devices share a given PV.

    Start from the closest example and change it — swap a parameter value, add a
    WHERE clause, widen or narrow the LIMIT — rather than composing a new query
    shape. Each example's ``description`` says what its parameters mean and how
    to vary them; ``get_schema`` lists the labels and relationship types if you
    need to go further than the examples reach.

    Does NOT require graph connectivity, so this is *not* a health check: a
    successful response says nothing about whether the store is reachable or
    seeded.

    Returns:
        JSON object with ``count``, ``examples`` (each ``{key, title,
        description, cypher, parameters}``, where ``parameters`` holds one value
        set per corpus under ``als`` and ``demo``) and ``notes`` (list of
        strings on how to run them).
    """
    try:
        examples = [
            {
                "key": example.key,
                "title": example.title,
                "description": example.description,
                "cypher": example.cypher,
                "parameters": {
                    corpus: dict(values) for corpus, values in example.parameters.items()
                },
            }
            for example in EXAMPLE_QUERIES
        ]

        return json.dumps(
            {
                "count": len(examples),
                "examples": examples,
                "notes": list(_NOTES),
            }
        )

    except ToolError:
        raise
    except Exception as exc:
        logger.exception("example_queries failed")
        return make_error(
            "internal_error",
            f"Failed to list graph example queries: {exc}",
            ["Check the MCP server logs for details."],
        )
