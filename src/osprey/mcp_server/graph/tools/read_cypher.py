"""MCP tool: read_cypher — run one read-only Cypher query against the graph.

PROMPT-PROVIDER: this tool's docstring is a static prompt visible to the agent.
  Facility-customizable: node labels, relationship types, property spellings.

The query is vetted by :mod:`~osprey.mcp_server.graph.gate` before the store is
touched, so a refused query costs no connection. Everything past the gate is the
server context's business: it owns the read transaction, the row cap, the query
timeout and the mapping from driver failures to envelopes. This module only
decides which envelope a caller sees and what the successful payload looks like.
"""

import json
import logging
from typing import Any

from fastmcp.exceptions import ToolError

from osprey.deployment.graphdb_service import GRAPHDB_SEED_COMMAND
from osprey.mcp_server.graph.gate import GateRefusal, vet_query
from osprey.mcp_server.graph.server import make_error, mcp
from osprey.mcp_server.graph.server_context import (
    QUERY_MAX_ROWS_CONFIG_KEY,
    GraphStoreError,
    get_server_context,
)

logger = logging.getLogger("osprey.mcp_server.graph.tools.read_cypher")

#: Remedies for a query the gate refused. They describe the posture rather than
#: the single token that tripped it, because the refusal message already names
#: that and the agent's next move is to pick a different shape entirely.
_GATE_SUGGESTIONS: tuple[str, ...] = (
    "Only read queries run here — MATCH / WHERE / RETURN and read-only CALL { … } subqueries.",
    "Extension procedures and functions (apoc.*, n10s.*, gds.*, dbms.*) and CSV "
    "import are not available through this tool.",
    "get_schema reports the labels, relationship types and properties this graph "
    "has; example_queries has a runnable query for each common question shape.",
)

_EMPTY_GRAPH_MESSAGE = (
    "The graph holds no Resource nodes, so no query can match: the store is "
    "running but its corpus was never loaded."
)


@mcp.tool()
def read_cypher(query: str, params: dict[str, Any] | None = None) -> str:
    """Run one read-only Cypher query against the facility knowledge graph.

    The graph is a NARAD-convention RDF corpus loaded through neosemantics, which
    shapes the names a query must use:

    * Every node carries the label ``Resource`` plus the local name of its RDF
      class — ``ChannelBinding`` for a PV binding, ``Class`` for an ontology
      class — so ``(d:Resource)`` matches devices and ``(b:ChannelBinding)``
      narrows to bindings.
    * Relationship types are UPPERCASED: ``HASBINDING``, ``READSSIGNAL``,
      ``WRITESSIGNAL``, ``SUBCLASSOF``, ``TYPE``. Writing ``hasBinding`` matches
      nothing and returns zero rows rather than an error.
    * Properties keep their original spelling: ``uri`` (every node's identity),
      ``fullPv``, ``sourceName``, ``sectionCode``, ``sPositionM``,
      ``ordinalInSection``, ``confidence``.

    Pass values through ``params`` as ``$name`` placeholders rather than pasting
    them into the query text — ``MATCH (d:Resource {sectionCode: $section})``
    with ``params={"section": "SR"}``. The curated examples are written that way,
    so an example plus its parameter set runs unedited.

    Read-only in both directions: a query that tries to write is refused by the
    read transaction, and extension procedures, functions and CSV import are
    refused before the query is sent.

    Results are bounded. At most ``services.graphdb.query_max_rows`` rows come
    back and ``truncated`` says whether more matched; the store cancels a query
    that outlives ``services.graphdb.query_timeout_s``. A truncated or timed-out
    answer means narrow the query — add a filter, bound the traversal, or
    aggregate — rather than retry it unchanged.

    Start from ``example_queries`` and adapt the closest one; call ``get_schema``
    when you need a label, relationship type or property the examples do not use.

    Args:
        query: The Cypher to run. Read-only clauses only.
        params: Query parameters as a JSON object keyed by placeholder name
            without the ``$``, or omitted when the query takes none.

    Returns:
        JSON object with ``columns`` (the RETURN names, in order), ``rows`` (one
        object per record), ``row_count`` and ``truncated``. A truncated result
        also carries ``guidance`` naming the row cap that cut it.
    """
    try:
        if not query or not query.strip():
            return make_error(
                "validation_error",
                "Empty Cypher query.",
                [
                    "Pass a read-only Cypher query, e.g. "
                    "'MATCH (d:Resource) RETURN d.sourceName LIMIT 10'.",
                    "example_queries lists runnable queries for the common question shapes.",
                ],
            )

        if params is not None and not isinstance(params, dict):
            return make_error(
                "validation_error",
                f"`params` must be a JSON object keyed by parameter name, got "
                f"{type(params).__name__}.",
                [
                    'Pass parameters as an object, e.g. {"section": "SR"} for a query '
                    "referencing $section.",
                    "Omit `params` entirely when the query has no $placeholders.",
                ],
            )

        try:
            vet_query(query)
        except GateRefusal as exc:
            return make_error("validation_error", str(exc), list(_GATE_SUGGESTIONS))

        ctx = get_server_context()
        result = ctx.run_read(query, params)

        # An empty answer from an empty store is a seeding gap, not a query that
        # matched nothing — say so, since no rewording of the query fixes it.
        if not result.rows and ctx.is_empty():
            return make_error(
                "no_results",
                _EMPTY_GRAPH_MESSAGE,
                [f"Seed it with `{GRAPHDB_SEED_COMMAND}`."],
            )

        payload: dict[str, Any] = {
            "columns": list(result.rows[0]) if result.rows else [],
            "rows": result.rows,
            "row_count": len(result.rows),
            "truncated": result.truncated,
        }
        if result.truncated:
            payload["guidance"] = [
                f"Result was cut at {QUERY_MAX_ROWS_CONFIG_KEY} = {ctx.query_max_rows} rows; "
                "add a LIMIT, aggregate, or narrow the MATCH."
            ]

        return json.dumps(payload, default=str)

    except GraphStoreError as exc:
        logger.warning("read_cypher failed: %s", exc)
        return make_error(exc.error_type, str(exc), exc.suggestions, details=exc.details)
    except ToolError:
        raise
    except Exception as exc:
        logger.exception("read_cypher failed")
        return make_error(
            "internal_error",
            f"Graph query failed: {exc}",
            ["Check the MCP server logs for details."],
        )
