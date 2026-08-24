"""MCP tool: get_schema — the labels, relationship types and properties in the graph.

PROMPT-PROVIDER: this tool's response is the vocabulary an agent writes Cypher
against. Unlike ``capabilities`` and ``example_queries`` it reads the live store,
so it is also the first call that reports an unreachable or unseeded graph.

Two deliberate departures from the obvious implementation, both because the
corpus carries bookkeeping the agent must never see or query:

* Property names come from **sampling nodes per label**, never from
  ``db.propertyKeys()``. That procedure returns every key registered anywhere in
  the database, including the seed marker's ``sha256``/``seededAt`` and
  neosemantics' own config keys, with no way to tell which label owns which —
  so its answer is both wrong and leaky.
* Labels neosemantics and the seeder use for their own state are dropped, along
  with anything else underscore-prefixed, so a future bookkeeping label is
  excluded the day it appears rather than the day someone remembers it.

Both rules live in
:mod:`osprey.services.facility_knowledge.seeder.prompt_snapshot`, whose
:func:`~osprey.services.facility_knowledge.seeder.prompt_snapshot.collect_schema`
this tool serves: the seed-time capture baked into the agent prompt and this
live answer are the same collection, so the exclusions cannot drift apart. This
module keeps aliases to the shared names so the vocabulary contract stays
importable from the tool that enforces it.
"""

from __future__ import annotations

import json
import logging

from fastmcp.exceptions import ToolError

from osprey.deployment.graphdb_service import GRAPHDB_SEED_COMMAND
from osprey.mcp_server.graph.server import make_error, mcp
from osprey.mcp_server.graph.server_context import GraphStoreError, get_server_context
from osprey.services.facility_knowledge.seeder.prompt_snapshot import (
    BOOKKEEPING_LABELS,
    BOOKKEEPING_PROPERTIES,
    collect_schema,
)

__all__ = [
    "BOOKKEEPING_LABELS",
    "BOOKKEEPING_PROPERTIES",
    "SCHEMA_SAMPLE_SIZE",
    "get_schema",
]

logger = logging.getLogger("osprey.mcp_server.graph.tools.get_schema")

#: Nodes read per label when collecting that label's property names.
#:
#: A bound rather than a full scan: the answer is a vocabulary for writing
#: queries, not a census, and every device of a class carries the same keys. Two
#: hundred is far past the point where a NARAD corpus stops revealing new ones
#: while staying a cheap query on a corpus of any size. The seed-time capture
#: passes ``None`` here instead — it runs once per seed and can afford the full
#: walk.
SCHEMA_SAMPLE_SIZE = 200

_EMPTY_MESSAGE = "The graph holds no Resource nodes — it is bootstrapped but not seeded."


@mcp.tool()
def get_schema() -> str:
    """Report the labels, relationship types, properties and prefixes this graph holds.

    Call this before writing Cypher. The facility knowledge graph is generated
    from an RDF corpus, so its vocabulary is not guessable: relationship types
    are uppercased by neosemantics, ``rdf:type`` appears both as a label and as
    an edge, and property names follow the source ontology rather than any
    naming convention you would invent. A query naming a label or property that
    does not exist returns zero rows rather than an error, so guessing produces
    confident wrong answers.

    Property lists are **sampled, not exhaustive**: each label's keys come from
    reading at most ``sample_size`` of its nodes. A key that only a handful of
    nodes carry can therefore be missing from the list — treat the lists as the
    vocabulary to start from, not as a schema to validate against. Labels and
    relationship types are complete.

    Store bookkeeping is excluded throughout: neosemantics' config and prefix
    nodes, the seed marker, and their properties are not part of the facility
    knowledge and querying them tells you nothing about the machine.

    Returns:
        JSON object with ``labels`` (sorted list), ``relationship_types``
        (sorted list), ``properties_by_label`` (label → sorted property names),
        ``prefixes`` (the NARAD namespace prefix map, for reading and building
        ``uri`` values) and ``naming`` (``note`` on the neosemantics naming
        rules, and the ``sample_size`` behind the property lists).

        On an unseeded graph, a ``no_results`` error envelope naming the command
        that loads the corpus.
    """
    try:
        ctx = get_server_context()

        # Asked before any schema query: on an unseeded store every listing
        # below is empty, and "no labels" reads as a broken query rather than
        # as the seeding gap it is.
        if ctx.is_empty():
            return make_error(
                "no_results",
                _EMPTY_MESSAGE,
                [f"Seed it with `{GRAPHDB_SEED_COMMAND}`"],
            )

        # run_read enforces the read-only transaction and the query timeout, so
        # binding it as the collection's Cypher seam keeps both guarantees on
        # every schema query.
        schema = collect_schema(
            lambda cypher, params: ctx.run_read(cypher, params or {}).rows,
            sample_size=SCHEMA_SAMPLE_SIZE,
        )
        return json.dumps(schema)

    except GraphStoreError as exc:
        logger.warning("get_schema failed: %s", exc)
        return make_error(exc.error_type, str(exc), exc.suggestions, details=exc.details)
    except ToolError:
        raise
    except Exception as exc:
        logger.exception("get_schema failed")
        return make_error(
            "internal_error",
            f"Failed to read the graph schema: {exc}",
            ["Check the MCP server logs for details."],
        )
