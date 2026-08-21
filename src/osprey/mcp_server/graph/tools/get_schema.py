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
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastmcp.exceptions import ToolError

from osprey.deployment.graphdb_service import GRAPHDB_SEED_COMMAND
from osprey.mcp_server.graph.server import make_error, mcp
from osprey.mcp_server.graph.server_context import GraphStoreError, get_server_context
from osprey.services.facility_knowledge.seeder import NARAD_PREFIXES

logger = logging.getLogger("osprey.mcp_server.graph.tools.get_schema")

#: Nodes read per label when collecting that label's property names.
#:
#: A bound rather than a full scan: the answer is a vocabulary for writing
#: queries, not a census, and every device of a class carries the same keys. Two
#: hundred is far past the point where a NARAD corpus stops revealing new ones
#: while staying a cheap query on a corpus of any size.
SCHEMA_SAMPLE_SIZE = 200

#: Labels holding store state rather than facility knowledge: neosemantics' graph
#: config and namespace-prefix nodes, and the seeder's own marker.
BOOKKEEPING_LABELS = frozenset({"_GraphConfig", "_NsPrefDef", "_OspreySeed"})

#: Property names that belong to the bookkeeping nodes above. Filtered on top of
#: the label exclusion as belt-and-braces: were one of these ever to land on a
#: knowledge node, listing it would invite the agent to query the seed marker.
BOOKKEEPING_PROPERTIES = frozenset({"sha256", "seededAt", "kind"})

_LABELS_CYPHER = "CALL db.labels() YIELD label RETURN label ORDER BY label"

_RELATIONSHIP_TYPES_CYPHER = (
    "CALL db.relationshipTypes() YIELD relationshipType "
    "RETURN relationshipType ORDER BY relationshipType"
)

_NAMING_NOTE = (
    "n10s applyNeo4jNaming: relationship types are uppercased (HASBINDING, "
    "READSSIGNAL, WRITESSIGNAL, SUBCLASSOF, TYPE); rdf:type is both a label and "
    "a TYPE edge to a Class node"
)

_EMPTY_MESSAGE = "The graph holds no Resource nodes — it is bootstrapped but not seeded."


def _is_reportable_label(label: str) -> bool:
    """Whether *label* names facility knowledge rather than store bookkeeping.

    Underscore-prefixed labels are excluded as a class, which already covers
    :data:`BOOKKEEPING_LABELS`; the named set is kept so the exclusion stays
    legible and survives a bookkeeping label that does not follow the
    convention.
    """
    return not label.startswith("_") and label not in BOOKKEEPING_LABELS


def _sample_cypher(label: str) -> str:
    """Build the bounded property-name sample for one label.

    The label is interpolated because Cypher takes no parameter in label
    position. Callers must have rejected any label containing a backtick first —
    that is the only character that could break out of the quoting.
    """
    return (
        f"MATCH (n:`{label}`) WITH n LIMIT $k UNWIND keys(n) AS key "
        "RETURN DISTINCT key ORDER BY key"
    )


def _column(rows: list[dict[str, Any]], key: str) -> list[str]:
    """Pull one column out of query rows, dropping nulls."""
    return [str(row[key]) for row in rows if row.get(key) is not None]


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

        labels = [
            label
            for label in _column(ctx.run_read(_LABELS_CYPHER).rows, "label")
            if _is_reportable_label(label)
        ]
        relationship_types = _column(
            ctx.run_read(_RELATIONSHIP_TYPES_CYPHER).rows, "relationshipType"
        )

        properties_by_label: dict[str, list[str]] = {}
        for label in labels:
            if "`" in label:
                # Unquotable in label position, so it cannot be sampled safely.
                # Left out of properties_by_label entirely rather than mapped to
                # an empty list, which would claim the label carries no
                # properties; the label itself still appears in ``labels``.
                logger.warning("Skipping property sample for label with a backtick: %r", label)
                continue
            keys = _column(
                ctx.run_read(_sample_cypher(label), {"k": SCHEMA_SAMPLE_SIZE}).rows, "key"
            )
            properties_by_label[label] = [key for key in keys if key not in BOOKKEEPING_PROPERTIES]

        return json.dumps(
            {
                "labels": labels,
                "relationship_types": relationship_types,
                "properties_by_label": properties_by_label,
                "prefixes": dict(NARAD_PREFIXES),
                "naming": {"note": _NAMING_NOTE, "sample_size": SCHEMA_SAMPLE_SIZE},
            }
        )

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
