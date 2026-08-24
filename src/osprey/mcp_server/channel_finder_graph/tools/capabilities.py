"""MCP tool: capabilities — what the graph paradigm expects of the channel finder.

PROMPT-PROVIDER: this tool's response is a static manifest read by the agent
before it plans a query. It deliberately does not dial the store, so a
successful response says nothing about whether the graph is reachable or seeded.

The main-agent graph server has a manifest of its own, and this is not a copy of
it. That one describes a general-purpose Cypher surface; this one describes a
*job*: an operator said a phrase, and the answer is the addresses a connector
can talk to. So it carries the four things an agent cannot guess and would
otherwise have to discover by running failing queries — how an address is
spelled and where it comes from, which prose exists to match a phrase against
and on which node, that synonyms are a list rather than a string, and how many
rows a query may return before it is cut off.

Everything here is a fact about the *corpus convention*, never about the data:
which corpus a deployment seeded is a store fact, so the manifest names the
property that carries it rather than answering it. The one live read is the row
cap and the query deadline, taken from the server context so a deployment that
raised either is not told the default. That read touches settings resolved at
startup, not the driver, which is what keeps this tool answerable when the graph
is unconfigured or down — the moment an agent most needs to be told what the
paradigm is.
"""

import json
import logging

from fastmcp.exceptions import ToolError

from osprey.deployment.graphdb_service import GRAPHDB_SEED_COMMAND
from osprey.mcp_server.channel_finder_graph.server import make_error, mcp
from osprey.mcp_server.graph.server_context import (
    QUERY_MAX_ROWS_CONFIG_KEY,
    QUERY_TIMEOUT_CONFIG_KEY,
    get_server_context,
)

logger = logging.getLogger("osprey.mcp_server.channel_finder_graph.tools.capabilities")

#: The channel-finder pipeline mode this server implements.
_PARADIGM = "graph"

#: The wire name every consumer reaches this server by, whichever paradigm is
#: selected. Reported so an agent reading the manifest can tell which of the
#: four channel finders it is talking to.
_SERVER_NAME = "channel-finder"

_DESCRIPTION = (
    "Find control-system channel addresses by writing read-only Cypher against "
    "the facility knowledge graph. There is no resolution API behind this "
    "server: the search strategy is yours, and the graph is the only source of "
    "an address."
)

#: Tools in reading order — the order the workflow below uses them.
_TOOLS: tuple[dict[str, str], ...] = (
    {
        "name": "example_queries",
        "purpose": "Return curated, runnable Cypher for the ways a channel question is asked.",
    },
    {
        "name": "get_schema",
        "purpose": "List the node labels, relationship types and property names this graph carries.",
    },
    {
        "name": "read_cypher",
        "purpose": "Run one read-only Cypher query and return the matching rows.",
    },
    {
        "name": "capabilities",
        "purpose": "Report this manifest: the paradigm, its tools, the graph's vocabulary and the query limits.",
    },
)

_WORKFLOW: tuple[str, ...] = (
    "Start at example_queries. Half the catalogue matches an operator's words "
    "against the prose the corpus carries; the other half walks the machine "
    "from a section, a device or an address you already have. Adapt the closest "
    "example instead of inventing a query shape.",
    "Call get_schema when no example fits, or when a query returned nothing and "
    "you need to know whether the property you matched on exists in this corpus "
    "at all. An empty result and a missing property look identical otherwise.",
    "Run the query with read_cypher. Writes, schema changes, extension "
    "procedures and CSV import are refused before the store is touched, so plan "
    "a read: MATCH / WHERE / RETURN and read-only CALL { … } subqueries.",
    "Hand back the addresses the graph returned. fullPv is the only source of a "
    "channel address — never assemble one from tokens, and never repair one by "
    "hand when a query came back empty.",
)

_ADDRESSES: dict[str, object] = {
    "property": "fullPv",
    "node_label": "ChannelBinding",
    "form": "colon-separated tokens, ending in the field the address reads or writes",
    "generated_grammar": {
        "tokens": ["ring", "system", "family", "device", "field", "subfield"],
        "example": "SR:MAG:DIPOLE:01:CURRENT:SP",
    },
    "notes": [
        "The six-token grammar is the generated demo corpus', and it is what "
        "fieldDescription and subfieldDescription explain. A corpus built from a "
        "real facility records whatever that facility calls its channels — "
        "als_gtb.ttl addresses are three tokens, such as GTL:BC1:Setpoint — so "
        "read the shape off the rows rather than assuming it.",
        "An address is a value in the graph, not a string you can derive. Take "
        "fullPv verbatim from a row; a token pattern that looks right for one "
        "corpus is a fabricated address in another.",
    ],
}

#: Prose a phrase search can match against, per node type. ``presence`` is the
#: part an agent cannot guess: these predicates are a *generator* convention, so
#: they exist in corpora built from a channel database and are simply absent
#: from one imported straight from a facility export.
_PROSE_IS_CORPUS_DEPENDENT = (
    "Present in generator-built corpora such as the demo machine, whose source "
    "channel database carries prose. A corpus imported from a facility export "
    "(als_gtb.ttl) carries none of them, and a phrase search against them "
    "returns no rows there — confirm with get_schema before concluding the "
    "phrase is wrong."
)

_DESCRIPTION_PREDICATES: dict[str, dict[str, object]] = {
    "binding": {
        "match": "(b:ChannelBinding)",
        "properties": ["description", "fieldDescription", "subfieldDescription"],
        "presence": _PROSE_IS_CORPUS_DEPENDENT,
    },
    "device": {
        "match": "(d:Resource)-[:HASBINDING]->(:ChannelBinding)",
        "properties": ["familyDescription", "systemDescription", "ringDescription"],
        "presence": _PROSE_IS_CORPUS_DEPENDENT,
    },
    "signal": {
        "match": "(:ChannelBinding)-[:READSSIGNAL|WRITESSIGNAL]->(s)",
        "properties": [],
        "presence": (
            "Never: a signal is keyed (FAMILY, FIELD, SUBFIELD) without a ring "
            "while the prose is ring-qualified, so it lives on the binding "
            "instead. Match a phrase on the binding and walk to the signal."
        ),
    },
}

_SYNONYMS: dict[str, object] = {
    "property": "altLabel",
    "node_label": "Class",
    "type": "list",
    "match": "ANY(l IN cls.altLabel WHERE toLower(l) = toLower($synonym))",
    "notes": [
        "The store is seeded with handleMultival: ARRAY for skos:altLabel, so "
        "the property is a list on every class even when it holds one entry. A "
        "scalar comparison or CONTAINS against altLabel matches nothing — that "
        "is a bug, not a shortcut.",
        "The comparison is on the whole synonym rather than a substring, which "
        "is what keeps 'pm' from matching 'pump'. Walk (sub:Class)-[:SUBCLASSOF*0..]->(cls) "
        "down from the class you matched so a broad word also finds every kind "
        "beneath it.",
    ],
}

_CORPUS: dict[str, object] = {
    "property": "facility",
    "carried_by": "device nodes",
    "read_with": (
        "MATCH (d:Resource) WHERE d.facility IS NOT NULL RETURN DISTINCT d.facility LIMIT 5"
    ),
    "notes": [
        "Which corpus a deployment seeded is a fact about the store, so this "
        "manifest cannot answer it. Every device carries it in facility; match "
        "that value case-insensitively against the corpus keys example_queries "
        "reports, and use that parameter set.",
        "Corpora differ in what they carry, not in the Cypher. The generated "
        "demo corpus comes from a channel database that carries prose, so it "
        "has every description predicate above; a corpus with no source "
        "database has none of them, and on that corpus a phrase search returns "
        "no rows while the structural queries still work.",
    ],
}

_NOTES: tuple[str, ...] = (
    "Read-only: writes, schema changes, extension procedures (apoc.*, n10s.*, "
    "gds.*, dbms.*) and LOAD CSV are refused before the query reaches the store.",
    "Every result is bounded by the row cap and the deadline in limits. A "
    "truncated result means narrow the query — add a WHERE, aggregate, or "
    "return fewer columns — rather than run it again unchanged.",
    "This manifest is static and never dials the store, so it is not a health "
    "check: a successful response says nothing about whether the graph is "
    "reachable or seeded. read_cypher and get_schema report that.",
    f"An empty graph is a seeding gap, not a query bug — run `{GRAPHDB_SEED_COMMAND}` "
    "to load the corpus.",
)


@mcp.tool()
def capabilities() -> str:
    """Report how this channel finder searches and what it needs from you.

    Returns the paradigm and the tools it offers, the order to use them in, and
    the corpus conventions a query has to respect: where an address comes from,
    which prose each kind of node carries for a phrase search, that class
    synonyms are a list, and how a deployment says which corpus it seeded. The
    row cap and the query deadline are this deployment's, with the config keys
    that set them.

    Does NOT require graph connectivity, so this is *not* a health check: a
    successful response says nothing about whether the store is reachable or
    seeded. Run ``read_cypher`` or ``get_schema`` for that.

    Returns:
        JSON object with ``paradigm``, ``server``, ``description``, ``tools``
        (list of ``{name, purpose}``), ``workflow``, ``addresses``,
        ``description_predicates``, ``synonyms``, ``corpus``, ``limits`` and
        ``notes``.
    """
    try:
        context = get_server_context()

        return json.dumps(
            {
                "paradigm": _PARADIGM,
                "server": _SERVER_NAME,
                "description": _DESCRIPTION,
                "tools": [dict(tool) for tool in _TOOLS],
                "workflow": list(_WORKFLOW),
                "addresses": _ADDRESSES,
                "description_predicates": _DESCRIPTION_PREDICATES,
                "synonyms": _SYNONYMS,
                "corpus": _CORPUS,
                "limits": {
                    "timeout_s": context.query_timeout_s,
                    "timeout_config_key": QUERY_TIMEOUT_CONFIG_KEY,
                    "max_rows": context.query_max_rows,
                    "max_rows_config_key": QUERY_MAX_ROWS_CONFIG_KEY,
                },
                "notes": list(_NOTES),
            }
        )

    except ToolError:
        raise
    except Exception as exc:
        logger.exception("capabilities failed")
        return make_error(
            "internal_error",
            f"Failed to report graph channel-finder capabilities: {exc}",
            ["Check the MCP server logs for details."],
        )
