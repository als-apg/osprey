"""MCP tool: example_queries — the channel finder's curated Cypher starting points.

PROMPT-PROVIDER: this tool's response is a static catalogue read by the agent
before it writes a query. Like ``capabilities`` it deliberately does not dial the
store, so it answers while the graph is down and a successful response says
nothing about whether the corpus is reachable or seeded.

It serves this paradigm's own catalogue — the channel-finding one, every query
about reaching a ``fullPv`` or starting from one — not the main-agent graph
server's survey of the store. Half of these examples search prose the generator
writes onto a corpus, so the notes say what happens on a corpus that carries
none: those examples return no rows, which must not be read as an empty
machine.
"""

import json
import logging

from fastmcp.exceptions import ToolError

from ..server import make_error, mcp
from .examples_data import EXAMPLE_QUERIES

logger = logging.getLogger("osprey.mcp_server.channel_finder_graph.tools.example_queries")

_NOTES: tuple[str, ...] = (
    "Each example carries its parameters separately from its Cypher. Pass the "
    "chosen parameter dict to read_cypher as `params` — never paste values into "
    "the query text.",
    "Each parameter set holds framework-default values that exist in the "
    "shipped demo machine. This tool never dials the store, so it cannot "
    "substitute this corpus's values; the *Graph at Hand* section of your "
    "prompt, baked at seed time, may carry the same examples with values "
    "captured from this corpus — prefer those. Failing both, take values "
    "from the rows the structural examples return.",
    "The search-by-meaning examples read prose the generator writes — the "
    "description predicates and the system token. A corpus imported straight "
    "from a facility export may carry none, and those examples then return no "
    "rows: confirm with get_schema before reading that as an empty machine.",
    "The catalogue reads from the loosest way in to the tightest: the earlier "
    "examples turn an operator's words into addresses, the later ones start "
    "from a class, a section, a device or an address that is already known.",
    "Every example already ends in a LIMIT, so an example that comes back "
    "truncated was truncated by the server's row cap, not by the query.",
    "Results are bounded by services.graphdb.query_max_rows; a truncated "
    "result means narrow the query rather than retry it unchanged.",
)


@mcp.tool()
def example_queries() -> str:
    """Return curated, runnable Cypher examples for finding channel addresses.

    Read this before writing Cypher against the facility knowledge graph. The
    examples are written against the real shape of the corpus — node labels,
    relationship types and property spellings that actually exist — so adapting
    one is reliable where inventing a query from scratch is guesswork. They fall
    into two halves:

    * *search by meaning* — match an operator's words against the prose the
      corpus carries: an address' own description, what its field and subfield
      mean, what a device family does, which system it belongs to;
    * *search by structure* — start from something already known: a class and
      everything under it, a whole section in beamline order, one device's
      addresses split into reads and writes, one address back to its device.

    Start from the closest example and change it — swap a parameter value, add a
    WHERE clause, widen or narrow the LIMIT — rather than composing a new query
    shape. Each example's ``description`` says what its parameters mean and how
    to vary them; ``get_schema`` lists the labels and relationship types if you
    need to go further than the examples reach.

    Corpora differ in what prose they carry: the search-by-meaning half reads
    predicates the generator writes, so on a corpus imported from a facility
    export it may return nothing while the structural half still answers.

    Does NOT require graph connectivity, so this is *not* a health check: a
    successful response says nothing about whether the store is reachable or
    seeded.

    Returns:
        JSON object with ``count``, ``examples`` (each ``{key, title,
        description, cypher, parameters}``, where ``parameters`` maps the
        query's placeholders to values that exist in the shipped demo machine)
        and ``notes`` (list of strings on how to run them).
    """
    try:
        examples = [
            {
                "key": example.key,
                "title": example.title,
                "description": example.description,
                "cypher": example.cypher,
                "parameters": dict(example.parameters),
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
            f"Failed to list channel-finder example queries: {exc}",
            ["Check the MCP server logs for details."],
        )
