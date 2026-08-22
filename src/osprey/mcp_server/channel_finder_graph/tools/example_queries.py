"""MCP tool: example_queries — the channel finder's curated Cypher starting points.

PROMPT-PROVIDER: this tool's response is a static catalogue read by the agent
before it writes a query. Like ``capabilities`` it deliberately does not dial the
store, so it answers while the graph is down and a successful response says
nothing about whether the corpus is reachable or seeded.

It serves this paradigm's own catalogue — the channel-finding one, every query
about reaching a ``fullPv`` or starting from one — not the main-agent graph
server's survey of the store. The two differ in a way the payload has to carry:
half of these examples search prose that only the generated demo corpus has, so
an example states the corpora it cannot run against in ``not_applicable`` rather
than leaving a missing key for the caller to interpret. A corpus is therefore
either supplied with real values or named as an exclusion, never present with an
empty parameter set — which would read as "runs here, takes no values" and
return zero rows that look like an empty machine.
"""

import json
import logging

from fastmcp.exceptions import ToolError

from ..server import make_error, mcp
from .examples_data import CORPORA, EXAMPLE_QUERIES

logger = logging.getLogger("osprey.mcp_server.channel_finder_graph.tools.example_queries")

_NOTES: tuple[str, ...] = (
    "Each example carries its parameters separately from its Cypher. Pass the "
    "chosen parameter dict to read_cypher as `params` — never paste values into "
    "the query text.",
    "Pick the parameter set matching the corpus this deployment seeded: `als` "
    "for the shipped ALS corpus, `demo` for the generated demo machine. The "
    "Cypher is the same either way; only the values differ.",
    "An example that lists the seeded corpus in `not_applicable` cannot run "
    "there — it searches prose that corpus does not carry — so skip it instead "
    "of inventing values for it. Every corpus is either supplied with "
    "parameters or named as an exclusion; a supplied set is never empty.",
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

    Corpora differ in what prose they carry, so an example that cannot run
    against a given corpus says so: use the parameter set for the corpus this
    deployment seeded, and skip any example that names it in ``not_applicable``.

    Does NOT require graph connectivity, so this is *not* a health check: a
    successful response says nothing about whether the store is reachable or
    seeded.

    Returns:
        JSON object with ``count``, ``corpora`` (the corpora the catalogue is
        written against), ``examples`` (each ``{key, title, description, cypher,
        parameters, not_applicable}``, where ``parameters`` holds one value set
        per corpus the example runs against and ``not_applicable`` lists the
        corpora it does not) and ``notes`` (list of strings on how to run them).
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
                "not_applicable": list(example.not_applicable),
            }
            for example in EXAMPLE_QUERIES
        ]

        return json.dumps(
            {
                "count": len(examples),
                "corpora": list(CORPORA),
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
