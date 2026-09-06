"""MCP tool: search_channels — keyword and facet lookup over the search index.

PROMPT-PROVIDER: this tool's docstring is a static prompt visible to the agent.
  Facility-customizable: section and system code spellings, signal names.

The other tools on this server ask the graph store a Cypher question. This one
reads the flat DuckDB index ``osprey build`` derives from the same corpus, which
is what makes "find the addresses that mention X" a scan of a few thousand rows
instead of a traversal the agent has to write. Search semantics are the index's
(:meth:`~osprey.services.channel_finder.graph_index.reader.GraphIndex.search`),
so this tool and the explorer's own page answer the same question the same way.

The index is opened once per process and held: it is a read-only file, the open
costs a driver import and a meta read, and a per-call open would pay both on
every keyword the agent tries. Only a successful open is cached — an absence is
re-probed, so a server that started before ``osprey build`` ran answers from the
index as soon as one exists.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Literal, get_args

from fastmcp.exceptions import ToolError

from osprey.deployment.graphdb_service import (
    GRAPHDB_BUILD_INDEX_COMMAND,
    GRAPHDB_INDEX_PATH_CONFIG_KEY,
    GRAPHDB_TTL_PATH_CONFIG_KEY,
    UNRESOLVED_INDEX_PATH_REMEDY,
    graph_corpus_configured,
    resolve_graph_index_path,
    unresolved_index_path_detail,
)
from osprey.services.channel_finder.graph_index.reader import (
    DEFAULT_PAGE_SIZE,
    EDGE_READS,
    EDGE_WRITES,
    GraphIndex,
    GraphIndexAbsence,
    open_graph_index,
)
from osprey.utils.workspace import load_osprey_config

from ..server import make_error, mcp

logger = logging.getLogger("osprey.mcp_server.channel_finder_graph.tools.search_channels")

#: Rows one page holds, as the reader defines it — the same number the HTTP
#: finder route pages by. Taken from the reader rather than from the route: the
#: route lives in :mod:`osprey.interfaces.channel_finder.database_api`, and
#: importing that module would pull FastAPI and the whole web app into an MCP
#: server process that serves no HTTP.
PAGE_SIZE = DEFAULT_PAGE_SIZE

#: How many values each facet list carries. The explorer's rail draws up to
#: five hundred; an agent reading a facet is deciding which filter to try next,
#: and ten is what fits that decision without spending the answer's budget on a
#: list it will not read.
FACET_CAP = 10

#: The directions a binding is filtered by, as the index, the rail and
#: ``directionOf`` in ``graph-finder-render.js`` all spell them. Written as a
#: ``Literal`` so the four values reach the tool's JSON schema, where an agent
#: reads them off the parameter instead of guessing and being refused.
Direction = Literal["R", "W", "RW", "none"]

#: The same four values as a tuple, for the hand validation below. A direct
#: caller is not held to the annotation, so the check stays; deriving the tuple
#: from the annotation is what keeps the two from drifting.
DIRECTIONS: tuple[str, ...] = get_args(Direction)

#: The opened index, held for the process's lifetime. ``None`` until the first
#: call, and left as ``None`` whenever the open failed, so an absence is
#: answered again rather than remembered.
_INDEX: GraphIndex | None = None

#: Serialises the first open. FastMCP runs a sync tool in a threadpool, so two
#: calls can reach an unopened index at once; without this both would open a
#: connection and one of them would be dropped on the floor still open.
_INDEX_LOCK = threading.Lock()


def _reset_index() -> None:
    """Close and forget the held index.

    Not part of the tool's contract — the server holds the index until the
    process ends. Tests call it to make each of them open its own.
    """
    global _INDEX
    with _INDEX_LOCK:
        if _INDEX is not None:
            _INDEX.close()
        _INDEX = None


def _absence_suggestions(absence: GraphIndexAbsence, config: dict[str, Any]) -> list[str]:
    """Remedies for an index that could not be opened.

    Three different gaps end here. An index written under another schema version
    is a stale file rather than a missing one, so the remedy is a rebuild by
    this version. A project with no corpus configured has nothing to build an
    index *from*, so telling it to run the build names the wrong step. A project
    that has a corpus is one build away from an answer.

    Args:
        absence: What :func:`open_graph_index` said about the path.
        config: The loaded project config, read only for the corpus key.

    Returns:
        The suggestion list the error envelope carries.
    """
    if absence.reason == "schema_mismatch":
        return [
            "The index was built under another schema version and must be rebuilt "
            f"by this one: run `{GRAPHDB_BUILD_INDEX_COMMAND}`.",
            "read_cypher answers the same questions against the graph store while "
            "there is no index.",
        ]

    if not graph_corpus_configured(config):
        return [
            f"No corpus is configured: set {GRAPHDB_TTL_PATH_CONFIG_KEY} to the facility's "
            "Turtle file, then build the index.",
            "read_cypher answers the same questions against the graph store while "
            "there is no index.",
        ]
    return [
        f"Build the index with `{GRAPHDB_BUILD_INDEX_COMMAND}`.",
        "`osprey build` renders the project and builds the index in one step.",
        f"{GRAPHDB_INDEX_PATH_CONFIG_KEY} says where the index is read from; the build "
        "writes it to the same place.",
        "read_cypher answers the same questions against the graph store while there is no index.",
    ]


def _get_index() -> GraphIndex:
    """Return the held index, opening it on first use.

    Raises:
        ToolError: The ``service_unavailable`` envelope, when the index is not
            there, cannot be read, or was written by another schema version.
    """
    global _INDEX
    if _INDEX is not None:
        return _INDEX

    # Checked again under the lock: a threadpool can put two first calls here at
    # once, and the loser must take the winner's index rather than open a second
    # connection that nothing would ever close.
    with _INDEX_LOCK:
        if _INDEX is not None:
            return _INDEX

        config = load_osprey_config() or {}
        # No config_dir: the render is found through OSPREY_CONFIG, which is how
        # an in-container consumer resolves it, and this server is one.
        try:
            path = resolve_graph_index_path(config)
        except ValueError as exc:
            # A malformed key is a config typo, not a server fault: say which key
            # and stop, rather than reporting it as an internal failure.
            make_error(
                "service_unavailable",
                unresolved_index_path_detail(exc),
                [UNRESOLVED_INDEX_PATH_REMEDY],
            )
        opened = open_graph_index(path)
        if isinstance(opened, GraphIndexAbsence):
            logger.warning("search_channels: no usable index at %s (%s)", path, opened.reason)
            make_error(
                "service_unavailable",
                opened.detail,
                _absence_suggestions(opened, config),
                details={"reason": opened.reason, "path": str(opened.path)},
            )
        _INDEX = opened
        return _INDEX


def _direction(edges: list[str]) -> str:
    """Return what a binding does, derived from its graph edges.

    The same derivation ``directionOf`` in ``graph-finder-render.js`` makes and
    the same spellings the ``dir`` facet counts under, so the value on a row is
    a value the ``direction`` argument filters on.

    Args:
        edges: The row's ``edges`` list, as the index stores it.

    Returns:
        ``R`` for a readback, ``W`` for a setpoint, ``RW`` for both, and
        ``none`` for a binding with no reading or writing edge at all.
    """
    names = {str(edge).upper() for edge in edges}
    reads = EDGE_READS in names
    writes = EDGE_WRITES in names
    if reads and writes:
        return "RW"
    if reads:
        return "R"
    if writes:
        return "W"
    return "none"


def _row(record: dict[str, Any]) -> dict[str, Any]:
    """Shape one index row into what the agent reads.

    ``device_uri`` and ``edges`` are dropped: the URI is the device card's
    argument, which this tool does not hand out, and the edges are what
    ``direction`` already says in one word.

    Args:
        record: One row of the search payload.

    Returns:
        The row, keyed as :func:`search_channels` documents it.
    """
    return {
        "fullPv": record.get("fullPv"),
        "description": record.get("description"),
        "device": record.get("device"),
        "section": record.get("section"),
        "system": record.get("system"),
        "direction": _direction(list(record.get("edges") or [])),
        "signals": list(record.get("signals") or []),
    }


@mcp.tool()
def search_channels(
    query: str = "",
    section: str | None = None,
    system: str | None = None,
    class_uri: str | None = None,
    signal: str | None = None,
    direction: Direction | None = None,
    page: int = 1,
) -> str:
    """Find control-system addresses by keyword and facet.

    This is the lookup tool: give it words an address, its description, its
    device, its signals or its device classes would contain, and it answers the
    matching channels with the facets that narrow them. Use it for "what is the
    PV for X", "which channels mention Y", "list the current setpoints in
    section Z". Use ``read_cypher`` instead for structural questions — what a
    device is connected to, how classes relate, everything one device carries —
    because those are traversals, and this tool only matches text and filters.

    Every token in ``query`` must match, and matching is case-insensitive over
    one text field per channel: its address, its description, its device name,
    its signal names and its class names and alternative labels. So
    ``"qf1 setpoint"`` keeps only channels whose text carries both words, in
    any order and in any of those fields.

    The five filters are ANDed with each other and with the query. ``class_uri``
    rolls its subclasses up, so filtering on an abstract class keeps every
    device under it. A filter takes one value; to ask about two sections, run
    the search twice or drop the filter and read the ``section`` facet.

    Results are bounded and paged: fifty rows per page, ``total`` matches in
    ``pages`` pages. Ask for the next page rather than rerunning the same query
    when a page is not enough, and narrow with a facet when ``total`` is large —
    the facets say which value would cut it most.

    Args:
        query: Words that must all appear in a channel's text. Empty matches
            every channel, which is how a facet-only search is asked.
        section: One section code to keep, e.g. ``"SR"``. A channel whose
            device is placed in no section is never kept by this filter.
        system: One system code to keep, e.g. ``"MAG"``.
        class_uri: One device-class URI to keep, as the ``class`` facet spells
            it. Subclasses of it are kept too.
        signal: One semantic signal name to keep, as the ``signal`` facet
            spells it.
        direction: One of ``R`` (readback), ``W`` (setpoint), ``RW`` (both) or
            ``none`` (a channel bound to no signal).
        page: 1-based page number.

    Returns:
        JSON object with ``total`` matching channels over ``devices`` distinct
        devices, the ``page`` of ``rows`` and how many ``pages`` there are, and
        ``facets`` — ``section``, ``system``, ``class``, ``signal`` and ``dir``.
        Each facet lists its ten most frequent values as ``{value, count}``
        entries ordered by count, and more values may exist beyond those ten:
        ``truncated`` is true when at least one facet was cut off that way. Each
        row carries ``fullPv`` (the address), ``description``, ``device``,
        ``section``, ``system``, ``direction`` and ``signals`` as ``{uri, name}``
        entries.
    """
    try:
        if direction is not None and direction not in DIRECTIONS:
            make_error(
                "validation_error",
                f"`direction` must be one of {', '.join(DIRECTIONS)}, not {direction!r}.",
                [
                    "R is a readback, W a setpoint, RW both, and none a channel bound "
                    "to no signal.",
                    "Omit `direction` to search every direction at once.",
                ],
            )

        if page < 1:
            make_error(
                "validation_error",
                f"`page` is 1-based and must be at least 1, not {page}.",
                ["Pass page=1 for the first page of results."],
            )

        index = _get_index()
        payload = index.search(
            tokens=query.lower().split(),
            sections=[section] if section else [],
            systems=[system] if system else [],
            cls=class_uri or None,
            signals=[signal] if signal else [],
            dirs=[direction] if direction else [],
            skip=(page - 1) * PAGE_SIZE,
            page_size=PAGE_SIZE,
            facet_cap=FACET_CAP,
        )

        return json.dumps(
            {
                "total": payload["total"],
                "devices": payload["devices"],
                "page": payload["page"],
                "pages": payload["pages"],
                "rows": [_row(record) for record in payload["rows"]],
                "facets": payload["facets"],
                "truncated": payload["truncated"],
            },
            default=str,
        )

    except ToolError:
        raise
    except Exception as exc:
        logger.exception("search_channels failed")
        make_error(
            "internal_error",
            f"Channel search failed: {exc}",
            ["Check the MCP server logs for details."],
        )
