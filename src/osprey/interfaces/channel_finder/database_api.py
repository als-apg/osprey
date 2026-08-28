"""Channel Finder Database REST API.

Exposes database operations via REST endpoints, adapting for each pipeline
type. The three file-backed paradigms (hierarchical, middle_layer, in_context)
are served by calling their database instances directly via app.state, avoiding
MCP server dependencies. The graph paradigm has no database file behind it: its
routes report the paradigm and point at the tools that read the store.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterable, Mapping
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from osprey.build.build_tiers import VALID_CHANNEL_FINDER_MODES
from osprey.deployment.graphdb_service import GRAPHDB_SEED_COMMAND, GRAPHDB_SERVICE_NAME
from osprey.mcp_server.graph.server_context import GraphStoreError
from osprey.registry.mcp import CHANNEL_FINDER_TOOLS_BY_PIPELINE

# The channel census comes from the shared census module rather than being
# spelled again, so the number the explorer shows is the number the benchmark
# harness counts; the relationship-vocabulary query comes from the seeder's
# schema snapshot, which is what bakes that same vocabulary into the agent
# prompt.
from osprey.services.channel_finder.graph_queries import GRAPH_CHANNEL_COUNT_CYPHER
from osprey.services.facility_knowledge.seeder.prompt_snapshot import RELATIONSHIP_TYPES_CYPHER

logger = logging.getLogger(__name__)

router = APIRouter()

#: The tools the graph paradigm serves, read from the registry that renders
#: them into the agent rather than spelled out again here, so the web UI can
#: only ever name the vocabulary the agent actually has.
GRAPH_PARADIGM_TOOLS: tuple[str, ...] = tuple(CHANNEL_FINDER_TOOLS_BY_PIPELINE["graph"])

#: Where a graph-mode reader is sent for the answers the file-backed paradigms
#: compute from a database. Reused verbatim by every route that has no
#: store-backed implementation, so the redirection reads the same everywhere.
_GRAPH_TOOL_HINT = (
    "Query the graph store with the read_cypher tool; get_schema describes what is in it."
)


def _graph_not_implemented(subject: str) -> HTTPException:
    """Return the 501 a route raises when graph mode has no store-backed answer.

    Not a 404: the route exists and the paradigm is served, there is simply no
    implementation of this particular question against a graph store yet. The
    detail names the tools that do answer it so the reply is a redirection
    rather than a dead end.
    """
    return HTTPException(
        status_code=501,
        detail=f"{subject} is not implemented for the graph paradigm. {_GRAPH_TOOL_HINT}",
    )


# ---------------------------------------------------------------------------
# Graph store queries
# ---------------------------------------------------------------------------

#: The class tree the explorer draws, one row per ``:Class`` node.
#:
#: ``rollup`` counts the devices that fall under a class *including* its
#: subclasses, which is what makes an abstract branch like ``Magnet`` show a
#: number even though nothing is typed directly as one. A "device" is defined
#: exactly as the curated example queries define it — a ``:Resource`` with at
#: least one channel binding — so the counts the explorer shows and the counts
#: the agent computes cannot disagree.
#:
#: The descent is bounded at ten hops rather than left unbounded: a corpus whose
#: ``SUBCLASSOF`` edges contain a cycle would otherwise walk forever, and no
#: real ontology nests device classes ten deep.
GRAPH_ONTOLOGY_CYPHER = """
MATCH (c:Class)
OPTIONAL MATCH (c)-[:SUBCLASSOF]->(p:Class)
WITH c, collect(DISTINCT p.uri) AS parents
OPTIONAL MATCH (sub:Class)-[:SUBCLASSOF*0..10]->(c)
OPTIONAL MATCH (d:Resource)-[:TYPE]->(sub)
WHERE (d)-[:HASBINDING]->(:ChannelBinding)
WITH c, parents, count(DISTINCT d) AS rollup
RETURN c.uri AS uri, c.altLabel AS altLabel, parents, rollup
ORDER BY uri
""".strip()

#: Devices in the store, counted the same way :data:`GRAPH_ONTOLOGY_CYPHER`
#: rolls them up: a bound ``:Resource``. Counted ``DISTINCT`` because a device
#: with several channels binds several times.
GRAPH_DEVICE_COUNT_CYPHER = (
    "MATCH (d:Resource)-[:HASBINDING]->(:ChannelBinding) RETURN count(DISTINCT d) AS n"
)

#: Semantic signals in the store — the readings and settings devices expose,
#: which is a different population from the channels that address them.
GRAPH_SIGNAL_COUNT_CYPHER = "MATCH (s:SemanticSignal) RETURN count(s) AS n"

#: How many sections of the machine the corpus covers. Devices without a
#: section code — anything not placed along the ring — are simply not counted.
GRAPH_SECTION_COUNT_CYPHER = (
    "MATCH (d:Resource) WHERE d.sectionCode IS NOT NULL RETURN count(DISTINCT d.sectionCode) AS n"
)

#: Row cap for the explorer's own reads, passed explicitly rather than left to
#: the store's default. That default bounds an *agent's* ad-hoc query, where a
#: couple of hundred rows is the point; a class tree is a fixed, small
#: population that must arrive whole or the drawn taxonomy silently loses
#: branches. Five hundred is well above any real ontology and still a bound.
_GRAPH_EXPLORE_MAX_ROWS = 500

#: Row cap for a census query. Each census aggregates to a single row, so
#: anything above one is already slack; ten leaves room for a query to grow a
#: grouping column without silently truncating, while still refusing to stream a
#: store that answers a count with a result set.
_GRAPH_COUNT_MAX_ROWS = 10

#: What an operator is told when the store answers but holds nothing. The
#: wording mirrors the read_cypher tool's empty-store envelope, so the web UI
#: and the agent name the same remedy.
_EMPTY_GRAPH_SUGGESTIONS = [
    f"The graph store is running but holds no corpus. Seed it with `{GRAPHDB_SEED_COMMAND}`.",
]

#: Detail and remedy for a graph-mode request that arrives with no store seam
#: at all — the app started without a graph context, so there is nothing to ask.
_NO_GRAPH_CONTEXT_DETAIL = "Graph store is not available."
_NO_GRAPH_CONTEXT_SUGGESTIONS = [
    f"Check that a 'services.{GRAPHDB_SERVICE_NAME}' block is configured and that the "
    "channel finder was started against it.",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _class_name(uri: str) -> str:
    """Return the display name of a class URI: its trailing fragment.

    Ontology URIs end in the class name after either a path separator or a
    fragment marker, and which one is used is the corpus author's choice rather
    than something the explorer should care about.

    Args:
        uri: The class URI as the store holds it.

    Returns:
        The text after the last ``/`` or ``#``, or the whole URI when it
        carries neither.
    """
    cut = max(uri.rfind("/"), uri.rfind("#"))
    return uri[cut + 1 :] if cut >= 0 else uri


def _prune_device_taxonomy(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Reduce raw ontology rows to the classes worth drawing as a device tree.

    The store's ``:Class`` nodes describe more than devices — signal and binding
    classes live in the same tree — and drawing all of them buries the taxonomy
    an operator came for. A class earns its place by either holding devices
    (its rollup is non-zero) or by being an abstract parent of a class that
    does; anything else is a leaf about something other than devices and is
    dropped.

    Args:
        rows: One mapping per ``:Class`` node, as
            :data:`GRAPH_ONTOLOGY_CYPHER` returns them — ``uri``, ``altLabel``,
            ``parents`` and ``rollup``.

    Returns:
        The surviving classes, each carrying its ``uri``, derived ``name``,
        ``altLabel`` list, ``parents`` list and ``rollup``, sorted by name.
    """
    materialised = list(rows)

    #: A class is abstract-but-wanted when another class declares it a parent.
    #: A row naming itself does not count, so a self-referential SUBCLASSOF edge
    #: cannot keep an otherwise empty class alive.
    parent_uris: set[str] = set()
    for row in materialised:
        own_uri = row.get("uri")
        for parent in row.get("parents") or []:
            if parent is not None and parent != own_uri:
                parent_uris.add(parent)

    kept: list[dict[str, Any]] = []
    for row in materialised:
        uri = row.get("uri")
        if uri is None:
            continue
        rollup = row.get("rollup") or 0
        if rollup == 0 and uri not in parent_uris:
            continue
        kept.append(
            {
                "uri": uri,
                "name": _class_name(uri),
                "altLabel": list(row.get("altLabel") or []),
                "parents": list(row.get("parents") or []),
                "rollup": rollup,
            }
        )

    kept.sort(key=lambda entry: (entry["name"], entry["uri"]))
    return kept


def _graph_error_payload(exc: GraphStoreError | None) -> tuple[int, dict[str, Any]]:
    """Turn a failed graph read into the status and body a route replies with.

    Every cause answers 503: whichever way the read failed — no config, store
    down, bad credential, query refused — the store did not serve this request,
    and the operator remedy travels in the body rather than in the status. The
    error's own :attr:`~GraphStoreError.error_type` and suggestions are copied
    verbatim, so the web UI shows the same remedy the agent is given.

    Args:
        exc: The error the store raised, or ``None`` when the app has no graph
            context at all and the read was never attempted.

    Returns:
        The HTTP status code and the response body.
    """
    if exc is None:
        return 503, {
            "detail": _NO_GRAPH_CONTEXT_DETAIL,
            "error_type": "service_unavailable",
            "suggestions": list(_NO_GRAPH_CONTEXT_SUGGESTIONS),
        }
    return 503, {
        "detail": str(exc),
        "error_type": exc.error_type,
        "suggestions": list(exc.suggestions),
    }


async def _read_device_taxonomy(ctx: Any) -> tuple[list[dict[str, Any]], bool]:
    """Read the store's class rows and reduce them to the device taxonomy.

    Shared by the two routes that need the tree — the ontology the explorer
    draws and the class count the statistics badge shows — so a change to how a
    class earns its place in the taxonomy cannot move one number without moving
    the other. The read runs off the event loop: the store's driver is
    synchronous, and awaiting it inline would stall every other request the app
    is serving for the length of the query.

    Args:
        ctx: The app's graph store context.

    Returns:
        The pruned taxonomy and whether the underlying read hit the row cap.

    Raises:
        GraphStoreError: Whatever the store raises when the read fails.
    """
    result = await asyncio.to_thread(
        ctx.run_read, GRAPH_ONTOLOGY_CYPHER, max_rows=_GRAPH_EXPLORE_MAX_ROWS
    )
    return _prune_device_taxonomy(result.rows), result.truncated


async def _serve_graph_read(request: Request, subject: str, read: Any) -> Any:
    """Answer a graph-paradigm request through the app's store context.

    The graph reads the app serves share everything around the read itself:
    where the context lives, what an app without one answers, and how a store
    failure travels. That contract lives here once, so the statistics and the
    ontology cannot drift into reporting the same broken store differently.

    A missing context means the app started without a usable store
    configuration: the read is never attempted, so there is no store error to
    report and the payload carries the configuration remedy instead. A
    :class:`GraphStoreError` means the store classified its own failure —
    unreachable, refused, timed out — and carries the operator remedy; both
    are copied verbatim rather than re-derived from the exception type here.

    Args:
        request: FastAPI request, carrying the app's graph context on its state.
        subject: What is being read, for the log lines.
        read: Coroutine function taking the context and returning the payload.

    Returns:
        Whatever *read* returns, or a 503 :class:`JSONResponse` carrying the
        remedy when the store could not serve the read.

    Raises:
        HTTPException: 500 when the read fails for a reason the store does not
            classify.
    """
    ctx = getattr(request.app.state, "graph_context", None)
    if ctx is None:
        status, payload = _graph_error_payload(None)
        return JSONResponse(status_code=status, content=payload)

    try:
        return await read(ctx)
    except GraphStoreError as exc:
        logger.warning("Graph %s read failed: %s", subject, exc)
        status, payload = _graph_error_payload(exc)
        return JSONResponse(status_code=status, content=payload)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to read the graph %s", subject)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def _graph_count(ctx: Any, cypher: str) -> int:
    """Run a census query off the event loop and return the number it counted.

    Args:
        ctx: The app's graph store context.
        cypher: A census query returning its single count as column ``n``.

    Returns:
        The count. A store that answers with no row at all — which a census
        over an empty label does on some drivers — counts as zero rather than
        failing the whole statistics read.

    Raises:
        GraphStoreError: Whatever the store raises when the read fails.
    """
    result = await asyncio.to_thread(ctx.run_read, cypher, max_rows=_GRAPH_COUNT_MAX_ROWS)
    if not result.rows:
        return 0
    return int(result.rows[0].get("n") or 0)


def _pipeline_type(request: Request) -> str:
    """Return the active pipeline type, or reject the request if it is not a paradigm.

    There is no default here: the paradigm the app resolved at startup is the
    only answer. Anything else — a paradigm name this build does not know, or
    no paradigm at all — is a configuration defect, and the route says so
    instead of quietly serving some other paradigm's data.
    """
    pipeline_type = getattr(request.app.state, "pipeline_type", None)
    if pipeline_type not in VALID_CHANNEL_FINDER_MODES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Active channel finder pipeline {pipeline_type!r} is not a known paradigm. "
                f"Set 'channel_finder.pipeline_mode' to one of: "
                f"{', '.join(VALID_CHANNEL_FINDER_MODES)}."
            ),
        )
    return pipeline_type


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class ValidateRequest(BaseModel):
    """Request body for channel validation."""

    channels: list[str]


class AddNodeRequest(BaseModel):
    """Request body for adding a hierarchical node."""

    level: str
    parent_selections: dict[str, str] = {}
    name: str
    description: str = ""


class EditNodeRequest(BaseModel):
    """Request body for editing a hierarchical node (name and/or description)."""

    level: str
    selections: dict[str, str] = {}
    old_name: str
    new_name: str | None = None
    description: str | None = None


class DeleteNodeRequest(BaseModel):
    """Request body for deleting a hierarchical node."""

    level: str
    selections: dict[str, str] = {}
    name: str


class EditExpansionRequest(BaseModel):
    """Request body for editing an instance-level expansion config."""

    level: str
    selections: dict[str, str] = {}
    pattern: str | None = None
    range_start: int | None = None
    range_end: int | None = None


class AddFamilyRequest(BaseModel):
    """Request body for adding a middle-layer family."""

    system: str
    family: str
    description: str = ""


class DeleteFamilyRequest(BaseModel):
    """Request body for deleting a middle-layer family."""

    system: str
    family: str


class AddMLChannelRequest(BaseModel):
    """Request body for adding a middle-layer channel."""

    system: str
    family: str
    field: str
    channel_name: str
    subfield: str | None = None


class DeleteMLChannelRequest(BaseModel):
    """Request body for deleting a middle-layer channel."""

    system: str
    family: str
    field: str
    channel_name: str
    subfield: str | None = None


class AddICChannelRequest(BaseModel):
    """Request body for adding an in-context channel."""

    channel_name: str
    address: str = ""
    description: str = ""


class UpdateICChannelRequest(BaseModel):
    """Request body for updating an in-context channel."""

    description: str | None = None
    address: str | None = None


# ---------------------------------------------------------------------------
# Common endpoints (all pipelines)
# ---------------------------------------------------------------------------


@router.get("/info")
async def get_info(request: Request):
    """Return pipeline type and pipeline-specific metadata."""
    # An unconfigured project has no paradigm at all — that is a reportable
    # state, not a request defect. The data routes still refuse through
    # ``_pipeline_type``; this route is how the UI learns what it is talking
    # to, so it must answer even when the answer is "nothing is configured".
    if getattr(request.app.state, "pipeline_type", None) is None:
        return {
            "pipeline_type": None,
            "available_pipelines": [],
            "graph_backed": False,
            "db_path": None,
            "metadata": {
                "error": (
                    "No channel-finder pipeline is configured. Set "
                    "'channel_finder.pipeline_mode' or configure a pipeline database."
                )
            },
        }
    pt = _pipeline_type(request)
    available = getattr(request.app.state, "available_pipelines", [pt])

    if pt == "graph":
        # Store-backed, so there is no database file to name and nothing local
        # to introspect. What the payload carries instead is the paradigm
        # itself: enough for the UI to draw the graph pane, and the tool names
        # that answer the questions a database file answers elsewhere.
        #
        # ``graph_store`` is the file-backed ``db_path``'s counterpart: where
        # the data lives and what was seeded into it. Both entries are reported
        # as ``None`` rather than withheld when unresolved, so the panel can
        # boot and say "no store" instead of failing to render at all when the
        # store is down or was never configured.
        ctx = getattr(request.app.state, "graph_context", None)
        return {
            "pipeline_type": pt,
            "available_pipelines": available,
            "graph_backed": True,
            "db_path": None,
            "tools": list(GRAPH_PARADIGM_TOOLS),
            "graph_store": {
                "uri": getattr(ctx, "uri", None) if ctx is not None else None,
                "ttl_filename": getattr(request.app.state, "graph_ttl_filename", None),
            },
            # The per-registry facility names are empty in graph mode; the one
            # the app resolved from config at startup is the right answer.
            "metadata": {"facility_name": getattr(request.app.state, "facility_name", "")},
        }

    info: dict = {
        "pipeline_type": pt,
        "available_pipelines": available,
        "graph_backed": False,
    }

    try:
        info["db_path"] = _get_db_path(request)
    except Exception:
        info["db_path"] = None

    try:
        db = _get_database(request)
        if pt == "hierarchical":
            info["metadata"] = {
                "hierarchy_levels": db.hierarchy_levels,
                "hierarchy_config": db.hierarchy_config,
                "naming_pattern": db.naming_pattern,
                "facility_name": _get_facility_name(request),
            }
        elif pt == "middle_layer":
            systems = db.list_systems()
            info["metadata"] = {"system_count": len(systems)}
        else:  # in_context
            stats = db.get_statistics()
            chunks = db.chunk_database(50)
            stats["total_chunks_at_50"] = len(chunks)
            stats["facility_name"] = _get_facility_name(request)
            info["metadata"] = stats

    except Exception as exc:
        logger.exception("Failed to get pipeline info")
        info["metadata"] = {"error": str(exc)}

    return info


class SwitchPipelineRequest(BaseModel):
    """Request body for switching the active pipeline type."""

    pipeline_type: str


@router.put("/pipeline")
async def switch_pipeline(request: Request, body: SwitchPipelineRequest):
    """Switch the active pipeline type at runtime (dev mode).

    Only allows switching to pipelines that were successfully initialized.
    A graph-mode app is answered before that check, so the reply names the
    paradigm and where its data lives instead of reporting an empty roster of
    alternatives — which is true but says nothing about why.
    """
    if _pipeline_type(request) == "graph":
        raise HTTPException(
            status_code=400,
            detail=(
                "Pipeline switching is not available for the graph paradigm; "
                "query the store with read_cypher."
            ),
        )

    available = getattr(request.app.state, "available_pipelines", [])
    if body.pipeline_type not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Pipeline '{body.pipeline_type}' not available. Available: {available}",
        )
    request.app.state.pipeline_type = body.pipeline_type
    logger.info("Switched active pipeline to %s", body.pipeline_type)
    return {"pipeline_type": body.pipeline_type}


@router.get("/statistics")
async def get_statistics(request: Request):
    """Return database statistics for the active pipeline."""
    pt = _pipeline_type(request)
    if pt == "graph":
        return await _serve_graph_read(request, "statistics", _read_graph_statistics)

    try:
        db = _get_database(request)
        if pt == "in_context":
            stats = db.get_statistics()
            chunks = db.chunk_database(50)
            stats["total_chunks_at_50"] = len(chunks)
            stats["facility_name"] = _get_facility_name(request)
            return stats
        else:
            return db.get_statistics()

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to get statistics")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/validate")
async def validate_channels(request: Request, body: ValidateRequest):
    """Validate channel names against the database."""
    pt = _pipeline_type(request)
    if pt == "graph":
        raise _graph_not_implemented("Channel validation")

    try:
        db = _get_database(request)
        if pt == "in_context":
            validation_results = db.validate_channels(body.channels)
            valid = db.get_valid_channels(validation_results)
            invalid = db.get_invalid_channels(validation_results)
            return {
                "total": len(body.channels),
                "valid_count": len(valid),
                "invalid_count": len(invalid),
                "valid_channels": valid,
                "invalid_channels": invalid,
                "results": validation_results,
            }
        else:  # hierarchical or middle_layer
            results = []
            valid_count = 0
            for ch in body.channels:
                is_valid = db.validate_channel(ch)
                results.append({"channel": ch, "valid": is_valid})
                valid_count += is_valid
            return {
                "results": results,
                "valid_count": valid_count,
                "invalid_count": len(body.channels) - valid_count,
                "total": len(body.channels),
            }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to validate channels")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Graph pipeline endpoints
# ---------------------------------------------------------------------------


async def _read_graph_statistics(ctx: Any) -> dict[str, int]:
    """Read the graph paradigm's answer to database statistics.

    The file-backed paradigms count rows in a database file; the graph counts
    populations in a store. Each census is a separate query rather than one
    aggregate, because the populations live under different labels and a single
    query joining them would multiply rows against each other.

    ``total_classes`` is deliberately not a census: it is the size of the same
    pruned taxonomy the explorer draws, so the badge and the tree cannot report
    a different number of device classes.

    Args:
        ctx: The app's graph store context.

    Returns:
        The five counts.

    Raises:
        GraphStoreError: Whatever the store raises when a read fails.
    """
    classes, _ = await _read_device_taxonomy(ctx)
    return {
        "total_devices": await _graph_count(ctx, GRAPH_DEVICE_COUNT_CYPHER),
        "total_channels": await _graph_count(ctx, GRAPH_CHANNEL_COUNT_CYPHER),
        "total_classes": len(classes),
        "total_signals": await _graph_count(ctx, GRAPH_SIGNAL_COUNT_CYPHER),
        "total_sections": await _graph_count(ctx, GRAPH_SECTION_COUNT_CYPHER),
    }


def _relationship_type_names(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    """Return the relationship-type names carried by schema-query rows.

    Args:
        rows: Rows as :data:`RELATIONSHIP_TYPES_CYPHER` returns them, each
            carrying a ``relationshipType``.

    Returns:
        The type names, in the order the store reported them, with any row
        missing the column dropped rather than rendered as a null entry.
    """
    names: list[str] = []
    for row in rows:
        name = row.get("relationshipType")
        if name:
            names.append(str(name))
    return names


@router.get("/graph/ontology")
async def graph_ontology(request: Request):
    """Return the device class tree and relationship vocabulary of the store.

    The graph paradigm's answer to the tree the file-backed paradigms read from
    a database file. Both reads run off the event loop: the store's driver is
    synchronous, and awaiting it inline would stall every other request the app
    is serving for the length of the query.

    A store that cannot be reached answers 503 carrying the error's own remedy
    rather than a bare status, and a store that answers with nothing is
    distinguished from one that is down — an empty corpus is a seeding gap, and
    the payload says which command closes it.

    Args:
        request: FastAPI request, carrying the app's graph context on its state.

    Returns:
        ``classes`` (the pruned device taxonomy), ``relationship_types``,
        ``truncated`` (either read hit the row cap), ``empty`` and
        ``suggestions``.

    Raises:
        HTTPException: 404 when the active paradigm is not the graph, 400 when
            no paradigm is configured at all, 500 when the read fails for a
            reason the store does not classify.
    """
    if _pipeline_type(request) != "graph":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")
    return await _serve_graph_read(request, "ontology", _read_graph_ontology)


async def _read_graph_ontology(ctx: Any) -> dict[str, Any]:
    """Read the class tree and relationship vocabulary the ontology route serves.

    Args:
        ctx: The app's graph store context.

    Returns:
        The ontology payload, in either its drawn or its empty-store shape.

    Raises:
        GraphStoreError: Whatever the store raises when a read fails.
    """
    classes, classes_truncated = await _read_device_taxonomy(ctx)
    relationships = await asyncio.to_thread(
        ctx.run_read, RELATIONSHIP_TYPES_CYPHER, max_rows=_GRAPH_EXPLORE_MAX_ROWS
    )

    # Nothing to draw has two very different causes. Ask the store which
    # one this is only when there is nothing to draw, so the common path
    # costs one round trip less.
    if not classes and await asyncio.to_thread(ctx.is_empty):
        return {
            "classes": [],
            "relationship_types": [],
            "truncated": False,
            "empty": True,
            "suggestions": list(_EMPTY_GRAPH_SUGGESTIONS),
        }

    return {
        "classes": classes,
        "relationship_types": _relationship_type_names(relationships.rows),
        "truncated": classes_truncated or relationships.truncated,
        "empty": False,
        "suggestions": [],
    }


# ---------------------------------------------------------------------------
# Hierarchical pipeline endpoints
# ---------------------------------------------------------------------------


@router.get("/explore/options")
async def explore_options(request: Request, level: str, selections: str | None = None):
    """Get available options at a hierarchy level.

    Args:
        request: FastAPI request.
        level: Hierarchy level name (e.g., "system", "device").
        selections: JSON-encoded dict of previous selections.
    """
    if _pipeline_type(request) != "hierarchical":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    try:
        db = _get_database(request)
        parsed_selections = json.loads(selections) if selections else None
        options = db.get_options_at_level(level, parsed_selections or {})
        return {"level": level, "options": options, "total": len(options)}

    except HTTPException:
        raise
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid selections JSON: {exc}") from exc
    except Exception as exc:
        logger.exception("Failed to get explore options")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/explore/build")
async def explore_build(request: Request, selections: str):
    """Build channel addresses from hierarchy selections.

    Args:
        request: FastAPI request.
        selections: JSON-encoded dict of hierarchy selections.
    """
    if _pipeline_type(request) != "hierarchical":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    try:
        db = _get_database(request)
        parsed_selections = json.loads(selections)
        channels = db.build_channels_from_selections(parsed_selections)
        valid = [ch for ch in channels if db.validate_channel(ch)]
        invalid = [ch for ch in channels if not db.validate_channel(ch)]
        return {
            "channels": channels,
            "total": len(channels),
            "valid": valid,
            "invalid": invalid,
            "valid_count": len(valid),
            "invalid_count": len(invalid),
        }

    except HTTPException:
        raise
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid selections JSON: {exc}") from exc
    except Exception as exc:
        logger.exception("Failed to build channels")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/explore/hierarchy-info")
async def explore_hierarchy_info(request: Request):
    """Get hierarchy structure information."""
    if _pipeline_type(request) != "hierarchical":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    try:
        db = _get_database(request)
        return {
            "hierarchy_levels": db.hierarchy_levels,
            "hierarchy_config": db.hierarchy_config,
            "naming_pattern": db.naming_pattern,
            "facility_name": _get_facility_name(request),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to get hierarchy info")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Middle layer pipeline endpoints
# ---------------------------------------------------------------------------


@router.get("/explore/systems")
async def explore_systems(request: Request):
    """List all systems in the channel database."""
    if _pipeline_type(request) != "middle_layer":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    try:
        db = _get_database(request)
        systems = db.list_systems()
        return {"systems": systems, "total": len(systems)}

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to list systems")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/explore/families")
async def explore_families(request: Request, system: str):
    """List device families in a system.

    Args:
        request: FastAPI request.
        system: System name (e.g., "SR" for Storage Ring).
    """
    if _pipeline_type(request) != "middle_layer":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    try:
        db = _get_database(request)
        families = db.list_families(system)
        return {"families": families, "total": len(families)}

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to list families")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/explore/fields")
async def explore_fields(
    request: Request,
    system: str,
    family: str,
    field: str | None = None,
):
    """Inspect fields of a device family.

    Args:
        request: FastAPI request.
        system: System name.
        family: Family name.
        field: Optional specific field to inspect.
    """
    if _pipeline_type(request) != "middle_layer":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    try:
        db = _get_database(request)
        fields = db.inspect_fields(system, family, field)
        return {"fields": fields}

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to inspect fields")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/explore/channels")
async def explore_channels(
    request: Request,
    system: str,
    family: str,
    field: str,
    subfield: str | None = None,
    sectors: str | None = None,
    devices: str | None = None,
):
    """Get channel names for a system/family/field path.

    Args:
        request: FastAPI request.
        system: System name.
        family: Family name.
        field: Field name (e.g., "Monitor", "Setpoint").
        subfield: Optional subfield name.
        sectors: Optional JSON-encoded list of sector numbers.
        devices: Optional JSON-encoded list of device numbers.
    """
    if _pipeline_type(request) != "middle_layer":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    try:
        db = _get_database(request)
        parsed_sectors = json.loads(sectors) if sectors else None
        parsed_devices = json.loads(devices) if devices else None
        channels = db.list_channel_names(
            system, family, field, subfield, parsed_sectors, parsed_devices
        )
        return {"channels": channels, "total": len(channels)}

    except HTTPException:
        raise
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid JSON in sectors or devices parameter: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Failed to list channels")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/explore/device-info")
async def explore_device_info(request: Request, system: str, family: str):
    """Get device arrangement info for a middle-layer family."""
    if _pipeline_type(request) != "middle_layer":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")
    try:
        db = _get_database(request)
        return db.get_device_info(system, family)
    except Exception as exc:
        logger.exception("Failed to get device info")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# In-context pipeline endpoints
# ---------------------------------------------------------------------------


@router.get("/channels")
async def get_channels(
    request: Request,
    chunk_idx: int | None = None,
    chunk_size: int = 50,
):
    """Get channels from the in-context database.

    Args:
        request: FastAPI request.
        chunk_idx: Optional chunk index (0-based). If omitted, returns all.
        chunk_size: Number of channels per chunk (default 50).
    """
    if _pipeline_type(request) != "in_context":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    try:
        db = _get_database(request)
        if chunk_idx is not None:
            chunks = db.chunk_database(chunk_size)
            if chunk_idx < 0 or chunk_idx >= len(chunks):
                raise HTTPException(
                    status_code=422,
                    detail=f"chunk_idx {chunk_idx} out of range (0-{len(chunks) - 1})",
                )
            chunk = chunks[chunk_idx]
            formatted = db.format_chunk_for_prompt(chunk)
            return {
                "chunk_idx": chunk_idx,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk),
                "channels": chunk,
                "formatted": formatted,
            }
        else:
            channels = db.get_all_channels()
            return {"channels": channels, "total": len(channels)}

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to get channels")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------


def _get_database(request: Request):
    """Get the database instance for the active pipeline type."""
    pt = _pipeline_type(request)
    databases = getattr(request.app.state, "databases", {})
    db = databases.get(pt)
    if db is None:
        raise HTTPException(status_code=503, detail=f"Database not available for pipeline '{pt}'")
    return db


def _get_db_path(request: Request) -> str:
    """Get the database file path for the active pipeline type."""
    return _get_database(request).db_path


def _get_facility_name(request: Request) -> str:
    """Get the facility name for the active pipeline type."""
    pt = _pipeline_type(request)
    facility_names = getattr(request.app.state, "facility_names", {})
    return facility_names.get(pt, "")


# ---------------------------------------------------------------------------
# Hierarchical CRUD endpoints
# ---------------------------------------------------------------------------


@router.post("/tree/node")
async def add_tree_node(request: Request, body: AddNodeRequest):
    """Add a new node at a hierarchy level."""
    if _pipeline_type(request) != "hierarchical":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    from osprey.services.channel_finder.core.base_database import DatabaseWriteError

    try:
        db = _get_database(request)
        return db.add_node(
            level=body.level,
            parent_selections=body.parent_selections,
            name=body.name,
            description=body.description,
        )
    except DatabaseWriteError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to add tree node")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.put("/tree/node")
async def edit_tree_node(request: Request, body: EditNodeRequest):
    """Edit a node's name and/or description at a hierarchy level."""
    if _pipeline_type(request) != "hierarchical":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    from osprey.services.channel_finder.core.base_database import DatabaseWriteError

    try:
        db = _get_database(request)
        return db.edit_node(
            level=body.level,
            selections=body.selections,
            old_name=body.old_name,
            new_name=body.new_name,
            description=body.description,
        )
    except DatabaseWriteError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to edit tree node")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete("/tree/node")
async def delete_tree_node(request: Request, body: DeleteNodeRequest):
    """Delete a node (and all descendants) at a hierarchy level."""
    if _pipeline_type(request) != "hierarchical":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    from osprey.services.channel_finder.core.base_database import DatabaseWriteError

    try:
        db = _get_database(request)
        return db.delete_node(
            level=body.level,
            selections=body.selections,
            name=body.name,
        )
    except DatabaseWriteError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to delete tree node")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/tree/impact")
async def tree_impact(request: Request, body: DeleteNodeRequest):
    """Preview the impact of deleting a hierarchy node."""
    if _pipeline_type(request) != "hierarchical":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    from osprey.services.channel_finder.core.base_database import DatabaseWriteError

    try:
        db = _get_database(request)
        impact = db.count_descendants(
            level=body.level,
            selections=body.selections,
            name=body.name,
        )
        return {
            "affected_channels": impact.get("channels", 0),
            "breakdown": {k: v for k, v in impact.items() if k != "channels"},
        }
    except DatabaseWriteError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to compute tree impact")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/tree/expansion")
async def get_tree_expansion(request: Request, level: str, selections: str | None = None):
    """Get the current expansion config for an instance-type level."""
    if _pipeline_type(request) != "hierarchical":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    from osprey.services.channel_finder.core.base_database import DatabaseWriteError

    try:
        parsed_selections = json.loads(selections) if selections else {}
        db = _get_database(request)
        return db.get_expansion(
            level=level,
            selections=parsed_selections,
        )
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid selections JSON: {exc}") from exc
    except DatabaseWriteError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to get expansion config")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.put("/tree/expansion")
async def edit_tree_expansion(request: Request, body: EditExpansionRequest):
    """Edit the expansion config for an instance-type level."""
    if _pipeline_type(request) != "hierarchical":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    from osprey.services.channel_finder.core.base_database import DatabaseWriteError

    try:
        db = _get_database(request)
        return db.edit_expansion(
            level=body.level,
            selections=body.selections,
            pattern=body.pattern,
            range_start=body.range_start,
            range_end=body.range_end,
        )
    except DatabaseWriteError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to edit expansion config")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Middle Layer CRUD endpoints
# ---------------------------------------------------------------------------


@router.post("/structure/family")
async def add_family(request: Request, body: AddFamilyRequest):
    """Add a new family to a system."""
    if _pipeline_type(request) != "middle_layer":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    from osprey.services.channel_finder.core.base_database import DatabaseWriteError

    try:
        db = _get_database(request)
        return db.add_family(
            system=body.system,
            family=body.family,
            description=body.description,
        )
    except DatabaseWriteError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to add family")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete("/structure/family")
async def delete_family(request: Request, body: DeleteFamilyRequest):
    """Delete a family and all its channels."""
    if _pipeline_type(request) != "middle_layer":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    from osprey.services.channel_finder.core.base_database import DatabaseWriteError

    try:
        db = _get_database(request)
        return db.delete_family(
            system=body.system,
            family=body.family,
        )
    except DatabaseWriteError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to delete family")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/structure/channel")
async def add_ml_channel(request: Request, body: AddMLChannelRequest):
    """Add a channel to a family's field."""
    if _pipeline_type(request) != "middle_layer":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    from osprey.services.channel_finder.core.base_database import DatabaseWriteError

    try:
        db = _get_database(request)
        return db.add_channel(
            system=body.system,
            family=body.family,
            field=body.field,
            channel_name=body.channel_name,
            subfield=body.subfield,
        )
    except DatabaseWriteError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to add ML channel")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete("/structure/channel")
async def delete_ml_channel(request: Request, body: DeleteMLChannelRequest):
    """Delete a channel from a family's field."""
    if _pipeline_type(request) != "middle_layer":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    from osprey.services.channel_finder.core.base_database import DatabaseWriteError

    try:
        db = _get_database(request)
        return db.delete_channel(
            system=body.system,
            family=body.family,
            field=body.field,
            channel_name=body.channel_name,
            subfield=body.subfield,
        )
    except DatabaseWriteError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to delete ML channel")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/structure/impact")
async def structure_impact(request: Request, body: DeleteFamilyRequest):
    """Preview the impact of deleting a middle-layer family."""
    if _pipeline_type(request) != "middle_layer":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    from osprey.services.channel_finder.core.base_database import DatabaseWriteError

    try:
        db = _get_database(request)
        count = db.count_family_channels(
            system=body.system,
            family=body.family,
        )
        return {"affected_channels": count}
    except DatabaseWriteError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to compute structure impact")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# In-Context CRUD endpoints
# ---------------------------------------------------------------------------


@router.post("/channels")
async def create_channel(request: Request, body: AddICChannelRequest):
    """Add a new channel to the in-context database."""
    if _pipeline_type(request) != "in_context":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    from osprey.services.channel_finder.core.base_database import DatabaseWriteError

    try:
        db = _get_database(request)
        return db.add_channel(
            channel=body.channel_name,
            address=body.address,
            description=body.description,
        )
    except DatabaseWriteError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to create channel")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.put("/channels/{channel_id:path}")
async def update_channel(channel_id: str, request: Request, body: UpdateICChannelRequest):
    """Update an in-context channel's description and/or address.

    Args:
        channel_id: Channel name (uses :path converter for colon-separated PV names).
        request: FastAPI request.
        body: Fields to update.
    """
    if _pipeline_type(request) != "in_context":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    from osprey.services.channel_finder.core.base_database import DatabaseWriteError

    try:
        db = _get_database(request)
        return db.update_channel(
            channel=channel_id,
            new_description=body.description,
            new_address=body.address,
        )
    except DatabaseWriteError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to update channel")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete("/channels/{channel_id:path}")
async def delete_channel(channel_id: str, request: Request):
    """Delete a channel from the in-context database.

    Args:
        channel_id: Channel name (uses :path converter for colon-separated PV names).
        request: FastAPI request.
    """
    if _pipeline_type(request) != "in_context":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    from osprey.services.channel_finder.core.base_database import DatabaseWriteError

    try:
        db = _get_database(request)
        return db.delete_channel(
            channel=channel_id,
        )
    except DatabaseWriteError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to delete channel")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
