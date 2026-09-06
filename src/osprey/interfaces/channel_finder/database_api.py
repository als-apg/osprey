"""Channel Finder Database REST API.

Exposes database operations via REST endpoints, adapting for each pipeline
type. The three file-backed paradigms (hierarchical, middle_layer, in_context)
are served by calling their database instances directly via app.state, avoiding
MCP server dependencies. The graph paradigm has no database file behind it: the
two routes that answer which channels exist — membership and enumeration — read
the channel roster the app resolved at startup, the explorer's search, class
tree and statistics read the search index the app opened, the device card
queries the store, and the rest report the paradigm and point at the tools
that read it.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from osprey.build.build_tiers import VALID_CHANNEL_FINDER_MODES
from osprey.deployment.graphdb_service import (
    DEFAULT_INDEX_PATH,
    GRAPHDB_BUILD_INDEX_COMMAND,
    GRAPHDB_SERVICE_NAME,
    GRAPHDB_TTL_PATH_CONFIG_KEY,
    UNRESOLVED_INDEX_PATH_REMEDY,
    graph_corpus_configured,
    unresolved_index_path_detail,
)
from osprey.mcp_server.graph.server_context import GraphStoreError
from osprey.registry.mcp import CHANNEL_FINDER_TOOLS_BY_PIPELINE
from osprey.services.channel_finder.graph_index.reader import (
    DEFAULT_PAGE_SIZE,
    GraphIndex,
    GraphIndexAbsence,
)

# The device card is the one explorer read still answered by the store, and its
# query comes from the shared query module so a test can import the text
# without starting the app. Search, the ontology and the statistics read the
# search index the app opened at startup instead.
from osprey.services.channel_finder.graph_index.taxonomy import (
    class_name as _class_name,
)
from osprey.services.channel_finder.graph_queries import GRAPH_DEVICE_CYPHER

if TYPE_CHECKING:
    from osprey.channel_roster import RosterAbsence

logger = logging.getLogger(__name__)

router = APIRouter()

#: The tools the graph paradigm serves, read from the registry that renders
#: them into the agent rather than spelled out again here, so the web UI can
#: only ever name the vocabulary the agent actually has.
GRAPH_PARADIGM_TOOLS: tuple[str, ...] = tuple(CHANNEL_FINDER_TOOLS_BY_PIPELINE["graph"])

# ---------------------------------------------------------------------------
# Graph explorer limits
# ---------------------------------------------------------------------------

#: How many entries each facet list of a search carries. The rail cannot draw
#: more than a few hundred values of one facet and still be a rail, and the
#: index reports a list it had to cut as ``truncated`` so the explorer can say
#: so. Five hundred is well above any real facet and still a bound.
_GRAPH_EXPLORE_MAX_ROWS = 500

#: Rows one page of the finder holds, as the reader defines it. Passed to the
#: index on every search and reported back in the answer, so the page offset the
#: route computes and the slice the index cuts cannot disagree.
_GRAPH_SEARCH_PAGE_SIZE = DEFAULT_PAGE_SIZE

#: How a channel binding relates to the signal it carries: read, written, both,
#: or neither. Declared as a type rather than checked in the body so a value
#: outside it is refused as a 422 by the request layer, before the index is
#: asked a question it has no answer for.
GraphDirection = Literal["R", "W", "RW", "none"]

#: What an operator is told when a device URI is not in the store. A miss is
#: not a store failure — the finder that produced the link is simply older than
#: the corpus behind it — so the remedy is to search again rather than to touch
#: the service.
_NO_DEVICE_SUGGESTIONS = [
    "Search again — the store may have been re-seeded.",
]

#: Detail and remedy for a graph-mode request that arrives with no store seam
#: at all — the app started without a graph context, so there is nothing to ask.
_NO_GRAPH_CONTEXT_DETAIL = "Graph store is not available."
_NO_GRAPH_CONTEXT_SUGGESTIONS = [
    f"Check that a 'services.{GRAPHDB_SERVICE_NAME}' block is configured and that the "
    "channel finder was started against it.",
]

#: Detail for a search-index request that arrives with no index handle at all —
#: the app is serving a paradigm whose lifespan never opened one.
_NO_GRAPH_INDEX_DETAIL = "The search index is not open."

#: What an operator does about an index that is missing, unreadable or stale.
#: Two facts: the verb that writes one, and the build that runs that verb as
#: part of rendering the project.
_NO_GRAPH_INDEX_SUGGESTIONS = [
    f"Build the index with `{GRAPHDB_BUILD_INDEX_COMMAND}`.",
    "`osprey build` renders the project and builds the index in one step.",
]

#: Said when a read of an *open* index failed: the file was there and was
#: built, so the build pair would be untrue. A request that raced shutdown
#: succeeds on retry; a file that went away under the connection does not, and
#: only then is a rebuild the remedy.
_INDEX_READ_FAILED_SUGGESTIONS = [
    f"Retry the request; if it keeps failing, rebuild the index with "
    f"`{GRAPHDB_BUILD_INDEX_COMMAND}` and restart the channel finder.",
]


class UnresolvedIndexPath(GraphIndexAbsence):
    """The absence a config that spells ``index_path`` wrongly resolves to.

    Its own type rather than a sentence to match on: the 503 builder has to
    tell this state apart from an index that is merely not built yet, because
    it answers no build remedy — a build would read the same malformed key —
    and recognising it by the tail of its prose would come undone the first
    time that prose is edited.

    An ``unreadable`` absence to every other reader, which is what it is: the
    config named no path this process could read the index from.
    """


def unresolved_index_path(exc: ValueError) -> UnresolvedIndexPath:
    """Build the absence a malformed ``index_path`` puts on the app's state.

    A config typo is not a missing artifact, and there is no resolved path to
    name — so the absence carries the key's default location, and its own
    sentence says which key to fix.

    Args:
        exc: What :func:`~osprey.deployment.graphdb_service.resolve_graph_index_path`
            refused with.

    Returns:
        The absence, carrying the resolver's own sentence and the remedy.
    """
    # The remedy travels in the sentence itself, so every surface that shows the
    # detail — the 503 body, the log line — shows the fix.
    return UnresolvedIndexPath(
        "unreadable",
        Path(DEFAULT_INDEX_PATH),
        f"{unresolved_index_path_detail(exc).rstrip('.')}. {UNRESOLVED_INDEX_PATH_REMEDY}",
    )


#: Said instead when the project configures no corpus: there is nothing to
#: build an index *from*, so naming the build verb would name the wrong step —
#: the build refuses in exactly that state. Whether an index *location* is
#: configured does not change that, so it is not consulted; the wording is the
#: MCP tool's, so both surfaces say the same thing.
_NO_GRAPH_CORPUS_SUGGESTIONS = [
    f"No corpus is configured: set {GRAPHDB_TTL_PATH_CONFIG_KEY} to the facility's Turtle "
    "file, then build the index.",
]

#: Said when the roster read failed in a way the roster does not model — the
#: app kept no result to render a reason from. Every modelled reason travels as
#: the absence's own sentence instead (see :func:`_roster_unavailable`).
_NO_ROSTER_DETAIL = (
    "The channel roster could not be read, so which channels this facility has is unknown."
)

#: What an operator edits when an enumeration route reports no roster. Named on
#: every such answer, including the ones whose reason already names a path: the
#: path says which file was unreadable, this says which key put it there.
_NO_ROSTER_SUGGESTIONS = [
    f"Check that 'services.{GRAPHDB_SERVICE_NAME}.ttl_path' names a readable "
    "knowledge-graph corpus and restart the channel finder.",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _graph_index_error_payload(
    absence: GraphIndexAbsence | None,
    config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Turn a missing search index into the body a route replies 503 with.

    Same three keys the store's own unavailable answer carries, for the same
    reason: the explorer branches on ``error_type`` and shows ``suggestions``,
    so an index that is not there and a store that is down must arrive in the
    same shape. The reason is the absence's own sentence — written once in
    :mod:`osprey.services.channel_finder.graph_index.reader` — and the remedy is
    added here, because only a surface knows what an operator can do about it.

    Which remedy depends on what the project has, decided the way the MCP
    tool decides it so the two surfaces never name different steps. An
    An
    :class:`UnresolvedIndexPath` — the absence a malformed ``index_path``
    resolves to — gets no further remedy, since a build would read the same
    malformed key. A deployment that
    configures no corpus has nothing to build an index *from*, and the build
    refuses in that state, so it is told to configure one. Everything else is
    one build away from an answer.

    Args:
        absence: Why there is no index, or ``None`` when the app holds no index
            handle at all and there is no modelled reason to render.
        config: The loaded project config, read only for the key that says
            whether a corpus is configured.

    Returns:
        The response body: the detail, ``service_unavailable``, and the remedy.
    """
    if isinstance(absence, UnresolvedIndexPath):
        suggestions: list[str] = []
    elif not graph_corpus_configured(config):
        suggestions = _NO_GRAPH_CORPUS_SUGGESTIONS
    else:
        suggestions = _NO_GRAPH_INDEX_SUGGESTIONS
    return {
        "detail": absence.detail if absence is not None else _NO_GRAPH_INDEX_DETAIL,
        "error_type": "service_unavailable",
        "suggestions": list(suggestions),
    }


def _index_unavailable(index: GraphIndex, exc: BaseException) -> bool:
    """Whether a failed read of an *open* index is unavailability, not a bug.

    Everything DuckDB raises — a file deleted under the open connection, a
    disk that went away — arrives as ``duckdb.Error``. A ``RuntimeError`` is
    the index reporting that it has been closed, which a request racing
    shutdown can see — but only when the index *is* closed: any other
    ``RuntimeError`` is a defect in the read and must reach the 500 path
    rather than hide behind an operator remedy.

    The driver is imported here rather than at module scope: by the time a read
    can fail the index is open, so ``duckdb`` is already in the process, and a
    deployment serving a file-backed paradigm never pays for it.

    Args:
        index: The index the read ran against.
        exc: What the read raised.

    Returns:
        ``True`` when the deployment cannot serve the read right now.
    """
    import duckdb

    if isinstance(exc, duckdb.Error):
        return True
    return isinstance(exc, RuntimeError) and index.closed


async def _serve_index_read(request: Request, subject: str, read: Any) -> Any:
    """Answer a graph-paradigm request from the search index the app opened.

    The mirror of :func:`_serve_graph_read` for the reads that come from the
    index rather than from the store: where the handle lives, what an app
    without a usable one answers, and how a failed read travels. Search, the
    ontology and the statistics share it, so they cannot drift into reporting
    the same missing index three ways.

    The read runs off the event loop. DuckDB's driver is synchronous, and
    awaiting a scan inline would stall every other request the app is serving
    for its duration.

    Args:
        request: FastAPI request, carrying the index on its app state.
        subject: What is being read, for the log lines.
        read: Callable taking the index and returning the payload, run in a
            worker thread.

    Returns:
        Whatever *read* returns, or a 503 :class:`JSONResponse` carrying the
        remedy when there is no index to read.

    Raises:
        HTTPException: 500 when the read fails for a reason the index does not
            model.
    """
    index = getattr(request.app.state, "graph_index", None)
    if index is None or isinstance(index, GraphIndexAbsence):
        # Read here rather than held on app state: this is the failure path, the
        # loader is a cached singleton, and the alternative is a copy of the
        # config kept alive for the two keys a 503 names.
        from osprey.utils.workspace import load_osprey_config

        return JSONResponse(
            status_code=503,
            content=_graph_index_error_payload(index, load_osprey_config()),
        )

    try:
        return await asyncio.to_thread(read, index)
    except HTTPException:
        raise
    except Exception as exc:
        if not _index_unavailable(index, exc):
            logger.exception("Failed to read the search index %s", subject)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        logger.warning("Search index %s read failed: %s", subject, exc)
        return JSONResponse(
            status_code=503,
            content={
                "detail": f"The search index at {index.path} could not be read: {exc}",
                "error_type": "service_unavailable",
                "suggestions": list(_INDEX_READ_FAILED_SUGGESTIONS),
            },
        )


async def _serve_graph_read(request: Request, subject: str, read: Any) -> Any:
    """Answer a graph-paradigm request through the app's store context.

    The store reads the app serves share everything around the read itself:
    where the context lives, what an app without one answers, and how a store
    failure travels. That contract lives here once — the device card is the
    read that uses it today, and a second one cannot drift into reporting the
    same broken store differently.

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


def _roster_unavailable(absence: RosterAbsence | None) -> JSONResponse:
    """Return the 503 an enumeration route answers with when there is no roster.

    503 rather than 501 or 404: the question is answerable and the route
    implements it — this deployment has nothing to answer it *from*, which is a
    state an operator can change. The reason is the roster's own sentence,
    rendered once in :mod:`osprey.channel_roster`, so the web body, the build
    fact and the log line cannot describe the same absence three ways.

    Args:
        absence: Why the roster is missing, or ``None`` when the app holds no
            roster result at all and there is no modelled reason to render.

    Returns:
        The 503, in the body shape the other graph routes answer an unavailable
        store in: detail, error type, and the remedy.
    """
    return JSONResponse(
        status_code=503,
        content={
            "detail": absence.message() if absence is not None else _NO_ROSTER_DETAIL,
            "error_type": "service_unavailable",
            "suggestions": list(_NO_ROSTER_SUGGESTIONS),
        },
    )


def _serve_from_roster(request: Request, answer: Callable[[], Any]) -> Any:
    """Answer a graph-paradigm request from the roster the app started with.

    The roster is read once, at lifespan, so a route only reads it off state
    here. Both routes that enumerate share what happens when there is nothing to
    read — an app whose corpus could not be resolved keeps no roster, and one
    whose corpus could not be parsed or holds no channel keeps the absence
    saying so — so neither can drift into reporting the same unstaged
    deployment differently.

    The gate is on the records, not on the absence: a source that enumerated
    its channels but cannot say which are settable carries both, and neither of
    these routes asks about direction — membership and enumeration are answers
    about which channels exist.

    Args:
        request: FastAPI request, carrying the app's roster on its state.
        answer: Returns the route's payload, read from the addresses the
            lifespan derived beside the roster.

    Returns:
        Whatever *answer* returns, or the 503 the absence renders.
    """
    roster = getattr(request.app.state, "channel_roster", None)
    if roster is None:
        return _roster_unavailable(None)
    if not roster.records:
        return _roster_unavailable(roster.absence)
    return answer()


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
        # The reader answers the badge counts off the index's own meta row.
        return await _serve_index_read(request, "statistics", lambda index: index.statistics())

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
    """Validate channel names against the paradigm's roster of channels.

    The file-backed paradigms ask their database. The graph paradigm asks the
    channel roster — the corpus the store is seeded from — because membership is
    a question about which channels the facility has, and that is one answer per
    build no matter who asks it.
    """
    pt = _pipeline_type(request)
    if pt == "graph":
        state = request.app.state
        return _serve_from_roster(
            request, lambda: _validate_against_roster(state.channel_address_index, body.channels)
        )

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


def _validate_against_roster(addresses: frozenset[str], channels: list[str]) -> dict[str, Any]:
    """Answer membership for *channels* against the roster's addresses.

    Direction plays no part: a settable channel and a readable one are equally
    real, and a membership answer that quietly dropped the readable half would
    report most of a facility as invalid.

    Args:
        addresses: Every address the roster enumerated, as the lifespan indexed
            them.
        channels: The addresses asked about, in the order they were asked.

    Returns:
        One result per channel plus the counts, in the shape the file-backed
        paradigms answer this route in.
    """
    results = [{"channel": channel, "valid": channel in addresses} for channel in channels]
    valid_count = sum(1 for result in results if result["valid"])
    return {
        "results": results,
        "valid_count": valid_count,
        "invalid_count": len(channels) - valid_count,
        "total": len(channels),
    }


@router.get("/graph/ontology")
async def graph_ontology(request: Request):
    """Return the device class tree of the search index.

    The graph paradigm's answer to the tree the file-backed paradigms read from
    a database file. The read runs off the event loop: the index's driver is
    synchronous, and awaiting it inline would stall every other request the app
    is serving for the length of the scan.

    An app without a usable index answers 503 carrying the build remedy rather
    than a bare status, and an index that binds nothing is distinguished from
    one that is missing — an empty corpus is a corpus gap, and the payload says
    which command closes it.

    Args:
        request: FastAPI request, carrying the index on its app state.

    Returns:
        ``classes`` (the pruned device taxonomy), ``relationship_types``,
        ``truncated``, ``empty`` and ``suggestions``.

    Raises:
        HTTPException: 404 when the active paradigm is not the graph, 400 when
            no paradigm is configured at all, 500 when the read fails for a
            reason the index does not model.
    """
    if _pipeline_type(request) != "graph":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")
    return await _serve_index_read(request, "ontology", _read_graph_ontology)


def _read_graph_ontology(index: GraphIndex) -> dict[str, Any]:
    """Read the class tree the ontology route serves.

    The rows are already the pruned device taxonomy: the build applied the
    pruning once, over the whole corpus, and the ``total_classes`` badge counts
    those same rows. They are not pruned again here — a second pass would drop
    an abstract parent whose only subclass the first pass removed, and the tree
    would then disagree with the badge.

    An index that binds nothing answers the empty-corpus shape the explorer
    already branches on: no classes to draw, and the remedy in
    ``suggestions``. The index still holds the ontology's class rows in that
    case, and blanking them keeps the answer the shape the store's empty
    answer had.

    Args:
        index: The search index the app opened.

    Returns:
        The ontology payload, in either its drawn or its empty-corpus shape.

    Raises:
        RuntimeError: If the index has been closed.
    """
    payload = index.ontology()
    if payload["empty"]:
        payload["classes"] = []
    return payload


# ---------------------------------------------------------------------------
# Graph finder: search and device
# ---------------------------------------------------------------------------


def _device_signal_groups(groups: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Reduce a device's signal groups to what the device card draws.

    The store hands back more than the card shows — a binding's protocol and
    the confidence it was matched with — so those are dropped here. The signal
    edges travel on untouched: the card derives direction from them exactly as
    the result table does, which is why the two cannot disagree.

    Args:
        groups: The ``signals`` list as :data:`GRAPH_DEVICE_CYPHER` returns it.

    Returns:
        One entry per signal, each carrying its bindings in store order.
    """
    return [
        {
            "name": group.get("name"),
            "uri": group.get("uri"),
            "bindings": [
                {
                    "fullPv": binding.get("fullPv"),
                    "edges": list(binding.get("edges") or []),
                    "description": binding.get("description"),
                    "subfieldDescription": binding.get("subfieldDescription"),
                    "fieldDescription": binding.get("fieldDescription"),
                }
                for binding in group.get("bindings") or []
            ],
        }
        for group in groups
    ]


@router.get("/graph/search")
async def graph_search(
    request: Request,
    q: str = Query(""),
    section: list[str] = Query([]),
    system: list[str] = Query([]),
    signal: list[str] = Query([]),
    dir: list[GraphDirection] = Query([]),
    cls: str | None = Query(None),
    page: int = Query(1, ge=1),
):
    """Answer one page of the graph finder, with the facets around it.

    The four multi-select facets arrive as repeated parameters — ``section=SR01C
    &section=SR02C`` — and are ORed within a facet and ANDed across facets by
    the index. ``cls`` is single-select and rolls its subclasses up, which is
    what lets a selection on an abstract branch stand for the devices under it.

    Args:
        request: FastAPI request, carrying the index on its app state.
        q: Free-text query, split on whitespace into tokens that must all match.
        section: Section codes to keep, or none for no section filter.
        system: System codes to keep.
        signal: Semantic signal names to keep.
        dir: Directions to keep, drawn from ``R``, ``W``, ``RW`` and ``none``.
        cls: One class URI, or nothing for no class filter.
        page: 1-based page of fifty rows.

    Returns:
        ``total`` and ``devices`` over every match, the ``page`` of ``rows``
        and the ``pages`` and ``page_size`` around it, the five ``facets``,
        whether a facet was ``truncated``, and — for an index that binds
        nothing — ``empty`` with the ``suggestions`` that name the corpus and
        the command that regenerates it.

    Raises:
        HTTPException: 404 when the active paradigm is not the graph, 400 when
            no paradigm is configured at all, 500 when the read fails for a
            reason the index does not model.
    """
    if _pipeline_type(request) != "graph":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")
    read = functools.partial(
        _read_graph_search,
        # The index matches folded text, so the tokens arrive folded; splitting
        # on whitespace drops the empty token an empty query would otherwise be.
        tokens=q.lower().split(),
        sections=list(section),
        systems=list(system),
        signals=list(signal),
        dirs=[str(value) for value in dir],
        # '' is a class no device is typed by; no filter is null.
        cls=cls or None,
        page=page,
    )
    return await _serve_index_read(request, "search", read)


def _read_graph_search(
    index: GraphIndex,
    *,
    tokens: list[str],
    sections: list[str],
    systems: list[str],
    signals: list[str],
    dirs: list[str],
    cls: str | None,
    page: int,
) -> dict[str, Any]:
    """Run the faceted search and answer the page it cuts.

    The index answers in the shape the finder reads, so nothing is reshaped
    here: only the request's 1-based page is turned into the row offset the
    index takes, and the caps the explorer draws to are passed along. The
    index asks one facet entry more than the cap itself, which is how a full
    list is told apart from a clipped one.

    Args:
        index: The search index the app opened.
        tokens: Lower-cased search tokens, all of which must match.
        sections: Section codes to keep, empty for no filter.
        systems: System codes to keep, empty for no filter.
        signals: Signal names to keep, empty for no filter.
        dirs: Directions to keep, empty for no filter.
        cls: One class URI, or ``None`` for no filter.
        page: 1-based page number.

    Returns:
        The search payload.

    Raises:
        RuntimeError: If the index has been closed.
    """
    return index.search(
        tokens=tokens,
        sections=sections,
        systems=systems,
        cls=cls,
        signals=signals,
        dirs=dirs,
        skip=(page - 1) * _GRAPH_SEARCH_PAGE_SIZE,
        page_size=_GRAPH_SEARCH_PAGE_SIZE,
        facet_cap=_GRAPH_EXPLORE_MAX_ROWS,
    )


@router.get("/graph/device")
async def graph_device(request: Request, uri: str = Query(..., min_length=1)):
    """Answer one device of the store, with its channels grouped by signal.

    Args:
        request: FastAPI request, carrying the app's graph context on its state.
        uri: The device URI, as a search row carries it in ``device_uri``.

    Returns:
        The device's properties — its class by name and by URI, its placement
        and its descriptions — and its ``signals``, each holding the bindings
        that carry it with the direction each one has.

    Raises:
        HTTPException: 404 when the active paradigm is not the graph, 400 when
            no paradigm is configured at all, 500 when the read fails for a
            reason the store does not classify.
    """
    if _pipeline_type(request) != "graph":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")
    return await _serve_graph_read(
        request, "device", functools.partial(_read_graph_device, uri=uri)
    )


async def _read_graph_device(ctx: Any, *, uri: str) -> Any:
    """Read one device off the event loop, or answer that the store has none.

    A URI the store does not hold answers with no row at all, which is a 404
    rather than an empty card. It travels in the same three-key shape every
    other graph answer uses — the panel branches on ``error_type``, not on the
    status — and carries its own remedy: a link the finder minted against an
    older corpus is a search away from being right again. The store is not
    asked whether it is empty: a device that is not there says nothing about
    whether anything else is.

    Args:
        ctx: The app's graph store context.
        uri: The device URI to read.

    Returns:
        The device payload, or a 404 :class:`JSONResponse` when the store holds
        no such device.

    Raises:
        GraphStoreError: Whatever the store raises when the read fails.
    """
    result = await asyncio.to_thread(ctx.run_read, GRAPH_DEVICE_CYPHER, {"uri": uri}, max_rows=1)
    if not result.rows:
        return JSONResponse(
            status_code=404,
            content={
                "detail": f"No device at {uri}",
                "error_type": "not_found",
                "suggestions": list(_NO_DEVICE_SUGGESTIONS),
            },
        )

    row = result.rows[0]
    class_uri = row.get("class")
    return {
        "uri": row.get("uri"),
        "device": row.get("device"),
        # The card shows the class the way the tree labels it, and keeps the
        # URI beside it because that is what the class filter is sent back as.
        "class": _class_name(class_uri) if class_uri else None,
        "class_uri": class_uri,
        "rawType": row.get("rawType"),
        "section": row.get("section"),
        "system": row.get("system"),
        "sPositionM": row.get("sPositionM"),
        "ordinalInSection": row.get("ordinalInSection"),
        "systemDescription": row.get("systemDescription"),
        "familyDescription": row.get("familyDescription"),
        "ringDescription": row.get("ringDescription"),
        "signals": _device_signal_groups(row.get("signals") or []),
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
    """Get the channels this facility has, as the active paradigm enumerates them.

    The in-context paradigm serves its database rows, optionally cut into the
    chunks its prompt is built from. The graph paradigm serves the channel
    roster's addresses: the corpus enumerates bindings rather than database
    rows, and an address is the whole record downstream — a device's name is its
    channel address.

    Every paradigm's items carry the channel under the ``channel`` key, so a
    reader of this route can name a channel without knowing which paradigm
    answered; the file-backed paradigms carry their remaining columns beside it.

    Args:
        request: FastAPI request.
        chunk_idx: Optional chunk index (0-based). If omitted, returns all.
            In-context only; see :func:`_roster_channels`.
        chunk_size: Number of channels per chunk (default 50).
    """
    pt = _pipeline_type(request)
    if pt == "graph":
        state = request.app.state
        return _serve_from_roster(
            request, lambda: _roster_channels(state.channel_addresses, chunk_idx)
        )
    if pt != "in_context":
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


def _roster_channels(addresses: tuple[str, ...], chunk_idx: int | None) -> dict[str, Any]:
    """Serve the roster's addresses, whole.

    Args:
        addresses: Every address the roster enumerated, in its order, as the
            lifespan deduplicated them.
        chunk_idx: The chunk asked for, which is always a mistake here.

    Returns:
        Every enumerated channel as a ``{"channel": address}`` item — the item
        shape this route answers in for every paradigm — and how many there
        are.

    Raises:
        HTTPException: 422 when a chunk is asked for. Chunking exists to cut the
            in-context paradigm's channel list into prompt-sized pieces; the
            graph paradigm builds no such prompt, so honouring the parameter
            would invent a chunking contract nothing on the other side has. Not
            a 400: the parameter is well-formed and inapplicable.
    """
    if chunk_idx is not None:
        raise HTTPException(
            status_code=422,
            detail=(
                "chunk_idx is not available for the graph paradigm: chunking cuts the "
                "in-context paradigm's channel list into prompt-sized pieces, which the "
                "graph paradigm does not build. Request the channels without it."
            ),
        )
    return {
        "channels": [{"channel": address} for address in addresses],
        "total": len(addresses),
    }


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
