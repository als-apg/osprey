"""OSPREY Channel Finder — FastAPI Application.

A web interface for exploring, searching, and managing control system channels.
Initializes the appropriate channel finder MCP pipeline registry based on config
and serves the frontend with REST and WebSocket endpoints.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.responses import FileResponse

from osprey.interfaces._app_setup import configure_interface_app
from osprey.utils.facility import resolve_facility_name
from osprey.utils.workspace import DEFAULT_AGENT_DATA_BASE_DIR

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

#: Repo-relative home of the feedback stores, used when the config names no
#: ``store_path``. Both stores are written while the agent runs, so they belong
#: in the durable STATE zone: ``data/`` is build-owned and checksummed into the
#: manifest, and ``build/`` is wiped and re-rendered by every build.
#:
#: Derived from :data:`~osprey.utils.workspace.DEFAULT_AGENT_DATA_BASE_DIR`
#: rather than spelled out, so this fallback cannot drift away from the root the
#: rest of the framework resolves. It is only a fallback: a config that sets
#: ``store_path`` wins, and the shipped templates set it.
FEEDBACK_DIR = f"{DEFAULT_AGENT_DATA_BASE_DIR}/feedback"


def _init_hierarchical_registry():
    """Build the hierarchical channel-finder registry."""
    from osprey.mcp_server.channel_finder_hierarchical.server_context import (
        initialize_cf_hier_context,
    )

    return initialize_cf_hier_context()


def _init_middle_layer_registry():
    """Build the middle-layer channel-finder registry."""
    from osprey.mcp_server.channel_finder_middle_layer.server_context import (
        initialize_cf_ml_context,
    )

    return initialize_cf_ml_context()


def _init_in_context_registry():
    """Build the in-context channel-finder registry."""
    from osprey.mcp_server.channel_finder_in_context.server_context import (
        initialize_cf_ic_context,
    )

    return initialize_cf_ic_context()


def _make_graph_context():
    """Build the graph-store context this app reads the graph paradigm through.

    The import is local because it pulls in the graph stack, which a deployment
    serving a file-backed paradigm has no use for. ``initialize()`` only resolves
    config — it dials nothing — so a store that is down costs startup nothing and
    surfaces as a failed read later, where the route can say so.

    Returns:
        An initialized graph-store context.

    Raises:
        GraphStoreError: The store's settings could not be resolved.
    """
    from osprey.mcp_server.graph.server_context import GraphContext

    context = GraphContext()
    context.initialize()
    return context


def _graph_ttl_filename(config) -> str | None:
    """Name the corpus the graph store was seeded from, for the UI to show.

    Only the file name travels: the configured path is relative to the config
    file's directory, so the rest of it names a location on the deployment host
    that means nothing to a browser.

    Args:
        config: The loaded project config.

    Returns:
        The corpus file name, or ``None`` when no ``services.graphdb`` block
        names one — including when the block is malformed, which is the graph
        store's problem to report and not a reason to refuse to serve.
    """
    from osprey.deployment.graphdb_service import resolve_graphdb_service_config

    try:
        graphdb_config = resolve_graphdb_service_config(config)
    except ValueError as exc:
        logger.warning("Could not read the services.graphdb block (%s)", exc)
        return None
    if graphdb_config is None or not graphdb_config.ttl_path:
        return None
    return Path(graphdb_config.ttl_path).name


#: How the web app brings up each channel-finder paradigm it can serve, in the
#: order the UI offers them. A paradigm is attempted only when the config has a
#: ``channel_finder.pipelines.<paradigm>`` block for it, so the absence of a
#: block is a quiet skip while a block that fails to come up is a loud warning.
#: Serving a new file-backed paradigm is one entry here plus its config block.
#:
#: The graph paradigm is deliberately absent: it opens no database file, so
#: there is no registry to build and nothing here to name. The lifespan serves
#: it from the resolved mode instead.
_PIPELINE_REGISTRY_INITIALIZERS: tuple[tuple[str, Callable[[], object]], ...] = (
    ("hierarchical", _init_hierarchical_registry),
    ("middle_layer", _init_middle_layer_registry),
    ("in_context", _init_in_context_registry),
)


def _create_lifespan(project_cwd: str | None = None):
    """Create a lifespan context manager that initializes the pipeline registry.

    Args:
        project_cwd: Project directory path for agent session use.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        import httpx

        from osprey.services.channel_finder.utils.detection import detect_pipeline_config
        from osprey.utils.workspace import load_osprey_config

        config = load_osprey_config()
        cf_config = config.get("channel_finder", {})

        # One resolver for the active paradigm, shared with the rest of the
        # framework: no local default that could disagree with it. Naming no
        # paradigm is legal but worth saying out loud, because what gets served
        # then depends on which pipeline happens to have a database configured.
        if "pipeline_mode" not in cf_config:
            logger.warning(
                "No 'channel_finder.pipeline_mode' configured; the active pipeline will be "
                "detected from the configured pipeline databases",
            )
        pipeline_type, _db_config = detect_pipeline_config(config)
        app.state.pipeline_type = pipeline_type
        app.state.project_cwd = project_cwd or str(Path.cwd())
        app.state.facility_name = resolve_facility_name(config, "")

        # Initialize all available pipeline registries so the UI can switch
        available: list[str] = []
        databases: dict[str, object] = {}
        facility_names: dict[str, str] = {}

        # The graph paradigm is store-backed rather than file-backed: its
        # channels live in a graph database reached over the network, so there
        # is no database file to open and no ``channel_finder.pipelines.graph``
        # block to read. It is served from the resolved mode alone, and the
        # empty ``databases`` map is the honest answer the routes report back —
        # the store is queried with the graph tools, not from this process.
        graph_backed = pipeline_type == "graph"
        app.state.graph_backed = graph_backed

        if graph_backed:
            available = ["graph"]
            logger.info(
                "Serving the graph paradigm: channels come from the graph store, which is "
                "queried with the read_cypher and get_schema tools, so no channel database "
                "is opened here",
            )

            # The store is the paradigm's only source, so the app holds one
            # context for it rather than resolving the connection per request.
            # A context that cannot be built leaves no attribute behind: the
            # routes answer 503 from its absence, and everything the app serves
            # that is not the graph keeps working.
            from osprey.mcp_server.graph.server_context import GraphStoreError

            try:
                app.state.graph_context = _make_graph_context()
            except GraphStoreError:
                logger.warning(
                    "The graph store context could not be built; graph queries will be "
                    "answered as unavailable",
                    exc_info=True,
                )
            app.state.graph_ttl_filename = _graph_ttl_filename(config)
        else:
            # ``pipelines`` is a key every rendered config carries, and a graph
            # render leaves it empty — so read it as "whichever blocks are
            # there", never as "a mapping is guaranteed".
            pipelines_config = cf_config.get("pipelines") or {}
            for mode, initialize_registry in _PIPELINE_REGISTRY_INITIALIZERS:
                if mode not in pipelines_config:
                    logger.debug(
                        "No 'channel_finder.pipelines.%s' block configured; not serving %s",
                        mode,
                        mode,
                    )
                    continue
                try:
                    registry = initialize_registry()
                except Exception:
                    logger.warning(
                        "Configured '%s' pipeline failed to initialize; it will not be offered",
                        mode,
                        exc_info=True,
                    )
                    continue
                databases[mode] = registry.database
                facility_names[mode] = registry.facility_name
                available.append(mode)
                logger.info("Initialized %s pipeline registry", mode)

        app.state.available_pipelines = available
        app.state.databases = databases
        app.state.facility_names = facility_names

        # Initialize feedback store if hierarchical pipeline has feedback enabled
        app.state.feedback_store = None
        if "hierarchical" in available:
            feedback_config = (
                (config.get("channel_finder", {}).get("pipelines") or {})
                .get("hierarchical", {})
                .get("feedback", {})
            )
            if feedback_config.get("enabled", False):
                from osprey.mcp_server.channel_finder_common import resolve_cf_state_path
                from osprey.services.channel_finder.feedback.store import FeedbackStore

                store_path = feedback_config.get(
                    "store_path", f"{FEEDBACK_DIR}/hierarchical_feedback.json"
                )
                # Anchored on the deployment REPO root, not on this process's
                # working directory. `project_cwd` defaults to `Path.cwd()`,
                # which is wherever the operator happened to be standing — so
                # the same deployment wrote its feedback to a different place
                # depending on where the command was typed, and under `build/`
                # it wrote into the zone the next build deletes.
                resolved = Path(resolve_cf_state_path(str(store_path)))
                app.state.feedback_store = FeedbackStore(str(resolved))
                logger.info("Initialized feedback store at %s", resolved)

        # Initialize pending review store (unconditional — captures come from hook)
        app.state.pending_review_store = None
        try:
            from osprey.mcp_server.channel_finder_common import resolve_cf_state_path
            from osprey.services.channel_finder.feedback.pending_store import (
                PendingReviewStore,
            )

            # Same anchor as the feedback store above, for the same reason.
            pr_path = Path(resolve_cf_state_path(f"{FEEDBACK_DIR}/pending_reviews.json"))
            app.state.pending_review_store = PendingReviewStore(str(pr_path))
            logger.info("Initialized pending review store at %s", pr_path)
        except Exception:
            logger.debug("Could not initialize pending review store", exc_info=True)

        # Fall back if the resolved pipeline didn't initialize. Graph is
        # outside this rule in both directions: it brings up no registry that
        # could fail, and it is never the substitute for a paradigm that did.
        if not graph_backed and pipeline_type not in available and available:
            logger.warning(
                "Pipeline %r is not among the pipelines that came up (%s); serving %s instead",
                pipeline_type,
                available,
                available[0],
            )
            pipeline_type = available[0]
            app.state.pipeline_type = pipeline_type

        app.state.http_client = httpx.AsyncClient(timeout=30.0)
        logger.info(
            "Channel Finder started (pipeline=%s, cwd=%s)",
            pipeline_type,
            app.state.project_cwd,
        )

        yield

        graph_context = getattr(app.state, "graph_context", None)
        if graph_context is not None:
            graph_context.shutdown()
        await app.state.http_client.aclose()

    return lifespan


def create_app(project_cwd: str | None = None) -> FastAPI:
    """Create the Channel Finder FastAPI application.

    Args:
        project_cwd: Project directory for agent session use.
            Falls back to current working directory.

    Returns:
        Configured FastAPI application.
    """
    from osprey.interfaces.channel_finder import (
        database_api,
        feedback_api,
        pending_review_api,
    )

    app = FastAPI(
        title="OSPREY Channel Finder",
        description="Web interface for exploring and managing control system channels",
        version="1.0.0",
        lifespan=_create_lifespan(project_cwd),
    )

    @app.get("/health")
    async def health():
        pipeline_type = getattr(app.state, "pipeline_type", "unknown")
        return {
            "status": "healthy",
            "service": "channel-finder",
            "pipeline_type": pipeline_type,
        }

    @app.get("/")
    async def root():
        return FileResponse(STATIC_DIR / "index.html")

    # Database REST API
    app.include_router(database_api.router, prefix="/api")

    # Feedback management API
    app.include_router(feedback_api.router, prefix="/api")

    # Pending review API
    app.include_router(pending_review_api.router, prefix="/api")

    configure_interface_app(app, static_dir=STATIC_DIR)

    return app
