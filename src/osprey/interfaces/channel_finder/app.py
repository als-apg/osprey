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
    from collections.abc import AsyncIterator

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


def _create_lifespan(project_cwd: str | None = None):
    """Create a lifespan context manager that initializes the pipeline registry.

    Args:
        project_cwd: Project directory path for agent session use.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        import httpx

        from osprey.utils.workspace import load_osprey_config

        # Load config and determine pipeline type
        config = load_osprey_config()
        pipeline_type = config.get("channel_finder", {}).get("pipeline_mode", "in_context")
        app.state.pipeline_type = pipeline_type
        app.state.project_cwd = project_cwd or str(Path.cwd())
        app.state.facility_name = resolve_facility_name(config, "")

        # Initialize all available pipeline registries so the UI can switch
        available: list[str] = []
        databases: dict[str, object] = {}
        facility_names: dict[str, str] = {}

        try:
            from osprey.mcp_server.channel_finder_hierarchical.server_context import (
                initialize_cf_hier_context,
            )

            registry = initialize_cf_hier_context()
            databases["hierarchical"] = registry.database
            facility_names["hierarchical"] = registry.facility_name
            available.append("hierarchical")
            logger.info("Initialized hierarchical pipeline registry")
        except Exception:
            logger.debug("Hierarchical pipeline not available", exc_info=True)

        try:
            from osprey.mcp_server.channel_finder_middle_layer.server_context import (
                initialize_cf_ml_context,
            )

            registry = initialize_cf_ml_context()
            databases["middle_layer"] = registry.database
            facility_names["middle_layer"] = registry.facility_name
            available.append("middle_layer")
            logger.info("Initialized middle_layer pipeline registry")
        except Exception:
            logger.debug("Middle layer pipeline not available", exc_info=True)

        try:
            from osprey.mcp_server.channel_finder_in_context.server_context import (
                initialize_cf_ic_context,
            )

            registry = initialize_cf_ic_context()
            databases["in_context"] = registry.database
            facility_names["in_context"] = registry.facility_name
            available.append("in_context")
            logger.info("Initialized in_context pipeline registry")
        except Exception:
            logger.debug("In-context pipeline not available", exc_info=True)

        app.state.available_pipelines = available
        app.state.databases = databases
        app.state.facility_names = facility_names

        # Initialize feedback store if hierarchical pipeline has feedback enabled
        app.state.feedback_store = None
        if "hierarchical" in available:
            feedback_config = (
                config.get("channel_finder", {})
                .get("pipelines", {})
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

        # Fall back if the configured pipeline didn't initialize
        if pipeline_type not in available and available:
            pipeline_type = available[0]
            app.state.pipeline_type = pipeline_type
            logger.warning("Configured pipeline unavailable, falling back to %s", pipeline_type)

        app.state.http_client = httpx.AsyncClient(timeout=30.0)
        logger.info(
            "Channel Finder started (pipeline=%s, cwd=%s)",
            pipeline_type,
            app.state.project_cwd,
        )

        yield

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
