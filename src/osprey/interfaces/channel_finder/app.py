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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def _create_lifespan(project_cwd: str | None = None):
    """Create a lifespan context manager that initializes the pipeline registry.

    Args:
        project_cwd: Project directory path for agent session use.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        import httpx

        from osprey.interfaces.channel_finder.agent_session import (
            cleanup_search_registry,
            init_search_registry,
        )
        from osprey.mcp_server.common import load_osprey_config

        # Load config and determine pipeline type
        config = load_osprey_config()
        pipeline_type = config.get("channel_finder", {}).get("pipeline_mode", "in_context")
        app.state.pipeline_type = pipeline_type
        app.state.project_cwd = project_cwd or str(Path.cwd())

        # Initialize all available pipeline registries so the UI can switch
        available: list[str] = []

        try:
            from osprey.services.channel_finder.mcp.hierarchical.registry import (
                initialize_cf_hier_registry,
            )

            initialize_cf_hier_registry()
            available.append("hierarchical")
            logger.info("Initialized hierarchical pipeline registry")
        except Exception:
            logger.debug("Hierarchical pipeline not available", exc_info=True)

        try:
            from osprey.services.channel_finder.mcp.middle_layer.registry import (
                initialize_cf_ml_registry,
            )

            initialize_cf_ml_registry()
            available.append("middle_layer")
            logger.info("Initialized middle_layer pipeline registry")
        except Exception:
            logger.debug("Middle layer pipeline not available", exc_info=True)

        try:
            from osprey.services.channel_finder.mcp.in_context.registry import (
                initialize_cf_ic_registry,
            )

            initialize_cf_ic_registry()
            available.append("in_context")
            logger.info("Initialized in_context pipeline registry")
        except Exception:
            logger.debug("In-context pipeline not available", exc_info=True)

        app.state.available_pipelines = available

        # Fall back if the configured pipeline didn't initialize
        if pipeline_type not in available and available:
            pipeline_type = available[0]
            app.state.pipeline_type = pipeline_type
            logger.warning("Configured pipeline unavailable, falling back to %s", pipeline_type)

        app.state.http_client = httpx.AsyncClient(timeout=30.0)
        init_search_registry()
        logger.info(
            "Channel Finder started (pipeline=%s, cwd=%s)",
            pipeline_type,
            app.state.project_cwd,
        )

        yield

        await cleanup_search_registry()
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
    from osprey.interfaces.channel_finder import agent_session, database_api

    app = FastAPI(
        title="OSPREY Channel Finder",
        description="Web interface for exploring and managing control system channels",
        version="1.0.0",
        lifespan=_create_lifespan(project_cwd),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Prevent browsers from caching JS/CSS (avoids stale code after updates)
    from starlette.middleware.base import BaseHTTPMiddleware

    class NoCacheStaticMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            response = await call_next(request)
            if request.url.path.startswith("/static/"):
                response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            return response

    app.add_middleware(NoCacheStaticMiddleware)

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

    # AI search routes (REST + WebSocket)
    app.include_router(agent_session.router)

    # Static files (must come last so it doesn't shadow API routes)
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app
