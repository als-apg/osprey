"""ARIEL Web Interface - FastAPI Application.

A production-grade web interface for ARIEL (Agentic Retrieval Interface
for Electronic Logbooks), providing search, browsing, and entry creation
for scientific logbook data.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from api.routes import router as api_router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)


def load_ariel_config() -> dict[str, Any]:
    """Load ARIEL configuration from config.yml.

    Looks for config in:
    1. /app/config.yml (Docker mount)
    2. CONFIG_FILE environment variable
    3. Current directory config.yml
    """
    config_paths = [
        Path("/app/config.yml"),
        Path(os.environ.get("CONFIG_FILE", "")),
        Path("config.yml"),
    ]

    for config_path in config_paths:
        if config_path.exists():
            logger.info(f"Loading config from {config_path}")
            with open(config_path) as f:
                config = yaml.safe_load(f)
                return config.get("ariel", {})

    raise RuntimeError(
        "No config.yml found. Set CONFIG_FILE environment variable "
        "or mount config.yml at /app/config.yml"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle.

    Initialize ARIEL service on startup, cleanup on shutdown.
    """
    from osprey.services.ariel_search import ARIELConfig, create_ariel_service

    logger.info("Starting ARIEL Web Interface...")

    # Load configuration
    config_dict = load_ariel_config()
    config = ARIELConfig.from_dict(config_dict)

    # Validate configuration
    errors = config.validate()
    if errors:
        raise RuntimeError(f"Configuration errors: {errors}")

    # Create and store service
    service = await create_ariel_service(config)
    app.state.ariel_service = service

    # Health check
    healthy, message = await service.health_check()
    if healthy:
        logger.info(f"ARIEL service ready: {message}")
    else:
        logger.warning(f"ARIEL service degraded: {message}")

    yield

    # Cleanup
    logger.info("Shutting down ARIEL Web Interface...")
    await service.__aexit__(None, None, None)


# Create FastAPI application
app = FastAPI(
    title="ARIEL Search Interface",
    description="Agentic Retrieval Interface for Electronic Logbooks",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(api_router)

# Static files directory
static_dir = Path(__file__).parent / "static"


# Mount static files
@app.get("/")
async def root():
    """Serve main index.html."""
    return FileResponse(static_dir / "index.html")


# Mount static assets
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/health")
async def health():
    """Simple health check endpoint."""
    if hasattr(app.state, "ariel_service"):
        healthy, message = await app.state.ariel_service.health_check()
        return {"status": "healthy" if healthy else "degraded", "message": message}
    return {"status": "starting", "message": "Service initializing"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # noqa: S104
        port=8085,
        reload=True,
        log_level="info",
    )
