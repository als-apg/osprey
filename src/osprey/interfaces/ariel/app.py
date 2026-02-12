"""ARIEL Web Interface - FastAPI Application.

A production-grade web interface for ARIEL (Agentic Retrieval Interface
for Electronic Logbooks), providing search, browsing, and entry creation
for scientific logbook data.
"""

from __future__ import annotations

import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = get_logger("ariel")

# Static files directory (relative to this module)
STATIC_DIR = Path(__file__).parent / "static"


def load_ariel_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load ARIEL configuration from config.yml.

    Looks for config in:
    1. Provided config_path argument
    2. /app/config.yml (Docker mount)
    3. CONFIG_FILE environment variable
    4. Current directory config.yml

    Environment variable overrides:
    - ARIEL_DATABASE_HOST: Override database host (for Docker networking)

    Args:
        config_path: Optional explicit path to config file.

    Returns:
        ARIEL configuration dictionary.

    Raises:
        RuntimeError: If no config file is found.
    """
    config_paths = [
        Path(config_path) if config_path else None,
        Path("/app/config.yml"),
        Path(os.environ.get("CONFIG_FILE", "")) if os.environ.get("CONFIG_FILE") else None,
        Path("config.yml"),
    ]

    for path in config_paths:
        if path and path.exists() and path.is_file():
            logger.info(f"Loading config from {path}")
            with open(path) as f:
                config = yaml.safe_load(f)
                ariel_config = config.get("ariel", {})

            # Apply environment variable overrides for Docker networking
            db_host_override = os.environ.get("ARIEL_DATABASE_HOST")
            if db_host_override and "database" in ariel_config:
                uri = ariel_config["database"].get("uri", "")
                if uri:
                    # Replace localhost or 127.0.0.1 with the override host
                    new_uri = re.sub(
                        r"@(localhost|127\.0\.0\.1):",
                        f"@{db_host_override}:",
                        uri,
                    )
                    if new_uri != uri:
                        logger.info(f"Overriding database host with {db_host_override}")
                        ariel_config["database"]["uri"] = new_uri

            return ariel_config

    raise RuntimeError(
        "No config.yml found. Set CONFIG_FILE environment variable "
        "or mount config.yml at /app/config.yml"
    )


def _create_lifespan(config_path: str | Path | None = None):
    """Create a lifespan context manager with the given config path.

    Args:
        config_path: Optional path to config file.

    Returns:
        Async context manager for FastAPI lifespan.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Manage application lifecycle.

        Initialize ARIEL service on startup, cleanup on shutdown.
        """
        from osprey.services.ariel_search import ARIELConfig, create_ariel_service

        logger.info("Starting ARIEL Web Interface...")

        # Initialize the framework-only registry before any search code runs.
        # ARIEL doesn't need an application registry (e.g. my_control_assistant/registry.py),
        # but config.yml may reference one. Pre-creating the singleton with no registry_path
        # prevents get_registry() from failing when the capabilities/agent/enhancement
        # code paths try to load it.
        try:
            import osprey.registry.manager as _reg_mod

            _reg_mod._registry = _reg_mod.RegistryManager(registry_path=None)
            _reg_mod._registry.initialize(silent=True)
            logger.info("Framework registry initialized for ARIEL")
        except Exception as e:
            logger.warning(f"Registry initialization failed (non-fatal): {e}")

        # Load configuration
        config_dict = load_ariel_config(config_path)
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

    return lifespan


def create_app(config_path: str | Path | None = None) -> FastAPI:
    """Create the ARIEL Web Interface FastAPI application.

    App factory for ASGI servers and testing.

    Args:
        config_path: Optional path to config.yml file. If not provided,
            will search standard locations.

    Returns:
        Configured FastAPI application instance.
    """
    from osprey.interfaces.ariel.api.routes import router as api_router

    app = FastAPI(
        title="ARIEL Search Interface",
        description="Agentic Retrieval Interface for Electronic Logbooks",
        version="1.0.0",
        lifespan=_create_lifespan(config_path),
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

    # Root route - serve index.html
    @app.get("/")
    async def root():
        """Serve main index.html."""
        return FileResponse(STATIC_DIR / "index.html")

    # Health check endpoint
    @app.get("/health")
    async def health():
        """Simple health check endpoint."""
        if hasattr(app.state, "ariel_service"):
            healthy, message = await app.state.ariel_service.health_check()
            return {"status": "healthy" if healthy else "degraded", "message": message}
        return {"status": "starting", "message": "Service initializing"}

    # Mount static assets
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app


def run_web(
    host: str = "127.0.0.1",
    port: int = 8085,
    reload: bool = False,
    config_path: str | None = None,
) -> None:
    """Run the ARIEL web interface.

    CLI entry point for launching the web server.

    Args:
        host: Host to bind to.
        port: Port to run on.
        reload: Enable auto-reload for development.
        config_path: Optional path to config file.
    """
    import uvicorn

    # For reload mode, we need to use string reference
    if reload:
        uvicorn.run(
            "osprey.interfaces.ariel.app:create_app",
            factory=True,
            host=host,
            port=port,
            reload=reload,
            log_level="info",
        )
    else:
        # For non-reload mode, we can pass the app directly
        app = create_app(config_path)
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
        )
