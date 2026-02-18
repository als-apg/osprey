"""OSPREY Web Terminal — FastAPI Application.

A browser-based split-pane interface with a real terminal (running Claude Code
via PTY) on the left and a live workspace file viewer on the right.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from osprey.interfaces.web_terminal.file_watcher import FileEventBroadcaster, WorkspaceWatcher
from osprey.interfaces.web_terminal.operator_session import OperatorRegistry
from osprey.interfaces.web_terminal.pty_manager import PtyRegistry
from osprey.interfaces.web_terminal.routes import router

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

STATIC_DIR = Path(__file__).parent / "static"

logger = __import__("logging").getLogger(__name__)


def _launch_artifact_server(app: FastAPI) -> None:
    """Auto-launch the artifact gallery server if configured."""
    try:
        from osprey.mcp_server.common import load_osprey_config
        from osprey.mcp_server.server_launcher import ensure_artifact_server

        config = load_osprey_config()
        art_config = config.get("artifact_server", {})
        host = art_config.get("host", "127.0.0.1")
        port = art_config.get("port", 8086)

        app.state.artifact_server_url = f"http://{host}:{port}"
        ensure_artifact_server()
        logger.info("Artifact server available at %s", app.state.artifact_server_url)
    except Exception:
        logger.warning("Could not auto-launch artifact server", exc_info=True)
        app.state.artifact_server_url = "http://127.0.0.1:8086"


def _launch_ariel_server(app: FastAPI) -> None:
    """Auto-launch the ARIEL logbook server if configured."""
    try:
        from osprey.mcp_server.common import load_osprey_config
        from osprey.mcp_server.server_launcher import ensure_ariel_server

        config = load_osprey_config()
        ariel_web = config.get("ariel", {}).get("web", {})
        host = ariel_web.get("host", "127.0.0.1")
        port = ariel_web.get("port", 8085)

        app.state.ariel_server_url = f"http://{host}:{port}"
        ensure_ariel_server()
        logger.info("ARIEL server available at %s", app.state.ariel_server_url)
    except Exception:
        logger.warning("Could not auto-launch ARIEL server", exc_info=True)
        app.state.ariel_server_url = None


def _launch_cui_server(app: FastAPI) -> None:
    """Auto-launch the CUI server subprocess if configured."""
    try:
        from osprey.mcp_server.common import load_osprey_config

        config = load_osprey_config()
        cui_config = config.get("cui_server", {})
        host = cui_config.get("host", "127.0.0.1")
        port = cui_config.get("port", 3001)

        app.state.cui_server_url = f"http://{host}:{port}"

        from osprey.interfaces.cui.launcher import ensure_cui_server

        ensure_cui_server(cwd=getattr(app.state, "project_cwd", None))
        logger.info("CUI server available at %s", app.state.cui_server_url)
    except Exception:
        logger.warning("Could not auto-launch CUI server", exc_info=True)
        app.state.cui_server_url = None


def _load_web_config(config_path: str | Path | None = None) -> dict:
    """Load web_terminal config section from config.yml."""
    config_paths = [
        Path(config_path) if config_path else None,
        Path(os.environ.get("CONFIG_FILE", "")) if os.environ.get("CONFIG_FILE") else None,
        Path("config.yml"),
    ]

    for path in config_paths:
        if path and path.exists() and path.is_file():
            with open(path) as f:
                config = yaml.safe_load(f) or {}
            return config.get("web_terminal", {})

    return {}


def _create_lifespan(
    config_path: str | Path | None = None,
    shell_command: str = "claude",
    project_dir: str | Path | None = None,
):
    """Create a lifespan context manager for the app."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        config = _load_web_config(config_path)

        app.state.shell_command = shell_command or config.get("shell") or "claude"
        app.state.pty_registry = PtyRegistry()
        app.state.operator_registry = OperatorRegistry()
        app.state.project_cwd = str(
            Path(project_dir).resolve() if project_dir else Path.cwd().resolve()
        )
        app.state.broadcaster = FileEventBroadcaster()
        app.state.active_panel = None

        # Resolve and store config_path for the settings API
        resolved_config_path = None
        for candidate in [
            Path(config_path) if config_path else None,
            Path(os.environ.get("CONFIG_FILE", "")) if os.environ.get("CONFIG_FILE") else None,
            Path("config.yml"),
        ]:
            if candidate and candidate.exists() and candidate.is_file():
                resolved_config_path = candidate.resolve()
                break
        app.state.config_path = resolved_config_path

        workspace_dir = Path(config.get("watch_dir") or "./osprey-workspace").resolve()
        app.state.workspace_dir = workspace_dir  # base path (file watcher watches all sessions)
        app.state.workspace_base = workspace_dir  # alias for clarity
        app.state.watcher = WorkspaceWatcher(workspace_dir, app.state.broadcaster)
        app.state.watcher.start()

        # Auto-launch the artifact gallery server
        _launch_artifact_server(app)

        # Auto-launch the ARIEL logbook server
        _launch_ariel_server(app)

        # Auto-launch the CUI server
        _launch_cui_server(app)

        yield

        app.state.watcher.stop()
        app.state.pty_registry.cleanup_all()
        await app.state.operator_registry.cleanup_all()

        # Stop CUI subprocess (not a daemon thread — must be explicitly stopped)
        try:
            from osprey.interfaces.cui.launcher import stop_cui_server

            stop_cui_server()
        except Exception:
            pass

    return lifespan


def create_app(
    config_path: str | Path | None = None,
    shell_command: str = "claude",
    project_dir: str | Path | None = None,
) -> FastAPI:
    """Create the Web Terminal FastAPI application.

    Args:
        config_path: Optional path to config.yml.
        shell_command: Shell command to spawn in the PTY.
        project_dir: Optional OSPREY project directory. When set, used as
            ``project_cwd`` instead of the current working directory.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="OSPREY Web Terminal",
        description="Browser-based terminal with live workspace viewer",
        version="1.0.0",
        lifespan=_create_lifespan(config_path, shell_command, project_dir),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Prevent browsers from caching JS/CSS (avoids stale code after updates)
    class NoCacheStaticMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            response = await call_next(request)
            if request.url.path.startswith("/static/"):
                response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            return response

    app.add_middleware(NoCacheStaticMiddleware)

    app.include_router(router)

    # Mount CUI reverse proxy (constrains sessions to this project)
    try:
        from osprey.interfaces.cui.proxy import create_cui_proxy_mount

        app.routes.append(create_cui_proxy_mount())
    except Exception:
        logger.warning("Could not mount CUI proxy", exc_info=True)

    @app.get("/")
    async def root():
        return FileResponse(STATIC_DIR / "index.html")

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app


def run_web(
    host: str = "127.0.0.1",
    port: int = 8087,
    shell_command: str = "claude",
    config_path: str | None = None,
    project_dir: str | None = None,
) -> None:
    """Run the web terminal server.

    Args:
        host: Host to bind to.
        port: Port to run on.
        shell_command: Shell command to spawn in the PTY.
        config_path: Optional path to config file.
        project_dir: Optional OSPREY project directory.
    """
    import uvicorn

    app = create_app(config_path=config_path, shell_command=shell_command, project_dir=project_dir)
    uvicorn.run(app, host=host, port=port, log_level="info")
