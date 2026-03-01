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

from osprey.interfaces.common_middleware import NoCacheStaticMiddleware
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
        from osprey.utils.workspace import load_osprey_config
        from osprey.infrastructure.server_launcher import ensure_artifact_server

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
        from osprey.utils.workspace import load_osprey_config
        from osprey.infrastructure.server_launcher import ensure_ariel_server

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


def _launch_tuning_server(app: FastAPI) -> None:
    """Auto-launch the tuning panel server if configured."""
    try:
        from osprey.utils.workspace import load_osprey_config
        from osprey.infrastructure.server_launcher import ensure_tuning_server

        config = load_osprey_config()
        tuning_web = config.get("tuning", {}).get("web", {})
        host = tuning_web.get("host", "127.0.0.1")
        port = tuning_web.get("port", 8090)

        app.state.tuning_server_url = f"http://{host}:{port}"
        ensure_tuning_server()
        logger.info("Tuning server available at %s", app.state.tuning_server_url)
    except Exception:
        logger.warning("Could not auto-launch tuning server", exc_info=True)
        app.state.tuning_server_url = None


def _launch_deplot_server(app: FastAPI) -> None:
    """Auto-launch the DePlot graph extraction service if configured."""
    try:
        from osprey.utils.workspace import load_osprey_config
        from osprey.infrastructure.server_launcher import ensure_deplot_server

        config = load_osprey_config()
        deplot = config.get("deplot", {})
        if not deplot:
            return
        host = deplot.get("host", "127.0.0.1")
        port = deplot.get("port", 8095)

        app.state.deplot_server_url = f"http://{host}:{port}"
        ensure_deplot_server()
        logger.info("DePlot server available at %s", app.state.deplot_server_url)
    except Exception:
        logger.warning("Could not auto-launch DePlot server", exc_info=True)
        app.state.deplot_server_url = None


def _launch_channel_finder_server(app: FastAPI) -> None:
    """Auto-launch the Channel Finder web server if configured."""
    try:
        from osprey.utils.workspace import load_osprey_config
        from osprey.infrastructure.server_launcher import ensure_channel_finder_server

        config = load_osprey_config()
        cf = config.get("channel_finder", {})
        if not cf:
            return
        cf_web = cf.get("web", {})
        host = cf_web.get("host", "127.0.0.1")
        port = cf_web.get("port", 8092)

        app.state.channel_finder_server_url = f"http://{host}:{port}"
        ensure_channel_finder_server()
        logger.info("Channel Finder server available at %s", app.state.channel_finder_server_url)
    except Exception:
        logger.warning("Could not auto-launch Channel Finder server", exc_info=True)
        app.state.channel_finder_server_url = None


def _launch_agentsview_server(app: FastAPI) -> None:
    """Auto-launch the agentsview session analytics server if configured."""
    try:
        from osprey.utils.workspace import load_osprey_config

        config = load_osprey_config()
        av_config = config.get("agentsview", {})
        host = av_config.get("host", "127.0.0.1")
        port = av_config.get("port", 8096)

        app.state.agentsview_server_url = f"http://{host}:{port}"
        app.state.agentsview_project = av_config.get("project")

        from osprey.interfaces.agentsview.launcher import ensure_agentsview

        ensure_agentsview()
        logger.info("agentsview available at %s", app.state.agentsview_server_url)
    except Exception:
        logger.warning("Could not auto-launch agentsview", exc_info=True)
        app.state.agentsview_server_url = None
        app.state.agentsview_project = None


def _launch_cui_server(app: FastAPI) -> None:
    """Auto-launch the CUI server subprocess if configured."""
    try:
        from osprey.utils.workspace import load_osprey_config

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


# Panel classification constants
UNIVERSAL_PANELS = {"artifacts", "session", "session-analytics"}
BUILTIN_PANELS = {"artifacts", "ariel", "tuning", "channel-finder", "session", "session-analytics"}


def _load_panel_config() -> tuple[set[str], list[dict]]:
    """Read web.panels from config.yml.

    Returns:
        (enabled_builtin_ids, custom_panel_defs)
    """
    try:
        from osprey.utils.workspace import load_osprey_config

        config = load_osprey_config()
    except Exception:
        return set(UNIVERSAL_PANELS), []

    panels_config = config.get("web", {}).get("panels", {})

    enabled = set(UNIVERSAL_PANELS)  # Always on
    custom = []

    for panel_id, spec in panels_config.items():
        if panel_id in BUILTIN_PANELS:
            if spec is True or (isinstance(spec, dict) and spec.get("enabled", True)):
                enabled.add(panel_id)
        else:
            custom.append(
                {
                    "id": panel_id,
                    "label": spec.get("label", panel_id.upper()),
                    "url": spec.get("url", ""),
                    "healthEndpoint": spec.get("health_endpoint"),
                }
            )

    return enabled, custom


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

        import uuid

        app.state.server_session_id = uuid.uuid4().hex[:12]
        app.state.shell_command = shell_command or config.get("shell") or "claude"
        max_bg = int(config.get("max_background_sessions", 5))
        app.state.pty_registry = PtyRegistry(max_background=max_bg)
        app.state.operator_registry = OperatorRegistry()
        app.state.project_cwd = str(
            Path(project_dir).resolve() if project_dir else Path.cwd().resolve()
        )
        app.state.broadcaster = FileEventBroadcaster()
        app.state.active_panel = None

        # Ensure OSPREY_CONFIG is set before any load_osprey_config() call
        if "OSPREY_CONFIG" not in os.environ:
            candidate = Path(app.state.project_cwd) / "config.yml"
            if candidate.exists():
                os.environ["OSPREY_CONFIG"] = str(candidate)
                logger.debug("Auto-set OSPREY_CONFIG=%s", candidate)

        # Clear any stale config cache (e.g. from web_cmd.py pre-lifespan call)
        from osprey.utils.workspace import reset_config_cache

        reset_config_cache()

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

        # Load panel config and conditionally launch servers
        enabled_panels, custom_panels = _load_panel_config()
        app.state.enabled_panels = enabled_panels
        app.state.custom_panels = custom_panels

        # Universal servers — always launched
        _launch_artifact_server(app)
        _launch_cui_server(app)
        _launch_agentsview_server(app)

        # Domain servers — template-controlled
        if "ariel" in enabled_panels:
            _launch_ariel_server(app)
        if "tuning" in enabled_panels:
            _launch_tuning_server(app)
        if "channel-finder" in enabled_panels:
            _launch_channel_finder_server(app)

        # Standalone services — always launched (not panel-tied)
        _launch_deplot_server(app)

        # Hook debug env — propagated to PTY/SDK sessions like OTEL
        app.state.hooks_env = {}
        try:
            from osprey.utils.workspace import load_osprey_config

            hooks_config = load_osprey_config().get("hooks", {})
            if hooks_config.get("debug"):
                app.state.hooks_env["OSPREY_HOOK_DEBUG"] = "1"
                logger.info("Hook debugging enabled (OSPREY_HOOK_DEBUG=1)")
        except Exception:
            logger.warning("Failed to read hooks config", exc_info=True)

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

        # Stop agentsview subprocess
        try:
            from osprey.interfaces.agentsview.launcher import stop_agentsview

            stop_agentsview()
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


def _open_browser_when_ready(url: str, timeout: float = 15.0) -> None:
    """Wait for the server to accept connections, then open the browser."""
    import socket
    import threading
    import time
    import webbrowser
    from urllib.parse import urlparse

    def _wait_and_open():
        parsed = urlparse(url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 8087
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((host, port), timeout=0.5):
                    break
            except OSError:
                time.sleep(0.3)
        else:
            return  # Server didn't start in time; skip browser open
        webbrowser.open(url)

    t = threading.Thread(target=_wait_and_open, daemon=True)
    t.start()


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

    url = f"http://{host}:{port}"
    _open_browser_when_ready(url)

    app = create_app(config_path=config_path, shell_command=shell_command, project_dir=project_dir)
    uvicorn.run(app, host=host, port=port, log_level="info")
