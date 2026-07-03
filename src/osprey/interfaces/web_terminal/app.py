"""OSPREY Web Terminal — FastAPI Application.

A browser-based split-pane interface with a real terminal (running Claude Code
via PTY) on the left and a live workspace file viewer on the right.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import httpx
import yaml
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from osprey.interfaces.common_middleware import ExceptionLoggingMiddleware, NoCacheStaticMiddleware
from osprey.interfaces.vendor import vendor_url
from osprey.interfaces.web_terminal.file_watcher import FileEventBroadcaster, WorkspaceWatcher
from osprey.interfaces.web_terminal.operator_session import OperatorRegistry
from osprey.interfaces.web_terminal.pty_manager import PtyRegistry
from osprey.interfaces.web_terminal.routes import router
from osprey.profiles.web_panels import BUILTIN_PANELS, UNIVERSAL_PANELS

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

STATIC_DIR = Path(__file__).parent / "static"

templates = Jinja2Templates(directory=str(STATIC_DIR))
templates.env.globals["vendor_url"] = vendor_url

logger = __import__("logging").getLogger(__name__)


def _launch_artifact_server(app: FastAPI) -> None:
    """Auto-launch the artifact gallery server if configured."""
    try:
        import os

        from osprey.infrastructure.server_launcher import ensure_artifact_server
        from osprey.utils.workspace import load_osprey_config

        config = load_osprey_config()
        art_config = config.get("artifact_server", {})
        host = art_config.get("host", "127.0.0.1")
        port = int(os.environ.get("OSPREY_ARTIFACT_SERVER_PORT", art_config.get("port", 8086)))

        app.state.artifact_server_url = f"http://{host}:{port}"
        ensure_artifact_server()
        logger.info("Artifact server available at %s", app.state.artifact_server_url)
    except Exception:
        logger.warning("Could not auto-launch artifact server", exc_info=True)
        app.state.artifact_server_url = "http://127.0.0.1:8086"


def _launch_ariel_server(app: FastAPI) -> None:
    """Auto-launch the ARIEL logbook server if configured."""
    try:
        from osprey.infrastructure.server_launcher import ensure_ariel_server
        from osprey.utils.workspace import load_osprey_config

        config = load_osprey_config()
        ariel_web = config.get("ariel", {}).get("web", {})
        host = ariel_web.get("host", "127.0.0.1")
        port = int(os.environ.get("OSPREY_ARIEL_PORT", ariel_web.get("port", 8085)))

        app.state.ariel_server_url = f"http://{host}:{port}"
        ensure_ariel_server()
        logger.info("ARIEL server available at %s", app.state.ariel_server_url)
    except Exception:
        logger.warning("Could not auto-launch ARIEL server", exc_info=True)
        app.state.ariel_server_url = None


def _launch_tuning_server(app: FastAPI) -> None:
    """Auto-launch the tuning panel server if configured."""
    try:
        from osprey.infrastructure.server_launcher import ensure_tuning_server
        from osprey.utils.workspace import load_osprey_config

        config = load_osprey_config()
        tuning_web = config.get("tuning", {}).get("web", {})
        host = tuning_web.get("host", "127.0.0.1")
        port = int(os.environ.get("OSPREY_TUNING_PORT", tuning_web.get("port", 8090)))

        app.state.tuning_server_url = f"http://{host}:{port}"
        ensure_tuning_server()
        logger.info("Tuning server available at %s", app.state.tuning_server_url)
    except Exception:
        logger.warning("Could not auto-launch tuning server", exc_info=True)
        app.state.tuning_server_url = None


def _launch_channel_finder_server(app: FastAPI) -> None:
    """Auto-launch the Channel Finder web server if configured."""
    try:
        from osprey.infrastructure.server_launcher import ensure_channel_finder_server
        from osprey.utils.workspace import load_osprey_config

        config = load_osprey_config()
        cf = config.get("channel_finder", {})
        if not cf:
            return
        cf_web = cf.get("web", {})
        host = cf_web.get("host", "127.0.0.1")
        port = int(os.environ.get("OSPREY_CHANNEL_FINDER_PORT", cf_web.get("port", 8092)))

        app.state.channel_finder_server_url = f"http://{host}:{port}"
        ensure_channel_finder_server()
        logger.info("Channel Finder server available at %s", app.state.channel_finder_server_url)
    except Exception:
        logger.warning("Could not auto-launch Channel Finder server", exc_info=True)
        app.state.channel_finder_server_url = None


def _launch_lattice_dashboard_server(app: FastAPI) -> None:
    """Auto-launch the lattice dashboard server if configured."""
    try:
        from osprey.infrastructure.server_launcher import ensure_lattice_dashboard_server
        from osprey.utils.workspace import load_osprey_config

        config = load_osprey_config()
        ld = config.get("lattice_dashboard", {})
        if not ld:
            return
        host = ld.get("host", "127.0.0.1")
        port = int(os.environ.get("OSPREY_LATTICE_DASHBOARD_PORT", ld.get("port", 8097)))

        app.state.lattice_dashboard_server_url = f"http://{host}:{port}"
        ensure_lattice_dashboard_server()
        logger.info("Lattice dashboard available at %s", app.state.lattice_dashboard_server_url)
    except Exception:
        logger.warning("Could not auto-launch lattice dashboard", exc_info=True)
        app.state.lattice_dashboard_server_url = None


def _load_panel_config() -> tuple[set[str], list[dict], str | None]:
    """Read web.panels and web.default_panel from config.yml.

    Returns:
        (enabled_builtin_ids, custom_panel_defs, default_panel_id_or_None)

        The default panel id is returned as declared by the profile/config;
        it is **not** validated here — the frontend treats an unknown id as
        a request to fall back to DEFAULT_PANEL_FALLBACK so a typo doesn't
        leave the user staring at a blank tabset.
    """
    try:
        from osprey.utils.workspace import load_osprey_config

        config = load_osprey_config()
    except Exception:
        return set(UNIVERSAL_PANELS), [], None

    web_config = config.get("web", {})
    panels_config = web_config.get("panels", {})
    default_panel = web_config.get("default_panel")

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
                    "path": spec.get("path", "/"),
                }
            )

    return enabled, custom, default_panel


class _PanelRuntimeConfig(NamedTuple):
    """Runtime-panel settings derived from config, plus the computed visible list."""

    allow_runtime_panels: bool
    runtime_panel_allowlist: list[str] | None
    visible_panels: list[str]


def _load_panel_runtime_config(
    enabled_panels: set[str], custom_panels: list[dict]
) -> _PanelRuntimeConfig:
    """Read runtime-panel settings and compute the visible-panel list.

    Honors per-panel ``hidden: true`` flags and the ``web.allow_runtime_panels`` /
    ``web.runtime_panel_allowlist`` knobs.  The raw config is re-read here rather
    than threaded through ``_load_panel_config``'s 3-tuple contract (which is
    relied on elsewhere, including tests).  Built-in panel specs are not retained
    by ``_load_panel_config`` — only the id lands in ``enabled`` — so hidden
    built-ins are tracked in a parallel set.

    ``visible_panels`` is the flat list of ids shown in the UI: enabled built-ins
    (minus hidden ones) followed by custom panels (minus hidden ones).  With no
    ``hidden`` flags it equals all enabled panels — backward compatible.

    Fails open: any config-read error yields the permissive defaults (nothing
    hidden, runtime registration off).
    """
    hidden_builtins: set[str] = set()
    hidden_custom_ids: set[str] = set()
    allow_runtime_panels = False
    runtime_panel_allowlist: list[str] | None = None
    try:
        from osprey.utils.workspace import load_osprey_config

        web_cfg = load_osprey_config().get("web", {})
        allow_runtime_panels = bool(web_cfg.get("allow_runtime_panels", False))
        allowlist_raw = web_cfg.get("runtime_panel_allowlist")
        if isinstance(allowlist_raw, list):
            # Lowercase at parse time so matching in _validate_panel_url is case-insensitive.
            runtime_panel_allowlist = [str(e).lower() for e in allowlist_raw]
        for pid, spec in web_cfg.get("panels", {}).items():
            if isinstance(spec, dict) and spec.get("hidden", False):
                if pid in BUILTIN_PANELS:
                    hidden_builtins.add(pid)
                else:
                    hidden_custom_ids.add(pid)
    except Exception:
        pass

    visible_panels = [p for p in enabled_panels if p not in hidden_builtins] + [
        cp["id"] for cp in custom_panels if cp["id"] not in hidden_custom_ids
    ]
    return _PanelRuntimeConfig(
        allow_runtime_panels=allow_runtime_panels,
        runtime_panel_allowlist=runtime_panel_allowlist,
        visible_panels=visible_panels,
    )


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


def _load_claude_code_config(config_path: str | Path | None = None) -> dict:
    """Load claude_code config section from config.yml.

    Mirrors :func:`_load_web_config` so the lifespan can derive the Claude
    Code launch argv (honoring ``claude_code.cli_version`` pins) even when
    no explicit ``shell_command`` was passed — e.g. under ``uvicorn --reload``
    where ``create_app`` is called with no arguments.
    """
    config_paths = [
        Path(config_path) if config_path else None,
        Path(os.environ.get("CONFIG_FILE", "")) if os.environ.get("CONFIG_FILE") else None,
        Path("config.yml"),
    ]

    for path in config_paths:
        if path and path.exists() and path.is_file():
            with open(path) as f:
                config = yaml.safe_load(f) or {}
            return config.get("claude_code", {})

    return {}


def _create_lifespan(
    config_path: str | Path | None = None,
    shell_command: list[str] | None = None,
    project_dir: str | Path | None = None,
):
    """Create a lifespan context manager for the app."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        from osprey.utils.claude_launcher import build_claude_launch_argv

        config = _load_web_config(config_path)

        import uuid

        app.state.server_session_id = uuid.uuid4().hex[:12]
        # Shell-command precedence — always normalized to list[str] so every
        # downstream consumer (websocket initial spawn + switch_session) can
        # safely unpack with [*base, ...]. The pin lookup lets --reload mode
        # honor claude_code.cli_version even though uvicorn's factory bypass
        # never lets web_cmd.py inject the argv.
        if shell_command:
            app.state.shell_command = list(shell_command)
        elif config.get("shell"):
            app.state.shell_command = [str(config["shell"])]
        else:
            app.state.shell_command = build_claude_launch_argv(
                _load_claude_code_config(config_path)
            )
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

        # ── Regenerate stale Claude Code artifacts on launch ──
        # config.yml is a build-time input: safety-critical fields (e.g. the
        # writes_enabled kill-switch baked into settings.json's permissions.deny)
        # only take effect once the artifacts are re-rendered. Regenerating here
        # — mirroring `osprey claude chat` — means an edited config.yml is honored
        # on the next server start. Fail open so a regen error never blocks launch.
        try:
            from osprey.cli.templates.manager import TemplateManager

            project_dir_for_regen = Path(app.state.project_cwd)
            changed = TemplateManager().regen_if_drift(project_dir_for_regen)
            if changed:
                logger.info(
                    "Regenerated %d stale Claude Code artifact(s): %s",
                    len(changed),
                    ", ".join(changed),
                )
        except Exception:  # noqa: BLE001 — never let regen block server startup
            logger.warning("Claude Code artifact regen on launch failed", exc_info=True)

        # ── Provider env injection ──
        from osprey.cli.claude_code_resolver import inject_provider_env, load_provider_spec

        if app.state.config_path:
            _project_dir = Path(app.state.config_path).parent
            # load_provider_spec expands ${VAR} in provider config before resolving.
            _spec = load_provider_spec(_project_dir)
            if _spec:
                inject_provider_env(os.environ, _spec, project_dir=_project_dir)

                # Start translation proxy for OpenAI-compatible providers
                if _spec.needs_proxy and _spec.upstream_base_url:
                    from osprey.infrastructure.proxy.lifecycle import start_proxy

                    proxy_port = start_proxy(
                        _spec.upstream_base_url,
                        os.environ.get(_spec.auth_env_var),
                    )
                    os.environ["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{proxy_port}"
                    logger.info(
                        "Translation proxy on :%d → %s",
                        proxy_port,
                        _spec.upstream_base_url,
                    )

        workspace_dir = Path(config.get("watch_dir") or "./_agent_data").resolve()
        app.state.workspace_dir = workspace_dir  # base path (file watcher watches all sessions)
        app.state.workspace_base = workspace_dir  # alias for clarity
        app.state.watcher = WorkspaceWatcher(workspace_dir, app.state.broadcaster)
        app.state.watcher.start()

        # Load panel config and conditionally launch servers
        enabled_panels, custom_panels, default_panel = _load_panel_config()
        app.state.enabled_panels = enabled_panels
        app.state.custom_panels = custom_panels
        app.state.default_panel = default_panel

        # Runtime-panel settings + visibility (honors hidden: true,
        # allow_runtime_panels, runtime_panel_allowlist).
        panel_runtime = _load_panel_runtime_config(enabled_panels, custom_panels)
        app.state.allow_runtime_panels = panel_runtime.allow_runtime_panels
        app.state.runtime_panel_allowlist = panel_runtime.runtime_panel_allowlist
        app.state.visible_panels = panel_runtime.visible_panels
        if panel_runtime.allow_runtime_panels and not panel_runtime.runtime_panel_allowlist:
            logger.warning(
                "web.allow_runtime_panels is enabled without a runtime_panel_allowlist — "
                "any http/https host on the internal network can be registered as a panel proxy."
            )

        # Universal servers — always launched
        _launch_artifact_server(app)

        # Domain servers — template-controlled
        if "ariel" in enabled_panels:
            _launch_ariel_server(app)
        if "tuning" in enabled_panels:
            _launch_tuning_server(app)
        if "channel-finder" in enabled_panels:
            _launch_channel_finder_server(app)
        if "lattice" in enabled_panels:
            _launch_lattice_dashboard_server(app)

        # Hook env placeholder — hooks read config.yml directly for
        # hot-reloadable settings (no env var propagation needed).
        app.state.hooks_env = {}

        # Shared httpx client for the panel reverse proxy.
        # trust_env=False prevents routing through the corporate HTTP proxy
        # (e.g. Squid) — all panel backends are container-local or on the
        # Docker network and must be reached directly.
        app.state.proxy_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=5.0),
            follow_redirects=True,
            trust_env=False,
        )

        yield

        await app.state.proxy_client.aclose()

        # Stop translation proxy if it was started
        from osprey.infrastructure.proxy.lifecycle import stop_proxy

        stop_proxy()

        app.state.watcher.stop()
        app.state.pty_registry.cleanup_all()
        await app.state.operator_registry.cleanup_all()

    return lifespan


def create_app(
    config_path: str | Path | None = None,
    shell_command: list[str] | None = None,
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
    app.add_middleware(ExceptionLoggingMiddleware)

    app.include_router(router)

    @app.get("/")
    async def root(request: Request):
        return templates.TemplateResponse(request, "index.html", {})

    # Mount shared fonts before /static (Starlette matches in declaration order)
    SHARED_FONTS_DIR = Path(__file__).parent.parent / "shared_fonts"
    if SHARED_FONTS_DIR.exists():
        app.mount("/static/fonts", StaticFiles(directory=SHARED_FONTS_DIR), name="shared-fonts")
    DESIGN_SYSTEM_STATIC_DIR = Path(__file__).parent.parent / "design_system" / "static"
    if DESIGN_SYSTEM_STATIC_DIR.exists():
        app.mount(
            "/design-system", StaticFiles(directory=DESIGN_SYSTEM_STATIC_DIR), name="design-system"
        )
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
    shell_command: list[str] | None = None,
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
