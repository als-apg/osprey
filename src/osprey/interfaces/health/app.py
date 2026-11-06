"""OSPREY System Health panel — FastAPI application factory.

The web surface over the P1 health framework: a sidecar that re-runs the same
suite the ``osprey health`` CLI runs, caches the report, and serves it to the
SYSTEM dashboard tab. Launched in-process by ``ServerLauncher`` and
reverse-proxied at ``/panel/system-health/`` — the ``channel_finder`` builtin
pattern.

Three collaborators are wired here and cached on ``app.state``:

* :class:`~osprey.health.loader.HealthConfigLoader` — the synchronous
  config-load phase (mtime-gated ``.env``/``config.yml`` → merged records);
* :class:`~osprey.health.lifecycle.HealthRuntimeLifecycle` — the sole
  owner of the control-system connector, with loop-affine teardown;
* :class:`~osprey.interfaces.health.engine.HealthCheckEngine` — the single-flight
  cache/scheduler backing ``/checks``.

**Guarded construction (okf invariant).** The factory never raises: a launcher
runs it on a swallowed daemon thread, so an exception would leave a silent dead
tab. Any failure wiring the engine degrades to ``app.state.engine = None`` —
``/health`` still returns 200 and ``/checks`` serves a constant degraded
envelope. A missing or invalid ``config.yml`` is *not* such a failure: the engine
handles it, serving core-category results with default cadence.

**Operator actions.** ``GET /mcp`` and ``POST /mcp/probe`` serve the MCP
servers' own description of themselves (spawning them only when nothing is
cached, or on an explicit re-probe — ``/checks`` never does);
``POST /reprobe`` re-runs part of the report through the engine's single
flight; ``POST /ping`` sends one one-token completion to one provider. Each is
a click, never a poll.

**Lifespan registers teardown only.** No suite runs at startup; the first
``/checks`` kicks the first refresh. Startup binds the owning event loop and
registers the ``atexit`` connector-teardown hook; shutdown cancels any in-flight
refresh and disconnects the connector on that loop, then unregisters the hook.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import Body, FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse

from osprey.health.config import parse_health_config
from osprey.health.models import CheckReport
from osprey.interfaces._app_setup import configure_interface_app

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from osprey.health.lifecycle import HealthRuntimeLifecycle
    from osprey.interfaces.health.engine import HealthCheckEngine

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

SERVICE_NAME = "system-health"

#: Per-server bound on one MCP introspection (spawn + initialize + tools/list).
_MCP_TIMEOUT_S = 30.0

#: Introspection keys the panel is not served: the server's environment can
#: carry secrets, and ``description`` repeats ``instructions``.
_PRIVATE_MCP_KEYS = ("env", "description")


#: ``hook_config.json`` lists naming a tool that writes; ``governed_tools`` is
#: the set the write chain gates.
_WRITE_LISTS = (
    "write_tools",
    "control_system_write_tools",
    "facility_write_tools",
    "mixed_read_write_tools",
)


def _public(server: dict[str, Any], hook_config: dict[str, Any] | None) -> dict[str, Any]:
    """A server as the panel is served it: no env, and each tool's hook coverage.

    With the render's ``hook_config.json`` (the write hooks' own contract),
    every tool carries ``hooks: {"write", "gated"}`` read from it, so the panel
    tags a tool as gated only when the chain actually gates it. Without the
    contract no tool carries ``hooks`` and the panel falls back to the tool's
    annotations or name.
    """
    out = {k: v for k, v in server.items() if k not in _PRIVATE_MCP_KEYS}
    if hook_config is None or not isinstance(out.get("tools"), list):
        return out
    writes = {t for key in _WRITE_LISTS for t in hook_config.get(key) or [] if isinstance(t, str)}
    gated = {t for t in hook_config.get("governed_tools") or [] if isinstance(t, str)}
    tools = []
    for tool in out["tools"]:
        full = f"mcp__{server.get('name')}__{tool.get('name')}"
        tools.append({**tool, "hooks": {"write": full in writes, "gated": full in gated}})
    out["tools"] = tools
    return out


def _hook_config(render: Path) -> dict[str, Any] | None:
    """The render's ``.claude/hooks/hook_config.json``, or ``None`` when unreadable."""
    import json

    try:
        data = json.loads((render / ".claude" / "hooks" / "hook_config.json").read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _build_engine(
    config_path: str | Path | None,
) -> tuple[HealthCheckEngine | None, HealthRuntimeLifecycle | None]:
    """Wire the loader → lifecycle → engine, or ``(None, None)`` on failure.

    Never raises: an unexpected wiring error degrades the panel to the
    engine-less guarded path rather than killing the launcher's daemon thread.
    Constructing these objects performs no config I/O — the first ``load`` is
    deferred to the first refresh — so this succeeds even with a missing config.
    """
    try:
        from osprey.health.lifecycle import HealthRuntimeLifecycle
        from osprey.health.loader import HealthConfigLoader
        from osprey.interfaces.health.engine import HealthCheckEngine

        loader = HealthConfigLoader(Path(config_path) if config_path is not None else None)
        lifecycle = HealthRuntimeLifecycle(restart_hint="restart the web terminal")
        engine = HealthCheckEngine(loader=loader, lifecycle=lifecycle, config_path=config_path)
        return engine, lifecycle
    except Exception:  # degrade to guarded mode, never raise.
        logger.warning(
            "system-health: could not wire the refresh engine; serving guarded app "
            "(/checks returns a degraded envelope).",
            exc_info=True,
        )
        return None, None


def _degraded_envelope() -> dict[str, Any]:
    """Constant ``/checks`` envelope for the engine-less guarded path.

    Uses the framework degraded defaults (``interval_s=60``) so a client polls at
    a sane cadence even when no engine could be wired.
    """
    settings = parse_health_config(None)
    base = CheckReport().to_dict()
    base["stale"] = True
    base["warming"] = False
    base["interval_s"] = settings.interval_s
    base["title"] = settings.title
    return base


def create_app(config_path: str | Path | None = None) -> FastAPI:
    """Create the System Health panel FastAPI application.

    Args:
        config_path: Explicit ``config.yml`` path. ``None`` (the default)
            resolves like the CLI (``OSPREY_CONFIG`` env, else ``./config.yml``)
            on each refresh, matching launcher/terminal parity.

    Returns:
        A configured FastAPI app. Guaranteed constructible even with a missing or
        invalid config; the refresh engine (or the guarded fallback) is cached on
        ``app.state.engine`` and its runtime owner on ``app.state.lifecycle``.
    """
    engine, lifecycle = _build_engine(config_path)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # Startup: capture the owning loop and arm the atexit connector teardown.
        if lifecycle is not None:
            lifecycle.bind_loop()
            lifecycle.register_atexit()
        try:
            yield
        finally:
            # Shutdown: cancel any in-flight refresh, disconnect on this loop,
            # and unregister the atexit hook (so stacked TestClients don't leak).
            if lifecycle is not None:
                await lifecycle.shutdown()

    app = FastAPI(
        title="OSPREY System Health",
        description="Web surface over the OSPREY health-check framework",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.engine = engine
    app.state.lifecycle = lifecycle

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Liveness probe — constant-time, executes no checks (200 even guarded).

        ``configured`` reflects whether the last completed refresh loaded a
        usable config; the tab's LED polls this for sidecar liveness only.
        """
        return {
            "status": "ok",
            "service": SERVICE_NAME,
            "configured": bool(engine is not None and engine.config_ok),
        }

    @app.get("/checks")
    async def checks(
        categories: Annotated[list[str] | None, Query()] = None,  # noqa: ARG001 - declared query parameter; the report covers every category
    ) -> dict[str, Any]:
        """Return the health-report envelope, constant-time in all states.

        ``categories`` is accepted and ignored (documented; no P2 consumer): the
        served report is always the full, unfiltered suite so the never-elevate
        cost-tiering invariant holds trivially.
        """
        if engine is None:
            return _degraded_envelope()
        return engine.get_checks()

    @app.get("/mcp", response_model=None)
    async def mcp_servers() -> dict[str, Any] | JSONResponse:
        """The MCP servers' introspection: cached, spawned only when nothing is cached."""
        return await _mcp(refresh=False, server=None)

    @app.post("/mcp/probe", response_model=None)
    async def mcp_probe(
        body: Annotated[dict[str, Any] | None, Body()] = None,
    ) -> dict[str, Any] | JSONResponse:
        """Re-introspect one server (``{"server": name}``) or every server (``{}``)."""
        server = (body or {}).get("server")
        return await _mcp(refresh=True, server=server if isinstance(server, str) else None)

    async def _mcp(*, refresh: bool, server: str | None) -> dict[str, Any] | JSONResponse:
        from osprey.mcp_server.introspect import get_mcp_servers_cached, peek_mcp_servers

        path = engine.mcp_json_path() if engine is not None else None
        if path is None or not path.exists():
            return {"probed_at": None, "servers": []}
        try:
            servers = await get_mcp_servers_cached(
                path, str(path.parent), timeout=_MCP_TIMEOUT_S, refresh=refresh, server=server
            )
        except Exception as exc:  # an unreadable .mcp.json is an answer, not a crash
            logger.warning("system-health: MCP introspection failed", exc_info=True)
            return JSONResponse({"detail": f"introspection failed: {exc}"}, status_code=500)
        _, probed_at = peek_mcp_servers(path)
        hook_config = _hook_config(path.parent)
        return {"probed_at": probed_at, "servers": [_public(s, hook_config) for s in servers]}

    @app.post("/reprobe", response_model=None)
    async def reprobe(
        body: Annotated[dict[str, Any] | None, Body()] = None,
    ) -> dict[str, Any] | JSONResponse:
        """Re-run the rows named (``checks``), one ``component``, or everything (``{}``)."""
        from osprey.interfaces.health.engine import RefreshSuppressedError

        if engine is None:
            return _degraded_envelope()
        body = body or {}
        checks = [c for c in body.get("checks") or [] if isinstance(c, str)]
        component = body.get("component") if isinstance(body.get("component"), str) else None
        try:
            return await engine.reprobe(checks=checks, component=component)
        except RefreshSuppressedError as exc:
            return JSONResponse({"detail": str(exc), "suppressed": True}, status_code=409)

    @app.post("/ping", response_model=None)
    async def ping(
        body: Annotated[dict[str, Any] | None, Body()] = None,
    ) -> dict[str, Any] | JSONResponse:
        """One one-token completion to ``{"provider": name}``."""
        from osprey.health.ping import ping_provider

        provider = (body or {}).get("provider")
        if not isinstance(provider, str) or not provider:
            return JSONResponse({"detail": "provider is required"}, status_code=422)
        return await ping_provider(engine.expanded if engine is not None else None, provider)

    @app.get("/", response_model=None)
    async def root() -> FileResponse | JSONResponse:
        """Serve the dashboard bundle's ``index.html`` (or a JSON placeholder)."""
        index_html = STATIC_DIR / "index.html"
        if index_html.exists():
            return FileResponse(index_html)
        return JSONResponse(
            {"service": SERVICE_NAME, "detail": "dashboard bundle not built; JSON API at /checks"}
        )

    # Shared CORS + middleware + static mounts (/design-system, /static/fonts,
    # /static) applied last so they wrap the fully-assembled app.
    configure_interface_app(app, static_dir=STATIC_DIR)

    return app
