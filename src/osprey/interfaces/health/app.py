"""OSPREY System Health panel — FastAPI application factory.

The web surface over the P1 health framework: a sidecar that re-runs the same
suite the ``osprey health`` CLI runs, caches the report, and serves it to the
SYSTEM dashboard tab. Launched in-process by ``ServerLauncher`` and
reverse-proxied at ``/panel/system-health/`` — the ``channel_finder`` builtin
pattern.

Three collaborators are wired here and cached on ``app.state``:

* :class:`~osprey.interfaces.health.loader.HealthConfigLoader` — the synchronous
  config-load phase (mtime-gated ``.env``/``config.yml`` → merged records);
* :class:`~osprey.interfaces.health.lifecycle.HealthRuntimeLifecycle` — the sole
  owner of the control-system connector, with loop-affine teardown;
* :class:`~osprey.interfaces.health.engine.HealthCheckEngine` — the single-flight
  cache/scheduler backing ``/checks``.

**Guarded construction (okf invariant).** The factory never raises: a launcher
runs it on a swallowed daemon thread, so an exception would leave a silent dead
tab. Any failure wiring the engine degrades to ``app.state.engine = None`` —
``/health`` still returns 200 and ``/checks`` serves a constant degraded
envelope. A missing or invalid ``config.yml`` is *not* such a failure: the engine
handles it, serving core-category results with default cadence.

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

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse

from osprey.health.config import parse_health_config
from osprey.health.models import CheckReport
from osprey.interfaces._app_setup import configure_interface_app

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from osprey.interfaces.health.engine import HealthCheckEngine
    from osprey.interfaces.health.lifecycle import HealthRuntimeLifecycle

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

SERVICE_NAME = "system-health"


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
        from osprey.interfaces.health.engine import HealthCheckEngine
        from osprey.interfaces.health.lifecycle import HealthRuntimeLifecycle
        from osprey.interfaces.health.loader import HealthConfigLoader

        loader = HealthConfigLoader(Path(config_path) if config_path is not None else None)
        lifecycle = HealthRuntimeLifecycle()
        engine = HealthCheckEngine(loader=loader, lifecycle=lifecycle, config_path=config_path)
        return engine, lifecycle
    except Exception:  # noqa: BLE001 — degrade to guarded mode, never raise.
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
        categories: Annotated[list[str] | None, Query()] = None,
    ) -> dict[str, Any]:
        """Return the health-report envelope, constant-time in all states.

        ``categories`` is accepted and ignored (documented; no P2 consumer): the
        served report is always the full, unfiltered suite so the never-elevate
        cost-tiering invariant holds trivially.
        """
        if engine is None:
            return _degraded_envelope()
        return engine.get_checks()

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
