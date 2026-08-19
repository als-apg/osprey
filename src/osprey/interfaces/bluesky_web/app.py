"""The bluesky-web sidecar's FastAPI app: the operator panel bundles plus a
shared HTTP client onto the Bluesky bridge.

What the app owns is serving and plumbing: its container healthcheck, a default
no-cache header on everything that does not set one itself, one
``httpx.AsyncClient`` and resolved
bridge URL published on ``app.state`` for every router to use, and the static
mounts for the panel bundles plus the shared design-system assets.

What it does NOT own is policy. The bridge-facing routers it composes — the read
proxy, the plan-draft relay and the plan-queue relay — relay to the bridge
verbatim, body and status code alike; the bridge decides what a request is
allowed to do. The one router that reaches no bridge is the channel catalog,
which serves the build-time ``channels.json`` mounted beside the config.

Panel mounts (panel bundles must agree with this mapping — see
``_PANEL_MOUNTS`` below):

- ``_BLUESKY_PANEL_DIR`` (bundle) -> ``/bluesky``

Stays import-clean of ``bluesky``/``ophyd``/``tiled`` at module scope, mirroring
``osprey.services.bluesky_bridge.app``.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from starlette.staticfiles import StaticFiles

from osprey.bluesky_bridge_connection import resolve_bridge_url
from osprey.interfaces._app_setup import configure_interface_app
from osprey.interfaces.bluesky_web import channels, draft_relay, queue_relay, read_proxy

# Panel bundle directories (relative to this module's directory) and the mount
# path each is served under. Directories are created on startup if absent, so
# the sidecar doesn't crash on a not-yet-authored bundle.
_PANELS_ROOT = Path(__file__).parent / "panels"

# The bundle directory behind the BLUESKY panel — plan composition, the queue,
# and the selected run's results.
_BLUESKY_PANEL_DIR = "bluesky"

# (mount path, bundle directory).
_PANEL_MOUNTS: tuple[tuple[str, str], ...] = (("/bluesky", _BLUESKY_PANEL_DIR),)


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    from osprey.utils.logger import configure_logging

    # Launched as `uvicorn ...:app`, bypassing every Osprey entry point.
    # Configuring on serve rather than on import keeps this module safe to
    # import from a library path.
    configure_logging()

    client = httpx.AsyncClient(timeout=15.0)
    _app.state.client = client
    # Resolved via the shared osprey.bluesky_bridge_connection helper so this
    # sidecar and the Bluesky MCP server agree on which bridge instance to talk
    # to.
    _app.state.bridge_url = resolve_bridge_url()
    try:
        yield
    finally:
        await client.aclose()


app = FastAPI(title="OSPREY Bluesky", lifespan=_lifespan)


@app.middleware("http")
async def _no_cache(request, call_next):  # type: ignore[no-untyped-def]
    """Forbid browser caching on everything this sidecar serves.

    The panel bundles ship unversioned filenames (``panel.js``, not
    ``panel-<hash>.js``), so a cached copy silently survives a container
    rebuild — an operator panel showing last week's UI. Everything else here
    is live JSON that must never be cached either. Header string matches
    ``osprey.interfaces.common_middleware.NoCacheStaticMiddleware``'s
    uncached branch (that middleware's path rules are interface-app specific
    — ``/static/``, ``/api/`` — and match nothing this sidecar mounts, hence
    the blanket rule; on a loopback service re-fetching is free).

    Filled in with ``setdefault``, never overridden, mirroring the web-terminal
    hub proxy's convention: a route that made its own caching decision (e.g. a
    revalidated ETag response) keeps it, and every route that made none still
    gets the full no-store string.
    """
    response = await call_next(request)
    response.headers.setdefault("Cache-Control", "no-cache, no-store, must-revalidate")
    return response


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# Wire the bridge read-proxy, the plan-draft relay, and the plan-queue relay
# onto the app. Each router reads the shared httpx client + bridge URL from
# ``app.state`` at request time (set in _lifespan). The queue relay carries the
# sidecar's whole write surface: enqueue, reorder, remove, start, stop, abort.
app.include_router(read_proxy.router)
app.include_router(draft_relay.router)
app.include_router(queue_relay.router)
# The channel catalog is the one router that talks to no bridge at all — it
# serves the build-time channels.json mounted beside the container's config.
app.include_router(channels.router)


for _mount_path, _panel_name in _PANEL_MOUNTS:
    _panel_dir = _PANELS_ROOT / _panel_name
    os.makedirs(_panel_dir, exist_ok=True)
    # Mount name keys off the mount PATH, not the bundle directory: the two
    # mounts sharing one bundle would otherwise collide on a single name.
    app.mount(
        _mount_path,
        StaticFiles(directory=_panel_dir, html=True),
        name=f"panel{_mount_path.replace('/', '-')}",
    )

# Serve the shared design-system assets (/design-system, /static/fonts) from
# this sidecar too: the panels are reached through the web-terminal reverse
# proxy at /panel/{id}, which rewrites a panel's root-absolute
# ``/design-system/…`` and ``/static/fonts/…`` references to
# ``/panel/{id}/design-system/…`` — i.e. back to THIS service — so the tokens,
# theme-boot, and fonts must be served here, exactly as every interface app
# does via the same shared helper.
configure_interface_app(app, static_dir=Path(__file__).parent / "static")
