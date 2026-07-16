"""The bluesky panels sidecar's FastAPI app: skeleton wiring for the operator
panel bundles plus a shared HTTP client onto the Bluesky bridge.

This is task 1.1 (sidecar-app-skeleton) of the Phase-6 "Operator Interfaces"
plan — it only wires the app shell: healthcheck, a shared ``httpx.AsyncClient``
resolved against the bridge, and static mounts for the three panel bundles.
The panel content itself (plan authoring, results, health) and the
read-proxy/execute routes onto the bridge are added by later tasks.

Panel mounts (task 3.2/3.3 must agree with this mapping — see
``_PANEL_MOUNTS`` below):

- ``panels/plan``    -> ``/plan``
- ``panels/results`` -> ``/results``
- ``panels/health``  -> ``/health-panel`` (NOT ``/health`` — that path is the
  sidecar's own healthcheck route, so the panel bundle is mounted at
  ``/health-panel`` to avoid the collision)

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

from osprey.interfaces._app_setup import configure_interface_app
from osprey.services.bluesky_panels import execute, read_proxy
from osprey.services.bluesky_panels import health as health_routes

_DEFAULT_BRIDGE_URL = "http://127.0.0.1:8090"

# Panel bundle directories (relative to this module's directory) and the
# mount path each is served under. Later tasks (3.2 plan authoring, 3.3
# results/health panels) author content into these directories; this task
# only wires the mounts so the sidecar doesn't crash on a not-yet-authored
# directory.
_PANELS_ROOT = Path(__file__).parent / "panels"
_PANEL_MOUNTS: dict[str, str] = {
    "plan": "/plan",
    "results": "/results",
    "health": "/health-panel",
}


def _resolve_bridge_url() -> str:
    """Resolve the Bluesky bridge base URL.

    Mirrors ``osprey.mcp_server.bluesky.server_context.BridgeContext._resolve_bridge_url``
    so the sidecar and the Bluesky MCP agree on which bridge instance to talk to.

    Resolution order:

    1. ``BLUESKY_BRIDGE_URL`` env var (full URL) — wins outright.
    2. ``bluesky.bridge_url`` in config.yml.
    3. ``http://127.0.0.1:8090`` default.
    """
    full = os.environ.get("BLUESKY_BRIDGE_URL")
    if full:
        return full.rstrip("/")

    from osprey.utils.workspace import load_osprey_config

    config = load_osprey_config()
    url = config.get("bluesky", {}).get("bridge_url", _DEFAULT_BRIDGE_URL)
    return str(url).rstrip("/")


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    client = httpx.AsyncClient(timeout=15.0)
    _app.state.client = client
    _app.state.bridge_url = _resolve_bridge_url()
    try:
        yield
    finally:
        await client.aclose()


app = FastAPI(title="OSPREY Bluesky Panels", lifespan=_lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# Wire the bridge read-proxy, the deterministic execute route, and the
# health-rollup router onto the app. Each router reads the shared httpx
# client + bridge URL from ``app.state`` at request time (set in _lifespan).
app.include_router(read_proxy.router)
app.include_router(execute.router)
app.include_router(health_routes.router)


for _panel_name, _mount_path in _PANEL_MOUNTS.items():
    _panel_dir = _PANELS_ROOT / _panel_name
    os.makedirs(_panel_dir, exist_ok=True)
    app.mount(
        _mount_path, StaticFiles(directory=_panel_dir, html=True), name=f"panel-{_panel_name}"
    )

# Serve the shared design-system assets (/design-system, /static/fonts) from
# this sidecar too: the panels are reached through the web-terminal reverse
# proxy at /panel/{id}, which rewrites a panel's root-absolute
# ``/design-system/…`` and ``/static/fonts/…`` references to
# ``/panel/{id}/design-system/…`` — i.e. back to THIS service — so the tokens,
# theme-boot, and fonts must be served here, exactly as every interface app
# does via the same shared helper.
configure_interface_app(app, static_dir=Path(__file__).parent / "static")
