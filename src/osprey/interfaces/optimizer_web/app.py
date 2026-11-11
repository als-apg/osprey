"""The optimizer-web sidecar's FastAPI app: the OPTIMIZER panel bundle plus a
shared HTTP client onto the optimizer service.

Shaped like ``osprey.interfaces.bluesky_web.app``. What the app owns is serving
and plumbing: its container healthcheck, a default no-cache header, one
``httpx.AsyncClient`` and the resolved service URL on ``app.state`` for every
router to use, and the mount for the panel bundle plus the shared
design-system assets. The bundle is served verbatim except its ``index.html``,
which is rendered so ``vendor_url()`` can resolve a vendored library to the CDN
or to a locally fetched copy.

What it does NOT own is policy. The read proxy and the write relay pass the
service's answers through verbatim; the service decides what a request may do.
The relay adds the two things only this process can: the owner the auth gate
matched, and the launch token on start and resume.

Panel mounts: ``panels/optimizer`` -> ``/optimizer``.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from starlette.responses import Response

from osprey.interfaces._app_setup import configure_interface_app
from osprey.interfaces._media_types import PinnedTypeStaticFiles
from osprey.interfaces.optimizer_web import read_proxy, relay
from osprey.interfaces.vendor import vendor_url
from osprey.optimizer_connection import OPTIMIZER_TIMEOUT_S, optimizer_url

_PANELS_ROOT = Path(__file__).parent / "panels"

#: The bundle directory behind the OPTIMIZER panel.
_OPTIMIZER_PANEL_DIR = "optimizer"

# (mount path, bundle directory).
_PANEL_MOUNTS: tuple[tuple[str, str], ...] = (("/optimizer", _OPTIMIZER_PANEL_DIR),)

#: Timeout for every relayed call: the MCP server's, since a start or a status
#: read waits on machine reads the service itself cuts off only at 60-75 s.
_CLIENT_TIMEOUT = httpx.Timeout(OPTIMIZER_TIMEOUT_S, connect=5.0)


def resolve_service_url() -> str:
    """The optimizer base URL this sidecar relays to.

    ``OPTIMIZER_URL`` (the compose file sets it to the in-network service key)
    wins; otherwise the rendered config's ``services.optimizer.port``, through
    the same resolver the optimizer MCP server uses. An unreadable config reads
    as no config, so a dev checkout still starts.
    """
    config: dict[str, Any] | None
    try:
        from osprey_connectors.workspace import load_osprey_config

        config = load_osprey_config()
    except Exception:
        config = None
    return optimizer_url(config)


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    from osprey_connectors.logger import configure_logging

    # Launched as `uvicorn ...:app`, bypassing every Osprey entry point.
    configure_logging()

    client = httpx.AsyncClient(timeout=_CLIENT_TIMEOUT)
    _app.state.client = client
    _app.state.optimizer_url = resolve_service_url()
    try:
        yield
    finally:
        await client.aclose()


app = FastAPI(title="OSPREY Optimizer", lifespan=_lifespan)


@app.middleware("http")
async def _no_cache(request, call_next):
    """Forbid browser caching on everything this sidecar serves.

    The bundle ships unversioned filenames and everything else is live JSON;
    see ``osprey.interfaces.bluesky_web.app._no_cache``. Filled in with
    ``setdefault``, so a route that set its own caching keeps it.
    """
    response = await call_next(request)
    response.headers.setdefault("Cache-Control", "no-cache, no-store, must-revalidate")
    return response


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


app.include_router(read_proxy.router)
app.include_router(relay.router)


def _panel_index_route(templates: Jinja2Templates) -> Callable:
    """Build the GET handler that renders one bundle's ``index.html``."""

    async def _index(request: Request) -> Response:
        return templates.TemplateResponse(request, "index.html", {})

    return _index


for _mount_path, _panel_name in _PANEL_MOUNTS:
    _panel_dir = _PANELS_ROOT / _panel_name
    os.makedirs(_panel_dir, exist_ok=True)
    _templates = Jinja2Templates(directory=str(_panel_dir))
    _templates.env.globals["vendor_url"] = vendor_url
    # Registered BEFORE the mount so the three spellings of the bundle root
    # reach the renderer rather than the raw, unrendered index.
    _index_route = _panel_index_route(_templates)
    for _index_path in (_mount_path, f"{_mount_path}/", f"{_mount_path}/index.html"):
        app.get(_index_path, include_in_schema=False)(_index_route)
    app.mount(
        _mount_path,
        PinnedTypeStaticFiles(directory=_panel_dir, html=True),
        name=f"panel{_mount_path.replace('/', '-')}",
    )

# The shared design-system assets (/design-system, /static/fonts), served here
# because the panel proxy rewrites a panel's root-absolute references back to
# this service, exactly as for the Bluesky sidecar. Also installs the auth gate
# that stamps the owner the relay forwards.
configure_interface_app(app, static_dir=Path(__file__).parent / "static")
