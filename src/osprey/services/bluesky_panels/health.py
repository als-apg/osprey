"""Full dependency healthcheck for the bluesky panels sidecar (task 1.4).

``GET /health/full`` fans out three concurrent probes onto the services the
sidecar depends on and returns a per-service breakdown plus a worst-of
rollup. A probe failure (timeout, connection refused, non-200, DNS failure)
degrades only that service's entry to ``unhealthy`` -- this endpoint never
returns a non-200 response of its own.

Probed services and their env-resolved targets (a sibling task, 3.1, wires
these into the compose env; panel-health, 2.3, consumes this response shape
-- both must agree on the env var names and the JSON shape used here):

- **Bridge** (HTTP) -- ``{request.app.state.bridge_url}/health``. The bridge
  URL is already env-resolved (from ``BLUESKY_BRIDGE_URL``) by the app
  skeleton in ``app.py``'s ``_lifespan``, so this probe reuses it rather than
  re-resolving from env itself.
- **Tiled** (HTTP) -- base URL from ``BLUESKY_PANELS_TILED_URL``
  (default ``http://tiled:8000``); probes ``{base}/healthz``.
- **VA IOC** (raw TCP, EPICS Channel Access) -- ``host:port`` from
  ``BLUESKY_PANELS_VA_ADDR`` (default ``virtual-accelerator:5064``); probed via
  a bare ``asyncio.open_connection`` TCP connect, NOT an HTTP request -- an
  EPICS IOC has no HTTP surface.

Only these three services are probed. The Bluesky MCP is stdio-only (no HTTP
port) and the web terminal is the sidecar itself, so neither is included.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import httpx
from fastapi import APIRouter, Request

router = APIRouter()

_DEFAULT_TILED_URL = "http://tiled:8000"
_DEFAULT_VA_ADDR = "virtual-accelerator:5064"

# Bounded per-probe timeout. Kept well under typical client/gateway timeouts
# so a single wedged dependency can't make /health/full itself hang.
_PROBE_TIMEOUT_S = 2.5

# Worst-of ordering for the rollup: higher rank wins.
_STATUS_RANK: dict[str, int] = {"ok": 0, "unhealthy": 1}


def _elapsed_ms(start: float) -> float:
    return (time.monotonic() - start) * 1000.0


async def _probe_http(name: str, url: str, client: httpx.AsyncClient) -> dict[str, Any]:
    """GET ``url`` and map the outcome to a service entry, never raising.

    A 200 is ``ok``; any other status, a timeout, or a connection-level error
    degrades this service to ``unhealthy`` (with a describing ``detail``).
    """
    start = time.monotonic()
    try:
        response = await client.get(url, timeout=_PROBE_TIMEOUT_S)
    except httpx.TimeoutException:
        return {
            "name": name,
            "status": "unhealthy",
            "detail": f"timed out probing {url}",
            "latency_ms": _elapsed_ms(start),
        }
    except httpx.RequestError as exc:
        return {
            "name": name,
            "status": "unhealthy",
            "detail": f"unreachable: {url} ({exc.__class__.__name__}: {exc})",
            "latency_ms": _elapsed_ms(start),
        }
    latency_ms = _elapsed_ms(start)
    if response.status_code == 200:
        return {"name": name, "status": "ok", "detail": f"200 from {url}", "latency_ms": latency_ms}
    return {
        "name": name,
        "status": "unhealthy",
        "detail": f"HTTP {response.status_code} from {url}",
        "latency_ms": latency_ms,
    }


async def _probe_bridge_http(request: Request) -> dict[str, Any]:
    """Probe the Bluesky bridge's own ``/health`` route over HTTP."""
    client: httpx.AsyncClient = request.app.state.client
    bridge_url: str = request.app.state.bridge_url
    return await _probe_http("bridge", f"{bridge_url}/health", client)


async def _probe_tiled_http(request: Request) -> dict[str, Any]:
    """Probe Tiled's ``/healthz`` route over HTTP."""
    client: httpx.AsyncClient = request.app.state.client
    base = os.environ.get("BLUESKY_PANELS_TILED_URL", _DEFAULT_TILED_URL).rstrip("/")
    return await _probe_http("tiled", f"{base}/healthz", client)


def _parse_va_addr(addr: str) -> tuple[str, int]:
    host, _, port_str = addr.rpartition(":")
    if not host:
        # No colon found (rpartition returns "" host) -- fall back to the
        # whole string as host with the EPICS CA default port.
        return addr, 5064
    try:
        port = int(port_str)
    except ValueError:
        return addr, 5064
    return host, port


async def _probe_va_tcp(_request: Request) -> dict[str, Any]:
    """Probe the virtual-accelerator IOC with a raw TCP connect.

    This is deliberately not an HTTP request -- EPICS Channel Access is a raw
    socket protocol, and the IOC has no HTTP surface to probe.
    """
    start = time.monotonic()
    addr = os.environ.get("BLUESKY_PANELS_VA_ADDR", _DEFAULT_VA_ADDR)
    host, port = _parse_va_addr(addr)
    try:
        _reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=_PROBE_TIMEOUT_S
        )
    except (OSError, TimeoutError) as exc:
        return {
            "name": "va_ioc",
            "status": "unhealthy",
            "detail": f"TCP connect to {addr} failed ({exc.__class__.__name__}: {exc})",
            "latency_ms": _elapsed_ms(start),
        }
    latency_ms = _elapsed_ms(start)
    try:
        writer.close()
        await writer.wait_closed()
    except OSError:
        # Best-effort teardown -- the connect already succeeded, so the
        # probe itself is a success regardless of close-time hiccups.
        pass
    return {
        "name": "va_ioc",
        "status": "ok",
        "detail": f"TCP connect to {addr} succeeded",
        "latency_ms": latency_ms,
    }


def _rollup(services: list[dict[str, Any]]) -> str:
    worst = "ok"
    for service in services:
        status = service["status"]
        if _STATUS_RANK.get(status, max(_STATUS_RANK.values())) > _STATUS_RANK[worst]:
            worst = status
    return worst


@router.get("/health/full")
async def health_full(request: Request) -> dict[str, Any]:
    services = list(
        await asyncio.gather(
            _probe_bridge_http(request),
            _probe_tiled_http(request),
            _probe_va_tcp(request),
        )
    )
    return {"services": services, "rollup": _rollup(services)}
