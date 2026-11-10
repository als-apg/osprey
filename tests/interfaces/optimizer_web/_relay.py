"""A sidecar router on a local app, with the optimizer faked by ``httpx.MockTransport``.

The read-proxy and relay suites each mount their router on a LOCAL FastAPI app
(never the package-level composed app), so no real network call is made and no
optimizer service needs to be running.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
from fastapi import APIRouter, FastAPI

OPTIMIZER_URL = "http://optimizer.test"


def build_router_app(
    router: APIRouter, handler: Callable[[httpx.Request], httpx.Response]
) -> FastAPI:
    """Mount ``router`` on a local app whose service client answers through ``handler``."""
    app = FastAPI()
    app.include_router(router)
    app.state.optimizer_url = OPTIMIZER_URL
    app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return app


def recording_app(
    router: APIRouter, status_code: int = 200, body: object = None, headers: dict | None = None
) -> tuple[FastAPI, list[httpx.Request], list[bytes]]:
    """An app whose mock service always answers ``(status_code, body, headers)``.

    Returns the requests it saw and their raw bodies, read eagerly.
    """
    seen: list[httpx.Request] = []
    contents: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        contents.append(request.content)
        return httpx.Response(status_code, json=body if body is not None else {}, headers=headers)

    return build_router_app(router, handler), seen, contents


def raising_handler(
    error: type[httpx.RequestError],
) -> Callable[[httpx.Request], httpx.Response]:
    """A service whose every request fails with ``error`` (no answer at all)."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise error("no answer", request=request)

    return handler
