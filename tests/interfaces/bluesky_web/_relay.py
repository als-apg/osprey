"""A sidecar relay router on a local app, with the bridge faked by ``httpx.MockTransport``.

The read proxy, draft relay and queue relay suites each mount their router on a
LOCAL FastAPI app (never the package-level composed app) so no real network call
is made and no real Bluesky bridge process needs to be running.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable

import httpx
from fastapi import APIRouter, FastAPI

BRIDGE_URL = "http://bridge.test"
VA_BRIDGE_URL = "http://bridge-va.test"

#: The per-lane bridge map of a two-lane deployment after its lifespan.
TWO_LANE_BRIDGE_URLS = {"bluesky": BRIDGE_URL, "bluesky_va": VA_BRIDGE_URL}


def build_router_app(
    router: APIRouter, handler: Callable[[httpx.Request], httpx.Response]
) -> FastAPI:
    """Mount ``router`` on a local app whose bridge client answers through ``handler``."""
    app = FastAPI()
    app.include_router(router)
    app.state.bridge_url = BRIDGE_URL
    app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return app


def refusing_handler(request: httpx.Request) -> httpx.Response:
    """A bridge that is not there."""
    raise httpx.ConnectError("connection refused", request=request)


class DelayedFrames(httpx.AsyncByteStream):
    """An async byte stream that yields ``frames`` with ``delay`` between them.

    Used to prove an SSE relay is a genuine streaming pass-through (not a
    buffer-then-forward), without a real multi-second sleep.
    """

    def __init__(self, frames: list[bytes], delay: float = 0.05) -> None:
        self._frames = frames
        self._delay = delay

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for i, frame in enumerate(self._frames):
            if i > 0:
                await asyncio.sleep(self._delay)
            yield frame
