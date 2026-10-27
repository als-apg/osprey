"""The real bridge app served on a real TCP port, for tests that need a socket."""

from __future__ import annotations

import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager

import uvicorn

from osprey.services.bluesky_bridge.app import app
from tests._live import free_port, wait_until


def _accepts_connections(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


@contextmanager
def live_bridge() -> Iterator[str]:
    """Run the real bridge app on a real TCP port in a background thread.

    It shares the test process's module-level singletons (a thread, not a
    subprocess), but requests travel over a real socket through the real ASGI
    app, so a missing or misrouted endpoint fails exactly as it would in
    production. A genuine SSE stream needs this: both `httpx.ASGITransport` and
    Starlette's `TestClient` drive the app to completion before handing back
    *any* response, so they hang on a never-ending generator.

    Cross-loop note: the bridge's module-level asyncio primitives lazily bind
    to whichever event loop first touches them. This is safe only because the
    loops are used strictly *sequentially*: the server thread's loop shuts down
    (``server.should_exit = True; t.join(...)``) before the next test's loop
    starts. Overlapping access from two live loops at once would fail with
    "Future ... attached to a different loop".
    """
    port = free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    wait_until(lambda: _accepts_connections(port), what=f"the bridge on port {port}")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        t.join(timeout=5)
