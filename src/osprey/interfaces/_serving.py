"""Supported helpers for serving a FastAPI interface on a throwaway port.

These utilities run any OSPREY interface ``create_app()`` on a free localhost
port in a background thread — the shared surface used by the real-browser
Playwright suites under ``tests/interfaces/`` *and* by the documentation
screenshot runner (``docs/screenshots/``). Both consumers import a real,
supported API here instead of reaching into test internals.

The module depends only on the standard library plus ``uvicorn``/``fastapi``
(both already core runtime dependencies); it pulls in no test-only packages, so
importing it from shipped code drags nothing extra along.
"""

from __future__ import annotations

import socket
import threading
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

import uvicorn

if TYPE_CHECKING:
    from collections.abc import Iterator

    from fastapi import FastAPI


def free_port() -> int:
    """Return an unused TCP port on 127.0.0.1."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_for_port(port: int, timeout: float = 10.0) -> None:
    """Block until the server accepts TCP connections, or raise RuntimeError."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"Server did not become ready on port {port} within {timeout}s")


@contextmanager
def run_app_server(app: FastAPI) -> Iterator[str]:
    """Run any FastAPI app on a free port in a background thread.

    Yields:
        The server's base URL, e.g. ``"http://127.0.0.1:54321"``.
    """
    port = free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    wait_for_port(port)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        t.join(timeout=5)
