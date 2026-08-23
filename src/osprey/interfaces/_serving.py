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
from contextlib import contextmanager, suppress
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


def _wait_until_serving(
    server: uvicorn.Server,
    thread: threading.Thread,
    thread_error: list[BaseException],
    port: int,
    timeout: float = 30.0,
) -> None:
    """Block until *server* is serving, or explain why it never will be.

    Waits on uvicorn's own ``started`` flag rather than probing the port, and
    treats the serving thread exiting as a terminal answer. A server that dies
    during startup — a lifespan that raises, a failed bind — otherwise looks
    exactly like one that is merely slow, so the caller pays the whole timeout
    and is then told the server "did not become ready", which points at the
    timeout rather than at the error that actually occurred.

    Because a dead server is detected immediately through the thread check,
    the timeout only ever bounds a *slow but alive* startup — app lifespans do
    real work (service imports, degraded-mode probes) that stretches under
    parallel test load, so the bound is generous rather than tight.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.started:
            return
        if not thread.is_alive():
            raise RuntimeError(
                f"Server exited during startup without listening on port {port}; "
                "the application's startup failed (see the server log above)"
            ) from (thread_error[0] if thread_error else None)
        time.sleep(0.02)
    raise RuntimeError(f"Server did not become ready on port {port} within {timeout}s")


@contextmanager
def run_app_server(app: FastAPI) -> Iterator[str]:
    """Run any FastAPI app on a free port in a background thread.

    Yields:
        The server's base URL, e.g. ``"http://127.0.0.1:54321"``.
    """
    # Bind the listening socket here and hand uvicorn the live socket, rather
    # than passing it a port number chosen earlier. ``free_port`` must close its
    # socket before returning the number, which leaves a window where anything
    # else on the machine can claim that port — and a readiness check that only
    # opens a TCP connection is satisfied by whoever won it. Owning the socket
    # for the server's whole lifetime removes the window and makes the yielded
    # URL provably this app's.
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    port = int(sock.getsockname()[1])

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread_error: list[BaseException] = []

    def _serve() -> None:
        try:
            server.run(sockets=[sock])
        except BaseException as exc:  # noqa: BLE001 - surfaced to the caller as the cause
            # Includes the SystemExit uvicorn raises when startup fails.
            thread_error.append(exc)

    t = threading.Thread(target=_serve, daemon=True)
    t.start()

    def _shutdown() -> None:
        server.should_exit = True
        t.join(timeout=5)
        with suppress(OSError):
            sock.close()

    try:
        _wait_until_serving(server, t, thread_error, port)
    except BaseException:
        _shutdown()
        raise

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        _shutdown()
