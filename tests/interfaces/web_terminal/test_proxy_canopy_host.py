"""The panel proxy never hands the browser's ``Host`` or ``Origin`` to the CANOPY sidecar.

CANOPY refuses a request whose ``Host`` names neither loopback nor a host it was
configured with, and compares a WebSocket's ``Origin`` against its own address.
OSPREY stays inside that rule because both proxy hops start a new client: the
upstream sees ``Host: 127.0.0.1:<port>``, the address the sidecar listens on, and
no ``Origin``, whatever the browser sent. Each test runs a real upstream on
loopback that records what arrives, published as the CANOPY panel's URL.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from osprey.profiles.sidecar_panels import CANOPY_PANEL_ID
from osprey.registry.web import panel_url_state_attr
from tests.interfaces.web_terminal._proxy_fakes import panel_app

#: What a browser on a terminal page served under a public name sends.
BROWSER_HOST = "terminal.facility.example"
BROWSER_ORIGIN = f"http://{BROWSER_HOST}"

#: The sidecar's own credential, as the launcher publishes it.
ACCESS = {"X-Canopy-Access": "front-access-key"}


@contextlib.contextmanager
def _http_upstream() -> Iterator[tuple[int, list[dict[str, str]]]]:
    """A loopback HTTP server that records each request's headers."""
    seen: list[dict[str, str]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append({k.lower(): v for k, v in self.headers.items()})
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # silence the per-request log line
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1], seen
    finally:
        server.shutdown()
        thread.join(10)


@contextlib.contextmanager
def _ws_upstream() -> Iterator[tuple[int, list[dict[str, str]]]]:
    """A loopback WebSocket server that records each handshake's headers and echoes."""
    import websockets

    seen: list[dict[str, str]] = []
    ready = threading.Event()
    state: dict[str, object] = {}

    async def _serve() -> None:
        state["loop"] = asyncio.get_running_loop()
        stop = asyncio.Event()
        state["stop"] = stop

        def record(connection, request):  # noqa: ARG001 - websockets' process_request hook
            seen.append({k.lower(): v for k, v in request.headers.raw_items()})

        async def handler(connection):
            async for frame in connection:
                await connection.send(frame)

        async with websockets.serve(handler, "127.0.0.1", 0, process_request=record) as server:
            state["port"] = server.sockets[0].getsockname()[1]
            ready.set()
            await stop.wait()

    thread = threading.Thread(target=lambda: asyncio.run(_serve()), daemon=True)
    thread.start()
    assert ready.wait(10), "the recording upstream did not start"
    try:
        yield int(state["port"]), seen  # type: ignore[call-overload]
    finally:
        state["loop"].call_soon_threadsafe(state["stop"].set)  # type: ignore[attr-defined]
        thread.join(10)


@contextlib.contextmanager
def _terminal_with_canopy_at(workspace_dir, port: int):
    """The whole app with the CANOPY panel published at ``127.0.0.1:<port>``."""
    for app, client in panel_app(workspace_dir, []):
        setattr(app.state, panel_url_state_attr(CANOPY_PANEL_ID), f"http://127.0.0.1:{port}")
        app.state.panel_auth_headers[CANOPY_PANEL_ID] = dict(ACCESS)
        yield client


def _assert_new_client(headers: dict[str, str], port: int) -> None:
    assert headers["host"] == f"127.0.0.1:{port}"
    assert "origin" not in headers
    assert BROWSER_HOST not in " ".join(headers.values())
    # Negative control: the hop did reach the sidecar with its own credential.
    assert headers["x-canopy-access"] == ACCESS["X-Canopy-Access"]


def test_the_http_hop_sends_the_sidecar_its_own_host_and_no_origin(workspace_dir):
    with _http_upstream() as (port, seen), _terminal_with_canopy_at(workspace_dir, port) as client:
        response = client.get(
            f"/panel/{CANOPY_PANEL_ID}/hare/v1/health",
            headers={"Host": BROWSER_HOST, "Origin": BROWSER_ORIGIN},
        )
    assert response.status_code == 200, response.text
    (headers,) = seen
    _assert_new_client(headers, port)


def test_the_websocket_hop_sends_the_sidecar_its_own_host_and_no_origin(workspace_dir):
    with _ws_upstream() as (port, seen), _terminal_with_canopy_at(workspace_dir, port) as client:
        with client.websocket_connect(
            f"/panel/{CANOPY_PANEL_ID}/hare/v1/ws",
            headers={"Host": BROWSER_HOST, "Origin": BROWSER_ORIGIN},
        ) as session:
            session.send_text("ping")
            assert session.receive_text() == "ping"
    (headers,) = seen
    _assert_new_client(headers, port)
