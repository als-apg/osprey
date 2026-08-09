"""Loopback HTTP plumbing shared by the Google Chat bridge's hermetic fakes.

The Chat and GCS fakes are ordinary ``http.server`` handlers bound to an
**ephemeral** loopback port (``127.0.0.1:0``), so any number of them can run
side by side and none of them collides with a developer's stack. Nothing here
needs a container runtime: a test that uses only these fakes must FAIL when they
cannot boot, never skip.

Two conventions the subclasses inherit and must not work around:

* **Every** response goes through :meth:`_FakeHandler.respond`, which always
  writes a ``Content-Length``. The handlers speak HTTP/1.1 with keep-alive, and
  a single response without a length would hang the client until its timeout
  rather than fail — so there is exactly one place that writes a response.
* Errors use the Google JSON error envelope (``{"error": {"code", "message",
  "status"}}``), because the client libraries under test parse that shape when
  they turn a non-2xx into an exception.
"""

from __future__ import annotations

import json
import threading
import urllib.parse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar

__all__ = ["FakeHttpService", "_FakeHandler"]


class _FakeHandler(BaseHTTPRequestHandler):
    """Request-handler base: response helpers plus request parsing.

    ``fake`` is injected as a class attribute on a per-instance subclass by
    :class:`FakeHttpService`, so a handler always reaches the server object that
    owns the recorded state without any global registry.
    """

    protocol_version = "HTTP/1.1"

    fake: ClassVar[Any]

    # -- response side ------------------------------------------------------

    def respond(
        self,
        status: int,
        body: bytes = b"",
        content_type: str = "application/json; charset=utf-8",
        headers: dict[str, str] | None = None,
    ) -> None:
        """The one and only response path (see the module docstring)."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if body and self.command != "HEAD":
            self.wfile.write(body)

    def respond_json(self, status: int, payload: Any) -> None:
        self.respond(status, json.dumps(payload).encode("utf-8"))

    def respond_error(self, status: int, message: str, api_status: str = "NOT_FOUND") -> None:
        """Google's JSON error envelope — what the real client libraries parse."""
        self.respond_json(
            status, {"error": {"code": status, "message": message, "status": api_status}}
        )

    # -- request side -------------------------------------------------------

    def read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def read_json(self) -> dict[str, Any]:
        raw = self.read_body()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def split_path(self) -> tuple[str, dict[str, str]]:
        """``(path, query)`` with the query flattened to its last value per key."""
        parsed = urllib.parse.urlsplit(self.path)
        query = {k: v[-1] for k, v in urllib.parse.parse_qs(parsed.query).items()}
        return parsed.path, query

    def log_message(self, *args: Any) -> None:  # noqa: A003 - http.server API
        """Silence per-request logging; failures surface through assertions."""


class FakeHttpService:
    """A threaded loopback HTTP server with a lifecycle tests can drive.

    Subclasses set :attr:`handler_class` and hold whatever state their routes
    record. Use as a context manager, or call :meth:`close` explicitly.
    """

    handler_class: ClassVar[type[_FakeHandler]]

    def __init__(self) -> None:
        handler = type(f"{type(self).__name__}Handler", (self.handler_class,), {"fake": self})
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._server.daemon_threads = True
        # The default 0.5 s poll interval is also the floor on how long close()
        # blocks, and these fakes are created per test — so poll tightly.
        self._thread = threading.Thread(
            target=partial(self._server.serve_forever, poll_interval=0.02),
            name=f"{type(self).__name__}-serve",
            daemon=True,
        )
        self._thread.start()

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
