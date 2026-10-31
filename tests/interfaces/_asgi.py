"""A raw-ASGI harness for driving middleware without a test client.

A middleware's input *is* a scope, and a raw one is immune to whatever default
credentials a shared test-client fixture may inject: a refusal test that a
fixture can silently turn into an admission is a test that stops testing.
"""

from __future__ import annotations

import asyncio
from typing import Any

OPERATOR_SECRET = "operator-secret-value"
PANEL_TOKEN = "panel-token-value"


class RecordingApp:
    """A downstream ASGI app that answers ``status`` and records what it saw."""

    def __init__(self, status: int = 200, *, explode: bool = False) -> None:
        self.status = status
        self.explode = explode
        self.scopes: list[dict[str, Any]] = []
        self.bodies: list[bytes] = []

    async def __call__(self, scope, receive, send) -> None:
        self.scopes.append(scope)
        if self.explode:
            raise RuntimeError("route blew up")
        if scope["type"] == "http":
            body = b""
            while True:
                message = await receive()
                if message["type"] != "http.request":
                    break
                body += message.get("body") or b""
                if not message.get("more_body", False):
                    break
            self.bodies.append(body)
            await send({"type": "http.response.start", "status": self.status, "headers": []})
            await send({"type": "http.response.body", "body": b"downstream"})
        elif scope["type"] == "websocket":
            await send({"type": "websocket.accept"})

    @property
    def called(self) -> bool:
        return bool(self.scopes)


def encode_headers(headers: dict[str, str] | None) -> list[tuple[bytes, bytes]]:
    return [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in (headers or {}).items()]


def http_scope(
    path: str = "/api/config",
    method: str = "GET",
    headers: dict[str, str] | None = None,
    *,
    app: Any = None,
    scheme: str = "http",
    raw_headers: list[tuple[bytes, bytes]] | None = None,
) -> dict[str, Any]:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": b"",
        "scheme": scheme,
        "headers": raw_headers if raw_headers is not None else encode_headers(headers),
        "client": ("127.0.0.1", 54321),
        "server": ("127.0.0.1", 8080),
        "app": app,
    }


def ws_scope(
    path: str = "/ws",
    headers: dict[str, str] | None = None,
    *,
    app: Any = None,
    extensions: dict[str, Any] | None = None,
    scheme: str = "ws",
) -> dict[str, Any]:
    scope = {
        "type": "websocket",
        "asgi": {"version": "3.0"},
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": b"",
        "scheme": scheme,
        "headers": encode_headers(headers),
        "client": ("127.0.0.1", 54321),
        "server": ("127.0.0.1", 8080),
        "app": app,
    }
    if extensions is not None:
        scope["extensions"] = extensions
    return scope


def drive(middleware, scope, incoming: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Run one connection through ``middleware`` and return everything it sent."""
    pending = list(incoming if incoming is not None else [{"type": "http.request", "body": b""}])
    sent: list[dict[str, Any]] = []

    async def receive():
        if pending:
            return pending.pop(0)
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    asyncio.run(middleware(scope, receive, send))
    return sent


def status_of(sent: list[dict[str, Any]]) -> int | None:
    for message in sent:
        if message["type"] in ("http.response.start", "websocket.http.response.start"):
            return message["status"]
    return None
