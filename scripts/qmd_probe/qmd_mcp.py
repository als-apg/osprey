"""Minimal MCP Streamable-HTTP client for a qmd daemon.

qmd 2.5.3 does not expose a REST ``POST /query`` endpoint. Its HTTP transport
speaks MCP Streamable HTTP on ``POST /mcp`` plus a plain ``GET /health``. A
session must be established with an ``initialize`` handshake; the server
returns an ``Mcp-Session-Id`` header that every later request must echo back.

Implemented on the standard library so the probe adds no dependencies.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

PROTOCOL_VERSION = "2025-06-18"


class QmdMcpError(RuntimeError):
    """Raised when the qmd daemon returns an error or an unusable response."""


class QmdMcpClient:
    """Session-aware MCP client for a qmd HTTP daemon.

    Args:
        base_url: Root URL of the daemon, e.g. ``http://localhost:8181``.
        timeout: Per-request timeout in seconds.
    """

    def __init__(self, base_url: str = "http://localhost:8181", timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session_id: str | None = None
        self._next_id = 0

    # -- transport ---------------------------------------------------------

    def _post(self, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, str]]:
        """POST one JSON-RPC message and decode the response.

        Returns:
            A tuple of (decoded JSON-RPC message or None for notifications,
            response headers).

        Raises:
            QmdMcpError: On transport failure or a JSON-RPC error reply.
        """
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            # The server negotiates between JSON and SSE; accept both.
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        request = urllib.request.Request(
            f"{self.base_url}/mcp", data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
                resp_headers = {k.lower(): v for k, v in response.headers.items()}
        except urllib.error.HTTPError as exc:  # pragma: no cover - network path
            raise QmdMcpError(f"HTTP {exc.code}: {exc.read().decode('utf-8')[:400]}") from exc
        except OSError as exc:  # pragma: no cover - network path
            raise QmdMcpError(f"qmd daemon unreachable at {self.base_url}: {exc}") from exc

        if not raw.strip():
            return None, resp_headers

        message = _decode(raw)
        if message is not None and "error" in message:
            raise QmdMcpError(str(message["error"]))
        return message, resp_headers

    def _rpc(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._next_id += 1
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": self._next_id, "method": method}
        if params is not None:
            payload["params"] = params
        message, headers = self._post(payload)
        if self._session_id is None and "mcp-session-id" in headers:
            self._session_id = headers["mcp-session-id"]
        if message is None:
            raise QmdMcpError(f"empty response to {method}")
        return message.get("result", {})

    # -- protocol ----------------------------------------------------------

    def connect(self) -> dict[str, Any]:
        """Perform the MCP initialize handshake and cache the session id.

        Returns:
            The server's ``initialize`` result payload.
        """
        result = self._rpc(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "osprey-qmd-probe", "version": "0"},
            },
        )
        self._next_id += 1
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return result

    def health(self) -> dict[str, Any]:
        """Read the daemon's liveness endpoint.

        Returns:
            The decoded ``GET /health`` payload.

        Raises:
            QmdMcpError: If the daemon is unreachable.
        """
        try:
            with urllib.request.urlopen(
                f"{self.base_url}/health", timeout=self.timeout
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except OSError as exc:
            raise QmdMcpError(f"health check failed: {exc}") from exc

    def list_tools(self) -> list[dict[str, Any]]:
        """List the tools the daemon exposes."""
        return self._rpc("tools/list").get("tools", [])

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Invoke one qmd MCP tool.

        Args:
            name: Tool name as reported by :meth:`list_tools`.
            arguments: Tool arguments.

        Returns:
            The raw ``tools/call`` result.
        """
        return self._rpc("tools/call", {"name": name, "arguments": arguments})


def _decode(raw: str) -> dict[str, Any] | None:
    """Decode either a plain JSON body or an SSE-framed one.

    Args:
        raw: Response body text.

    Returns:
        The decoded JSON-RPC message, or None if the body carried no data.
    """
    text = raw.strip()
    if text.startswith("{"):
        return json.loads(text)

    # Server-sent events framing: one or more "data: {...}" lines.
    for line in text.splitlines():
        if line.startswith("data:"):
            chunk = line[len("data:") :].strip()
            if chunk:
                return json.loads(chunk)
    return None
