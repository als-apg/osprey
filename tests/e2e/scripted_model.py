"""A scripted stand-in for the model, for tests whose subject is a gate.

A gate test needs one thing from the model: that a particular tool call is
made. :class:`ScriptedModel` is a local endpoint speaking the Messages format
that makes exactly the calls a test scripts, so the real Claude Code delivers
them through its real permission engine, hooks and MCP servers.

A script is a list of turns; a turn is a list of :class:`ScriptedCall`. Which
turn a request gets is read off the conversation it carries — the number of
assistant messages already in it — so a request the CLI repeats is answered
the same way twice. Once the script is used up the endpoint answers with
:data:`FINAL_TEXT` and the run ends.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

#: The text turn that ends a run once the script is used up.
FINAL_TEXT = "Scripted run complete."

#: The credential a scripted run presents. A run with no key in its
#: environment falls back to whatever login the machine holds, and would send
#: that to the endpoint and load that account's connectors into the session.
SCRIPTED_AUTH_TOKEN = "scripted-model-token"

_MESSAGES_ROUTE = "/v1/messages"
_COUNT_TOKENS_ROUTE = "/v1/messages/count_tokens"


@dataclass(frozen=True)
class ScriptedCall:
    """One tool call the scripted model makes."""

    name: str
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecordedRequest:
    """What one Messages request carried.

    Attributes:
        offered_tools: Names of the tools the request offered the model.
        tool_results: Every ``tool_result`` block in the conversation so far.
        credential: The credential presented, whichever header carried it —
            the bearer token of ``Authorization`` or the value of
            ``x-api-key`` — or ``None`` when the request carried neither.
        turn_index: The script turn the request asked for; ``None`` for a
            request that offered no tools, which is not an agent turn.
    """

    offered_tools: list[str]
    tool_results: list[dict[str, Any]]
    credential: str | None
    turn_index: int | None


class ScriptedModel:
    """A local Messages endpoint that replays a script of tool calls.

    Args:
        script: The turns to deliver, in order.

    Raises:
        ValueError: If a turn has no calls.
    """

    def __init__(self, script: Sequence[Sequence[ScriptedCall]]) -> None:
        self._script = [list(turn) for turn in script]
        for index, turn in enumerate(self._script):
            if not turn:
                raise ValueError(f"turn {index} of the script has no calls")
        self.requests: list[RecordedRequest] = []
        self.unexpected: list[str] = []
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Bind an ephemeral loopback port and serve in a daemon thread."""
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_for(self))
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop serving and release the port."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    @property
    def base_url(self) -> str:
        """The URL an Anthropic client is pointed at."""
        assert self._server is not None, "the scripted model is not started"
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    @property
    def offered_tools(self) -> list[str]:
        """The tools the agent's first turn request offered."""
        for request in self.requests:
            if request.turn_index is not None:
                return request.offered_tools
        raise AssertionError("the agent never asked the scripted model for a turn")

    def assert_script_consumed(self) -> None:
        """Fail unless every turn was delivered and the agent came back after the last.

        A run that stops early leaves forbidden calls undelivered, and a test
        that only looks for successful forbidden calls would pass on their
        absence.
        """
        asked = {r.turn_index for r in self.requests if r.turn_index is not None}
        for index in range(len(self._script)):
            assert index in asked, f"turn {index} of the script was never delivered"
        assert len(self._script) in asked, (
            "the agent never returned the results of the last scripted turn"
        )

    def answer(
        self, body: dict[str, Any], credential: str | None
    ) -> tuple[list[dict[str, Any]], str]:
        """Record a request and return the content blocks and stop reason for it."""
        messages = [m for m in body.get("messages") or [] if isinstance(m, dict)]
        tools = [t.get("name", "") for t in body.get("tools") or [] if isinstance(t, dict)]
        index = sum(1 for m in messages if m.get("role") == "assistant") if tools else None
        results = [
            block
            for m in messages
            if m.get("role") == "user" and isinstance(m.get("content"), list)
            for block in m["content"]
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        with self._lock:
            self.requests.append(RecordedRequest(tools, results, credential, index))
        if index is not None and index < len(self._script):
            blocks = [
                {
                    "type": "tool_use",
                    "id": f"toolu_scripted_{index:02d}_{n:02d}",
                    "name": call.name,
                    "input": dict(call.input),
                }
                for n, call in enumerate(self._script[index])
            ]
            return blocks, "tool_use"
        return [{"type": "text", "text": FINAL_TEXT}], "end_turn"

    def note_unexpected(self, method: str, path: str) -> None:
        """Record a request the endpoint refuses to serve."""
        with self._lock:
            self.unexpected.append(f"{method} {path}")


def _handler_for(model: ScriptedModel) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        # A connection a client holds open is dropped, so no handler thread outlives its
        # request for long.
        timeout = 30

        def log_message(self, format: str, *args: Any) -> None:
            """Silence the per-request stderr line."""

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _refuse(self, method: str) -> None:
            model.note_unexpected(method, self.path)
            self._json(
                404, {"type": "error", "error": {"type": "not_found_error", "message": self.path}}
            )

        def do_HEAD(self) -> None:
            # The CLI's reachability check; it reads only the status.
            self.send_response(200)
            self.send_header("content-length", "0")
            self.end_headers()

        def do_GET(self) -> None:
            self._refuse("GET")

        def do_POST(self) -> None:
            raw = self.rfile.read(int(self.headers.get("content-length", 0)))
            route = self.path.split("?")[0]
            if route.endswith(_COUNT_TOKENS_ROUTE):
                self._json(200, {"input_tokens": 1})
                return
            if not route.endswith(_MESSAGES_ROUTE):
                self._refuse("POST")
                return
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                body = None
            if not isinstance(body, dict):
                self._json(
                    400,
                    {
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "message": "body is not a JSON object",
                        },
                    },
                )
                return
            presented = self._presented()
            if len(presented) > 1:
                model.note_unexpected("POST", f"{self.path} with two different credentials")
                self._json(
                    401,
                    {
                        "type": "error",
                        "error": {
                            "type": "authentication_error",
                            "message": "the request carries two different credentials",
                        },
                    },
                )
                return
            blocks, stop = model.answer(body, next(iter(presented), None))
            message = {
                "id": f"msg_scripted_{len(model.requests):04d}",
                "type": "message",
                "role": "assistant",
                "model": body.get("model", "scripted"),
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
            if not body.get("stream"):
                self._json(200, {**message, "content": blocks, "stop_reason": stop})
                return
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("cache-control", "no-cache")
            self.send_header("connection", "close")
            self.end_headers()
            self._event("message_start", {"message": message})
            for index, block in enumerate(blocks):
                if block["type"] == "text":
                    start = {"type": "text", "text": ""}
                    delta = {"type": "text_delta", "text": block["text"]}
                else:
                    start = {**block, "input": {}}
                    delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
                self._event("content_block_start", {"index": index, "content_block": start})
                self._event("content_block_delta", {"index": index, "delta": delta})
                self._event("content_block_stop", {"index": index})
            self._event(
                "message_delta",
                {
                    "delta": {"stop_reason": stop, "stop_sequence": None},
                    "usage": {"output_tokens": 1},
                },
            )
            self._event("message_stop", {})
            self.close_connection = True

        def _presented(self) -> set[str]:
            """Every distinct credential the request carries, in either header."""
            presented = set()
            auth = self.headers.get("authorization")
            if auth:
                presented.add(auth[len("Bearer ") :] if auth.startswith("Bearer ") else auth)
            key = self.headers.get("x-api-key")
            if key:
                presented.add(key)
            return presented

        def _event(self, name: str, payload: dict[str, Any]) -> None:
            data = json.dumps({"type": name, **payload})
            self.wfile.write(f"event: {name}\ndata: {data}\n\n".encode())
            self.wfile.flush()

    return Handler
