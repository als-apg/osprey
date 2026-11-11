"""Wire format of the dispatch worker bus.

A message is multipart. On the worker's ``DEALER`` it is::

    [kind] [json-body] [blob?]

and on the dispatcher's ``ROUTER`` the peer's routing id comes first. ``kind``
is one of the ASCII verbs below; ``json-body`` is a UTF-8 JSON object; ``blob``
is an optional raw frame for bytes (an artifact's contents), so bytes never
travel through base64 inside JSON.

Requests carry ``rid``, a request id, and their reply (``OK`` or ``ERR``)
echoes it. ``ERR`` carries an HTTP status number on purpose: the dispatcher
maps it to the same exception classes the HTTP worker client raises, and the
bridges' facade passes it through unchanged.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

#: Worker → dispatcher, on connect and every :data:`HEARTBEAT_SEC`. Carries the
#: worker's whole state, so a dispatcher that restarted rebuilds its view from
#: the next round of heartbeats with no separate handshake.
HELLO = b"HELLO"
#: Dispatcher → worker, every :data:`HEARTBEAT_SEC`.
PING = b"PING"
#: Dispatcher → worker requests.
DISPATCH = b"DISPATCH"
STATUS = b"STATUS"
CANCEL = b"CANCEL"
RUNS = b"RUNS"
ARTIFACTS = b"ARTIFACTS"
ARTIFACT = b"ARTIFACT"
#: Worker → dispatcher replies.
OK = b"OK"
ERR = b"ERR"
#: Worker → dispatcher: one event of one run, pushed as the run produces it.
EVENT = b"EVENT"

#: The requests a worker serves.
REQUESTS: frozenset[bytes] = frozenset({DISPATCH, STATUS, CANCEL, RUNS, ARTIFACTS, ARTIFACT})

#: Seconds between heartbeats, in both directions.
HEARTBEAT_SEC = 2.0
#: Heartbeats a worker may miss before the dispatcher expires it.
LIVENESS = 3

#: The event types that end a run's stream.
TERMINAL_EVENTS: frozenset[str] = frozenset({"done", "error"})


class ProtocolError(ValueError):
    """A message that is not in the bus's wire format."""


class BusError(Exception):
    """A request the far end answered with ``ERR``.

    Attributes:
        status: The HTTP status number the error stands for.
        detail: The far end's machine-readable or human detail, never a trace.
    """

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


@dataclass(frozen=True)
class Message:
    """One decoded message, without the routing id."""

    kind: bytes
    body: dict[str, Any]
    blob: bytes | None = None


def new_rid() -> str:
    """A fresh request id."""
    return uuid.uuid4().hex


def encode(kind: bytes, body: dict[str, Any], blob: bytes | None = None) -> list[bytes]:
    """The frames of one message, without the routing id."""
    frames = [kind, json.dumps(body, separators=(",", ":")).encode()]
    if blob is not None:
        frames.append(blob)
    return frames


def decode(frames: list[bytes]) -> Message:
    """Decode the frames of one message, without the routing id.

    Raises:
        ProtocolError: The frames are not a bus message.
    """
    if len(frames) not in (2, 3):
        raise ProtocolError(f"expected 2 or 3 frames, got {len(frames)}")
    try:
        body = json.loads(frames[1])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"body is not JSON: {exc}") from None
    if not isinstance(body, dict):
        raise ProtocolError("body is not a JSON object")
    return Message(frames[0], body, frames[2] if len(frames) == 3 else None)


def ok(rid: str, body: Any, blob: bytes | None = None) -> list[bytes]:
    """The frames of a success reply to request ``rid``."""
    return encode(OK, {"rid": rid, "body": body}, blob)


def err(rid: str, status: int, detail: str) -> list[bytes]:
    """The frames of an error reply to request ``rid``."""
    return encode(ERR, {"rid": rid, "status": status, "detail": detail})
