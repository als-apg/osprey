"""The wire between the Channels panel and its value child, and the row both read.

A frame is a 4-byte big-endian length and that many bytes of strict JSON. The
request is ``{"names": [str]}`` and nothing else: a read of at most
:data:`MAX_NAMES` channels is the only thing the wire can say. The response is a
list of :class:`ValueRow`, one per name, in request order.

The bounds add up: a request frame holds at most :data:`MAX_REQUEST_BYTES` of
names, and a row holds at most :data:`MAX_VALUE_BYTES` of value, so a response
always fits :data:`MAX_FRAME_BYTES`.
"""

from __future__ import annotations

import asyncio
import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

#: The most channels one batch reads (D9).
MAX_NAMES = 50

#: The largest request frame: far more than 50 channel names need.
MAX_REQUEST_BYTES = 64 * 1024

#: The largest encoded value a row carries; a larger one is not sent.
MAX_VALUE_BYTES = 64 * 1024

#: The largest frame either side reads.
MAX_FRAME_BYTES = 8 * 1024 * 1024

#: The child's module, as ``python -m`` takes it.
CHILD_MODULE = "osprey.interfaces.channel_finder.directory_values_child"

#: The argument naming the inherited listening socket's file descriptor.
SOCKET_FD_ARG = "--socket-fd"

#: What the child's last line on stderr starts with when it exits 78; the rest
#: of the line is the reason.
START_REFUSED_PREFIX = "start refused: "

OUTCOME_FOUND = "found"
OUTCOME_NO_ANSWER = "no_answer"
OUTCOME_UNAVAILABLE = "unavailable"

_OUTCOMES = (OUTCOME_FOUND, OUTCOME_NO_ANSWER, OUTCOME_UNAVAILABLE)

_HEADER = struct.Struct(">I")


class FrameError(ValueError):
    """A frame or the message in it is not one this wire allows."""


@dataclass(frozen=True)
class ValueRow:
    """One channel's current value, or why there is none."""

    name: str
    outcome: str
    value: Any = None
    timestamp: str | None = None
    severity: int | None = None
    status: str | None = None
    reason: str | None = None

    def to_wire(self) -> dict[str, Any]:
        """The row as JSON fields; ``reason`` is present only when there is one."""
        row = {
            "name": self.name,
            "outcome": self.outcome,
            "value": self.value,
            "timestamp": self.timestamp,
            "severity": self.severity,
            "status": self.status,
        }
        if self.reason is not None:
            row["reason"] = self.reason
        return row

    @classmethod
    def from_wire(cls, row: Mapping[str, Any]) -> ValueRow:
        """Read one row from its JSON fields.

        Raises:
            FrameError: The row names no channel or an unknown outcome.
        """
        name, outcome = row.get("name"), row.get("outcome")
        if not isinstance(name, str) or outcome not in _OUTCOMES:
            raise FrameError(f"not a value row: name={name!r} outcome={outcome!r}")
        return cls(
            name=name,
            outcome=outcome,
            value=row.get("value"),
            timestamp=row.get("timestamp"),
            severity=row.get("severity"),
            status=row.get("status"),
            reason=row.get("reason"),
        )


def wire_value(value: Any) -> Any:
    """*value* as strict JSON can carry it.

    Arrays and their scalars become lists and numbers, bytes become text, and a
    non-finite float becomes its name, since neither the frame nor the panel's
    JSON response has a spelling for it.
    """
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", "replace")
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): wire_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [wire_value(item) for item in value]
    return str(value)


def _refuse_constant(name: str) -> None:
    """Refuse the JSON constants ``json.loads`` accepts and strict JSON does not have."""
    raise FrameError(f"{name} is not strict JSON")


def _encode(message: Any) -> bytes:
    return json.dumps(message, separators=(",", ":"), allow_nan=False).encode("utf-8")


def encoded_size(message: Any) -> int:
    """How many bytes *message* takes in a frame's body."""
    return len(_encode(message))


def encode_frame(message: Any) -> bytes:
    """One frame holding *message*.

    Raises:
        FrameError: The message is over :data:`MAX_FRAME_BYTES`.
    """
    body = _encode(message)
    if len(body) > MAX_FRAME_BYTES:
        raise FrameError(f"frame of {len(body)} bytes is over {MAX_FRAME_BYTES}")
    return _HEADER.pack(len(body)) + body


async def read_frame(reader: asyncio.StreamReader, *, limit: int = MAX_FRAME_BYTES) -> Any:
    """The next message on *reader*, or ``None`` when the peer closed between frames.

    Raises:
        FrameError: The frame is over *limit*, cut short, or not JSON.
    """
    try:
        header = await reader.readexactly(_HEADER.size)
    except asyncio.IncompleteReadError as exc:
        if not exc.partial:
            return None
        raise FrameError("frame header cut short") from exc
    length = _HEADER.unpack(header)[0]
    if length > limit:
        raise FrameError(f"frame of {length} bytes is over {limit}")
    try:
        body = await reader.readexactly(length)
    except asyncio.IncompleteReadError as exc:
        raise FrameError("frame body cut short") from exc
    try:
        return json.loads(body, parse_constant=_refuse_constant)
    except FrameError:
        raise
    except (ValueError, RecursionError) as exc:
        raise FrameError("frame is not JSON") from exc


def encode_request(names: Sequence[str]) -> bytes:
    """The frame asking for *names*."""
    return encode_frame({"names": list(names)})


def decode_request(message: Any) -> list[str]:
    """The names a request asks for.

    A request says exactly one thing, so a field beside ``names`` is refused
    rather than ignored.

    Raises:
        FrameError: The message is not a read of at most :data:`MAX_NAMES` names.
    """
    if not isinstance(message, dict) or set(message) != {"names"}:
        raise FrameError("a request is {names: [str]} and nothing else")
    names = message["names"]
    if not isinstance(names, list) or not all(isinstance(n, str) and n for n in names):
        raise FrameError("names must be a list of non-empty strings")
    if len(names) > MAX_NAMES:
        raise FrameError(f"{len(names)} names is over the batch limit of {MAX_NAMES}")
    return names


def encode_response(rows: Sequence[ValueRow]) -> bytes:
    """The frame answering a request with *rows*."""
    return encode_frame([row.to_wire() for row in rows])


def decode_response(message: Any) -> list[ValueRow]:
    """The rows of a response.

    Raises:
        FrameError: The message is not a list of value rows.
    """
    if not isinstance(message, list) or not all(isinstance(row, dict) for row in message):
        raise FrameError("a response is a list of value rows")
    return [ValueRow.from_wire(row) for row in message]
