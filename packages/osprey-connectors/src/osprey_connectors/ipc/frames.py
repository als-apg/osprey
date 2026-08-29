"""Length-prefixed frame codec for the connector-host IPC channel.

This module is the *contract* spoken over the pipe pair between the parent
process (the proxy connector) and the connector-host child that owns the
control-system client library. It is a pure codec: bytes in, bytes out. There
is no process management, no asyncio, no pipe handling here, so both sides can
be tested against the wire format without spawning anything.

Wire layout
-----------
Every frame is self-delimiting. All integers are unsigned 32-bit big-endian::

    +-------+-----------+------------+---------------+----------------+
    | MAGIC | total_len | header_len | header (JSON) | blob segments  |
    | 4 B   | 4 B       | 4 B        | header_len B  | rest of frame  |
    +-------+-----------+------------+---------------+----------------+

``total_len`` counts every byte that follows it — the ``header_len`` field,
the header, and all blob segments — so a reader knows how much to buffer
before it can parse anything. The header is UTF-8 JSON and carries the
structured envelope; ``header["blobs"]`` is the ordered list of segment byte
lengths, which slices the trailing binary region. Binary values (numpy arrays)
never enter the JSON: the header holds only an index into that list.

Three frame kinds exist, each carrying a ``request_id`` so a parent may have
several requests outstanding and match replies arriving in any order:

``request``
    ``{request_id, method, kwargs}``. The codec is deliberately
    *method-agnostic* — it neither enumerates nor validates method names, so
    the child and proxy can grow their call surface without a codec change.
``result``
    ``{request_id, value}``, where the value is any supported Python value
    (see below), including the ``dict[str, ChannelValue]`` a batched
    multi-channel read returns.
``error``
    ``{request_id, class_tag, message, fields}``. Exceptions cross the
    boundary *typed*: ``fields`` are the structured attributes enumerated by
    :mod:`osprey_connectors.ipc.exceptions`, and decoding calls the real
    constructor, so ``.attempted_value``, ``.reason`` and the min/max/step
    bounds survive and the parent's error rendering is unchanged. That module
    owns the registry — which classes cross typed, which fields each carries,
    and how an unrecognised ``class_tag`` fails closed to a
    :class:`ConnectionError` naming the original tag and message.

No pickle, ever
---------------
Nothing in this module imports :mod:`pickle`, and every array crosses the
boundary as ``.npy`` bytes written by ``np.save`` and read by ``np.load``,
both called with ``allow_pickle=False``. That is a hard house rule: the
child runs control-system code and the frames it emits must never be able to
execute code in the parent. Both halves live in the single ``_npy_bytes`` /
``_npy_load`` pair so the rule is checkable by test.

Supported values
----------------
``None``, ``bool``, ``int``, ``float``, ``str``, :class:`datetime.datetime`
(ISO 8601), lists/tuples and dicts of the above (dict keys are coerced to
strings; tuples decode as lists), numpy arrays and numpy scalars (dtype and
shape preserved), and the connector dataclasses
:class:`~osprey_connectors.control_system.base.ChannelValue`,
``ChannelMetadata`` and ``ChannelWriteResult``, which decode as real
instances of those classes. Anything else is refused at encode
time rather than silently degraded.
"""

from __future__ import annotations

import json
import struct
import uuid
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from datetime import datetime
from io import BytesIO
from typing import Any

import numpy as np

from osprey_connectors.control_system.base import (
    ChannelMetadata,
    ChannelValue,
    ChannelWriteResult,
)
from osprey_connectors.ipc.exceptions import exception_fields, reconstruct_exception

__all__ = [
    "MAGIC",
    "MAX_FRAME_BYTES",
    "PROTOCOL_VERSION",
    "ErrorFrame",
    "FrameDecodeError",
    "FrameEncodeError",
    "FrameReader",
    "RequestFrame",
    "ResultFrame",
    "decode_frame",
    "encode_error",
    "encode_request",
    "encode_result",
    "new_request_id",
]

#: Frame preamble. A desynchronised stream is detected here instead of being
#: read as a wildly wrong length.
MAGIC = b"OSPF"

#: Bumped only for an incompatible envelope change; decode refuses anything else.
PROTOCOL_VERSION = 1

#: Refuse absurd lengths rather than trying to allocate them. Generous enough
#: for a detector image, small enough that a corrupt length cannot exhaust RAM.
MAX_FRAME_BYTES = 256 * 1024 * 1024

_HEADER_STRUCT = struct.Struct(">I")

#: Marker key identifying a tagged (non-scalar) value node in the JSON header.
_TAG = "__osprey_ipc__"


class FrameEncodeError(TypeError):
    """A value was handed to the codec that the wire format cannot carry."""


class FrameDecodeError(ValueError):
    """A frame could not be parsed: bad magic, bad version, or unknown tag."""


@dataclass(frozen=True)
class RequestFrame:
    """A decoded call request. ``method`` is never validated by the codec."""

    request_id: str
    method: str
    kwargs: dict[str, Any]


@dataclass(frozen=True)
class ResultFrame:
    """A decoded successful reply."""

    request_id: str
    value: Any


@dataclass(frozen=True)
class ErrorFrame:
    """A decoded failure reply.

    ``exception`` is a real, reconstructed exception instance ready to be
    raised by the parent; ``class_tag`` and ``message`` are kept alongside it
    so a caller can still see what the child actually sent — which matters for
    the fail-closed case, where ``exception`` is a ``ConnectionError`` standing
    in for a class this codec does not know.
    """

    request_id: str
    class_tag: str
    message: str
    exception: BaseException


# Dataclasses reconstructed as real instances on the far side.
_DATACLASSES: dict[str, type] = {
    "ChannelValue": ChannelValue,
    "ChannelMetadata": ChannelMetadata,
    "ChannelWriteResult": ChannelWriteResult,
}


def new_request_id() -> str:
    """A fresh request id. Short, opaque, and unique across both processes."""
    return uuid.uuid4().hex


# --------------------------------------------------------------------------
# .npy payloads — the only place arrays are serialized, so the no-pickle rule
# is enforced (and testable) in exactly two lines.
# --------------------------------------------------------------------------


def _npy_bytes(array: np.ndarray) -> bytes:
    """Serialize an array to ``.npy`` bytes, never enabling pickle."""
    buffer = BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _npy_load(data: bytes) -> np.ndarray:
    """Read ``.npy`` bytes back, never enabling pickle."""
    buffer = BytesIO(data)
    return np.load(buffer, allow_pickle=False)


# --------------------------------------------------------------------------
# Value encoding
# --------------------------------------------------------------------------


def _encode_value(value: Any, blobs: list[bytes]) -> Any:
    """Turn ``value`` into a JSON node, appending any binary payloads to ``blobs``."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.ndarray):
        blobs.append(_npy_bytes(value))
        return {_TAG: "ndarray", "blob": len(blobs) - 1}
    if isinstance(value, np.generic):
        blobs.append(_npy_bytes(np.asarray(value)))
        return {_TAG: "npscalar", "blob": len(blobs) - 1}
    if isinstance(value, datetime):
        return {_TAG: "datetime", "iso": value.isoformat()}
    if isinstance(value, (list, tuple)):
        return [_encode_value(item, blobs) for item in value]
    if isinstance(value, dict):
        return {
            _TAG: "dict",
            "items": [[str(key), _encode_value(item, blobs)] for key, item in value.items()],
        }
    tag = type(value).__name__
    if _DATACLASSES.get(tag) is type(value):
        return {
            _TAG: tag,
            "fields": {
                spec.name: _encode_value(getattr(value, spec.name), blobs)
                for spec in dataclass_fields(value)
            },
        }
    raise FrameEncodeError(f"IPC frames cannot carry a value of type {tag!r}")


def _decode_value(node: Any, blobs: list[bytes]) -> Any:
    """Inverse of :func:`_encode_value`."""
    if node is None or isinstance(node, (bool, int, float, str)):
        return node
    if isinstance(node, list):
        return [_decode_value(item, blobs) for item in node]
    if not isinstance(node, dict):
        raise FrameDecodeError(f"unsupported JSON node of type {type(node).__name__!r}")

    tag = node.get(_TAG)
    if tag is None:
        raise FrameDecodeError("untagged object in frame payload")
    if tag == "ndarray":
        return _npy_load(_blob(blobs, node["blob"]))
    if tag == "npscalar":
        return _npy_load(_blob(blobs, node["blob"]))[()]
    if tag == "datetime":
        return datetime.fromisoformat(node["iso"])
    if tag == "dict":
        return {key: _decode_value(item, blobs) for key, item in node["items"]}
    cls = _DATACLASSES.get(tag)
    if cls is None:
        raise FrameDecodeError(f"unknown value tag {tag!r} in frame payload")
    return cls(**{name: _decode_value(item, blobs) for name, item in node["fields"].items()})


def _blob(blobs: list[bytes], index: Any) -> bytes:
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(blobs):
        raise FrameDecodeError(f"blob index {index!r} outside the frame's {len(blobs)} segments")
    return blobs[index]


# --------------------------------------------------------------------------
# Frame encoding / decoding
# --------------------------------------------------------------------------


def _pack(header: dict[str, Any], blobs: list[bytes]) -> bytes:
    header["v"] = PROTOCOL_VERSION
    header["blobs"] = [len(blob) for blob in blobs]
    payload = json.dumps(header).encode("utf-8")
    body = _HEADER_STRUCT.pack(len(payload)) + payload + b"".join(blobs)
    if len(body) > MAX_FRAME_BYTES:
        raise FrameEncodeError(f"frame of {len(body)} bytes exceeds the {MAX_FRAME_BYTES} limit")
    return MAGIC + _HEADER_STRUCT.pack(len(body)) + body


def encode_request(request_id: str, method: str, kwargs: dict[str, Any] | None = None) -> bytes:
    """Encode a call request. ``method`` is passed through unexamined."""
    blobs: list[bytes] = []
    header = {
        "kind": "request",
        "request_id": request_id,
        "method": method,
        "kwargs": _encode_value(dict(kwargs or {}), blobs),
    }
    return _pack(header, blobs)


def encode_result(request_id: str, value: Any) -> bytes:
    """Encode a successful reply carrying ``value``."""
    blobs: list[bytes] = []
    header = {
        "kind": "result",
        "request_id": request_id,
        "value": _encode_value(value, blobs),
    }
    return _pack(header, blobs)


def encode_error(request_id: str, exc: BaseException) -> bytes:
    """Encode a failure reply, keeping the exception's structured fields.

    An exception outside the registry is still encoded under its own class
    name with an empty field set, so the far side can report what it was
    rather than pretending the call succeeded.
    """
    blobs: list[bytes] = []
    header = {
        "kind": "error",
        "request_id": request_id,
        "class_tag": type(exc).__name__,
        "message": str(exc),
        "fields": _encode_value(exception_fields(exc), blobs),
    }
    return _pack(header, blobs)


def decode_frame(frame: bytes) -> RequestFrame | ResultFrame | ErrorFrame:
    """Decode one complete frame, magic prefix included."""
    if len(frame) < 12 or not frame.startswith(MAGIC):
        raise FrameDecodeError("frame does not start with the expected magic prefix")
    (body_len,) = _HEADER_STRUCT.unpack_from(frame, 4)
    body = frame[8:]
    if len(body) != body_len:
        raise FrameDecodeError(f"frame declares {body_len} body bytes but carries {len(body)}")
    (header_len,) = _HEADER_STRUCT.unpack_from(body, 0)
    if header_len > len(body) - 4:
        raise FrameDecodeError("frame header length runs past the end of the frame")
    try:
        header = json.loads(body[4 : 4 + header_len].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrameDecodeError(f"frame header is not valid JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise FrameDecodeError("frame header is not a JSON object")
    if header.get("v") != PROTOCOL_VERSION:
        raise FrameDecodeError(f"frame protocol version {header.get('v')!r} != {PROTOCOL_VERSION}")

    blobs: list[bytes] = []
    offset = 4 + header_len
    for size in header.get("blobs", []):
        blobs.append(bytes(body[offset : offset + size]))
        if len(blobs[-1]) != size:
            raise FrameDecodeError("frame blob segment is truncated")
        offset += size

    kind = header.get("kind")
    request_id = header.get("request_id")
    if not isinstance(request_id, str):
        raise FrameDecodeError("frame is missing its request id")
    if kind == "request":
        return RequestFrame(
            request_id=request_id,
            method=header["method"],
            kwargs=_decode_value(header["kwargs"], blobs),
        )
    if kind == "result":
        return ResultFrame(request_id=request_id, value=_decode_value(header["value"], blobs))
    if kind == "error":
        class_tag = str(header.get("class_tag", ""))
        message = str(header.get("message", ""))
        fields = _decode_value(header.get("fields", {_TAG: "dict", "items": []}), blobs)
        return ErrorFrame(
            request_id=request_id,
            class_tag=class_tag,
            message=message,
            exception=reconstruct_exception(class_tag, message, fields),
        )
    raise FrameDecodeError(f"unknown frame kind {kind!r}")


class FrameReader:
    """Reassembles frames from a byte stream delivered in arbitrary chunks.

    A pipe hands over whatever bytes happen to be available, which splits
    frames at meaningless boundaries and coalesces several into one read.
    ``feed`` absorbs a chunk and returns every frame that is now complete, in
    order; partial trailing bytes stay buffered for the next call.
    """

    def __init__(self, max_frame_bytes: int = MAX_FRAME_BYTES) -> None:
        self._buffer = bytearray()
        self._max_frame_bytes = max_frame_bytes

    def __len__(self) -> int:
        """Bytes currently buffered as an incomplete frame."""
        return len(self._buffer)

    def feed(self, data: bytes) -> list[RequestFrame | ResultFrame | ErrorFrame]:
        """Absorb ``data`` and return the frames it completed."""
        self._buffer.extend(data)
        return [decode_frame(raw) for raw in self._take_raw()]

    def _take_raw(self) -> list[bytes]:
        frames: list[bytes] = []
        while len(self._buffer) >= 8:
            if bytes(self._buffer[:4]) != MAGIC:
                raise FrameDecodeError("stream is out of sync: no frame magic at the read point")
            (body_len,) = _HEADER_STRUCT.unpack_from(self._buffer, 4)
            if body_len > self._max_frame_bytes:
                raise FrameDecodeError(
                    f"frame declares {body_len} bytes, above the {self._max_frame_bytes} limit"
                )
            total = 8 + body_len
            if len(self._buffer) < total:
                break
            frames.append(bytes(self._buffer[:total]))
            del self._buffer[:total]
        return frames
