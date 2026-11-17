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
    (see below), including the ``ChannelReadBatch`` a batched multi-channel
    read returns. A well-framed reply whose value its own class
    refuses to construct decodes as an ``error`` for that request alone, with
    a :class:`~osprey_connectors.errors.ConnectorInternalError`; the stream
    goes on.
``error``
    ``{request_id, class_tag, message, fields}``. Exceptions cross the
    boundary *typed*: ``fields`` are the structured attributes enumerated by
    :mod:`osprey_connectors.ipc._exceptions`, and decoding calls the real
    constructor, so ``.attempted_value``, ``.reason`` and the min/max/step
    bounds survive and the parent's error rendering is unchanged. That module
    owns the registry — which classes cross typed, which fields each carries,
    and how an unrecognised ``class_tag`` fails closed to a
    :class:`~osprey_connectors.errors.ConnectorInternalError` naming the
    original tag and message.

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
shape preserved), and the package's records:

* the connector records a connector's calls return, ``ChannelValue``,
  ``ChannelMetadata``, ``ChannelReadBatch`` and ``ChannelWriteResult``, and the
  ``LimitsViolation`` a limits refusal carries, named by class name and
  carried as their fields in declaration order. Each decodes as a real
  instance of its class, validated by the class itself. A subclass is
  refused: only the class the far side rebuilds crosses.
* a :class:`~osprey_connectors.errors.ChannelError`, the one exception a value
  may hold (a batch read's failure for one channel), as an ``error`` node
  carrying the same ``class_tag``, ``message`` and ``fields`` an error frame
  does. It decodes through the same registry, so an unknown tag or a refused
  field set fails closed to a ``ConnectorInternalError``, which no record
  field typed for a ``ChannelError`` accepts.

Anything else is refused at encode time rather than silently degraded.

A record decodes through its class's lax validation: the wire is JSON-shaped,
so a tuple arrives as a list and an enum member as its string, and the field's
declared type turns each back into what the sender held.

Failures
--------
A stream that has lost its framing (no magic at the read point, an absurd
length) cannot be read past. A frame that is delimited but cannot be decoded
can: :meth:`FrameReader._feed_raw` hands frames over undecoded, and
:func:`_request_id_of` reads the request such a frame named, so the
connector-host child refuses that request alone and keeps serving.
"""

from __future__ import annotations

import json
import struct
import uuid
from collections.abc import Mapping
from datetime import datetime
from io import BytesIO
from typing import TypeAlias, TypeGuard

import numpy as np
from pydantic import BaseModel, ConfigDict

from osprey_connectors.control_system._base import (
    ChannelMetadata,
    ChannelReadBatch,
    ChannelValue,
    ChannelWriteResult,
)
from osprey_connectors.errors import (
    ChannelError,
    ConnectorInternalError,
    ConnectorsError,
    LimitsViolation,
)
from osprey_connectors.ipc._exceptions import (
    _registered_tag,
    exception_fields,
    reconstruct_exception,
)
from osprey_connectors.model import FrozenModel

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
#: 2: a write result carries ``verdict_reason`` and a blocked write's error
#: frame its typed reason, which a version-1 peer would drop.
#: 3: a limits refusal crosses as a write result carrying its
#: ``LimitsViolation`` (a version-2 peer raised it as an error frame), a blocked
#: write's error frame carries that violation, a failed write's error frame
#: carries its outcome alone, without the reason word, and a batch read answers
#: a ``ChannelReadBatch`` whose failures cross as error nodes.
PROTOCOL_VERSION = 3

#: Refuse absurd lengths rather than trying to allocate them. Generous enough
#: for a detector image, small enough that a corrupt length cannot exhaust RAM.
MAX_FRAME_BYTES = 256 * 1024 * 1024

_HEADER_STRUCT = struct.Struct(">I")


def _length_at(buffer: bytes | bytearray, offset: int) -> int:
    """The length word :data:`_HEADER_STRUCT` packed at *offset*; the caller checked it is there."""
    return int.from_bytes(buffer[offset : offset + _HEADER_STRUCT.size], "big")


#: Marker key identifying a tagged (non-scalar) value node in the JSON header.
_TAG = "__osprey_ipc__"

#: A node of a frame's JSON header, as ``json`` writes it.
_Json: TypeAlias = None | bool | int | float | str | list["_Json"] | dict[str, "_Json"]


class FrameEncodeError(ConnectorsError, TypeError):
    """A value was handed to the codec that the wire format cannot carry."""


class FrameDecodeError(ConnectorsError, ValueError):
    """A frame could not be parsed: bad magic, bad version, or unknown tag."""


class _UnrebuildableValueError(FrameDecodeError):
    """A well-framed value whose own class refused the decoded fields."""

    def __init__(self, class_tag: str, cause: Exception) -> None:
        self.class_tag = class_tag
        self.cause = cause
        super().__init__(
            f"connector host sent a {class_tag} reply the parent cannot rebuild: {cause!r}"
        )


class RequestFrame(FrozenModel):
    """A decoded call request. ``method`` and ``kwargs`` are never validated by the codec:
    the child refuses a bad one in that request's error frame."""

    request_id: str
    method: object
    kwargs: object


class ResultFrame(FrozenModel):
    """A decoded successful reply."""

    request_id: str
    value: object


class ErrorFrame(FrozenModel):
    """A decoded failure reply.

    ``exception`` is a real, reconstructed exception instance ready to be
    raised by the parent; ``class_tag`` and ``message`` are kept alongside it
    so a caller can still see what the child actually sent — which matters for
    the fail-closed cases, where ``exception`` is a ``ConnectorInternalError``
    standing in for a class this codec does not know, or for a reply value it
    could not rebuild (``class_tag`` then names the value's class).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_id: str
    class_tag: str
    message: str
    exception: BaseException


#: The records a frame may carry, by the class name that tags them. A frame can
#: rebuild only one of these, never an arbitrary class the far side happens to
#: load.
_RECORDS: dict[str, type[BaseModel]] = {
    cls.__name__: cls
    for cls in (
        ChannelValue,
        ChannelMetadata,
        ChannelReadBatch,
        ChannelWriteResult,
        LimitsViolation,
    )
}


def new_request_id() -> str:
    """A fresh request id. Short, opaque, and unique across both processes."""
    return uuid.uuid4().hex


# --------------------------------------------------------------------------
# .npy payloads — the only place arrays are serialized, so the no-pickle rule
# is enforced (and testable) in exactly two lines.
# --------------------------------------------------------------------------


#: An array of any shape and element type: the one kind of blob a frame carries.
#: Quoted because ``np.generic`` takes no subscript at run time.
_Array: TypeAlias = "np.ndarray[tuple[int, ...], np.dtype[np.generic[object]]]"


def _npy_bytes(array: _Array) -> bytes:
    """Serialize an array to ``.npy`` bytes, never enabling pickle."""
    buffer = BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _npy_load(data: bytes) -> _Array:
    """Read ``.npy`` bytes back, never enabling pickle."""
    buffer = BytesIO(data)
    loaded = np.load(buffer, allow_pickle=False)
    if not isinstance(loaded, np.ndarray):
        raise FrameDecodeError("frame blob is not a single .npy array")
    return loaded


# --------------------------------------------------------------------------
# Value encoding
# --------------------------------------------------------------------------


def _is_json_object(node: object) -> TypeGuard[dict[str, object]]:
    """Whether *node* is a JSON object: a dict keyed by strings, as ``json`` writes one."""
    return isinstance(node, dict) and all(isinstance(key, str) for key in node)


def _is_json_array(node: object) -> TypeGuard[list[object]]:
    """Whether *node* is a JSON array."""
    return isinstance(node, list)


def _encode_value(value: object, blobs: list[bytes]) -> _Json:
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
    if isinstance(value, ChannelError):
        return {
            _TAG: "error",
            "class_tag": _registered_tag(value) or type(value).__name__,
            "message": str(value),
            "fields": _encode_value(exception_fields(value), blobs),
        }
    tag = type(value).__name__
    record = _RECORDS.get(tag)
    if record is None or type(value) is not record:
        raise FrameEncodeError(f"IPC frames cannot carry a value of type {tag!r}")
    return {
        _TAG: tag,
        "fields": {
            name: _encode_value(getattr(value, name), blobs) for name in record.model_fields
        },
    }


def _decode_value(node: object, blobs: list[bytes]) -> object:
    """Inverse of :func:`_encode_value`."""
    if node is None or isinstance(node, (bool, int, float, str)):
        return node
    if _is_json_array(node):
        return [_decode_value(item, blobs) for item in node]
    if not _is_json_object(node):
        raise FrameDecodeError(f"unsupported JSON node of type {type(node).__name__!r}")

    tag = node.get(_TAG)
    if tag is None:
        raise FrameDecodeError("untagged object in frame payload")
    if tag == "ndarray":
        return _npy_load(_blob(blobs, node["blob"]))
    if tag == "npscalar":
        return _npy_load(_blob(blobs, node["blob"]))[()]
    if tag == "datetime":
        iso = node["iso"]
        if not isinstance(iso, str):
            raise FrameDecodeError(f"datetime node carries a {type(iso).__name__}, not a string")
        return datetime.fromisoformat(iso)
    if tag == "dict":
        return _decode_items(node["items"], blobs)
    if tag == "error":
        return _decode_error(node, blobs)
    record = _RECORDS.get(tag) if isinstance(tag, str) else None
    if record is None:
        raise FrameDecodeError(f"unknown value tag {tag!r} in frame payload")
    fields = node["fields"]
    if not _is_json_object(fields):
        raise FrameDecodeError(f"{tag} node fields are not a JSON object")
    kwargs = {name: _decode_value(item, blobs) for name, item in fields.items()}
    # Lax, so each field's declared type restores what the JSON-shaped wire
    # flattened (a tuple, an enum member).
    try:
        return record.model_validate(kwargs, strict=False)
    except (TypeError, ValueError) as exc:
        raise _UnrebuildableValueError(str(tag), exc) from exc


def _decode_error(node: dict[str, object], blobs: list[bytes]) -> BaseException:
    """The exception an ``error`` node describes, rebuilt as an error frame's is."""
    class_tag, message = node.get("class_tag"), node.get("message")
    if not isinstance(class_tag, str) or not isinstance(message, str):
        raise FrameDecodeError("error node does not name its class and message as strings")
    fields = _decode_value(node.get("fields", {_TAG: "dict", "items": []}), blobs)
    if not isinstance(fields, dict):
        raise FrameDecodeError("error node fields are not a mapping")
    return reconstruct_exception(class_tag, message, fields)


def _decode_items(items: object, blobs: list[bytes]) -> dict[object, object]:
    """The dict a ``dict`` node's ``[key, value]`` pairs describe."""
    if not _is_json_array(items):
        raise FrameDecodeError(f"dict node items are a {type(items).__name__}, not a list")
    decoded: dict[object, object] = {}
    for pair in items:
        if not _is_json_array(pair) or len(pair) != 2:
            raise FrameDecodeError(f"dict node item {pair!r} is not a [key, value] pair")
        key, item = pair
        decoded[key] = _decode_value(item, blobs)
    return decoded


def _blob(blobs: list[bytes], index: object) -> bytes:
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(blobs):
        raise FrameDecodeError(f"blob index {index!r} outside the frame's {len(blobs)} segments")
    return blobs[index]


# --------------------------------------------------------------------------
# Frame encoding / decoding
# --------------------------------------------------------------------------


def _pack(header: dict[str, _Json], blobs: list[bytes]) -> bytes:
    header["v"] = PROTOCOL_VERSION
    header["blobs"] = [len(blob) for blob in blobs]
    payload = json.dumps(header).encode("utf-8")
    body = _HEADER_STRUCT.pack(len(payload)) + payload + b"".join(blobs)
    if len(body) > MAX_FRAME_BYTES:
        raise FrameEncodeError(f"frame of {len(body)} bytes exceeds the {MAX_FRAME_BYTES} limit")
    return MAGIC + _HEADER_STRUCT.pack(len(body)) + body


def encode_request(request_id: str, method: str, kwargs: Mapping[str, object]) -> bytes:
    """Encode a call request. ``method`` is passed through unexamined."""
    blobs: list[bytes] = []
    header: dict[str, _Json] = {
        "kind": "request",
        "request_id": request_id,
        "method": method,
        "kwargs": _encode_value(dict(kwargs), blobs),
    }
    return _pack(header, blobs)


def encode_result(request_id: str, value: object) -> bytes:
    """Encode a successful reply carrying ``value``."""
    blobs: list[bytes] = []
    header: dict[str, _Json] = {
        "kind": "result",
        "request_id": request_id,
        "value": _encode_value(value, blobs),
    }
    return _pack(header, blobs)


def encode_error(request_id: str, exc: BaseException) -> bytes:
    """Encode a failure reply, keeping the exception's structured fields.

    The class tag is that of the nearest registered class in the exception's
    MRO, the class its fields are read by, so a subclass of a vocabulary class
    decodes as that class. An exception with no registered class in its MRO is
    still encoded under its own class name with an empty field set, so the far
    side can report what it was rather than pretending the call succeeded: it
    decodes as a ``ConnectorInternalError`` naming that class.
    """
    blobs: list[bytes] = []
    header: dict[str, _Json] = {
        "kind": "error",
        "request_id": request_id,
        "class_tag": _registered_tag(exc) or type(exc).__name__,
        "message": str(exc),
        "fields": _encode_value(exception_fields(exc), blobs),
    }
    return _pack(header, blobs)


def _header(frame: bytes) -> tuple[dict[str, object], bytes, int]:
    """A complete frame's JSON header, its body, and where its blobs start in that body."""
    if len(frame) < 12 or not frame.startswith(MAGIC):
        raise FrameDecodeError("frame does not start with the expected magic prefix")
    body_len = _length_at(frame, 4)
    body = frame[8:]
    if len(body) != body_len:
        raise FrameDecodeError(f"frame declares {body_len} body bytes but carries {len(body)}")
    header_len = _length_at(body, 0)
    if header_len > len(body) - 4:
        raise FrameDecodeError("frame header length runs past the end of the frame")
    try:
        header: object = json.loads(body[4 : 4 + header_len].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrameDecodeError(f"frame header is not valid JSON: {exc}") from exc
    if not _is_json_object(header):
        raise FrameDecodeError("frame header is not a JSON object")
    return header, body, 4 + header_len


def _request_id_of(frame: bytes) -> str | None:
    """The request a complete frame names, or ``None`` when its header names none it can read.

    For answering a frame that would not decode: the header is read as far as
    the request id and no further, and nothing about the frame raises here.
    """
    try:
        header = _header(frame)[0]
    except FrameDecodeError:
        return None
    request_id = header.get("request_id")
    return request_id if isinstance(request_id, str) else None


def decode_frame(frame: bytes) -> RequestFrame | ResultFrame | ErrorFrame:
    """Decode one complete frame, magic prefix included."""
    header, body, offset = _header(frame)
    if header.get("v") != PROTOCOL_VERSION:
        raise FrameDecodeError(f"frame protocol version {header.get('v')!r} != {PROTOCOL_VERSION}")

    blobs: list[bytes] = []
    sizes = header.get("blobs", [])
    if not _is_json_array(sizes):
        raise FrameDecodeError(f"frame blob sizes are a {type(sizes).__name__}, not a list")
    for size in sizes:
        if not isinstance(size, int):
            raise FrameDecodeError(f"frame blob size {size!r} is not a byte count")
        blobs.append(bytes(body[offset : offset + size]))
        if len(blobs[-1]) != size:
            raise FrameDecodeError("frame blob segment is truncated")
        offset += size

    kind = header.get("kind")
    request_id = header.get("request_id")
    if not isinstance(request_id, str):
        raise FrameDecodeError("frame is missing its request id")
    if kind == "request":
        method = header["method"]
        try:
            kwargs = _decode_value(header["kwargs"], blobs)
        except _UnrebuildableValueError as exc:
            raise FrameDecodeError(
                f"request {request_id!r} carries a {exc.class_tag} the connector host "
                f"cannot rebuild: {exc.cause!r}"
            ) from exc
        return RequestFrame(request_id=request_id, method=method, kwargs=kwargs)
    if kind == "result":
        try:
            value = _decode_value(header["value"], blobs)
        except _UnrebuildableValueError as exc:
            return ErrorFrame(
                request_id=request_id,
                class_tag=exc.class_tag,
                message=str(exc),
                exception=ConnectorInternalError(str(exc), class_tag=exc.class_tag),
            )
        return ResultFrame(request_id=request_id, value=value)
    if kind == "error":
        class_tag = str(header.get("class_tag", ""))
        message = str(header.get("message", ""))
        fields = _decode_value(header.get("fields", {_TAG: "dict", "items": []}), blobs)
        if not isinstance(fields, dict):
            raise FrameDecodeError("error frame fields are not a mapping")
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
        return [decode_frame(raw) for raw in self._feed_raw(data)]

    def _feed_raw(self, data: bytes) -> list[bytes]:
        """Absorb ``data`` and return the frames it completed, still encoded.

        For a reader that decodes each frame itself, so that one it cannot
        decode costs that frame alone.

        Raises:
            FrameDecodeError: The stream has lost its framing.
        """
        self._buffer.extend(data)
        return self._take_raw()

    def _take_raw(self) -> list[bytes]:
        frames: list[bytes] = []
        while len(self._buffer) >= 8:
            if bytes(self._buffer[:4]) != MAGIC:
                raise FrameDecodeError("stream is out of sync: no frame magic at the read point")
            body_len = _length_at(self._buffer, 4)
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
