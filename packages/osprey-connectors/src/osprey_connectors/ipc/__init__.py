"""Inter-process plumbing for the connector-host child.

The child process owns the control-system client library; the parent talks to
it over a pipe pair. :mod:`osprey_connectors.ipc.frames` holds the wire format
both sides speak — a pure codec, with no process management in it —
and :mod:`osprey_connectors.ipc.exceptions` holds the typed-exception registry
that codec uses, so a failure crosses the boundary with its fields intact.
"""

from osprey_connectors.ipc.exceptions import (
    CHANNEL_LIMITS_VIOLATION_FIELDS,
    CHANNEL_WRITE_BLOCKED_FIELDS,
    EXCEPTION_SPECS,
    ExceptionSpec,
    decode_exception,
    encode_exception,
    exception_fields,
    reconstruct_exception,
)
from osprey_connectors.ipc.frames import (
    ErrorFrame,
    FrameDecodeError,
    FrameEncodeError,
    FrameReader,
    RequestFrame,
    ResultFrame,
    decode_frame,
    encode_error,
    encode_request,
    encode_result,
    new_request_id,
)

__all__ = [
    "CHANNEL_LIMITS_VIOLATION_FIELDS",
    "CHANNEL_WRITE_BLOCKED_FIELDS",
    "EXCEPTION_SPECS",
    "ErrorFrame",
    "ExceptionSpec",
    "FrameDecodeError",
    "FrameEncodeError",
    "FrameReader",
    "RequestFrame",
    "ResultFrame",
    "decode_exception",
    "decode_frame",
    "encode_error",
    "encode_exception",
    "encode_request",
    "encode_result",
    "exception_fields",
    "new_request_id",
    "reconstruct_exception",
]
