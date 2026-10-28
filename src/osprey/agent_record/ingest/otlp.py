"""The OTLP receiver: Claude Code's log events become receiver-owned spool lines.

agent-record receives OTLP itself. Only logs are taken; the traces and metrics
endpoints answer 404, because their facts are derived from entries and storing
them would record a fact twice. A request is handled in four steps:

**Decode.** ``application/x-protobuf`` and ``application/json`` bodies, plain or
``Content-Encoding: gzip``, are decoded into an ``ExportLogsServiceRequest``. The
body is capped at :data:`MAX_BODY_BYTES` both on the wire and decompressed: a
larger one answers 413, an unknown type or encoding 415, an undecodable body 400.

**Sanitise.** Every sender-set attribute whose key starts with ``osprey.`` is
discarded at resource, scope and log level, and so is a ``session.id`` or
``terminal.session`` that is not a string matching
:data:`~osprey.agent_record.contract.TOKEN_RE`. An event's time is its
``time_unix_nano``, or its observed time when that is zero; an event older than
the retention or more than :data:`FUTURE_SKEW` ahead of the receiver's clock is
dropped.

**Map.** Each kept event becomes one line of its own kind: ``tool_decision`` a
``decision`` carrying its ``tool_use_id``, ``api_request`` a ``usage`` carrying
only what the transcript lacks (:data:`USAGE_FIELDS`), ``api_error`` an
``error``. Every other event is dropped, because the transcript carries the
content. A line is keyed ``otel:<identity>:<session>:<event name>:<content id>``
(:func:`~osprey.agent_record.contract.otel_key`): the content id is the
``tool_use_id`` of a decision, and otherwise a hash of the event's sanitised
attributes without ``event.sequence``, which restarts with every CLI process.
An event without a valid ``session.id`` cannot be keyed and is dropped.

**Write.** The batch is written fsynced as one sealed segment under
``<spool root>/.otlp/<login>/`` (:func:`~osprey.agent_record.writer.write_received`)
before the receiver answers 200; when that write fails it answers 503, so the
exporter retries. The receiver never touches the record backend: the spool
consumer commits the segment, so a backend outage loses nothing.

Authentication belongs to the hosting service, which hands the authenticated
login in (:func:`otlp_app`'s ``login_of``, or :meth:`OtlpReceiver.receive_logs`).
"""

from __future__ import annotations

import json
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from google.protobuf import json_format
from google.protobuf.message import DecodeError
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import (
    ExportLogsServiceRequest,
    ExportLogsServiceResponse,
)
from opentelemetry.proto.common.v1.common_pb2 import AnyValue
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from osprey.agent_record import contract as c
from osprey.agent_record import writer
from osprey.agent_record.config import DEFAULT_RETENTION_DAYS
from osprey.agent_record.identities import login_source_identity

#: The largest request body taken, compressed or decompressed.
MAX_BODY_BYTES = 20 * 1024 * 1024

#: How far ahead of the receiver's clock an event may claim to be.
FUTURE_SKEW = timedelta(hours=1)

PROTOBUF_CONTENT_TYPE = "application/x-protobuf"
JSON_CONTENT_TYPE = "application/json"

LOGS_PATH = "/v1/logs"
TRACES_PATH = "/v1/traces"
METRICS_PATH = "/v1/metrics"

#: Sender attributes with this prefix are OSPREY's to set, never a sender's.
RESERVED_ATTRIBUTE_PREFIX = "osprey."
SESSION_ATTRIBUTE = "session.id"
TERMINAL_SESSION_ATTRIBUTE = "terminal.session"
_SESSION_KEYS = (SESSION_ATTRIBUTE, TERMINAL_SESSION_ATTRIBUTE)
EVENT_NAME_ATTRIBUTE = "event.name"
#: The prefix of a Claude Code event's log body, ``claude_code.<event name>``.
EVENT_BODY_PREFIX = "claude_code."

EVENT_TOOL_DECISION = "tool_decision"
EVENT_API_REQUEST = "api_request"
EVENT_API_ERROR = "api_error"

#: The entry kind each exported event becomes; every other event is dropped.
EVENT_KINDS: Mapping[str, str] = {
    EVENT_TOOL_DECISION: c.KIND_DECISION,
    EVENT_API_REQUEST: c.KIND_USAGE,
    EVENT_API_ERROR: c.KIND_ERROR,
}

#: The attributes each kind's body carries: only what the transcript lacks.
DECISION_FIELDS = ("decision", "source", "tool_name", "tool_source")
USAGE_FIELDS = ("cost_usd", "duration_ms", "ttft_ms", "speed", "effort")
ERROR_FIELDS = ("error", "status_code", "attempt", "duration_ms", "model")
_BODY_FIELDS: Mapping[str, tuple[str, ...]] = {
    c.KIND_DECISION: DECISION_FIELDS,
    c.KIND_USAGE: USAGE_FIELDS,
    c.KIND_ERROR: ERROR_FIELDS,
}

#: Log-record fields OTLP/JSON spells as hex, which the protobuf JSON parser
#: would read as base64; nothing here uses them.
_HEX_ID_FIELDS = ("traceId", "spanId")


class OtlpRefused(Exception):
    """A request answered with *status* before anything is written."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def _media_type(content_type: str | None) -> str:
    return (content_type or "").split(";", 1)[0].strip().lower()


def _decompress(body: bytes, encoding: str | None) -> bytes:
    encoding = (encoding or "identity").strip().lower()
    if encoding == "identity":
        return body
    if encoding != "gzip":
        raise OtlpRefused(415, f"unsupported content encoding {encoding!r}")
    inflater = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
    try:
        out = inflater.decompress(body, MAX_BODY_BYTES + 1)
        if len(out) > MAX_BODY_BYTES or inflater.unconsumed_tail:
            raise OtlpRefused(413, f"decompressed body exceeds {MAX_BODY_BYTES} bytes")
        out += inflater.flush()
    except zlib.error as exc:
        raise OtlpRefused(400, f"body is not valid gzip: {exc}") from None
    if len(out) > MAX_BODY_BYTES:
        raise OtlpRefused(413, f"decompressed body exceeds {MAX_BODY_BYTES} bytes")
    if not inflater.eof:
        raise OtlpRefused(400, "body is truncated gzip")
    return out


def _drop_hex_ids(data: Any) -> None:
    for resource_logs in data.get("resourceLogs") or ():
        for scope_logs in (resource_logs or {}).get("scopeLogs") or ():
            for record in (scope_logs or {}).get("logRecords") or ():
                if isinstance(record, dict):
                    for name in _HEX_ID_FIELDS:
                        record.pop(name, None)


def decode_logs(
    body: bytes, content_type: str | None, content_encoding: str | None = None
) -> ExportLogsServiceRequest:
    """Decode an OTLP logs request body.

    Raises:
        OtlpRefused: 413 for an oversize body, 415 for an unknown type or
            encoding, 400 for a body that does not decode.
    """
    if len(body) > MAX_BODY_BYTES:
        raise OtlpRefused(413, f"body exceeds {MAX_BODY_BYTES} bytes")
    media = _media_type(content_type)
    if media not in (PROTOBUF_CONTENT_TYPE, JSON_CONTENT_TYPE):
        raise OtlpRefused(415, f"unsupported content type {media or '(none)'!r}")
    raw = _decompress(body, content_encoding)
    request = ExportLogsServiceRequest()
    try:
        if media == PROTOBUF_CONTENT_TYPE:
            request.ParseFromString(raw)
        else:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("an OTLP/JSON body is a JSON object")
            _drop_hex_ids(data)
            json_format.ParseDict(data, request, ignore_unknown_fields=True)
    except (DecodeError, json_format.ParseError, ValueError, TypeError) as exc:
        raise OtlpRefused(400, f"body is not an OTLP logs request: {exc}") from None
    return request


def _any_value(value: AnyValue) -> Any:
    which = value.WhichOneof("value")
    if which is None:
        return None
    if which == "array_value":
        return [_any_value(v) for v in value.array_value.values]
    if which == "kvlist_value":
        return {kv.key: _any_value(kv.value) for kv in value.kvlist_value.values}
    if which == "bytes_value":
        return value.bytes_value.hex()
    return getattr(value, which)


def _sanitised(key_values: Any) -> dict[str, Any]:
    """Attributes without ``osprey.*`` keys and without a non-token session."""
    out: dict[str, Any] = {}
    for kv in key_values:
        if kv.key.startswith(RESERVED_ATTRIBUTE_PREFIX):
            continue
        value = _any_value(kv.value)
        if kv.key in _SESSION_KEYS and not c.is_token(value):
            continue
        out[kv.key] = value
    return out


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------


def _event_name(record: Any, attributes: Mapping[str, Any]) -> str:
    name = attributes.get(EVENT_NAME_ATTRIBUTE)
    if isinstance(name, str) and name:
        return name
    if isinstance(record.event_name, str) and record.event_name:
        return record.event_name
    body = _any_value(record.body) if record.HasField("body") else None
    if isinstance(body, str) and body.startswith(EVENT_BODY_PREFIX):
        return body[len(EVENT_BODY_PREFIX) :]
    return ""


def _event_time(record: Any) -> datetime | None:
    nanos = record.time_unix_nano or record.observed_time_unix_nano
    if not nanos:
        return None
    seconds, rest = divmod(nanos, 1_000_000_000)
    return datetime.fromtimestamp(seconds, tz=UTC) + timedelta(microseconds=rest // 1000)


@dataclass(frozen=True)
class _Kept:
    kind: str
    event_name: str
    when: datetime
    attributes: Mapping[str, Any]
    session: str
    terminal_session: str | None


def map_logs(
    request: ExportLogsServiceRequest,
    identity: str,
    *,
    now: datetime,
    retention_days: int,
) -> list[writer.ReceivedLine]:
    """The receiver lines of *request*, sanitised, clamped and keyed for *identity*."""
    floor = now - timedelta(days=retention_days)
    ceiling = now + FUTURE_SKEW
    lines: list[writer.ReceivedLine] = []
    for resource_logs in request.resource_logs:
        resource = _sanitised(resource_logs.resource.attributes)
        for scope_logs in resource_logs.scope_logs:
            for record in scope_logs.log_records:
                kept = _keep(record, resource, floor=floor, ceiling=ceiling)
                if kept is not None:
                    lines.append(_line(kept, identity))
    return lines


def _keep(
    record: Any, resource: Mapping[str, Any], *, floor: datetime, ceiling: datetime
) -> _Kept | None:
    attributes = _sanitised(record.attributes)
    event_name = _event_name(record, attributes)
    kind = EVENT_KINDS.get(event_name)
    if kind is None:
        return None
    when = _event_time(record)
    if when is None or when < floor or when > ceiling:
        return None
    session = attributes.get(SESSION_ATTRIBUTE, resource.get(SESSION_ATTRIBUTE))
    if session is None:
        return None
    return _Kept(
        kind=kind,
        event_name=event_name,
        when=when,
        attributes=attributes,
        session=session,
        terminal_session=attributes.get(
            TERMINAL_SESSION_ATTRIBUTE, resource.get(TERMINAL_SESSION_ATTRIBUTE)
        ),
    )


def _line(kept: _Kept, identity: str) -> writer.ReceivedLine:
    attributes = kept.attributes
    tool_use_id = attributes.get("tool_use_id") if kept.kind == c.KIND_DECISION else None
    if not c.is_token(tool_use_id):
        tool_use_id = None
    key = c.otel_key(
        identity,
        kept.session,
        kept.event_name,
        tool_use_id=tool_use_id,
        attributes=attributes,
    )
    body = {name: attributes[name] for name in _BODY_FIELDS[kept.kind] if name in attributes}
    ids: dict[str, Any] = {"ts": kept.when, "session_id": kept.session}
    if kept.terminal_session is not None:
        ids["terminal_session"] = kept.terminal_session
    if tool_use_id is not None:
        ids["tool_use_id"] = tool_use_id
    return writer.ReceivedLine(kind=kept.kind, body=body, key=key, ids=ids)


# ---------------------------------------------------------------------------
# The receiver
# ---------------------------------------------------------------------------


def _text(status: int, message: str) -> Response:
    return Response(message, status_code=status, media_type="text/plain")


class OtlpReceiver:
    """Receive OTLP logs into ``<spool_root>/.otlp/<login>/``.

    Args:
        spool_root: The spool root agent-record owns.
        config: The rendered deployment config; maps a login to its identity.
        retention_days: Events older than this are dropped.
        now: The clock, returning an aware ``datetime``.
    """

    def __init__(
        self,
        spool_root: Path | str,
        config: Mapping[str, Any] | None = None,
        *,
        retention_days: int = DEFAULT_RETENTION_DAYS,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(retention_days, int) or retention_days < 1:
            raise ValueError(f"retention_days {retention_days!r} must be a positive int")
        self.spool_root = Path(spool_root)
        self.config = config
        self.retention_days = retention_days
        self._now = now or (lambda: datetime.now(UTC))

    def handle(
        self,
        login: str,
        body: bytes,
        content_type: str | None,
        content_encoding: str | None = None,
    ) -> Response:
        """Answer one logs request from *login*; blocks on the fsynced write."""
        try:
            identity = login_source_identity(self.config, login)
        except ValueError as exc:
            return _text(403, str(exc))
        try:
            request = decode_logs(body, content_type, content_encoding)
        except OtlpRefused as exc:
            return _text(exc.status, exc.message)
        lines = map_logs(request, identity, now=self._now(), retention_days=self.retention_days)
        if writer.write_received(self.spool_root, login, lines) is None:
            return _text(503, "the batch could not be stored; retry")
        return _success(_media_type(content_type))

    async def receive_logs(self, request: Request, login: str) -> Response:
        """The ``POST /v1/logs`` handler for an authenticated *login*."""
        try:
            body = await _read_capped(request)
        except OtlpRefused as exc:
            return _text(exc.status, exc.message)
        return await run_in_threadpool(
            self.handle,
            login,
            body,
            request.headers.get("content-type"),
            request.headers.get("content-encoding"),
        )


def _success(media: str) -> Response:
    response = ExportLogsServiceResponse()
    if media == JSON_CONTENT_TYPE:
        return Response(json_format.MessageToJson(response), media_type=JSON_CONTENT_TYPE)
    return Response(response.SerializeToString(), media_type=PROTOBUF_CONTENT_TYPE)


async def _read_capped(request: Request) -> bytes:
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        raise OtlpRefused(413, f"body exceeds {MAX_BODY_BYTES} bytes")
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BODY_BYTES:
            raise OtlpRefused(413, f"body exceeds {MAX_BODY_BYTES} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


async def not_exported(_request: Request) -> Response:
    """Traces and metrics are not taken."""
    return _text(404, "only OTLP logs are received")


def otlp_app(receiver: OtlpReceiver, *, login_of: Callable[[Request], str | None]) -> Starlette:
    """An ASGI app serving ``/v1/logs``, and 404 on ``/v1/traces`` and ``/v1/metrics``.

    Args:
        receiver: The receiver the logs go to.
        login_of: The hosting service's authentication: the request's login,
            or ``None`` when it is not authenticated (answered 401).
    """

    async def logs(request: Request) -> Response:
        login = login_of(request)
        if login is None:
            return _text(401, "not authenticated")
        return await receiver.receive_logs(request, login)

    return Starlette(
        routes=[
            Route(LOGS_PATH, logs, methods=["POST"]),
            Route(TRACES_PATH, not_exported, methods=["POST"]),
            Route(METRICS_PATH, not_exported, methods=["POST"]),
        ]
    )


__all__ = [
    "FUTURE_SKEW",
    "MAX_BODY_BYTES",
    "OtlpReceiver",
    "OtlpRefused",
    "decode_logs",
    "map_logs",
    "not_exported",
    "otlp_app",
]
