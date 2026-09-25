"""Send each full tool-call record to the telemetry store as one OTLP log.

No endpoint of its own: the record goes where the agent harness's telemetry
already goes, read from the environment every stdio MCP server inherits
(``CLAUDE_CODE_ENABLE_TELEMETRY``, ``OTEL_EXPORTER_OTLP_ENDPOINT``,
``OTEL_EXPORTER_OTLP_HEADERS``, ``OTEL_RESOURCE_ATTRIBUTES``). Each record is
posted as OTLP/HTTP JSON to ``<endpoint>/v1/logs`` with the attribute
``session.id`` — the harness's own name for the conversation id — so the line
joins the harness's events for the same conversation.

The post happens on one daemon thread fed by a bounded queue, so a slow or dead
endpoint never blocks the call being recorded. A full queue drops the record
and warns once; a failed post is logged, one warning per error class. The
``grpc`` protocol is not spoken here: it emits nothing and warns once. The
durable ledger (``tool_call.jsonl``) is authoritative; this copy is for search.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["SCOPE_NAME", "emit"]

#: The instrumentation scope every record is sent under.
SCOPE_NAME = "osprey.audit.tool_call"

#: The event name a record is searchable by.
EVENT_NAME = "osprey.tool_call"

#: The ``service.name`` used when the resource attributes name none.
DEFAULT_SERVICE_NAME = "osprey-mcp"

#: How many records may wait for the sender before new ones are dropped.
QUEUE_SIZE = 64

#: Per-post timeout, seconds.
POST_TIMEOUT_S = 5.0

_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=QUEUE_SIZE)
_sender: threading.Thread | None = None
_sender_lock = threading.Lock()
_warned: set[str] = set()


def _warn_once(key: str, message: str, *args: Any) -> None:
    if key in _warned:
        return
    _warned.add(key)
    logger.warning(message, *args)


def _enabled() -> bool:
    """The gate the harness's own emitter uses: telemetry on, and an endpoint."""
    if os.environ.get("CLAUDE_CODE_ENABLE_TELEMETRY") != "1":
        return False
    return bool(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"))


def _resource_attributes() -> list[dict[str, Any]]:
    attributes: dict[str, str] = {}
    for pair in (os.environ.get("OTEL_RESOURCE_ATTRIBUTES") or "").split(","):
        key, _sep, value = pair.partition("=")
        if key.strip() and value.strip():
            attributes[key.strip()] = value.strip()
    attributes.setdefault("service.name", DEFAULT_SERVICE_NAME)
    return [{"key": key, "value": {"stringValue": value}} for key, value in attributes.items()]


def _attribute(key: str, value: Any) -> dict[str, Any]:
    return {"key": key, "value": {"stringValue": "" if value is None else str(value)}}


def _payload(record: dict[str, Any]) -> dict[str, Any]:
    """The OTLP ``resourceLogs`` document carrying *record*."""
    return {
        "resourceLogs": [
            {
                "resource": {"attributes": _resource_attributes()},
                "scopeLogs": [
                    {
                        "scope": {"name": SCOPE_NAME},
                        "logRecords": [
                            {
                                "timeUnixNano": str(time.time_ns()),
                                "body": {
                                    "stringValue": json.dumps(
                                        record, separators=(",", ":"), default=str
                                    )
                                },
                                "attributes": [
                                    _attribute("event.name", EVENT_NAME),
                                    _attribute("session.id", record.get("session_id")),
                                    _attribute("tool_use_id", record.get("tool_use_id")),
                                    _attribute("tool_name", record.get("subject")),
                                    _attribute("osprey.server", record.get("server")),
                                    _attribute("osprey.decision", record.get("decision")),
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    }


def _post(document: dict[str, Any]) -> None:
    """POST one document to the logs endpoint. Raises on failure."""
    import httpx

    from osprey.build.claude_code_telemetry import _parse_header_map

    endpoint = (os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or "").rstrip("/")
    headers = _parse_header_map(os.environ.get("OTEL_EXPORTER_OTLP_HEADERS") or "")
    headers["Content-Type"] = "application/json"
    response = httpx.post(
        f"{endpoint}/v1/logs", json=document, headers=headers, timeout=POST_TIMEOUT_S
    )
    response.raise_for_status()


def _drain() -> None:
    while True:
        document = _queue.get()
        try:
            _post(document)
        except Exception as exc:
            logger.debug("Could not send a tool_call record", exc_info=True)
            _warn_once(
                f"post:{type(exc).__name__}",
                "Could not send tool_call records to the telemetry store (%s); "
                "the audit ledger still holds them",
                type(exc).__name__,
            )
        finally:
            _queue.task_done()


def _ensure_sender() -> None:
    global _sender
    with _sender_lock:
        if _sender is None or not _sender.is_alive():
            _sender = threading.Thread(target=_drain, name="osprey-tool-call-otlp", daemon=True)
            _sender.start()


def emit(record: dict[str, Any]) -> None:
    """Queue *record* for the telemetry store. Never raises, never blocks."""
    try:
        if not _enabled():
            return
        if (os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL") or "").strip() == "grpc":
            _warn_once(
                "grpc",
                "OTEL_EXPORTER_OTLP_PROTOCOL is grpc; tool_call records are sent over "
                "OTLP/HTTP only and are not sent",
            )
            return
        _ensure_sender()
        try:
            _queue.put_nowait(_payload(record))
        except queue.Full:
            _warn_once(
                "full",
                "The tool_call telemetry queue is full; records are dropped from the "
                "telemetry store and kept in the audit ledger",
            )
    except Exception:
        logger.debug("Could not queue a tool_call record", exc_info=True)
