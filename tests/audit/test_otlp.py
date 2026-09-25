"""The OTLP emission of the full tool-call record, against a local HTTP server."""

from __future__ import annotations

import json
import logging
import queue
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from osprey.audit import otlp

RECORD = {
    "surface": "tool_call",
    "session_id": "conv-1",
    "tool_use_id": "toolu_1",
    "subject": "mcp__controls__channel_read",
    "server": "controls",
    "decision": "allowed",
    "result": {"value": 1.5},
}


class _Collector:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        collector = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # http.server spelling
                length = int(self.headers.get("Content-Length") or 0)
                collector.requests.append(
                    {
                        "path": self.path,
                        "headers": dict(self.headers),
                        "body": json.loads(self.rfile.read(length)),
                    }
                )
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *_args):
                return

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/api/default/"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture(autouse=True)
def _fresh_emitter(monkeypatch):
    """A fresh queue, sender and warn-once set for every test."""
    monkeypatch.setattr(otlp, "_queue", queue.Queue(maxsize=otlp.QUEUE_SIZE))
    monkeypatch.setattr(otlp, "_sender", None)
    monkeypatch.setattr(otlp, "_warned", set())
    for name in (
        "OTEL_EXPORTER_OTLP_HEADERS",
        "OTEL_EXPORTER_OTLP_PROTOCOL",
        "OTEL_RESOURCE_ATTRIBUTES",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def collector(monkeypatch):
    server = _Collector()
    monkeypatch.setenv("CLAUDE_CODE_ENABLE_TELEMETRY", "1")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", server.endpoint)
    yield server
    server.close()


def _attributes(log_record: dict) -> dict[str, str]:
    return {item["key"]: item["value"]["stringValue"] for item in log_record["attributes"]}


def test_one_record_is_one_otlp_log(collector, monkeypatch):
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "deployment=demo")
    otlp.emit(RECORD)
    otlp._queue.join()

    (request,) = collector.requests
    assert request["path"] == "/api/default/v1/logs"
    assert request["headers"]["Content-Type"] == "application/json"
    (resource_logs,) = request["body"]["resourceLogs"]
    resource = _attributes(resource_logs["resource"])
    assert resource == {"deployment": "demo", "service.name": otlp.DEFAULT_SERVICE_NAME}
    (scope_logs,) = resource_logs["scopeLogs"]
    assert scope_logs["scope"]["name"] == "osprey.audit.tool_call"
    (log_record,) = scope_logs["logRecords"]
    assert json.loads(log_record["body"]["stringValue"]) == RECORD
    assert _attributes(log_record) == {
        "event.name": "osprey.tool_call",
        "session.id": "conv-1",
        "tool_use_id": "toolu_1",
        "tool_name": "mcp__controls__channel_read",
        "osprey.server": "controls",
        "osprey.decision": "allowed",
    }


def test_the_resolver_headers_are_sent(collector, monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "Authorization=Basic abc=,stream-name=x")
    otlp.emit(RECORD)
    otlp._queue.join()

    (request,) = collector.requests
    assert request["headers"]["Authorization"] == "Basic abc="
    assert request["headers"]["stream-name"] == "x"


def test_telemetry_off_sends_nothing(collector, monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_ENABLE_TELEMETRY", "0")
    otlp.emit(RECORD)
    otlp._queue.join()

    assert collector.requests == []
    assert otlp._sender is None


def test_grpc_sends_nothing_and_warns_once(collector, monkeypatch, caplog):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
    with caplog.at_level(logging.WARNING, logger=otlp.__name__):
        otlp.emit(RECORD)
        otlp.emit(RECORD)
    otlp._queue.join()

    assert collector.requests == []
    assert len([r for r in caplog.records if "grpc" in r.getMessage()]) == 1


def _dead_endpoint() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def test_a_dead_endpoint_never_raises(monkeypatch, caplog):
    monkeypatch.setenv("CLAUDE_CODE_ENABLE_TELEMETRY", "1")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", _dead_endpoint())
    with caplog.at_level(logging.WARNING, logger=otlp.__name__):
        otlp.emit(RECORD)
        otlp.emit(RECORD)
        otlp._queue.join()

    warnings = [r for r in caplog.records if "telemetry store" in r.getMessage()]
    assert len(warnings) == 1


@pytest.mark.usefixtures("collector")
def test_a_full_queue_drops_and_warns_once(monkeypatch, caplog):
    monkeypatch.setattr(otlp, "_ensure_sender", lambda: None)
    monkeypatch.setattr(otlp, "_queue", queue.Queue(maxsize=2))
    with caplog.at_level(logging.WARNING, logger=otlp.__name__):
        for _ in range(5):
            otlp.emit(RECORD)

    assert otlp._queue.qsize() == 2
    assert len([r for r in caplog.records if "queue is full" in r.getMessage()]) == 1
