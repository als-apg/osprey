"""The OTLP receiver: logs only (traces and metrics 404), http/protobuf and
http/json with gzip under one body cap, ``osprey.*`` and non-token sessions
discarded, events clamped to the retention window, one ``decision``, ``usage``
or ``error`` line per event keyed by content (never ``event.sequence``), and a
batch written fsynced to ``.otlp/<login>/`` before 200, or 503.
"""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from google.protobuf import json_format
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue
from starlette.testclient import TestClient

from osprey.agent_record import contract as c
from osprey.agent_record import writer
from osprey.agent_record.backends.embedded import EmbeddedBackend
from osprey.agent_record.ingest import otlp
from osprey.agent_record.ingest.otlp import OtlpReceiver, otlp_app
from osprey.agent_record.ingest.spool import SpoolConsumer
from tests.agent_record.test_identities import _config

NOW = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
CONFIG = _config(workers=1, lanes=True)
LOGIN = "alice"
SESSION = "bb1af18d-a6d1-4ce8-a1f6-b088b297e9b8"
TERMINAL = "contract-d733b7c87b7d"
PROMPT = "548c2f8b-ff4c-4833-9104-de5f670ec4c7"
FIXTURE = Path(__file__).parents[1] / "fixtures" / "telemetry" / "claude_cli_session.otlp.json"
JSON = {"content-type": "application/json"}
PROTO = {"content-type": "application/x-protobuf"}


@pytest.fixture(autouse=True)
def _fresh_writer(monkeypatch):
    writer.seal_all()
    monkeypatch.delenv("OSPREY_SPOOL_DIR", raising=False)
    yield
    writer.seal_all()


@pytest.fixture
def backend(tmp_path):
    with EmbeddedBackend(tmp_path / "record.duckdb") as b:
        yield b


@pytest.fixture
def root(tmp_path) -> Path:
    path = tmp_path / "spool"
    path.mkdir()
    return path


def _client(root, *, login: str | None = LOGIN, **kwargs) -> TestClient:
    kwargs.setdefault("now", lambda: NOW)
    receiver = OtlpReceiver(root, CONFIG, **kwargs)
    return TestClient(otlp_app(receiver, login_of=lambda request: login))


def _consumer(backend, root) -> SpoolConsumer:
    return SpoolConsumer(backend, root, config=CONFIG, now=lambda: NOW)


def _entries(backend) -> list[c.Entry]:
    return backend.select(c.ScopeFilter(), c.Query())


def _by_key(backend) -> dict[str, c.Entry]:
    return {e.key: e for e in _entries(backend)}


def _segments(root: Path, login: str = LOGIN) -> list[Path]:
    directory = root / c.OTLP_SPOOL_DIR / login
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if c.parse_segment_name(p.name))


def _lines(root: Path, login: str = LOGIN) -> list[c.SpoolLine]:
    out = []
    for segment in _segments(root, login):
        out += [c.parse_spool_line(raw) for raw in segment.read_bytes().splitlines()]
    return out


# --- payload builders (OTLP/JSON shape) ------------------------------------------------


def _value(value) -> dict:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    return {"stringValue": value}


def _attrs(mapping: dict) -> list[dict]:
    return [{"key": k, "value": _value(v)} for k, v in mapping.items()]


def _nanos(when: datetime) -> str:
    return str(int(when.timestamp() * 1_000_000) * 1000)


def _record(event: str, attributes: dict, *, when: datetime | None = None, **fields) -> dict:
    when = when or NOW - timedelta(minutes=5)
    attributes = {"event.name": event, **attributes}
    if "session.id" not in attributes:
        attributes["session.id"] = SESSION
    attributes = {k: v for k, v in attributes.items() if v is not None}
    record = {
        "timeUnixNano": _nanos(when),
        "observedTimeUnixNano": _nanos(when),
        "body": {"stringValue": f"claude_code.{event}"},
        "attributes": _attrs(attributes),
        "traceId": "a509b693621c08253705b9eee5279ad1",
        "spanId": "139c87f1ae8d2c9b",
    }
    record.update(fields)
    return record


def _decision(
    tool_use_id: str = "toolu_1", seq: int = 1, when: datetime | None = None, **attrs
) -> dict:
    return _record(
        "tool_decision",
        {
            "event.sequence": seq,
            "prompt.id": PROMPT,
            "decision": "accept",
            "source": "config",
            "tool_name": "Bash",
            "tool_use_id": tool_use_id,
            "tool_source": "builtin",
            "tool_parameters": "{}",
            **attrs,
        },
        when=when,
    )


def _usage(cost: float = 0.0034, seq: int = 2, when: datetime | None = None, **attrs) -> dict:
    when = when or NOW - timedelta(minutes=4)
    return _record(
        "api_request",
        {
            "event.sequence": seq,
            "event.timestamp": when.isoformat(),
            "prompt.id": PROMPT,
            "model": "claude-haiku-5-5",
            "input_tokens": 2,
            "output_tokens": 84,
            "cost_usd": cost,
            "duration_ms": 1878,
            "ttft_ms": 1657,
            "speed": "normal",
            "effort": "medium",
            **attrs,
        },
        when=when,
    )


def _error(seq: int = 3, **attrs) -> dict:
    return _record(
        "api_error",
        {
            "event.sequence": seq,
            "model": "claude-haiku-5-5",
            "error": "overloaded",
            "status_code": "529",
            "duration_ms": 12,
            "attempt": 1,
            **attrs,
        },
    )


def _request(*records: dict, resource: dict | None = None, scope: dict | None = None) -> dict:
    resource = {"terminal.session": TERMINAL, "service.name": "claude-code"} | (resource or {})
    return {
        "resourceLogs": [
            {
                "resource": {"attributes": _attrs(resource)},
                "scopeLogs": [
                    {
                        "scope": {
                            "name": "com.anthropic.claude_code.events",
                            "attributes": _attrs(scope or {}),
                        },
                        "logRecords": list(records),
                    }
                ],
            }
        ]
    }


def _post_json(client, payload: dict, **headers):
    return client.post("/v1/logs", content=json.dumps(payload).encode(), headers=JSON | headers)


def _protobuf(payload: dict) -> bytes:
    payload = json.loads(json.dumps(payload))
    for rl in payload["resourceLogs"]:
        for sl in rl["scopeLogs"]:
            for record in sl["logRecords"]:
                record.pop("traceId", None)
                record.pop("spanId", None)
    return json_format.ParseDict(payload, ExportLogsServiceRequest()).SerializeToString()


# --- the source's five cases -----------------------------------------------------------


def test_a_resumed_session_whose_sequence_restarts_at_zero_drops_no_event(backend, root):
    client = _client(root)
    first = _request(_usage(0.1, seq=0), _decision("toolu_a", seq=1))
    resumed = _request(
        _usage(0.2, seq=0, when=NOW - timedelta(minutes=1)), _decision("toolu_b", seq=1)
    )
    assert _post_json(client, first).status_code == 200
    assert _post_json(client, resumed).status_code == 200
    report = _consumer(backend, root).consume()
    assert report.appended == 4 and report.duplicates == 0
    kinds = sorted(e.kind for e in _entries(backend))
    assert kinds == [c.KIND_DECISION, c.KIND_DECISION, c.KIND_USAGE, c.KIND_USAGE]


def test_two_api_requests_in_one_prompt_are_two_usage_entries(backend, root):
    client = _client(root)
    payload = _request(
        _usage(0.0034, seq=8, when=NOW - timedelta(minutes=3)),
        _usage(0.0003, seq=12, when=NOW - timedelta(minutes=2)),
    )
    assert _post_json(client, payload).status_code == 200
    _consumer(backend, root).consume()
    usage = sorted((e for e in _entries(backend)), key=lambda e: e.ts)
    assert [e.kind for e in usage] == [c.KIND_USAGE, c.KIND_USAGE]
    assert [e.body["cost_usd"] for e in usage] == [0.0034, 0.0003]
    assert set(usage[0].body) == set(otlp.USAGE_FIELDS), (
        "usage carries only what the transcript lacks"
    )
    assert usage[0].session_id == SESSION and usage[0].identity == LOGIN


def test_the_same_batch_posted_twice_is_one_entry_per_event(backend, root):
    client = _client(root)
    payload = _request(_decision("toolu_a"), _usage(), _error())
    assert _post_json(client, payload).status_code == 200
    assert _post_json(client, payload).status_code == 200
    report = _consumer(backend, root).consume()
    assert report.appended == 3 and report.duplicates == 3
    assert sorted(e.kind for e in _entries(backend)) == [
        c.KIND_DECISION,
        c.KIND_ERROR,
        c.KIND_USAGE,
    ]


class _Flaky:
    """A backend that fails every call while ``down``."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.down = True

    def __getattr__(self, name):
        attr = getattr(self.inner, name)
        if not callable(attr):
            return attr

        def call(*args, **kwargs):
            if self.down:
                raise c.RecordBackendUnavailable("backend is down")
            return attr(*args, **kwargs)

        return call


def test_a_backend_outage_keeps_the_batch_and_commits_it_later(backend, root):
    client = _client(root)
    response = _post_json(client, _request(_decision("toolu_a"), _usage()))
    assert response.status_code == 200, "the receiver answers without the backend"
    flaky = _Flaky(backend)
    consumer = SpoolConsumer(flaky, root, config=CONFIG, now=lambda: NOW)
    report = consumer.consume()
    assert report.appended == 0 and report.backend_error
    assert len(_segments(root)) == 1, "the batch is kept"
    flaky.down = False
    assert consumer.consume().appended == 2
    assert _segments(root) == []
    assert len(_entries(backend)) == 2


def test_a_spool_write_failure_answers_503(tmp_path):
    not_a_dir = tmp_path / "spool"
    not_a_dir.write_text("a file where the spool root should be")
    response = _post_json(_client(not_a_dir), _request(_usage()))
    assert response.status_code == 503


def test_a_spool_write_failure_reported_by_the_writer_answers_503(root, monkeypatch):
    monkeypatch.setattr(writer, "write_received", lambda *a, **k: None)
    response = _post_json(_client(root), _request(_usage()))
    assert response.status_code == 503


@pytest.mark.parametrize("path", ["/v1/traces", "/v1/metrics"])
def test_traces_and_metrics_answer_404(root, path):
    response = _client(root).post(path, content=b"{}", headers=JSON)
    assert response.status_code == 404
    assert _segments(root) == []


# --- the session rule on both keys -----------------------------------------------------


@pytest.mark.parametrize("bad", ["has space", "x" * 129, "a:b", "", 7])
def test_a_non_token_session_id_is_discarded_and_the_event_is_dropped(root, bad):
    record = _usage()
    record["attributes"] = [a for a in record["attributes"] if a["key"] != "session.id"]
    record["attributes"].append({"key": "session.id", "value": _value(bad)})
    response = _post_json(_client(root), _request(record))
    assert response.status_code == 200
    assert _lines(root) == [], "an event without a valid session cannot be keyed"


def test_a_non_token_log_session_id_falls_back_to_a_valid_resource_one(root):
    record = _usage(**{"session.id": "not a token"})
    _post_json(_client(root), _request(record, resource={"session.id": "res-session"}))
    (line,) = _lines(root)
    assert line.session_id == "res-session"
    assert line.key.split(":")[2] == "res-session"


@pytest.mark.parametrize("bad", ["has space", "x" * 129, "a/b", 3])
def test_a_non_token_terminal_session_is_discarded_at_both_levels(root, bad):
    record = _usage(**{"terminal.session": bad})
    _post_json(_client(root), _request(record, resource={"terminal.session": bad}))
    (line,) = _lines(root)
    assert line.terminal_session is None
    assert line.key.endswith(":" + _content_id(_usage())), "the bad value is not hashed"


def test_a_valid_terminal_session_is_kept_from_the_log_or_the_resource(root):
    client = _client(root)
    _post_json(client, _request(_usage(**{"terminal.session": "log-term"})))
    _post_json(client, _request(_decision("toolu_r")))
    terminals = sorted(line.terminal_session for line in _lines(root))
    assert terminals == sorted(["log-term", TERMINAL])


# --- sender attributes, clamp, mapping -------------------------------------------------


def test_sender_set_osprey_attributes_are_discarded_at_every_level(root):
    forged = _usage(**{"osprey.user": "bob", "osprey.on_behalf_of": "carol"})
    _post_json(
        _client(root),
        _request(forged, resource={"osprey.user": "bob"}, scope={"osprey.user": "bob"}),
    )
    (line,) = _lines(root)
    assert line.key == c.otel_key(LOGIN, SESSION, "api_request", attributes=_plain(_usage()))
    assert line.on_behalf_of is None
    assert "bob" not in json.dumps(line.body)


def test_events_outside_the_retention_window_are_clamped(root):
    client = _client(root, retention_days=14)
    records = [
        _usage(0.1, when=NOW - timedelta(days=14, minutes=1)),
        _usage(0.2, when=NOW + timedelta(hours=1, minutes=1)),
        _usage(0.3, when=NOW + timedelta(minutes=50)),
        _usage(0.4, when=NOW - timedelta(days=13)),
    ]
    assert _post_json(client, _request(*records)).status_code == 200
    assert sorted(line.body["cost_usd"] for line in _lines(root)) == [0.3, 0.4]


def test_an_event_without_a_time_uses_its_observed_time(root):
    when = NOW - timedelta(minutes=7)
    record = _usage(when=when)
    record["timeUnixNano"] = "0"
    _post_json(_client(root), _request(record))
    (line,) = _lines(root)
    assert c.parse_ts(line.ts) == when


def test_an_event_with_no_time_at_all_is_dropped(root):
    record = _usage()
    record["timeUnixNano"] = "0"
    record["observedTimeUnixNano"] = "0"
    assert _post_json(_client(root), _request(record)).status_code == 200
    assert _lines(root) == []


def test_a_decision_is_keyed_by_its_tool_use_id_and_carries_it(backend, root):
    _post_json(_client(root), _request(_decision("toolu_42")))
    _consumer(backend, root).consume()
    key = c.otel_key(LOGIN, SESSION, "tool_decision", tool_use_id="toolu_42")
    entry = _by_key(backend)[key]
    assert entry.kind == c.KIND_DECISION
    assert entry.tool_use_id == "toolu_42"
    assert entry.body == {
        "decision": "accept",
        "source": "config",
        "tool_name": "Bash",
        "tool_source": "builtin",
    }


def test_an_error_event_becomes_an_error_entry(root):
    _post_json(_client(root), _request(_error()))
    (line,) = _lines(root)
    assert line.kind == c.KIND_ERROR
    assert line.body["error"] == "overloaded" and line.body["status_code"] == "529"


def test_content_events_are_dropped(root):
    records = [
        _record(name, {"prompt": "hello", "event.sequence": 4})
        for name in (
            "user_prompt",
            "api_request_body",
            "api_response_body",
            "tool_result",
            "assistant_response",
            "plugin_loaded",
        )
    ]
    response = _post_json(_client(root), _request(*records))
    assert response.status_code == 200
    assert _segments(root) == [], "an all-dropped batch writes nothing"


def test_the_event_name_falls_back_to_the_body(root):
    record = _usage()
    record["attributes"] = [a for a in record["attributes"] if a["key"] != "event.name"]
    _post_json(_client(root), _request(record))
    (line,) = _lines(root)
    assert line.kind == c.KIND_USAGE and ":api_request:" in line.key


def test_the_real_cli_capture_yields_one_decision_and_two_usage(backend, root):
    capture = json.loads(FIXTURE.read_text())
    logs = [r["body"] for r in capture["requests"] if r["signal"] == "logs"]
    newest = max(
        int(rec["timeUnixNano"])
        for body in logs
        for rl in body["resourceLogs"]
        for sl in rl["scopeLogs"]
        for rec in sl["logRecords"]
    )
    shift = int((NOW - timedelta(minutes=1)).timestamp() * 1e9) - newest
    client = _client(root)
    for body in logs:
        for rl in body["resourceLogs"]:
            for sl in rl["scopeLogs"]:
                for rec in sl["logRecords"]:
                    for field in ("timeUnixNano", "observedTimeUnixNano"):
                        rec[field] = str(int(rec[field]) + shift)
        assert _post_json(client, body).status_code == 200
    _consumer(backend, root).consume()
    kinds = sorted(e.kind for e in _entries(backend))
    assert kinds == [c.KIND_DECISION, c.KIND_USAGE, c.KIND_USAGE]


# --- decoding ----------------------------------------------------------------------------


def test_protobuf_and_json_bodies_yield_the_same_lines(tmp_path):
    payload = _request(_decision("toolu_p"), _usage())
    json_root, proto_root = tmp_path / "j", tmp_path / "p"
    assert _post_json(_client(json_root), payload).status_code == 200
    response = _client(proto_root).post("/v1/logs", content=_protobuf(payload), headers=PROTO)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-protobuf")
    assert [ln.key for ln in _lines(json_root)] == [ln.key for ln in _lines(proto_root)]


def test_a_json_request_gets_a_json_answer(root):
    response = _post_json(_client(root), _request(_usage()))
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {}


def test_a_gzip_body_is_accepted(root):
    body = gzip.compress(_protobuf(_request(_usage())))
    response = _client(root).post(
        "/v1/logs", content=body, headers=PROTO | {"content-encoding": "gzip"}
    )
    assert response.status_code == 200
    assert len(_lines(root)) == 1


def test_the_body_cap_is_twenty_mebibytes():
    assert otlp.MAX_BODY_BYTES == 20 * 1024 * 1024


def test_an_oversize_body_answers_413(root, monkeypatch):
    monkeypatch.setattr(otlp, "MAX_BODY_BYTES", 64)
    response = _post_json(_client(root), _request(_usage()))
    assert response.status_code == 413
    assert _segments(root) == []


def test_a_gzip_body_inflating_past_the_cap_answers_413(root, monkeypatch):
    monkeypatch.setattr(otlp, "MAX_BODY_BYTES", 4096)
    bomb = gzip.compress(b"\0" * 100_000)
    assert len(bomb) < 4096
    response = _client(root).post(
        "/v1/logs", content=bomb, headers=PROTO | {"content-encoding": "gzip"}
    )
    assert response.status_code == 413


@pytest.mark.parametrize(
    ("headers", "body", "status"),
    [
        ({"content-type": "text/plain"}, b"{}", 415),
        (JSON | {"content-encoding": "br"}, b"{}", 415),
        (JSON, b"not json", 400),
        (JSON, b"[]", 400),
        (PROTO, b"\xff\xff\xff", 400),
        (PROTO | {"content-encoding": "gzip"}, b"not gzip", 400),
    ],
)
def test_bad_requests_are_refused_before_anything_is_written(root, headers, body, status):
    response = _client(root).post("/v1/logs", content=body, headers=headers)
    assert response.status_code == status
    assert _segments(root) == []


def test_an_unauthenticated_request_answers_401(root):
    response = _post_json(_client(root, login=None), _request(_usage()))
    assert response.status_code == 401


def test_a_reader_login_sends_nothing(root):
    response = _post_json(_client(root, login="reader.ops"), _request(_usage()))
    assert response.status_code == 403
    assert not (root / c.OTLP_SPOOL_DIR).exists()


def test_retention_days_must_be_a_positive_int(root):
    with pytest.raises(ValueError):
        OtlpReceiver(root, CONFIG, retention_days=0)


# --- helpers that recompute a key from a builder record ---------------------------------


def _plain(record: dict) -> dict:
    """The record's attributes as the receiver sees them after sanitising."""
    out = {}
    for a in record["attributes"]:
        value = json_format.ParseDict(a["value"], AnyValue())
        out[a["key"]] = getattr(value, value.WhichOneof("value"))
    return out


def _content_id(record: dict) -> str:
    return c.otel_content_id(_plain(record))
