"""The record client against a real agent-record service over an embedded backend.

Pins every read method round-tripping (the client's answer equals what the
reader answers in-process for the same scope, the drain included), the method
list and parameters matching the service's routes, a narrowed ``Source`` read
and each history viewer reading what D18 grants, a non-history login naming a
viewer refused as ``unavailable``, a stopped service, a wrong password, a slow
service and a body the client cannot read all answering ``unavailable`` within
the timeout without raising, paging a timeline to its end through ``next``
yielding each entry once, ``entries(source=...)`` and
``recent_audit(written_by="self")`` over the wire, ``from_env`` and the
:class:`~tests.agent_record.fakes.FakeRecordClient` double; and
``health_sources()`` read by any login, ``export()`` run by the host and
operator logins and refused to every other, both answering unavailable on a
stopped, unconfigured or odd service and naming why an export failed.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import time
from collections.abc import Iterator
from datetime import UTC, date, datetime

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse

from osprey.agent_record import contract as c
from osprey.agent_record import credentials as creds
from osprey.agent_record import identities as ids
from osprey.agent_record import reader as rd
from osprey.agent_record import service
from osprey.agent_record.backends.embedded import EmbeddedBackend
from osprey.agent_record.client import (
    DEFAULT_TIMEOUT,
    READ_NAMES,
    URL_ENV,
    ClientAnswer,
    RecordClient,
)
from osprey.interfaces._serving import run_app_server
from osprey_connectors_tests._harness._ports import reserved_port
from tests.agent_record.fakes import FakeRecordClient
from tests.agent_record.test_identities import _config
from tests.agent_record.test_reader import BRIDGE, QS, R1, R2, S1, S3, W1, _at, _e, _seed

AUDITOR = {"name": "auditor", "kinds": [c.KIND_AUDIT]}
EXTRA = [
    _e("alice", "sp:alice:dbg1", c.KIND_DEBUG, 60, {"source": "hook", "n": 1}),
    _e("alice", "sp:alice:dbg2", c.KIND_DEBUG, 61, {"source": "sdk", "n": 2}),
    _e("alice", "sp:alice:dbg3", c.KIND_DEBUG, 62, {"source": "hook", "n": 3}),
    _e("alice", "sp:alice:aud1", c.KIND_AUDIT, 70, {"surface": "tool"}),
]


def _token(login: str) -> str:
    return f"token-of-{login}-0123456789"


def _b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


@pytest.fixture(scope="module")
def config(tmp_path_factory) -> dict:
    cfg = _config(workers=2, bridges=("teams_bridge",), lanes=True, readers=[AUDITOR])
    cfg["project_root"] = str(tmp_path_factory.mktemp("project"))
    return cfg


@pytest.fixture(scope="module")
def backend(tmp_path_factory):
    with EmbeddedBackend(tmp_path_factory.mktemp("record") / "record.duckdb") as b:
        b.append(_seed() + EXTRA)
        yield b


@pytest.fixture(scope="module")
def url(config, backend, tmp_path_factory):
    root = tmp_path_factory.mktemp("service")
    (root / "archive").mkdir(mode=0o700)
    lines = "".join(
        creds.verifier_line(login, _token(login)) + "\n" for login in ids.credential_table(config)
    )
    env = {
        creds.VERIFIER_ENV: _b64(lines),
        creds.SECRETS_ENV: _b64(json.dumps(["s3cr3t-token-value-0123456789"])),
        service.SPOOL_ROOT_ENV: str(root / "spool"),
        service.RECORD_DIR_ENV: str(root / "record"),
        service.TRANSCRIPTS_DIR_ENV: str(root / "transcripts"),
        service.ARCHIVE_DIR_ENV: str(root / "archive"),
    }
    with run_app_server(service.create_app(config=config, backend=backend, env=env)) as base:
        yield base


@pytest.fixture(scope="module")
def reader(backend, config) -> rd.RecordReader:
    return rd.RecordReader(backend, config)


def _client(url: str, login: str, *, viewer: str | None = None, **kw) -> RecordClient:
    return RecordClient(url, login, _token(login), viewer=viewer, **kw)


def _same(answer: ClientAnswer, expected: rd.ReadAnswer) -> None:
    """*answer* carries exactly what the reader answers in-process."""
    wire = expected.to_json()
    assert answer.unavailable is None, answer.unavailable
    assert answer.data == wire["data"]
    assert (answer.complete, answer.more, answer.next, answer.newer) == (
        wire["complete"],
        wire["more"],
        wire["next"],
        wire["newer"],
    )
    assert answer.reason == wire["reason"]


def _calls():
    """Each read method, by name: the client call and the reader call it mirrors."""
    return {
        "list_sessions": (
            lambda k: k.list_sessions(limit=5),
            lambda r, s: r.list_sessions(s, limit=5),
        ),
        "session_users": (lambda k: k.session_users(), lambda r, s: r.session_users(s)),
        "timeline": (
            lambda k: k.timeline(S1, limit=3),
            lambda r, s: r.timeline(s, S1, limit=3),
        ),
        "agent_timeline": (
            lambda k: k.agent_timeline(S1, "ag1"),
            lambda r, s: r.agent_timeline(s, S1, "ag1"),
        ),
        "full_entry": (
            lambda k: k.full_entry("cc:alice:sess-alice-1:u1:0"),
            lambda r, s: r.full_entry(s, "cc:alice:sess-alice-1:u1:0"),
        ),
        "run": (lambda k: k.run(R1), lambda r, s: r.run(s, R1)),
        "run_timeline": (lambda k: k.run_timeline(R1), lambda r, s: r.run_timeline(s, R1)),
        "recent_audit": (
            lambda k: k.recent_audit(surface="preflight", since=_at(0)),
            lambda r, s: r.recent_audit(s, surface="preflight", since=_at(0)),
        ),
        "bridge_history": (
            lambda k: k.bridge_history(conversation="c1", limit=10),
            lambda r, s: r.bridge_history(s, conversation="c1", limit=10),
        ),
        "entries": (
            lambda k: k.entries(c.KIND_FEEDBACK, limit=2),
            lambda r, s: r.entries(s, c.KIND_FEEDBACK, limit=2),
        ),
    }


# --- the method surface ----------------------------------------------------------------


def test_the_client_has_one_method_per_read_route_taking_its_parameters():
    assert set(READ_NAMES) == set(service.READ_METHODS)
    assert set(_calls()) == set(READ_NAMES)
    common = {"wait", "narrow_session", "narrow_runs"}
    for name, method in service.READ_METHODS.items():
        params = set(inspect.signature(getattr(RecordClient, name)).parameters) - {"self"}
        assert params == set(method.required) | set(method.optional) | common, name
        reader_params = set(inspect.signature(getattr(rd.RecordReader, name)).parameters)
        assert params - common <= reader_params - {"self", "scope"}, name


def test_the_default_timeout_outlasts_a_drained_read():
    assert DEFAULT_TIMEOUT > rd.WAIT_SECONDS + rd.READ_SECONDS


# --- round trips -----------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(READ_NAMES))
def test_each_method_round_trips(url, reader, name):
    client_call, reader_call = _calls()[name]
    answer = client_call(_client(url, "operator"))
    _same(answer, reader_call(reader, c.All()))
    assert answer.data and answer.complete, "the seed gives every method something to answer"


def test_a_drained_read_round_trips(url, reader):
    client = _client(url, "alice")
    for wait in (c.Drain(S1), c.Drain(), c.WaitMark(drain=c.Drain(S1))):
        answer = client.timeline(S1, wait=wait)
        _same(answer, reader.timeline(c.Own("alice"), S1))
        assert answer.complete


def test_entries_source_filters_before_the_limit_over_the_wire(url):
    client = _client(url, "alice")
    newest = client.entries(c.KIND_DEBUG, limit=1, source="hook")
    assert [e["key"] for e in newest.data] == ["sp:alice:dbg3"] and newest.more
    sdk = client.entries(c.KIND_DEBUG, source="sdk")
    assert [e["key"] for e in sdk.data] == ["sp:alice:dbg2"] and not sdk.more


def test_recent_audit_written_by_self_is_the_login_own_identity(url):
    alice = _client(url, "alice")
    seen = {e["key"] for e in alice.recent_audit().data}
    assert {f"sp:{QS}:pf1", "sp:alice:aud1"} <= seen
    own = alice.recent_audit(written_by="self")
    assert [e["key"] for e in own.data] == ["sp:alice:aud1"]
    bridge = _client(url, "teams-bridge").recent_audit(written_by="self")
    assert bridge.available and bridge.data == []
    history = _client(url, "history", viewer="all").recent_audit(written_by="self")
    assert history.unavailable and "400" in history.unavailable
    wrong = alice.recent_audit(written_by="bob")  # type: ignore[arg-type]
    assert wrong.unavailable and "400" in wrong.unavailable


# --- D18 over the wire -----------------------------------------------------------------


def test_a_narrowed_source_reads_only_its_session_or_runs(url, reader):
    worker = _client(url, W1)
    narrowed = worker.timeline(S3, narrow_session=S3)
    _same(narrowed, reader.timeline(c.Source(W1, c.Narrow(session_id=S3)), S3))
    assert narrowed.data, "the worker's own session is visible"
    assert worker.timeline(S1, narrow_session=S3).data == []
    by_run = worker.run(R1, narrow_runs={R1})
    _same(by_run, reader.run(c.Source(W1, c.Narrow(run_ids=frozenset({R1}))), R1))
    assert by_run.data and by_run.data["accept"]
    both = worker.list_sessions(narrow_runs=[R1, R2])
    _same(both, reader.list_sessions(c.Source(W1, c.Narrow(run_ids=frozenset({R1, R2})))))
    assert (
        worker.run(R1, narrow_runs={"run-other"}).data
        == reader.run(c.Source(W1, c.Narrow(run_ids=frozenset({"run-other"}))), R1).to_json()[
            "data"
        ]
    )
    refused = _client(url, "alice").timeline(S1, narrow_session=S1)
    assert refused.unavailable and "400" in refused.unavailable


@pytest.mark.parametrize(
    ("viewer", "scope"),
    [
        ("all", c.All()),
        ("own:alice", c.Own("alice")),
        ("own:bob", c.Own("bob")),
        ("reader:auditor", c.Reader("auditor", frozenset({c.KIND_AUDIT}))),
    ],
)
def test_a_history_viewer_reads_what_d18_grants(url, reader, viewer, scope):
    history = _client(url, "history", viewer=viewer)
    _same(history.list_sessions(), reader.list_sessions(scope))
    _same(history.recent_audit(), reader.recent_audit(scope))
    _same(history.timeline(S1), reader.timeline(scope, S1))


def test_history_without_a_viewer_is_unavailable(url):
    answer = _client(url, "history").list_sessions()
    assert answer.unavailable and "403" in answer.unavailable
    assert not answer.complete and answer.data is None


@pytest.mark.parametrize("login", ["alice", W1, BRIDGE, "operator", "reader.auditor"])
def test_a_non_history_login_naming_a_viewer_is_unavailable(url, login):
    answer = _client(url, login, viewer="all").recent_audit()
    assert answer.unavailable and "403" in answer.unavailable
    assert answer.data is None and not answer.complete


# --- failures answer unavailable, never raise ------------------------------------------


@pytest.fixture
def closed_port() -> Iterator[int]:
    """A loopback port nothing listens on: reserved for the test, and never bound."""
    with reserved_port() as port:
        yield port


def test_a_stopped_service_is_unavailable_within_the_timeout(closed_port):
    client = RecordClient(f"http://127.0.0.1:{closed_port}", "alice", "x", timeout=2.0)
    started = time.monotonic()
    for name, (call, _) in _calls().items():
        answer = call(client)
        assert answer.unavailable and "unreachable" in answer.unavailable, name
        assert answer.data is None and not answer.complete
    assert time.monotonic() - started < 2.0 * len(READ_NAMES)


def test_a_wrong_password_is_unavailable_within_the_timeout(url):
    client = RecordClient(url, "alice", "not-the-token", timeout=5.0)
    started = time.monotonic()
    answer = client.list_sessions()
    assert time.monotonic() - started < 5.0
    assert answer.unavailable and "401" in answer.unavailable
    unknown = RecordClient(url, "mallory", "x", timeout=5.0).list_sessions()
    assert unknown.unavailable and "401" in unknown.unavailable


def test_a_malformed_url_is_unavailable():
    for url in ("", "not a url", "ftp://record"):
        answer = RecordClient(url, "alice", "x", timeout=1.0).list_sessions()
        assert answer.unavailable, url


@pytest.fixture(scope="module")
def odd_url():
    """A service answering what the record never does."""
    app = FastAPI()

    @app.get("/v1/read/{name}")
    async def read(name: str):
        if name == "list_sessions":
            return PlainTextResponse("not json")
        if name == "session_users":
            return JSONResponse(["a", "list"])
        if name == "timeline":
            return JSONResponse({"status": "ok", "complete": "yes", "more": False, "data": []})
        if name == "agent_timeline":
            return JSONResponse({"status": "ok", "complete": True, "more": False})
        if name == "full_entry":
            await asyncio.sleep(3)
            return JSONResponse({})
        if name == "run":
            return JSONResponse({"status": "unavailable", "complete": False, "reason": "down"}, 503)
        if name == "run_timeline":
            return PlainTextResponse("bad gateway", 502)
        return JSONResponse(
            {
                "status": "ok",
                "complete": False,
                "more": True,
                "next": None,
                "newer": False,
                "reason": "scan limit",
                "data": [1],
            }
        )

    with run_app_server(app) as base:
        yield base


def test_a_body_the_client_cannot_read_is_unavailable(odd_url):
    client = RecordClient(odd_url, "alice", "x", timeout=1.0)
    for answer in (
        client.list_sessions(),
        client.session_users(),
        client.timeline(S1),
        client.agent_timeline(S1, "ag1"),
    ):
        assert answer.unavailable
        assert any(w in answer.unavailable for w in ("without JSON", "no object", "cannot read"))
        assert answer.data is None and not answer.complete
    assert client.run(R1) == ClientAnswer(unavailable="down")
    gateway = client.run_timeline(R1)
    assert gateway.unavailable and "502" in gateway.unavailable


def test_a_slow_service_is_unavailable_at_the_timeout(odd_url):
    client = RecordClient(odd_url, "alice", "x", timeout=0.5)
    started = time.monotonic()
    answer = client.full_entry("k")
    assert time.monotonic() - started < 2.5
    assert answer.unavailable and "within 0.5 s" in answer.unavailable


def test_an_incomplete_answer_keeps_its_reason(odd_url):
    answer = RecordClient(odd_url, "alice", "x", timeout=1.0).entries(c.KIND_FEEDBACK)
    assert answer.available and not answer.complete and answer.more
    assert answer.reason == "scan limit" and answer.data == [1]


# --- paging ----------------------------------------------------------------------------


def _node_keys(nodes) -> list[str]:
    out = []
    for node in nodes:
        if node.get("key"):
            out.append(node["key"])
        for part in ("decision", "result"):
            if node.get(part):
                out.append(node[part]["key"])
        if node.get("subagent"):
            out.extend(_node_keys(node["subagent"]["timeline"]))
    return out


@pytest.mark.parametrize("limit", [1, 2, 3])
def test_paging_to_the_end_through_next_yields_each_entry_once(url, limit):
    client = _client(url, "alice")
    whole = client.timeline(S1)
    assert whole.available and whole.next is None
    expected = _node_keys(whole.data)
    seen: list[str] = []
    after = None
    pages = 0
    while True:
        page = client.timeline(S1, after=after, limit=limit)
        assert page.available, page.unavailable
        assert not page.newer
        seen.extend(_node_keys(page.data))
        pages += 1
        assert pages < 100
        if page.next is None:
            break
        after = page.next
    assert pages > 1
    assert len(seen) == len(set(seen)), "no entry is paged twice"
    assert seen == expected


def test_a_cursor_the_service_cannot_read_is_unavailable(url):
    answer = _client(url, "alice").timeline(S1, after="not-a-cursor")
    assert answer.unavailable and "400" in answer.unavailable


# --- from_env --------------------------------------------------------------------------


def test_from_env_reads_the_url_and_the_login_pair(url, reader):
    env = {URL_ENV: url, creds.LOGIN_ENV: f"alice:{_token('alice')}"}
    client = RecordClient.from_env(env)
    assert client.login == "alice" and client.url == url
    _same(client.entries(c.KIND_FEEDBACK), reader.entries(c.Own("alice"), c.KIND_FEEDBACK))
    history = RecordClient.from_env(
        {URL_ENV: url, creds.LOGIN_ENV: f"history:{_token('history')}"}, viewer="own:alice"
    )
    _same(history.list_sessions(), reader.list_sessions(c.Own("alice")))


@pytest.mark.parametrize(
    ("env", "missing"),
    [
        ({}, URL_ENV),
        ({URL_ENV: "http://127.0.0.1:1"}, creds.LOGIN_ENV),
        ({URL_ENV: "http://127.0.0.1:1", creds.LOGIN_ENV: "alice"}, "<login>:<token>"),
        ({URL_ENV: "http://127.0.0.1:1", creds.LOGIN_ENV: ":tok"}, "<login>:<token>"),
        ({URL_ENV: "http://127.0.0.1:1", creds.LOGIN_ENV: "alice:"}, "<login>:<token>"),
    ],
)
def test_from_env_without_a_configuration_answers_unavailable(env, missing):
    client = RecordClient.from_env(env)
    for name, (call, _) in _calls().items():
        answer = call(client)
        assert answer.unavailable and missing in answer.unavailable, name
    assert "tok" not in repr(client)


def test_the_password_is_not_in_the_repr(url):
    assert _token("alice") not in repr(_client(url, "alice"))


# --- the fake --------------------------------------------------------------------------


def test_the_fake_has_the_client_surface_and_records_each_call():
    fake = FakeRecordClient(
        {"timeline": [ClientAnswer(data=[1], complete=True, next="c1"), ClientAnswer(data=[2])]}
    )
    for name in READ_NAMES:
        assert callable(getattr(fake, name)), name
    first = fake.timeline(S1, limit=1)
    second = fake.timeline(S1, after="c1", limit=1)
    third = fake.timeline(S1, after="c2")
    assert (first.data, first.next, second.data, third.data) == ([1], "c1", [2], [2])
    assert fake.calls_to("timeline") == [
        {"session_id": S1, "limit": 1},
        {"session_id": S1, "after": "c1", "limit": 1},
        {"session_id": S1, "after": "c2"},
    ]
    assert fake.entries(c.KIND_DEBUG, source="hook") == ClientAnswer(data=[], complete=True)
    assert fake.calls[-1].name == "entries"
    assert fake.calls[-1].kwargs == {"kind": c.KIND_DEBUG, "source": "hook"}
    with pytest.raises(TypeError):
        fake.timeline(S1, since="yesterday")


def test_the_fake_answers_unavailable_and_computed_answers():
    down = FakeRecordClient(unavailable="record down")
    assert down.run(R1) == ClientAnswer(unavailable="record down")
    computed = FakeRecordClient({"run": lambda run_id, **_: ClientAnswer(data={"id": run_id})})
    assert computed.run(R1, narrow_runs={R1}).data == {"id": R1}
    assert computed.calls_to("run") == [{"run_id": R1, "narrow_runs": {R1}}]


# --- health and the export -------------------------------------------------------------


@pytest.mark.parametrize("login", ["alice", "operator", "builder", "history"])
def test_any_login_reads_the_health_report(url, login):
    answer = _client(url, login, timeout=10.0).health_sources()
    assert answer.unavailable is None, answer.unavailable
    assert answer.complete
    assert {"backend", "spool", "transcripts", "export", "jobs_clock"} <= set(answer.data)
    assert answer.data["backend"]["ok"] is True
    assert set(answer.data["export"]) >= {"last_exported_day", "lag_days", "last_error"}


@pytest.mark.parametrize("login", ["operator", "builder"])
def test_the_host_and_operator_logins_run_the_export(url, login):
    answer = _client(url, login, timeout=30.0).export()
    assert answer.unavailable is None, answer.unavailable
    assert answer.complete and isinstance(answer.data, list)
    for day in answer.data:
        assert date.fromisoformat(day) < datetime.now(UTC).date()
    report = _client(url, login, timeout=10.0).health_sources().data["export"]
    assert report["lag_days"] == 0 and report["last_error"] is None


@pytest.mark.parametrize("login", ["alice", "history", "teams-bridge"])
def test_every_other_login_is_refused_the_export(url, login):
    answer = _client(url, login, timeout=10.0).export()
    assert answer.unavailable and "403" in answer.unavailable
    assert answer.data is None and not answer.complete


def test_health_and_export_on_a_stopped_service_are_unavailable(closed_port):
    client = RecordClient(f"http://127.0.0.1:{closed_port}", "operator", "x", timeout=2.0)
    for answer in (client.health_sources(), client.export()):
        assert answer.unavailable and "unreachable" in answer.unavailable
        assert answer.data is None and not answer.complete


def test_health_and_export_with_a_wrong_password_are_unavailable(url):
    client = RecordClient(url, "operator", "not-the-token", timeout=5.0)
    for answer in (client.health_sources(), client.export()):
        assert answer.unavailable and "401" in answer.unavailable


def test_health_and_export_unconfigured_answer_unavailable():
    client = RecordClient.from_env({})
    for answer in (client.health_sources(), client.export()):
        assert answer.unavailable and "not configured" in answer.unavailable


@pytest.fixture(scope="module")
def odd_report_url():
    """A service answering health and export bodies the record never sends."""
    app = FastAPI()

    @app.get("/health/sources")
    async def sources():
        return JSONResponse(["not", "an", "object"])

    @app.post("/jobs/export")
    async def export():
        return JSONResponse({"status": "ok"})

    with run_app_server(app) as base:
        yield base


def test_a_report_the_client_cannot_read_is_unavailable(odd_report_url):
    client = RecordClient(odd_report_url, "operator", "x", timeout=2.0)
    sources = client.health_sources()
    assert sources.unavailable and "no object" in sources.unavailable
    exported = client.export()
    assert exported.unavailable and "cannot read" in exported.unavailable


def test_an_export_that_fails_names_why(monkeypatch, config, tmp_path):
    from osprey.agent_record import archive

    def broken(record, now, root):  # noqa: ARG001 - run_export signature
        raise c.RecordBackendError("the record is down")

    monkeypatch.setattr(archive, "run_export", broken)
    lines = "".join(
        creds.verifier_line(login, _token(login)) + "\n" for login in ids.credential_table(config)
    )
    env = {
        creds.VERIFIER_ENV: _b64(lines),
        creds.SECRETS_ENV: _b64(json.dumps(["s3cr3t-token-value-0123456789"])),
        service.SPOOL_ROOT_ENV: str(tmp_path / "spool"),
        service.RECORD_DIR_ENV: str(tmp_path / "record"),
        service.TRANSCRIPTS_DIR_ENV: str(tmp_path / "transcripts"),
        service.ARCHIVE_DIR_ENV: str(tmp_path / "archive"),
    }
    with run_app_server(service.create_app(config=config, env=env)) as base:
        answer = _client(base, "operator", timeout=10.0).export()
    assert answer.unavailable and "the record is down" in answer.unavailable
