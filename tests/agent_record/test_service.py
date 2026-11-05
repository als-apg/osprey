"""The agent-record service over httpx's ASGI transport.

Pins HTTP Basic against the verifier (refused without a credential, 503 while
the verifier is missing, ``/health`` open), the viewer header honoured only for
the ``history`` login, every D18 credential reading exactly its scope through
the read API, an OTLP batch committed by the spool consumer and then readable,
spool lines written while the service was down ingested after a restart, and a
backend outage that answers reads ``unavailable`` while ``/health/sources``
shows the kept backlog. Also the lifespan's jobs: run at start with the
archive root and retention window, ``jobs_ready`` on ``/health``, a running
job blocking neither start-up nor reads, shutdown waiting for it before the
backend closes, a failed export kept as ``last_error``, and the jobs clock
override; and the ``osprey agent-record serve`` registration. The jobs'
schedule itself is pinned in ``test_scheduler.py``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from click.testing import CliRunner

from osprey.agent_record import archive, retention, service, writer
from osprey.agent_record import contract as c
from osprey.agent_record import credentials as creds
from osprey.agent_record import identities as ids
from osprey.agent_record.backends.embedded import EmbeddedBackend
from tests.agent_record.test_identities import _config
from tests.agent_record.test_otlp_receiver import _decision, _request, _usage
from tests.agent_record.test_spool_consumer import _line, _sealed

SECRET = "s3cr3t-token-value-0123456789"
NOW = datetime.now(UTC).replace(microsecond=0)
SESSION = "bb1af18d-a6d1-4ce8-a1f6-b088b297e9b8"  # the OTLP builders' session


@pytest.fixture(autouse=True)
def _fresh_writer(monkeypatch):
    writer.seal_all()
    monkeypatch.delenv("OSPREY_SPOOL_DIR", raising=False)
    yield
    writer.seal_all()


@pytest.fixture
def config(tmp_path) -> dict:
    cfg = _config(
        workers=1,
        bridges=("teams_bridge",),
        readers=[{"name": "auditor", "kinds": [c.KIND_AUDIT]}],
    )
    cfg["project_root"] = str(tmp_path)
    return cfg


def _token(login: str) -> str:
    return f"token-of-{login}-0123456789"


def _b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


@pytest.fixture
def env(tmp_path, config) -> dict[str, str]:
    lines = "".join(
        creds.verifier_line(login, _token(login)) + "\n" for login in ids.credential_table(config)
    )
    return {
        creds.VERIFIER_ENV: _b64(lines),
        creds.SECRETS_ENV: _b64(json.dumps([SECRET])),
        service.SPOOL_ROOT_ENV: str(tmp_path / "spool"),
        service.RECORD_DIR_ENV: str(tmp_path / "record"),
        service.TRANSCRIPTS_DIR_ENV: str(tmp_path / "transcripts"),
    }


@pytest.fixture
def backend(tmp_path):
    with EmbeddedBackend(tmp_path / "injected.duckdb") as b:
        yield b


def _auth(login: str, token: str | None = None) -> dict[str, str]:
    raw = f"{login}:{token if token is not None else _token(login)}"
    return {"Authorization": "Basic " + _b64(raw)}


@asynccontextmanager
async def _serving(app) -> AsyncIterator[httpx.AsyncClient]:
    """The app with its lifespan running (ASGITransport does not run it)."""
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://record") as client:
            yield client


async def _eventually(probe, *, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while True:
        value = await probe()
        if value:
            return value
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.05)


def _entry(identity: str, name: str, kind: str = c.KIND_AUDIT, **fields) -> c.Entry:
    return c.Entry(
        key=c.spool_key(identity, name),
        kind=kind,
        ts=NOW - timedelta(minutes=5),
        identity=identity,
        body={"what": name},
        inserted_at=NOW - timedelta(minutes=5),
        **fields,
    )


SEED = [
    _entry("alice", "a1", session_id="s-alice"),
    _entry("bob", "b1", session_id="s-bob"),
    _entry("dispatch-worker-1", "w1", session_id="s-worker"),
    _entry("teams-bridge", "t1"),
    _entry("alice", "f1", kind=c.KIND_FEEDBACK),
]
AUDITS = {"a1", "b1", "w1", "t1"}


def _names(answer: dict) -> set[str]:
    assert answer["status"] == "ok", answer
    return {item["body"]["what"] for item in answer["data"]}


# --- authentication -----------------------------------------------------------------


async def test_health_is_open_and_an_unauthenticated_read_is_refused(config, env, backend):
    async with _serving(service.create_app(config=config, backend=backend, env=env)) as client:
        health = (await client.get("/health")).json()
        assert health["status"] == "ok" and set(health) == {"status", "jobs_ready"}
        missing = await client.get("/v1/read/recent_audit")
        assert missing.status_code == 401
        assert missing.headers["www-authenticate"].startswith("Basic")
        wrong = await client.get("/v1/read/recent_audit", headers=_auth("alice", "not-it"))
        assert wrong.status_code == 401
        unknown = await client.get("/v1/read/recent_audit", headers=_auth("mallory", "x"))
        assert unknown.status_code == 401
        assert (await client.get("/health/sources")).status_code == 401
        otlp = await client.post("/v1/logs", content=b"{}")
        assert otlp.status_code == 401


async def test_a_missing_verifier_answers_503_everywhere_but_health(config, env, backend):
    del env[creds.VERIFIER_ENV]
    app = service.create_app(config=config, backend=backend, env=env)
    async with _serving(app) as client:
        health = await client.get("/health")
        assert health.status_code == 503 and health.json()["status"] == "unconfigured"
        for response in (
            await client.get("/v1/read/recent_audit", headers=_auth("operator")),
            await client.get("/health/sources", headers=_auth("operator")),
            await client.post("/v1/logs", content=b"{}", headers=_auth("alice")),
        ):
            assert response.status_code == 503
        assert app.state.runtime is None, "nothing is ingested while unconfigured"


async def test_a_verifier_login_with_no_grant_is_refused(config, env, backend):
    env[creds.VERIFIER_ENV] = _b64(creds.verifier_line("stranger", _token("stranger")) + "\n")
    async with _serving(service.create_app(config=config, backend=backend, env=env)) as client:
        response = await client.get("/v1/read/recent_audit", headers=_auth("stranger"))
        assert response.status_code == 403


@pytest.mark.parametrize("login", ["alice", "dispatch-worker-1", "operator", "reader.auditor"])
async def test_a_non_history_credential_naming_a_viewer_is_refused(config, env, backend, login):
    async with _serving(service.create_app(config=config, backend=backend, env=env)) as client:
        headers = _auth(login) | {ids.VIEWER_HEADER: "all"}
        response = await client.get("/v1/read/recent_audit", headers=headers)
        assert response.status_code == 403


@pytest.mark.parametrize("viewer", [None, "nobody-form", "reader:ghost", "own:"])
async def test_history_without_a_valid_viewer_reads_nothing(config, env, backend, viewer):
    async with _serving(service.create_app(config=config, backend=backend, env=env)) as client:
        headers = _auth("history") | ({ids.VIEWER_HEADER: viewer} if viewer is not None else {})
        response = await client.get("/v1/read/recent_audit", headers=headers)
        assert response.status_code == 403


# --- scope --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("login", "viewer", "audits", "feedback"),
    [
        ("alice", None, {"a1"}, {"f1"}),
        ("bob", None, {"b1"}, set()),
        ("builder", None, set(), set()),
        ("dispatch-worker-1", None, {"w1"}, set()),
        ("teams-bridge", None, {"t1"}, set()),
        ("sidecar", None, set(), set()),
        ("operator", None, AUDITS, {"f1"}),
        ("reader.auditor", None, AUDITS, set()),
        ("history", "own:bob", {"b1"}, set()),
        ("history", "own:alice", {"a1"}, {"f1"}),
        ("history", "all", AUDITS, {"f1"}),
        ("history", "reader:auditor", AUDITS, set()),
    ],
)
async def test_each_credential_reads_exactly_its_scope(
    config, env, backend, login, viewer, audits, feedback
):
    backend.append(SEED)
    headers = _auth(login) | ({ids.VIEWER_HEADER: viewer} if viewer else {})
    async with _serving(service.create_app(config=config, backend=backend, env=env)) as client:
        audit = await client.get("/v1/read/recent_audit", headers=headers)
        assert audit.status_code == 200
        assert _names(audit.json()) == audits
        listed = await client.get(
            "/v1/read/entries", params={"kind": c.KIND_FEEDBACK}, headers=headers
        )
        assert _names(listed.json()) == feedback


def test_every_d18_login_is_pinned_by_the_scope_cases(config):
    """A login added to the credential table without a scope case here is a gap."""
    pinned = {"alice", "bob", "builder", "dispatch-worker-1", "teams-bridge", "sidecar"}
    pinned |= {"operator", "reader.auditor", "history"}
    assert set(ids.credential_table(config)) == pinned


async def test_a_source_narrows_but_nothing_else_does(config, env, backend):
    backend.append(SEED)
    async with _serving(service.create_app(config=config, backend=backend, env=env)) as client:
        worker = _auth("dispatch-worker-1")
        own = await client.get(
            "/v1/read/recent_audit", params={"narrow_session": "s-worker"}, headers=worker
        )
        assert _names(own.json()) == {"w1"}
        other = await client.get(
            "/v1/read/recent_audit", params={"narrow_session": "s-alice"}, headers=worker
        )
        assert _names(other.json()) == set()
        widen = await client.get(
            "/v1/read/recent_audit", params={"narrow_session": "s-alice"}, headers=_auth("alice")
        )
        assert widen.status_code == 400


async def test_bad_read_arguments_answer_400_and_unknown_methods_404(config, env, backend):
    async with _serving(service.create_app(config=config, backend=backend, env=env)) as client:
        op = _auth("operator")
        for path, params in [
            ("/v1/read/timeline", {}),
            ("/v1/read/recent_audit", {"limit": "-1"}),
            ("/v1/read/recent_audit", {"since": "yesterday"}),
            ("/v1/read/recent_audit", {"wait": "{not json"}),
            ("/v1/read/recent_audit", {"wait": json.dumps({"nope": 1})}),
            ("/v1/read/recent_audit", {"bogus": "1"}),
            ("/v1/read/entries", {"kind": "no-such-kind"}),
            ("/v1/read/timeline", {"session_id": "s-alice", "after": "not-a-cursor"}),
            ("/v1/read/timeline", {"session_id": "s-alice", "after": "e30"}),
            ("/v1/read/timeline", {"session_id": "s-alice", "since": "2026-10-23T12:00:00"}),
            (
                "/v1/read/agent_timeline",
                {"session_id": "s-alice", "agent_id": "ag1", "after": "%%%"},
            ),
        ]:
            response = await client.get(path, params=params, headers=op)
            assert response.status_code == 400, (path, params, response.text)
        assert (await client.get("/v1/read/drop_tables", headers=op)).status_code == 404


async def test_every_reader_method_has_a_route(config, env, backend):
    backend.append(SEED)
    async with _serving(service.create_app(config=config, backend=backend, env=env)) as client:
        op = _auth("operator")
        args = {
            "session_id": "s-alice",
            "agent_id": "agent-1",
            "key": SEED[0].key,
            "run_id": "run-1",
            "kind": c.KIND_AUDIT,
        }
        for name, method in service.READ_METHODS.items():
            params = {k: args[k] for k in method.required}
            response = await client.get(f"/v1/read/{name}", params=params, headers=op)
            assert response.status_code == 200, (name, response.text)
            assert response.json()["status"] == "ok"


async def test_a_timeline_pages_over_http_through_its_next_cursor(config, env, backend):
    entries = [_entry("alice", f"p{n}", session_id="s-paged") for n in range(5)]
    backend.append(entries)
    async with _serving(service.create_app(config=config, backend=backend, env=env)) as client:
        seen: list[str] = []
        params = {"session_id": "s-paged", "limit": "2"}
        for _ in range(10):
            response = await client.get("/v1/read/timeline", params=params, headers=_auth("alice"))
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["newer"] is False and body["more"] == (body["next"] is not None)
            seen.extend(item["key"] for item in body["data"])
            if body["next"] is None:
                break
            params = {"session_id": "s-paged", "limit": "2", "after": body["next"]}
        assert seen == sorted(e.key for e in entries)


# --- OTLP ---------------------------------------------------------------------------


async def test_an_otlp_post_is_committed_and_readable(config, env):
    app = service.create_app(config=config, env=env)
    async with _serving(app) as client:
        payload = _request(_decision("toolu_svc"), _usage())
        response = await client.post(
            "/v1/logs",
            content=json.dumps(payload).encode(),
            headers=_auth("alice") | {"content-type": "application/json"},
        )
        assert response.status_code == 200

        async def visible():
            answer = await client.get(
                "/v1/read/timeline", params={"session_id": SESSION}, headers=_auth("alice")
            )
            body = answer.json()
            return body if body["status"] == "ok" and body["data"] else None

        timeline = await _eventually(visible)
        assert "toolu_svc" in json.dumps(timeline)
        hidden = await client.get(
            "/v1/read/timeline", params={"session_id": SESSION}, headers=_auth("bob")
        )
        assert hidden.json()["data"] == []


async def test_traces_and_metrics_answer_404(config, env, backend):
    async with _serving(service.create_app(config=config, backend=backend, env=env)) as client:
        for path in ("/v1/traces", "/v1/metrics"):
            response = await client.post(path, content=b"{}", headers=_auth("alice"))
            assert response.status_code == 404


# --- restart and outage -------------------------------------------------------------


async def test_spool_lines_written_while_down_are_ingested_after_a_restart(config, env, tmp_path):
    spool = tmp_path / "spool"
    first = service.create_app(config=config, env=env)
    async with _serving(first):
        pass
    line = _line(session_id="s-later", body={"what": "while-down"})
    _sealed(spool / "alice", line)
    second = service.create_app(config=config, env=env)
    async with _serving(second) as client:

        async def ingested():
            answer = await client.get("/v1/read/recent_audit", headers=_auth("alice"))
            return answer.json()["status"] == "ok" and _names(answer.json()) == {"while-down"}

        await _eventually(ingested)
    assert not list((spool / "alice").iterdir()), "the committed segment is deleted"


class _Down:
    """A backend that is unreachable for every call."""

    def _fail(self, *_args, **_kwargs):
        raise c.RecordBackendUnavailable("backend is down")

    append = select = delete_arrived_before = export_parquet = _fail
    get_state = put_state = _fail

    def health(self) -> c.BackendHealth:
        return c.BackendHealth(ok=False, backend="down", detail="backend is down")


async def test_an_outage_answers_unavailable_and_health_shows_the_backlog(config, env, tmp_path):
    spool = tmp_path / "spool"
    segment = _sealed(spool / "alice", _line(), _line())
    app = service.create_app(config=config, backend=_Down(), env=env)
    async with _serving(app) as client:
        read = await client.get("/v1/read/recent_audit", headers=_auth("alice"))
        assert read.status_code == 503
        assert read.json()["status"] == "unavailable"

        async def backlog():
            body = (await client.get("/health/sources", headers=_auth("operator"))).json()
            return body if body["spool"]["backend_error"] else None

        body = await _eventually(backlog)
        assert body["backend"]["ok"] is False
        source = body["spool"]["sources"][str(spool / "alice")]
        assert source["identity"] == "alice"
        assert source["backlog_lines"] == 2
        assert source["last_seen"] is not None
        assert body["spool"]["backlog_lines"] == 2
        # Every root but the host's, which the host session forwards through the spool.
        assert set(body["transcripts"]) == set(ids.transcript_roots(config)) - {
            ids.HOST_TRANSCRIPT_ROOT
        }
    assert segment.exists(), "the spool is kept through the outage"


async def test_an_injected_backend_is_left_open_and_an_opened_one_is_closed(config, env, tmp_path):
    with EmbeddedBackend(tmp_path / "mine.duckdb") as mine:
        async with _serving(service.create_app(config=config, backend=mine, env=env)):
            pass
        assert mine.health().ok, "the caller's backend is still open"
    async with _serving(service.create_app(config=config, env=env)):
        pass
    path = tmp_path / "record" / service.EMBEDDED_FILENAME
    with EmbeddedBackend(path) as reopened:
        assert reopened.health().ok, "the lifespan closed the store it opened"


def test_the_clickhouse_backend_needs_its_endpoint(tmp_path):
    record = service.parse_agent_record_block({"agent_record": {"backend": "clickhouse"}})
    with pytest.raises(ValueError, match=service.CLICKHOUSE_URL_ENV):
        service.open_backend(record, tmp_path, {})


# --- jobs ---------------------------------------------------------------------------


class _Jobs:
    """Fakes for both jobs; the export can be held until released."""

    def __init__(self, monkeypatch, *, hold: bool = False, fail: bool = False):
        self.exports: list[tuple] = []
        self.retentions: list[tuple] = []
        self.started = threading.Event()
        self.release = threading.Event()
        if not hold:
            self.release.set()
        self._fail = fail
        monkeypatch.setattr(archive, "run_export", self.export)
        monkeypatch.setattr(retention, "run_retention", self.retention)

    def export(self, record, now, archive_root):
        self.started.set()
        assert self.release.wait(10)
        self.exports.append((record, now, archive_root))
        if self._fail:
            raise RuntimeError("archive unreachable")
        return []

    def retention(self, record, now, retention_days, archive_root):
        self.retentions.append((record, now, retention_days, archive_root))
        return 0


async def _jobs_ready(client) -> bool:
    return (await client.get("/health")).json().get("jobs_ready") is True


async def test_the_jobs_run_at_start_with_the_archive_root_and_retention_days(
    config, env, backend, tmp_path, monkeypatch
):
    config.setdefault("agent_record", {})["retention_days"] = 5
    env[service.ARCHIVE_DIR_ENV] = str(tmp_path / "archive")
    jobs = _Jobs(monkeypatch)
    async with _serving(service.create_app(config=config, backend=backend, env=env)) as client:
        await _eventually(lambda: _jobs_ready(client))
        assert (await client.get("/health")).json() == {"status": "ok", "jobs_ready": True}
    [(record, now, root)] = jobs.exports
    assert record is backend and root == tmp_path / "archive"
    assert now.tzinfo is not None and abs(now - datetime.now(UTC)) < timedelta(minutes=5)
    assert jobs.retentions == [(backend, now, 5, tmp_path / "archive")]


async def test_a_running_job_blocks_neither_start_up_nor_reads(config, env, backend, monkeypatch):
    backend.append(SEED)
    jobs = _Jobs(monkeypatch, hold=True)
    try:
        app = service.create_app(config=config, backend=backend, env=env)
        async with _serving(app) as client:
            assert await asyncio.to_thread(jobs.started.wait, 10)
            assert (await client.get("/health")).json() == {"status": "ok", "jobs_ready": False}
            answer = await client.get("/v1/read/recent_audit", headers=_auth("operator"))
            assert answer.status_code == 200
            assert _names(answer.json()) == AUDITS
            jobs.release.set()
            await _eventually(lambda: _jobs_ready(client))
    finally:
        jobs.release.set()


async def test_shutdown_waits_for_a_running_job_before_the_backend_closes(
    config, env, tmp_path, monkeypatch
):
    events: list[str] = []
    jobs = _Jobs(monkeypatch, hold=True)

    class Recorded(EmbeddedBackend):
        def close(self):
            events.append("closed")
            super().close()

    def export(_record, _now, _archive_root):
        jobs.started.set()
        assert jobs.release.wait(10)
        events.append("export done")
        return []

    monkeypatch.setattr(archive, "run_export", export)
    monkeypatch.setattr(
        service, "open_backend", lambda record, record_dir, env: Recorded(tmp_path / "r.duckdb")
    )
    app = service.create_app(config=config, env=env)
    try:
        async with app.router.lifespan_context(app):
            assert await asyncio.to_thread(jobs.started.wait, 10)
            asyncio.get_running_loop().call_later(0.3, jobs.release.set)
    finally:
        jobs.release.set()
    assert events == ["export done", "closed"]


async def test_a_failed_export_is_kept_as_last_error_and_skips_retention(
    config, env, backend, monkeypatch
):
    jobs = _Jobs(monkeypatch, fail=True)
    app = service.create_app(config=config, backend=backend, env=env)
    async with _serving(app) as client:
        await _eventually(lambda: _jobs_ready(client))
        assert app.state.jobs.last_error == "RuntimeError: archive unreachable"
    assert len(jobs.exports) == 1 and jobs.retentions == []


async def test_the_jobs_clock_sets_every_runs_now_and_is_warned_about(
    config, env, backend, monkeypatch, caplog
):
    env[service.JOBS_CLOCK_ENV] = "2030-01-02"
    jobs = _Jobs(monkeypatch)
    app = service.create_app(config=config, backend=backend, env=env)
    with caplog.at_level("WARNING", logger=service.__name__):
        async with _serving(app) as client:
            await _eventually(lambda: _jobs_ready(client))
    fixed = datetime(2030, 1, 2, tzinfo=UTC)
    assert [now for _r, now, _root in jobs.exports] == [fixed]
    assert app.state.jobs_clock() == fixed
    assert any(service.JOBS_CLOCK_ENV in record.getMessage() for record in caplog.records)


def test_a_jobs_clock_that_is_not_an_iso_date_is_refused(config, env):
    env[service.JOBS_CLOCK_ENV] = "tomorrow"
    with pytest.raises(ValueError, match=service.JOBS_CLOCK_ENV):
        service.create_app(config=config, env=env)


def test_without_a_jobs_clock_the_jobs_read_the_wall_clock(config, env):
    app = service.create_app(config=config, env=env)
    assert abs(app.state.jobs_clock() - datetime.now(UTC)) < timedelta(minutes=1)


# --- CLI ----------------------------------------------------------------------------


def test_agent_record_serve_is_registered_and_lazy():
    from osprey.cli.main import LazyGroup, cli

    group = LazyGroup(name="test")
    assert "agent-record" in group.list_commands(None)
    command = group.get_command(None, "agent-record")
    assert command is not None and "serve" in command.commands
    result = CliRunner().invoke(cli, ["agent-record", "serve", "--help"])
    assert result.exit_code == 0
    assert "--host" in result.output and "--port" in result.output


# --- list_sessions ``before``: the session list's pages over the wire ---------------------

_PAGED_OLD = "33333333-cccc-4ccc-8ccc-333333333333"
_PAGED_NEW = "22222222-bbbb-4bbb-8bbb-222222222222"


def _turn(session: str, n: int, kind: str, at: datetime, text: str) -> c.Entry:
    return c.Entry(
        key=f"cc:operator:{session}:u{n}:0",
        kind=kind,
        ts=at,
        identity="operator",
        body={"text": text},
        inserted_at=at,
        session_id=session,
    )


def _paged_seed(base: datetime) -> list[c.Entry]:
    rows = [
        _turn(_PAGED_OLD, 1, c.KIND_PROMPT, base, "the old one"),
        _turn(_PAGED_OLD, 2, c.KIND_ANSWER, base + timedelta(seconds=1), "ok"),
        _turn(_PAGED_NEW, 0, c.KIND_PROMPT, base + timedelta(minutes=1), "newer"),
    ]
    rows += [
        _turn(_PAGED_NEW, n, c.KIND_ANSWER, base + timedelta(minutes=1, seconds=n), f"a{n}")
        for n in range(1, 30)
    ]
    return rows


@pytest.fixture
def served_record(config, env, backend, monkeypatch):
    """The service on a real port over the embedded backend, scanning 10 rows a read."""
    from functools import partial

    from osprey.agent_record import reader as rd
    from osprey.interfaces._serving import run_app_server

    monkeypatch.setattr(
        service, "RecordReader", partial(rd.RecordReader, limits=rd.Limits(scan_rows=10))
    )
    with run_app_server(service.create_app(config=config, backend=backend, env=env)) as base:
        yield base


def test_before_round_trips_and_reaches_sessions_past_the_newest_scan(served_record, backend):
    from osprey.agent_record.client import RecordClient

    base = NOW - timedelta(hours=2)
    backend.append(_paged_seed(base))
    client = RecordClient(served_record, "operator", _token("operator"), timeout=10.0)

    newest = client.list_sessions()
    assert newest.available and not newest.complete
    assert [r["session_id"] for r in newest.data] == [_PAGED_NEW]
    assert newest.data[0]["truncated"] is True

    older = client.list_sessions(before=base + timedelta(minutes=1))
    assert older.available and older.complete, older.reason
    assert [r["session_id"] for r in older.data] == [_PAGED_OLD]
    assert older.data[0]["prompt"] == "the old one"
    assert older.data[0]["message_count"] == 2


def test_an_older_service_refuses_before_and_the_client_says_so(served_record, monkeypatch):
    from osprey.agent_record.client import RecordClient

    monkeypatch.setitem(
        service.READ_METHODS, "list_sessions", service._Method(optional=("since", "limit"))
    )
    client = RecordClient(served_record, "operator", _token("operator"), timeout=10.0)
    answer = client.list_sessions(before=NOW)
    assert not answer.available and answer.refused
    assert "unknown parameter(s): before" in (answer.unavailable or "")
    # A service that is simply down is not a refusal.
    down = RecordClient("http://127.0.0.1:9", "operator", _token("operator"), timeout=1.0)
    assert not down.list_sessions().refused


async def test_list_sessions_takes_before_and_refuses_a_bad_one(config, env, backend):
    backend.append(_paged_seed(NOW - timedelta(hours=2)))
    async with _serving(service.create_app(config=config, backend=backend, env=env)) as client:
        op = _auth("operator")
        ok = await client.get(
            "/v1/read/list_sessions",
            params={"before": (NOW - timedelta(hours=1)).isoformat()},
            headers=op,
        )
        assert ok.status_code == 200, ok.text
        assert [r["session_id"] for r in ok.json()["data"]] == [_PAGED_NEW, _PAGED_OLD]
        bad = await client.get("/v1/read/list_sessions", params={"before": "soon"}, headers=op)
        assert bad.status_code == 400 and "before" in bad.json()["reason"]


def test_known_sessions_round_trips_and_asks_again_past_a_cut(served_record, backend):
    from osprey.agent_record.client import RecordClient
    from osprey.mcp_server import session as session_record

    backend.append(_paged_seed(NOW - timedelta(hours=2)))
    client = RecordClient(served_record, "operator", _token("operator"), timeout=10.0)
    unknown = "44444444-dddd-4ddd-8ddd-444444444444"

    narrowed = client.list_sessions(session_ids=[_PAGED_OLD, unknown], limit=2)
    assert narrowed.available and narrowed.complete, narrowed.reason
    assert [r["session_id"] for r in narrowed.data] == [_PAGED_OLD]

    # _PAGED_NEW alone fills the 10-row scan; the second read finds _PAGED_OLD.
    known, _ = session_record.known_sessions([_PAGED_NEW, _PAGED_OLD, unknown], client=client)
    assert known == {_PAGED_NEW, _PAGED_OLD}

    down = RecordClient("http://127.0.0.1:9", "operator", _token("operator"), timeout=1.0)
    assert session_record.known_sessions([_PAGED_OLD], client=down)[0] is None
    assert session_record.known_sessions([], client=down)[0] == set()


async def test_list_sessions_refuses_an_empty_session_ids(config, env, backend):
    async with _serving(service.create_app(config=config, backend=backend, env=env)) as client:
        bad = await client.get(
            "/v1/read/list_sessions", params={"session_ids": " , "}, headers=_auth("operator")
        )
        assert bad.status_code == 400 and "session_ids" in bad.json()["reason"]
