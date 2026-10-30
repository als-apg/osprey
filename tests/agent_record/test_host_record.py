"""The host-local agent record: one service, launched on the host by ``ensure_host_record``.

Pinned here, with real ``osprey agent-record serve`` processes over the
embedded backend: two concurrent ensures share one process; the process is
gone when the last holder returns, the starter or a reuser that outlived it;
a port that answers 503 or refuses the host login (another project's record)
is never spawned over; a live ``stack-owner`` marker is never spawned under,
and a stale one is removed; the host root read through
``OSPREY_AGENT_RECORD_HOST_PROJECTS`` never reads a sibling project; while the
stack owns the record the host session forwards the same files under the same
state names and keys, so the stack's record files nothing twice, resumes from
its own position, skips a busy round, scrubs the build's secrets, and runs as a
holder starts and ends, every supervisor round and before a drained read of a
host launch's children; ``record.duckdb`` is
group-writable; the ClickHouse refusal and a missing service file create and
spawn nothing; ``serve`` without ``--lease-dir`` keeps running and keeps the
compose ``/health`` body.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import yaml
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from osprey.agent_record import credentials as creds
from osprey.agent_record import host as hr
from osprey.agent_record import identities as ids
from osprey.agent_record import service
from osprey.agent_record.backends.embedded import EmbeddedBackend
from osprey.agent_record.contract import Query, ScopeFilter, transcript_state_name
from osprey.agent_record.ingest.spool import SpoolConsumer
from osprey.agent_record.ingest.tailer import TranscriptTailer
from osprey.deployment.compose_generator import agent_record_host_projects_dir
from osprey.interfaces._serving import run_app_server
from osprey_connectors_tests._harness._ports import reserved_port

HOST = "builder"
SESSION = "5f0c9a3e-1111-4222-8333-944455556666"
SIBLING_SESSION = "0a0b0c0d-2222-4333-8444-955566667777"


def _line(session: str, text: str) -> bytes:
    obj = {
        "parentUuid": None,
        "isSidechain": False,
        "userType": "external",
        "cwd": "/somewhere",
        "sessionId": session,
        "version": "2.1.300",
        "type": "user",
        "message": {"role": "user", "content": text},
        "uuid": f"u-{session[:4]}-{text}",
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
    }
    return (json.dumps(obj) + "\n").encode("utf-8")


class Project:
    """A deployment repo in a temp dir: rendered config, credentials, host projects dir."""

    def __init__(self, root: Path, *, port: int, backend: str = "embedded"):
        self.root = root
        self.port = port
        self.config: dict = {
            "project_name": "hostrec",
            "project_root": str(root),
            "build_dir": "./build",
            "deployment": {"host_identity": HOST},
            "agent_record": {"port": self.port, "backend": backend},
        }
        (root / "build").mkdir(parents=True, exist_ok=True)
        (root / "build" / "config.yml").write_text(yaml.safe_dump(self.config))
        creds.ensure_agent_record_credentials(root, self.config, reader_keys=False)
        self.host_projects = agent_record_host_projects_dir(self.config, root)
        assert self.host_projects is not None
        self.state_dir = hr.host_state_dir(root, self.config)

    def ensure(self, **kwargs):
        return hr.ensure_host_record(self.root, self.config, **kwargs)

    def process(self) -> tuple[int, str] | None:
        return hr._running_process(self.state_dir)


@pytest.fixture
def claude_dir(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "claude"
    path.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(path))
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)
    return path


@pytest.fixture
def record_port() -> Iterator[int]:
    """The port the host-local record a test may start binds: the config names it
    before the process exists, so it is reserved for the whole test."""
    with reserved_port() as port:
        yield port


@pytest.fixture
def project(tmp_path, claude_dir, record_port) -> Iterator[Project]:
    proj = Project(tmp_path / "repo", port=record_port)
    assert proj.host_projects.parent == claude_dir / "projects"
    yield proj
    # Never leave a process behind, whatever a test did.
    if proj.state_dir.exists():
        hr.stop_host_process(proj.state_dir, wait_s=10)


def _health_once_the_jobs_ran(url: str, wait_s: float = 30.0) -> dict:
    """The ``/health`` body once the jobs' start-up run has finished.

    The jobs start after the first ingest pass, so ``jobs_ready`` may still be
    false the moment the record turns ready.
    """
    deadline = time.monotonic() + wait_s
    body = httpx.get(url, trust_env=False).json()
    while body.get("jobs_ready") is False and time.monotonic() < deadline:
        time.sleep(0.05)
        body = httpx.get(url, trust_env=False).json()
    return body


def _gone(pid: int, start: str, wait_s: float = 10.0) -> bool:
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if not hr._alive(pid, start):
            return True
        time.sleep(0.05)
    return False


def _sources(record: hr.HostRecord) -> dict:
    response = httpx.get(
        f"{record.url}/health/sources", auth=(record.login, record.token), trust_env=False
    )
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------------------
# Lifetime
# ---------------------------------------------------------------------------


def test_two_concurrent_ensures_share_one_process(project):
    barrier = threading.Barrier(2)
    release = threading.Event()
    seen: list[tuple[hr.HostRecord, tuple[int, str] | None]] = []
    errors: list[BaseException] = []

    def hold() -> None:
        try:
            barrier.wait()
            with project.ensure() as record:
                seen.append((record, project.process()))
                release.wait(60)
        except BaseException as exc:  # surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=hold) for _ in range(2)]
    for t in threads:
        t.start()
    deadline = time.monotonic() + 90
    while len(seen) < 2 and not errors and time.monotonic() < deadline:
        time.sleep(0.05)
    release.set()
    for t in threads:
        t.join(60)
    assert not errors, errors
    assert len(seen) == 2
    (first, proc_a), (second, proc_b) = seen
    assert first.url == second.url == f"http://127.0.0.1:{project.port}"
    assert sorted([first.started, second.started]) == [False, True]
    assert proc_a is not None and proc_a == proc_b
    assert _gone(*proc_a)


def test_the_process_is_gone_when_the_last_holder_returns(project):
    with project.ensure() as record:
        assert record.started
        running = project.process()
        assert running is not None
        assert _health_once_the_jobs_ran(f"{record.url}/health") == {
            "status": "ok",
            "ready": True,
            "jobs_ready": True,
        }
    assert _gone(*running, wait_s=0.5), "the stop must be synchronous"
    assert not (project.state_dir / hr.PROCESS_FILE_NAME).exists()
    assert hr.live_leases(project.state_dir / hr.LEASE_DIR_NAME) == []
    with pytest.raises(httpx.ConnectError):
        httpx.get(f"{record.url}/health", trust_env=False)


def test_a_reuser_that_outlives_the_starter_leaves_no_process(project):
    starter = project.ensure()
    first = starter.__enter__()
    assert first.started
    running = project.process()
    assert running is not None
    with project.ensure() as second:
        assert not second.started
        starter.__exit__(None, None, None)
        assert hr._alive(*running), "a live reuser keeps the process"
        _sources(second)
    assert _gone(*running, wait_s=0.5)


def test_the_service_stops_itself_when_its_holder_dies(project):
    # A holder that dies without releasing (here: its lease names a dead pid)
    # leaves the lease watch to end the process.
    script = (
        "import sys, time\n"
        "from osprey.agent_record import host as hr\n"
        "import yaml\n"
        "cfg = yaml.safe_load(open(sys.argv[2]))\n"
        "cm = hr.ensure_host_record(sys.argv[1], cfg)\n"
        "cm.__enter__()\n"
        "print('ready', flush=True)\n"
        "time.sleep(600)\n"
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", script, str(project.root), str(project.root / "build/config.yml")],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "ready"
        running = project.process()
        assert running is not None
    finally:
        holder.send_signal(signal.SIGKILL)
        holder.wait()
    assert _gone(*running, wait_s=service.LEASE_POLL_S * 15)


# ---------------------------------------------------------------------------
# Never spawned over a live port
# ---------------------------------------------------------------------------


def _stub(health_status: int, sources_status: int) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok"}, status_code=health_status)

    @app.get("/health/sources")
    async def sources() -> JSONResponse:
        return JSONResponse({"status": "refused"}, status_code=sources_status)

    return app


@pytest.mark.usefixtures("claude_dir")
@pytest.mark.parametrize(
    ("health_status", "sources_status", "why"),
    [(503, 503, "503"), (200, 401, "refused the host login")],
    ids=["unhealthy", "another-projects-login"],
)
def test_a_live_port_that_is_not_ours_is_never_spawned_over(
    tmp_path, health_status, sources_status, why
):
    with run_app_server(_stub(health_status, sources_status)) as base:
        port = int(base.rsplit(":", 1)[1])
        project = Project(tmp_path / "repo", port=port)
        with pytest.raises(hr.HostRecordRefused, match=why), project.ensure(wait_s=1.0):
            pass
    assert project.process() is None
    assert not (project.state_dir / hr.PROCESS_FILE_NAME).exists()
    assert not (project.root / "var" / "agent_record").exists()


def test_a_live_stack_marker_is_never_spawned_under(project, monkeypatch):
    monkeypatch.setattr(hr, "_container_running", lambda name: name == "hostrec-agent-record")
    project.state_dir.mkdir(parents=True, exist_ok=True)
    (project.state_dir / hr.MARKER_FILE_NAME).write_text("hostrec-agent-record\n")
    with pytest.raises(hr.HostRecordRefused, match="stack owns"), project.ensure(wait_s=1.0):
        pass
    assert project.process() is None
    assert (project.state_dir / hr.MARKER_FILE_NAME).exists()


def test_a_stale_stack_marker_is_removed_and_the_record_starts(project, monkeypatch):
    monkeypatch.setattr(hr, "_container_running", lambda name: False)
    project.state_dir.mkdir(parents=True, exist_ok=True)
    (project.state_dir / hr.MARKER_FILE_NAME).write_text("hostrec-agent-record\n")
    with project.ensure() as record:
        assert record.started
    assert not (project.state_dir / hr.MARKER_FILE_NAME).exists()


# ---------------------------------------------------------------------------
# Transcript roots
# ---------------------------------------------------------------------------


def _state_names(path: Path, names: list[str]) -> dict[str, bool]:
    with EmbeddedBackend(path) as backend:
        return {name: backend.get_state(name) is not None for name in names}


def _own_the_record(project: Project, monkeypatch) -> None:
    """Play ``osprey up``'s ``stack-owner`` marker, its container running."""
    monkeypatch.setattr(hr, "_container_running", lambda name: name == "hostrec-agent-record")
    project.state_dir.mkdir(parents=True, exist_ok=True)
    (project.state_dir / hr.MARKER_FILE_NAME).write_text("hostrec-agent-record\n")


def _forwarded(project: Project) -> list[Path]:
    spool = project.root / "var" / "spool" / HOST
    return sorted(spool.glob("*.sealed")) if spool.is_dir() else []


def test_a_takeover_files_nothing_twice_and_reads_no_sibling(project, monkeypatch):
    encoded = project.host_projects.name
    project.host_projects.mkdir(parents=True, exist_ok=True)
    (project.host_projects / f"{SESSION}.jsonl").write_bytes(_line(SESSION, "hello"))
    sibling = project.host_projects.parent / "-some-other-project"
    sibling.mkdir()
    (sibling / f"{SIBLING_SESSION}.jsonl").write_bytes(_line(SIBLING_SESSION, "secret"))

    ours = transcript_state_name(HOST, f"{encoded}/{SESSION}.jsonl")
    theirs = transcript_state_name(HOST, f"-some-other-project/{SIBLING_SESSION}.jsonl")

    # Host-local: the real process reads the host projects dir.
    with project.ensure() as record:
        transcripts = _sources(record)["transcripts"]
        # Ready means the first scan has committed: the one file is in already.
        assert set(transcripts) == {"host"}
        assert transcripts["host"]["error"] is None
        assert transcripts["host"]["files"] == 1
        assert transcripts["host"]["appended"] >= 1
    store = project.root / "var/agent_record/record.duckdb"
    assert _state_names(store, [ours, theirs]) == {ours: True, theirs: False}

    # `osprey up`: the host session forwards the same file, under the same
    # state name, from a position of its own that starts at nothing.
    _own_the_record(project, monkeypatch)
    report = hr.forward_host_transcripts(project.root, project.config)
    assert report is not None and report.errors == {}
    assert report.appended >= 1
    state = json.loads((project.state_dir / hr.FORWARD_STATE_FILE_NAME).read_text())
    assert set(state) == {ours}

    # The stack's record ingests the spool into the same store: every
    # forwarded entry is one the host-local record already filed.
    with EmbeddedBackend(store) as backend:
        before = {entry.key for entry in backend.select(ScopeFilter(), Query())}
        consumed = SpoolConsumer(
            backend, project.root / "var" / "spool", config=project.config
        ).consume()
        after = {entry.key for entry in backend.select(ScopeFilter(), Query())}
    assert consumed.refused == 0
    assert (consumed.appended, consumed.duplicates) == (0, report.appended)
    assert after == before
    assert _forwarded(project) == []


def test_without_a_stack_marker_nothing_is_forwarded(project):
    project.host_projects.mkdir(parents=True, exist_ok=True)
    (project.host_projects / f"{SESSION}.jsonl").write_bytes(_line(SESSION, "hello"))
    assert hr.forward_host_transcripts(project.root, project.config) is None
    assert _forwarded(project) == []
    assert not (project.state_dir / hr.FORWARD_STATE_FILE_NAME).exists()


def test_a_forward_resumes_from_its_own_position(project, monkeypatch):
    _own_the_record(project, monkeypatch)
    project.host_projects.mkdir(parents=True, exist_ok=True)
    path = project.host_projects / f"{SESSION}.jsonl"
    path.write_bytes(_line(SESSION, "one"))
    first = hr.forward_host_transcripts(project.root, project.config)
    assert first is not None and first.appended == 1
    assert hr.forward_host_transcripts(project.root, project.config).appended == 0
    with path.open("ab") as handle:
        handle.write(_line(SESSION, "two"))
    assert hr.forward_host_transcripts(project.root, project.config).appended == 1
    assert len(_forwarded(project)) == 2


def test_a_busy_forward_skips_the_round(project, monkeypatch):
    _own_the_record(project, monkeypatch)
    project.host_projects.mkdir(parents=True, exist_ok=True)
    (project.host_projects / f"{SESSION}.jsonl").write_bytes(_line(SESSION, "hello"))
    with hr._forward_lock(project.state_dir, 0.0) as had:
        assert had
        assert hr.forward_host_transcripts(project.root, project.config) is None
    assert _forwarded(project) == []


def test_a_forwarded_entry_is_scrubbed_of_the_build_secrets(project, monkeypatch):
    secret = "s3cr3t-forwarded-value-0123456789"
    service_file = project.root / creds.SERVICE_FILE
    verifier = creds.read_env_value(service_file, creds.VERIFIER_ENV)
    service_file.write_text(creds.render_service_env([verifier or ""], [secret]))
    _own_the_record(project, monkeypatch)
    project.host_projects.mkdir(parents=True, exist_ok=True)
    (project.host_projects / f"{SESSION}.jsonl").write_bytes(_line(SESSION, secret))
    assert hr.forward_host_transcripts(project.root, project.config).appended == 1
    (segment,) = _forwarded(project)
    (line,) = segment.read_text().splitlines()
    assert secret not in json.dumps(json.loads(line)["body"])


def test_a_holder_forwards_as_it_ends_while_the_stack_owns_the_record(project, monkeypatch):
    project.host_projects.mkdir(parents=True, exist_ok=True)
    path = project.host_projects / f"{SESSION}.jsonl"
    path.write_bytes(_line(SESSION, "before"))
    calls: list[float] = []
    real = hr.forward_host_transcripts

    def counting(root, config, *, wait_s=0.0):
        calls.append(wait_s)
        return real(root, config, wait_s=wait_s)

    monkeypatch.setattr(hr, "forward_host_transcripts", counting)
    with project.ensure():
        # No marker: the host-local record reads the file itself.
        assert _forwarded(project) == []
        _own_the_record(project, monkeypatch)
        with path.open("ab") as handle:
            handle.write(_line(SESSION, "after"))
    # The one-shot holder asked as it started (no marker yet: nothing forwarded)
    # and as it ended, when the whole file went to the spool. The host-local
    # process the marker now keeps may already have ingested the segment, so
    # the forwarder's own position is what is read.
    assert calls == [hr.FORWARD_WAIT_S, hr.FORWARD_WAIT_S]
    state = json.loads((project.state_dir / hr.FORWARD_STATE_FILE_NAME).read_text())
    (position,) = state.values()
    assert position["offset"] == path.stat().st_size


def test_the_supervisor_forwards_every_round_while_the_stack_owns_the_record(project, monkeypatch):
    rounds: list[Path] = []
    monkeypatch.setattr(
        hr, "forward_host_transcripts", lambda root, config, **_: rounds.append(Path(root))
    )
    with project.ensure(supervise_s=0.05):
        deadline = time.monotonic() + 10
        while len(rounds) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
    assert len(rounds) >= 2
    assert set(rounds) == {project.root}


def test_a_drained_read_forwards_first_in_a_host_launchs_children(project, monkeypatch):
    from osprey.agent_record.client import RecordClient
    from osprey.agent_record.contract import Drain

    forwarded: list[str] = []
    monkeypatch.setattr(hr, "forward_for_drain", forwarded.append)
    client = RecordClient("http://127.0.0.1:1", "builder", "token")
    config_path = str(project.root / "build" / "config.yml")

    monkeypatch.delenv(hr.FORWARD_CONFIG_ENV, raising=False)
    client.list_sessions(wait=Drain())
    assert forwarded == []

    monkeypatch.setenv(hr.FORWARD_CONFIG_ENV, config_path)
    client.list_sessions()
    assert forwarded == []
    client.list_sessions(wait=Drain())
    assert forwarded == [config_path]


def test_forward_for_drain_reads_the_rendered_config(project, monkeypatch):
    _own_the_record(project, monkeypatch)
    project.host_projects.mkdir(parents=True, exist_ok=True)
    (project.host_projects / f"{SESSION}.jsonl").write_bytes(_line(SESSION, "hello"))
    hr.forward_for_drain(project.root / "build" / "config.yml")
    assert len(_forwarded(project)) == 1


def test_the_service_reads_only_the_host_root_on_the_host(project, tmp_path):
    settings = service._settings(
        {**project.config, "services": {}},
        {
            service.HOST_PROJECTS_ENV: str(project.host_projects),
            service.TRANSCRIPTS_DIR_ENV: str(tmp_path / "nothing-mounted"),
        },
    )
    roots, prefixes = service._transcript_roots(project.config, settings)
    assert roots == {"host": project.host_projects}
    assert prefixes == {"host": f"{project.host_projects.name}/"}


def test_a_host_root_that_does_not_exist_yet_is_empty_not_an_error(tmp_path):
    with EmbeddedBackend(tmp_path / "r.duckdb") as backend:
        tailer = TranscriptTailer(
            backend,
            {"host": tmp_path / "projects" / "-enc"},
            {"deployment": {"host_identity": HOST}},
            prefixes={"host": "-enc/"},
        )
        report = tailer.poll_once()
    assert report.errors == {}
    assert tailer.health["host"].error is None


def test_a_prefix_must_be_a_relative_directory_path(tmp_path):
    with EmbeddedBackend(tmp_path / "r.duckdb") as backend:
        for bad in ("-enc", "/abs/", "../up/"):
            with pytest.raises(ValueError):
                TranscriptTailer(
                    backend,
                    {"host": tmp_path},
                    {"deployment": {"host_identity": HOST}},
                    prefixes={"host": bad},
                )


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def test_the_store_is_group_writable(project):
    with project.ensure():
        pass
    store = project.root / "var" / "agent_record" / service.EMBEDDED_FILENAME
    assert store.exists()
    assert stat.S_IMODE(store.stat().st_mode) & stat.S_IWGRP
    assert stat.S_IMODE(store.parent.stat().st_mode) & stat.S_IWGRP


def _tree(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*")}


@pytest.mark.usefixtures("claude_dir")
def test_the_clickhouse_refusal_creates_no_file(tmp_path, record_port):
    project = Project(tmp_path / "repo", port=record_port, backend="clickhouse")
    before = _tree(project.root)
    with pytest.raises(hr.HostRecordRefused, match="osprey up") as info, project.ensure():
        pass
    assert "agent_record.backend: embedded" in str(info.value)
    assert _tree(project.root) == before
    assert not (project.root / "var").exists()


def test_a_missing_service_file_spawns_nothing(project):
    (project.root / creds.SERVICE_FILE).unlink()
    with pytest.raises(hr.HostRecordRefused, match="osprey build"), project.ensure(wait_s=2.0):
        pass
    assert project.process() is None
    assert not (project.root / "var" / "agent_record").exists()
    with pytest.raises(httpx.ConnectError):
        httpx.get(f"http://127.0.0.1:{project.port}/health", trust_env=False)


@pytest.mark.usefixtures("claude_dir")
def test_a_deployment_without_a_host_identity_is_refused(tmp_path):
    config = {"project_name": "p", "project_root": str(tmp_path)}
    with pytest.raises(hr.HostRecordRefused, match="host_identity"):
        with hr.ensure_host_record(tmp_path, config):
            pass


# ---------------------------------------------------------------------------
# serve without --lease-dir
# ---------------------------------------------------------------------------


def test_serve_without_a_lease_dir_keeps_running(project):
    env = hr._child_env(
        project.root, project.config, project.state_dir, project.root / "build/config.yml"
    )
    child = subprocess.Popen(
        [sys.executable, "-m", "osprey.cli.main", "agent-record", "serve"],
        cwd=project.root,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        url = f"http://127.0.0.1:{project.port}/health"
        deadline = time.monotonic() + 60
        body = None
        while time.monotonic() < deadline:
            try:
                body = httpx.get(url, trust_env=False).json()
                break
            except httpx.ConnectError:
                time.sleep(0.1)
        assert body is not None
        assert _health_once_the_jobs_ran(url) == {"status": "ok", "jobs_ready": True}
        time.sleep(service.LEASE_POLL_S * 3)
        assert child.poll() is None
    finally:
        os.killpg(child.pid, signal.SIGTERM)
        child.wait(30)


# ---------------------------------------------------------------------------
# Process identity, the lease watch, waiting and the release
# ---------------------------------------------------------------------------


def test_a_start_time_reads_the_same_under_any_time_zone(monkeypatch):
    pid = os.getpid()
    seen = set()
    for zone in ("UTC", "Asia/Tokyo", "America/Los_Angeles"):
        monkeypatch.setenv("TZ", zone)
        seen.add(hr._process_start(pid))
    monkeypatch.delenv("TZ")
    seen.add(hr._process_start(pid))
    assert len(seen) == 1 and None not in seen


def test_a_lease_written_under_one_zone_is_live_under_another(tmp_path, monkeypatch):
    lease_dir = tmp_path / "state" / hr.LEASE_DIR_NAME
    monkeypatch.setenv("TZ", "Asia/Tokyo")
    path = hr._write_lease(lease_dir)
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    assert [lease.path for lease in hr.live_leases(lease_dir)] == [path]


def test_a_live_pid_with_another_start_time_is_a_dead_lease(tmp_path):
    lease_dir = tmp_path / "state" / hr.LEASE_DIR_NAME
    lease_dir.mkdir(parents=True)
    hr._write_identity(
        lease_dir / f"{os.getpid()}-x.lease", os.getpid(), "Thu Jan  1 00:00:00 1970"
    )
    assert hr.live_leases(lease_dir) == []
    assert not hr.any_live_lease(lease_dir)
    hr._prune_leases(lease_dir)
    assert list(lease_dir.iterdir()) == []


def test_the_repr_keeps_the_token_out():
    record = hr.HostRecord("http://127.0.0.1:1", "host-builder", "s3cr3t-token")
    assert "s3cr3t-token" not in repr(record)
    assert "s3cr3t-token" not in str(record)


@pytest.fixture
def held_locks() -> Iterator[None]:
    yield
    while hr._HELD_UNTIL_EXIT:
        os.close(hr._HELD_UNTIL_EXIT.pop())


@pytest.mark.usefixtures("held_locks")
def test_the_idle_stop_is_decided_under_the_ensure_lock(tmp_path):
    state_dir = tmp_path / "state"
    lease_dir = state_dir / hr.LEASE_DIR_NAME
    lease_dir.mkdir(parents=True)
    # Busy lock (an ensure is deciding): no stop, even with no lease.
    with hr.host_record_lock(state_dir, wait_s=0):
        assert not hr.claim_idle_stop(lease_dir)
    # A live lease: no stop, and the lock is not kept.
    lease = hr._write_lease(lease_dir)
    assert not hr.claim_idle_stop(lease_dir)
    with hr.host_record_lock(state_dir, wait_s=0):
        pass
    # No live lease, free lock: stop, and the lock is kept until exit.
    lease.unlink()
    assert hr.claim_idle_stop(lease_dir)
    with pytest.raises(hr.HostRecordRefused, match="not free"):
        with hr.host_record_lock(state_dir, wait_s=0):
            pass


@pytest.mark.usefixtures("held_locks")
def test_the_lease_watch_skips_a_round_while_an_ensure_holds_the_lock(tmp_path, monkeypatch):
    import asyncio

    monkeypatch.setattr(service, "LEASE_POLL_S", 0.02)
    state_dir = tmp_path / "state"
    lease_dir = state_dir / hr.LEASE_DIR_NAME
    lease_dir.mkdir(parents=True)
    stops: list[float] = []

    async def watch_for(seconds: float) -> None:
        task = asyncio.create_task(service._watch_leases(lease_dir, lambda: stops.append(1)))
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(asyncio.shield(task), seconds)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    with hr.host_record_lock(state_dir, wait_s=0):
        asyncio.run(watch_for(0.3))
    assert stops == []
    asyncio.run(watch_for(2.0))
    assert stops == [1]


class _Alive:
    """A spawned child that keeps running."""

    pid = 999_999
    returncode = None

    def poll(self):
        return None


def _scripted_probe(monkeypatch, script, *, first=None):
    """Make ``_probe`` answer by elapsed time: *script* is ``[(until_s, state), ...]``.

    *first*, when given, is the answer to the first probe whatever the clock
    says, and the script's clock starts at that probe.
    """
    calls: list[float] = []

    def probe(*_args):
        calls.append(time.monotonic())
        if first is not None and len(calls) == 1:
            return first, first
        elapsed = time.monotonic() - calls[0]
        for until, state in script:
            if elapsed < until:
                return state, state
        return script[-1][1], script[-1][1]

    monkeypatch.setattr(hr, "_probe", probe)
    monkeypatch.setattr(hr, "_PROBE_INTERVAL_S", 0.01)


def test_the_starter_waits_out_its_own_childs_first_pass(tmp_path, monkeypatch):
    state_dir = tmp_path / "state"
    stops: list[Path] = []
    monkeypatch.setattr(hr, "_spawn", lambda *a: _Alive())
    monkeypatch.setattr(hr, "stop_host_process", lambda d, **k: stops.append(d) or True)
    _scripted_probe(monkeypatch, [(0.8, hr._STARTING), (99, hr._READY)], first=hr._DOWN)
    record = hr.HostRecord("http://127.0.0.1:1", "host-builder", "t")
    lease, started = hr._acquire(
        tmp_path, {}, state_dir, record, 1, True, 0.2, tmp_path / "config.yml"
    )
    assert started and lease.exists()
    assert stops == [], "a still-alive child in its first pass is never stopped"


@pytest.mark.usefixtures("claude_dir")
def test_a_foreign_port_that_is_not_ready_is_refused_at_the_health_bound(tmp_path, monkeypatch):
    state_dir = tmp_path / "state"
    _scripted_probe(monkeypatch, [(99, hr._STARTING)])
    monkeypatch.setattr(hr, "_spawn", lambda *a: pytest.fail("spawned over a live port"))
    record = hr.HostRecord("http://127.0.0.1:1", "host-builder", "t")
    t0 = time.monotonic()
    with pytest.raises(hr.HostRecordRefused, match="not usable after 0 s: starting"):
        hr._acquire(tmp_path, {}, state_dir, record, 1, True, 0.2, tmp_path / "config.yml")
    assert time.monotonic() - t0 < 5


def _flaky_stub(failures: int) -> FastAPI:
    app = FastAPI()
    calls = {"n": 0}

    @app.get("/health")
    async def health() -> JSONResponse:
        calls["n"] += 1
        if calls["n"] <= failures:
            return JSONResponse({"status": "starting"}, status_code=503)
        return JSONResponse({"status": "ok"})

    @app.get("/health/sources")
    async def sources() -> JSONResponse:
        return JSONResponse({})

    return app


@pytest.mark.usefixtures("claude_dir")
def test_an_unhealthy_port_that_turns_healthy_is_reused(tmp_path):
    with run_app_server(_flaky_stub(failures=3)) as base:
        port = int(base.rsplit(":", 1)[1])
        project = Project(tmp_path / "repo", port=port)
        with project.ensure(wait_s=10.0) as record:
            assert not record.started
            assert project.process() is None
    assert not (project.root / "var" / "agent_record").exists()


def test_a_busy_lock_at_release_never_masks_the_callers_error(project, monkeypatch):
    class Boom(Exception):
        pass

    real_lock = hr.host_record_lock
    calls = {"n": 0}

    @contextlib.contextmanager
    def lock(state_dir, **kwargs):
        calls["n"] += 1
        if calls["n"] > 1:  # the release's lock
            raise hr.HostRecordRefused("the agent-record lock was not free")
        with real_lock(state_dir, **kwargs):
            yield

    monkeypatch.setattr(hr, "host_record_lock", lock)
    with pytest.raises(Boom), project.ensure():
        raise Boom


def test_shutdown_waits_for_the_first_pass_before_closing_the_backend(
    project, tmp_path, monkeypatch
):
    from fastapi.testclient import TestClient

    env = hr._child_env(
        project.root, project.config, project.state_dir, project.root / "build/config.yml"
    )
    lease_dir = project.state_dir / hr.LEASE_DIR_NAME
    hr._write_lease(lease_dir)
    events: list[str] = []
    started = threading.Event()

    def slow_consume(_self):
        started.set()
        time.sleep(0.5)
        events.append("first pass done")

    class Recorded(EmbeddedBackend):
        def close(self):
            events.append("closed")
            super().close()

    monkeypatch.setattr(service.SpoolConsumer, "consume", slow_consume)
    monkeypatch.setattr(
        service, "open_backend", lambda record, record_dir, env: Recorded(tmp_path / "r.duckdb")
    )
    app = service.create_app(config=project.config, env=env, lease_dir=lease_dir, stop=lambda: None)
    with TestClient(app) as client:
        assert started.wait(10)
        # The jobs start only after the first pass, so neither is ready yet.
        assert client.get("/health").json() == {
            "status": "ok",
            "ready": False,
            "jobs_ready": False,
        }
    assert events == ["first pass done", "closed"]


def test_a_failed_first_pass_answers_503_so_no_one_waits_on_it(project, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    env = hr._child_env(
        project.root, project.config, project.state_dir, project.root / "build/config.yml"
    )
    lease_dir = project.state_dir / hr.LEASE_DIR_NAME
    hr._write_lease(lease_dir)

    def broken(_self):
        raise RuntimeError("spool unreadable")

    monkeypatch.setattr(service.SpoolConsumer, "consume", broken)
    monkeypatch.setattr(
        service, "open_backend", lambda record, record_dir, env: EmbeddedBackend(tmp_path / "r.db")
    )
    app = service.create_app(config=project.config, env=env, lease_dir=lease_dir, stop=lambda: None)
    with TestClient(app) as client:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            response = client.get("/health")
            if response.status_code == 503:
                break
            time.sleep(0.02)
        assert response.status_code == 503
        assert response.json() == {"status": "error", "ready": False}


def test_a_missing_host_login_names_the_build_and_the_run_time_mount(tmp_path, record_port):
    """A bare container of the project image holds no credential file: the
    refusal says to build, or to mount the build's login file where it is read."""
    proj = Project(tmp_path / "repo", port=record_port)
    login_path = proj.root / creds.login_file(ids.login_for(HOST))
    login_path.unlink()

    why = hr.host_record_refusal(proj.root, proj.config)

    assert why == (
        f"the host login file {login_path} is missing: run `osprey build`, or, in a "
        f"container, mount the build's login file at {login_path}"
    )


def test_a_missing_service_file_names_the_build_and_the_run_time_mount(tmp_path, record_port):
    proj = Project(tmp_path / "repo", port=record_port)
    service_path = proj.root / creds.SERVICE_FILE
    service_path.unlink()

    why = hr.host_record_refusal(proj.root, proj.config)

    assert why == (
        f"{service_path} is missing or incomplete, so agent-record cannot start: run "
        f"`osprey build`, or, in a container, mount the build's service file at {service_path}"
    )
