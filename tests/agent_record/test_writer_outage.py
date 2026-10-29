"""Every writer keeps working while agent-record is stopped, and loses nothing.

Each producer is driven through its own production entry point — the audit
writer, a hook process (its audit and debug lines), a sidecar login, a dispatch
accept and terminal, a bridge turn, a feedback report and an ``api_call`` — with
no agent-record service running and every record client aimed at a closed port.
Every call succeeds, the sidecar's logins answer at their normal speed, and the
lines wait in the spool. A real service over the embedded backend is then
started on the same spool root; it records every spooled line exactly once and
keeps ingesting what is written while it runs.

A second test stops the service in the middle of a consume — after a batch was
stored but before its checkpoint was — and restarts it over the same record
file: no entry is duplicated and none is lost.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import subprocess
import sys
import time
from collections import Counter
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.agent_record import contract as c
from osprey.agent_record import credentials as creds
from osprey.agent_record import identities as ids
from osprey.agent_record import service, writer
from osprey.agent_record.backends.embedded import EmbeddedBackend
from osprey.agent_record.client import URL_ENV
from osprey.agent_record.ingest.spool import BATCH_LINES
from osprey.audit import writer as audit_writer
from osprey.audit.envelope import DECISION_ALLOWED, POSTURE_SOURCE_PROCESS
from osprey.bridges.core.history import RECORD_UNAVAILABLE, ConversationHistory
from osprey.interfaces.web_terminal.feedback_composer import CONTEXT_RECORD_UNAVAILABLE
from osprey.interfaces.web_terminal.routes.feedback import router as feedback_router
from osprey.mcp_server.dispatch_worker import dispatch_api, sdk_runner
from osprey.models.logging import log_api_call
from osprey.services.auth_sidecar.app import create_app as create_sidecar_app
from osprey.services.auth_sidecar.passwords import hash_password
from osprey.services.auth_sidecar.revocation import STATE_DIR_ENV
from osprey_connectors.identity import AUDIT_IDENTITY_ENV, TERMINAL_USER_ENV
from osprey_connectors_tests._harness._ports import reserved_port
from tests.agent_record.test_identities import _config

HOOKS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "osprey" / "templates" / "claude_code" / "claude"
) / "hooks"

TERMINAL = "alice"
SIDECAR = ids.SIDECAR_IDENTITY
DISPATCH = "dispatch-worker-1"
BRIDGE = ids.BRIDGE_IDENTITIES["teams_bridge"]

#: The longest a sidecar login may take; a login that waited on the record would not.
LOGIN_BUDGET_S = 5.0

EXTERNAL_ORIGIN = "https://terminals.example.org"
ALICE_PASSWORD = "alice-password"
SIDECAR_ENV = {
    "OSPREY_AUTH_METHOD": "password",
    "OSPREY_AUTH_SESSION_SECRET": "session-secret-value",
    "OSPREY_AUTH_SESSION_LIFETIME": "3600",
    "OSPREY_AUTH_USERS": "alice,bob",
    "OSPREY_AUTH_PW_HASH_ALICE": hash_password(ALICE_PASSWORD),
    "OSPREY_AUTH_EXTERNAL_ORIGIN": EXTERNAL_ORIGIN,
    "OSPREY_AUTH_TLS_ENABLED": "true",
}

AUDIT_FIELDS = {
    "surface": "mcp_python",
    "posture": "sandbox",
    "posture_source": POSTURE_SOURCE_PROCESS,
    "session": None,
    "subject": "mcp__python__execute",
    "decision": DECISION_ALLOWED,
    "reason": "allowed",
}

FEEDBACK_SESSION = "3f2504e0-4f89-41d3-9a0c-0305e82c3301"

#: Hook processes start from these variables only, so no developer marker leaks in.
_HOOK_BASE_VARS = ("PATH", "PYTHONPATH", "SYSTEMROOT", "TEMP", "TMP", "TMPDIR", "LANG")


# --- fixtures -----------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_writer():
    writer.seal_all()
    yield
    writer.seal_all()


@pytest.fixture
def config(tmp_path) -> dict:
    cfg = _config(workers=1, bridges=("teams_bridge",))
    cfg["project_root"] = str(tmp_path / "project")
    return cfg


def _token(login: str) -> str:
    return f"token-of-{login}-0123456789"


def _b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


@pytest.fixture
def record_env(tmp_path, config, spool_zone) -> dict[str, str]:
    """The service's environment: the writers' spool root and a record directory."""
    lines = "".join(
        creds.verifier_line(login, _token(login)) + "\n" for login in ids.credential_table(config)
    )
    return {
        creds.VERIFIER_ENV: _b64(lines),
        creds.SECRETS_ENV: _b64(json.dumps(["s3cr3t-token-value-0123456789"])),
        service.SPOOL_ROOT_ENV: str(spool_zone.root),
        service.RECORD_DIR_ENV: str(tmp_path / "record"),
        service.TRANSCRIPTS_DIR_ENV: str(tmp_path / "transcripts"),
    }


@pytest.fixture
def stopped_record(monkeypatch) -> Iterator[str]:
    """Every record client in this process aims at a port nothing listens on,
    reserved for the test and never bound."""
    with reserved_port() as port:
        url = f"http://127.0.0.1:{port}"
        monkeypatch.setenv(URL_ENV, url)
        monkeypatch.setenv(creds.LOGIN_ENV, f"{TERMINAL}:{_token(TERMINAL)}")
        yield url


@asynccontextmanager
async def _serving(app: FastAPI) -> AsyncIterator[FastAPI]:
    """The service with its lifespan (and so its ingest loop) running."""
    async with app.router.lifespan_context(app):
        yield app


async def _eventually(probe, *, timeout: float = 20.0):
    deadline = time.monotonic() + timeout
    while True:
        value = probe()
        if value:
            return value
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.05)


def _segments(root: Path) -> list[Path]:
    """Every open or sealed segment under the spool *root*."""
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name.endswith((c.OPEN_SUFFIX, c.SEALED_SUFFIX))
    )


def _spooled_keys(root: Path) -> Counter[str]:
    """The key every spooled line will be recorded under, counted."""
    keys: Counter[str] = Counter()
    for segment in _segments(root):
        identity = segment.parent.name
        for raw in segment.read_bytes().splitlines():
            if raw.strip():
                line = c.parse_spool_line(raw)
                keys[c.spool_entry_key(line, identity, receiver_owned=False)] += 1
    return keys


def _writer_errors(root: Path) -> list[str]:
    return [
        f"{path}: {path.read_text()}" for path in root.rglob(c.WRITER_ERRORS_FILE) if path.is_file()
    ]


def _recorded(record_dir: Path) -> list[c.Entry]:
    """Everything the record file holds, read after the service has closed it."""
    with EmbeddedBackend(record_dir / service.EMBEDDED_FILENAME) as backend:
        return backend.select(c.ScopeFilter(), c.Query())


def _stored_twice(entries: list[c.Entry]) -> list[tuple[str, int]]:
    counts = Counter((entry.key, entry.chunk.index) for entry in entries)
    return [pair for pair, n in counts.items() if n > 1]


# --- the writers, each through its production entry point --------------------------


def _write_audit(spool_zone) -> None:
    spool_zone.spool_for(TERMINAL)
    assert audit_writer.record(**AUDIT_FIELDS), "the audit writer stored nothing"
    writer.seal_all()


def _run_hook(spool_zone, project: Path) -> None:
    """A writes-check hook process refusing a write, with hook debug on."""
    directory = spool_zone.spool_for(TERMINAL)
    child_env = {name: os.environ[name] for name in _HOOK_BASE_VARS if name in os.environ}
    child_env.update(
        {
            "CLAUDE_PROJECT_DIR": str(project),
            "OSPREY_TERMINAL_USER": TERMINAL,
            "OSPREY_AGENT_DATA_ROOT": str(project / "var" / "agent_data"),
            "OSPREY_EXECUTION_MODE": "readonly",
            "OSPREY_HOOK_DEBUG": "1",
            c.SPOOL_DIR_ENV: str(directory),
        }
    )
    result = subprocess.run(
        [sys.executable, str(HOOKS_DIR / "osprey_writes_check.py")],
        input=json.dumps(
            {
                "tool_name": "mcp__controls__channel_write",
                "tool_input": {},
                "session_id": "conv-outage",
                "tool_use_id": "toolu_outage",
            }
        ),
        capture_output=True,
        text=True,
        env=child_env,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    decisions = [
        json.loads(line)["hookSpecificOutput"]["permissionDecision"]
        for line in result.stdout.splitlines()
        if line.strip().startswith("{")
    ]
    assert decisions == ["deny"], result.stdout


def _sidecar_login(spool_zone, monkeypatch, tmp_path) -> float:
    """One password login through the sidecar; how long it took."""
    state = tmp_path / "var" / "state" / SIDECAR
    state.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(STATE_DIR_ENV, str(state))
    monkeypatch.setenv(AUDIT_IDENTITY_ENV, SIDECAR)
    monkeypatch.delenv(TERMINAL_USER_ENV, raising=False)
    spool_zone.spool_for(SIDECAR)
    with TestClient(create_sidecar_app(SIDECAR_ENV), base_url="https://testserver") as client:
        started = time.monotonic()
        response = client.post(
            "/auth/login",
            data={"user": "alice", "next": "", "password": ALICE_PASSWORD},
            headers={"Origin": EXTERNAL_ORIGIN},
            follow_redirects=False,
        )
        elapsed = time.monotonic() - started
    assert response.status_code == 303, response.text
    writer.seal_all()
    monkeypatch.delenv(AUDIT_IDENTITY_ENV, raising=False)
    return elapsed


async def _dispatch(spool_zone, monkeypatch, tmp_path) -> str:
    """Accept one run and let it end; its run id."""
    spool_zone.spool_for(DISPATCH)
    monkeypatch.setattr(dispatch_api, "_runs", {})
    monkeypatch.setattr(dispatch_api, "_queues", {})
    monkeypatch.setattr(dispatch_api, "_tasks", {})
    monkeypatch.setattr(dispatch_api, "_run_input_seam", {})
    monkeypatch.setattr(dispatch_api, "_settled", dispatch_api.OrderedDict())
    monkeypatch.setenv("OSPREY_PROJECT_DIR", str(tmp_path / "project"))
    monkeypatch.setattr(dispatch_api, "describe_run_artifacts", lambda run_id: [])
    monkeypatch.setattr(dispatch_api, "describe_run_input_artifacts", lambda run_id: [])

    async def _agent(**_kw: Any) -> dict[str, Any]:
        return {
            "status": "completed",
            "text_output": "the answer",
            "tool_calls": [],
            "error": None,
            "duration_sec": 0.25,
            "cost_usd": 0.03,
            "num_turns": 2,
        }

    monkeypatch.setattr(sdk_runner, "run_dispatch", _agent)
    response = await dispatch_api.dispatch(
        dispatch_api.DispatchRequest(prompt="why did BPM 3 drift?", allowed_tools=[], owner="alice")
    )
    task = dispatch_api._tasks.get(response.run_id)
    if task is not None:
        await task
    writer.seal_all()
    return response.run_id


def _bridge_turns(spool_zone, run_id: str) -> None:
    spool_zone.spool_for(BRIDGE)
    history = ConversationHistory()
    replay = history.read("teams:conv-outage")
    assert replay.record == RECORD_UNAVAILABLE, "the stopped record answered a read"
    assert history.append("teams:conv-outage", "why did BPM 3 drift?", run_id=run_id)
    assert history.append_failed("teams:conv-outage", "and BPM 4?")
    writer.seal_all()


def _feedback(spool_zone, tmp_path) -> None:
    spool_zone.spool_for(TERMINAL)
    app = FastAPI()
    app.include_router(feedback_router)
    app.state.project_cwd = str(tmp_path / "project")
    app.state.workspace_dir = tmp_path / "workspace"
    app.state.terminal_user = TERMINAL
    app.state.app_name = "OSPREY Outage Rig"
    app.state.transcript_map = {}
    app.state.transcript_map_provisional = False
    with TestClient(app) as client:
        response = client.post(
            "/api/feedback",
            json={
                "text": "the terminal froze while the record was down",
                "channel": "local",
                "include_metadata": True,
                "include_context": True,
                "session_id": FEEDBACK_SESSION,
                "scrollback": "$ osprey web\n",
            },
        )
    assert response.status_code == 200, response.text
    assert response.json()["context_status"] == CONTEXT_RECORD_UNAVAILABLE
    writer.seal_all()


def _api_call(spool_zone, monkeypatch) -> None:
    spool_zone.spool_for(TERMINAL)
    monkeypatch.setattr(
        "osprey.models.logging.get_config_value",
        lambda *args, **kwargs: {"api_calls": {"save_all": True}},
    )
    log_api_call(
        message="test input message",
        result="test output",
        provider="anthropic",
        model_id="claude-3-sonnet",
        max_tokens=1024,
        temperature=0.7,
    )
    writer.seal_all()


# --- the outage ---------------------------------------------------------------------


@pytest.mark.usefixtures("stopped_record")
async def test_every_writer_spools_while_the_record_is_stopped_and_each_entry_lands_once(
    config, record_env, spool_zone, monkeypatch, tmp_path
):
    project = tmp_path / "project"
    project.mkdir()
    (project / "profile.yml").write_text("name: writer-outage-test\n")
    root = spool_zone.root

    _write_audit(spool_zone)
    _run_hook(spool_zone, project)
    login_times = [_sidecar_login(spool_zone, monkeypatch, tmp_path)]
    run_id = await _dispatch(spool_zone, monkeypatch, tmp_path)
    _bridge_turns(spool_zone, run_id)
    _feedback(spool_zone, tmp_path)
    _api_call(spool_zone, monkeypatch)
    writer.seal_all()

    assert _writer_errors(root) == [], "a writer failed while the record was stopped"
    spooled = _spooled_keys(root)
    assert spooled, "nothing was spooled"
    assert [key for key, n in spooled.items() if n > 1] == [], "a line was spooled twice"
    assert c.run_key(run_id, c.RUN_ACCEPT) in spooled
    assert c.run_key(run_id, c.RUN_TERMINAL) in spooled
    by_identity = Counter(key.split(":")[1] for key in spooled if key.startswith("sp:"))
    assert by_identity[SIDECAR] >= 1, "the sidecar login wrote no entry"
    assert by_identity[BRIDGE] == 2

    record_dir = Path(record_env[service.RECORD_DIR_ENV])
    async with _serving(service.create_app(config=config, env=record_env)):
        await _eventually(lambda: not _segments(root))
        login_times.append(_sidecar_login(spool_zone, monkeypatch, tmp_path))
        await _eventually(lambda: not _segments(root))

    assert all(t < LOGIN_BUDGET_S for t in login_times), login_times

    entries = _recorded(record_dir)
    assert _stored_twice(entries) == [], "an entry was recorded twice"
    recorded = {entry.key for entry in entries}
    assert set(spooled) <= recorded, f"lost: {sorted(set(spooled) - recorded)}"
    extra = recorded - set(spooled)
    assert len(extra) == 1, f"recorded beyond the outage's lines: {sorted(extra)}"
    (live_login,) = extra
    assert live_login.startswith(f"sp:{SIDECAR}:"), "the login while serving was recorded"

    kinds = Counter(entry.kind for entry in entries if entry.key in spooled)
    assert kinds[c.KIND_AUDIT] >= 3, kinds  # the audit writer, the hook, the sidecar
    assert kinds[c.KIND_RUN] == 2, kinds
    assert kinds[c.KIND_BRIDGE_TURN] == 2, kinds
    assert kinds[c.KIND_FEEDBACK] == 1, kinds
    assert kinds[c.KIND_API_CALL] == 1, kinds
    hook_debug = [e for e in entries if e.kind == c.KIND_DEBUG and e.body.get("source") == "hook"]
    assert hook_debug, "the hook's debug line was not recorded"
    by_writer = {e.identity for e in entries if e.kind == c.KIND_AUDIT}
    assert {TERMINAL, SIDECAR} <= by_writer, by_writer


# --- a restart in the middle of a consume ------------------------------------------


class _DiesMidConsume:
    """The record, until the process dies between a batch's append and its checkpoint.

    The first spool checkpoint is stored; the second batch's entries are
    appended and then its checkpoint fails, and from then on every call fails —
    the service is gone as far as the record can tell.
    """

    def __init__(self, backend: EmbeddedBackend, *, die_at_checkpoint: int = 2) -> None:
        self._backend = backend
        self._die_at = die_at_checkpoint
        self._checkpoints = 0
        self.dead = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    def _alive(self) -> None:
        if self.dead:
            raise c.RecordBackendUnavailable("the service stopped")

    def append(self, entries):
        self._alive()
        return self._backend.append(entries)

    def put_state(self, name, value):
        self._alive()
        if name.startswith(f"{c.SPOOL_STATE_PREFIX}:"):
            self._checkpoints += 1
            if self._checkpoints >= self._die_at:
                self.dead = True
                raise c.RecordBackendUnavailable("the service stopped mid-consume")
        return self._backend.put_state(name, value)

    def get_state(self, name):
        self._alive()
        return self._backend.get_state(name)

    def select(self, *args, **kwargs):
        self._alive()
        return self._backend.select(*args, **kwargs)


async def test_a_restart_mid_consume_neither_duplicates_nor_loses_an_entry(
    config, record_env, spool_zone
):
    root = spool_zone.root
    spool_zone.spool_for(TERMINAL)
    lines = 3 * BATCH_LINES  # three of the consumer's batches, in one segment
    for _ in range(lines):
        assert audit_writer.record(**AUDIT_FIELDS)
    writer.seal_all()
    spooled = _spooled_keys(root)
    assert len(spooled) == lines
    assert len(_segments(root)) == 1

    record_dir = Path(record_env[service.RECORD_DIR_ENV])
    record_dir.mkdir(parents=True)
    with EmbeddedBackend(record_dir / service.EMBEDDED_FILENAME) as backend:
        dying = _DiesMidConsume(backend)
        async with _serving(service.create_app(config=config, backend=dying, env=record_env)):
            await _eventually(lambda: dying.dead)
        before_restart = backend.select(c.ScopeFilter(), c.Query())
    # The first batch is stored and checkpointed; the second is stored but its
    # checkpoint is not, so the restart reads it again.
    assert len(before_restart) == 2 * BATCH_LINES, "the stop did not fall inside the consume"
    assert _segments(root), "the uncommitted segment is kept"

    async with _serving(service.create_app(config=config, env=record_env)):
        await _eventually(lambda: not _segments(root))

    entries = _recorded(record_dir)
    assert _stored_twice(entries) == [], "an entry was recorded twice"
    assert {entry.key for entry in entries} == set(spooled)
