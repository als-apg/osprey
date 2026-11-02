"""The host-local agent record on a real hello-world build, with no container.

One deployment per module: ``osprey init --preset hello-world`` and ``osprey
build``, run as the shipped commands with ``CLAUDE_CONFIG_DIR`` and ``HOME``
pointed into the module's temp tree. Nothing runs ``osprey up``. A ``docker``
and a ``podman`` stand-in lead ``PATH``: each logs every call and answers
``inspect`` with "running" only for a container name the test has declared
running, so the module can prove that no container is started and can play
the stack's agent-record without a container runtime.

Pinned here, against real ``osprey agent-record serve`` processes:

- A host ``osprey web`` (the real server, as a subprocess), a host ``osprey
  chat`` (the real command, running a stand-in ``claude`` that writes a
  Claude Code transcript into ``build/``'s project directory and exports
  Claude Code's OTLP decision and usage events with the environment it was
  handed) and a host ``osprey query`` (the real command, its agent run
  replaced) each get a host-local record: their tool call is in the session
  log and the session list, the turn's thinking is in the chat's
  turns, and their OTLP decision and usage events are in the record. No
  container is started, and no record process is left once the session ends.
- ``osprey feedback list`` answers and leaves no process.
- With ``backend: clickhouse`` and no stack, ``web``, ``chat`` and ``query``
  refuse, naming the fix, and no ``record.duckdb`` appears.
- With a stack (an ``agent-record serve`` the test starts the way ``osprey
  up`` hands the record over, under the real takeover and hand-back
  functions, reading no host transcript as the container reads none): a
  launcher uses the stack's record and starts none; a host chat's session
  reaches it through the spool, forwarded by the chat itself; a host
  session that started first keeps working after ``osprey up``, and the step
  it takes then is committed by the stack; after ``osprey down`` mid-session
  the next read answers again, the step still in it; two concurrent host
  sessions share one record; no record process is left after the last one
  ends.

The live variant, with a real model and real containers, is
``tests/e2e/test_host_record_e2e.py``.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import yaml
from click.testing import CliRunner

from osprey.agent_record import credentials as creds
from osprey.agent_record import host as hr
from osprey.agent_record import identities as ids
from osprey.agent_record.client import URL_ENV, RecordClient
from osprey.agent_record.contract import (
    KIND_DECISION,
    KIND_USAGE,
    SPOOL_DIR_ENV,
    Drain,
    spool_key,
)
from osprey.agent_runner.project_paths import CLAUDE_CONFIG_DIR_ENV, claude_project_dir
from osprey.interfaces.web_terminal.routes import session as session_routes
from osprey.mcp_server import session as session_record

pytestmark = [pytest.mark.slow]

#: This module's port block; the record's own port is pinned to a free port.
PORT_BASE = 25700
CLAUDE_VERSION = "2.1.300"
#: The MCP tool every session here calls; the session log lists MCP calls.
TOOL_NAME = "mcp__controls__channel_read"
#: From a fact being written to it being readable.
READ_TIMEOUT_S = 90.0
#: How long a verb may take.
VERB_TIMEOUT_S = 600
#: The supervisor probes every :data:`hr.SUPERVISE_INTERVAL_S`; a restart
#: then waits out the process's first pass.
RESTART_TIMEOUT_S = 120.0

#: Variables a developer shell or an earlier test may carry that would change
#: what a spawned ``osprey`` verb resolves.
_SCRUBBED_ENV = (
    "CLAUDECODE",
    URL_ENV,
    creds.LOGIN_ENV,
    SPOOL_DIR_ENV,
    "OSPREY_CONFIG",
    "OSPREY_TERMINAL_SESSION",
    "OSPREY_TERMINAL_USER",
    "OSPREY_AGENT_RECORD_OFF",
    "OSPREY_WEB_PORT",
    hr.FORWARD_CONFIG_ENV,
)

#: The container runtime stand-in: logs each call; ``inspect`` of a name the
#: test declared running (a file of that name in ``running/``) answers true.
_RUNTIME = """#!/bin/sh
printf '%s\\n' "$*" >> "{log}"
if [ "$1" = inspect ] && [ -n "$4" ] && [ -f "{running}/$4" ]; then
  echo true
  exit 0
fi
exit 1
"""

#: The ``claude`` stand-in ``osprey chat`` runs: one turn of a Claude Code
#: session, written where Claude Code writes it, exported where the launcher
#: says, then held open until the test releases it.
_CLAUDE = """#!{python}
import json, os, sys, time, urllib.request, uuid
from pathlib import Path
sys.path.insert(0, {tests_root!r})
if sys.argv[1:] == ["--version"]:
    print("{version} (Claude Code)")
    sys.exit(0)
from tests.agent_record.test_host_record_integration import otlp_request, transcript
from osprey.agent_runner.project_paths import claude_project_dir

session = str(uuid.uuid4())
tool_use_id = "toolu_" + uuid.uuid4().hex[:20]
project = claude_project_dir(Path.cwd())
project.mkdir(parents=True, exist_ok=True)
(project / (session + ".jsonl")).write_text(transcript(session, tool_use_id, "chat"))
endpoint = os.environ.get("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", "")
header = os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", "")
status = None
if endpoint and header.startswith("Authorization="):
    body = json.dumps(otlp_request(session, tool_use_id, "chat")).encode()
    request = urllib.request.Request(
        endpoint, data=body, method="POST",
        headers={{"Content-Type": "application/json",
                  "Authorization": header.removeprefix("Authorization=")}},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        status = response.status
keys = ("OSPREY_AGENT_RECORD_URL", "OSPREY_SPOOL_DIR", "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT")
report = {{"session": session, "tool_use_id": tool_use_id, "otlp_status": status,
           "env": {{k: os.environ.get(k) for k in keys}}}}
Path(os.environ["HSA_CLAUDE_REPORT"]).write_text(json.dumps(report))
release = Path(os.environ["HSA_CLAUDE_RELEASE"])
deadline = time.monotonic() + 300
while not release.exists() and time.monotonic() < deadline:
    time.sleep(0.1)
"""

#: A host session holding the record the way ``chat``/``query`` do
#: (``host_record_launch``, supervisor included), taking a spool step on
#: demand: ``step <session> <message>`` on stdin, ``end`` to return.
_HOLDER = """
import json, os, sys
from pathlib import Path
import yaml
from osprey.agent_record import writer
from osprey.agent_record.contract import KIND_DEBUG, SPOOL_DIR_ENV
from osprey.agent_runner.provider_env import host_record_launch
repo = Path(sys.argv[1])
config_path = repo / "build" / "config.yml"
config = yaml.safe_load(config_path.read_text())
with host_record_launch(repo, config, config_path=config_path) as record:
    print(json.dumps({"url": record.url, "started": record.started}), flush=True)
    for line in sys.stdin:
        words = line.split()
        if not words or words[0] == "end":
            break
        if words[0] == "step":
            entry = writer.write(KIND_DEBUG, {"message": words[2]}, session_id=words[1])
            writer.seal_all()
            print(json.dumps({"entry": entry}), flush=True)
print(json.dumps({"released": True}), flush=True)
"""


# --------------------------------------------------------------------------- #
# What a Claude Code session leaves behind
# --------------------------------------------------------------------------- #


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _line(session: str, uid: str, kind: str, content: Any, parent: str | None) -> str:
    role = "user" if kind == "user" else "assistant"
    message: dict[str, Any] = {"role": role, "content": content}
    if role == "assistant":
        message["model"] = "claude-opus-5-5"
        message["usage"] = {"input_tokens": 3, "output_tokens": 5}
    return (
        json.dumps(
            {
                "parentUuid": parent,
                "isSidechain": False,
                "userType": "external",
                "cwd": "/somewhere",
                "sessionId": session,
                "version": CLAUDE_VERSION,
                "type": kind,
                "message": message,
                "uuid": uid,
                "timestamp": _now(),
            }
        )
        + "\n"
    )


def thinking_text(label: str) -> str:
    return f"The {label} session reads the channel first."


def transcript(session: str, tool_use_id: str, label: str) -> str:
    """One turn: a prompt, thinking and an MCP tool call, its result, the answer."""
    tag = session[:8]
    return "".join(
        (
            _line(session, f"u1-{tag}", "user", f"What does the {label} channel read?", None),
            _line(
                session,
                f"a1-{tag}",
                "assistant",
                [
                    {"type": "thinking", "thinking": thinking_text(label), "signature": "c2ln"},
                    {
                        "type": "tool_use",
                        "id": tool_use_id,
                        "name": TOOL_NAME,
                        "input": {"channel": "HW:READBACK"},
                    },
                ],
                f"u1-{tag}",
            ),
            _line(
                session,
                f"u2-{tag}",
                "user",
                [{"type": "tool_result", "tool_use_id": tool_use_id, "content": "0.41"}],
                f"a1-{tag}",
            ),
            _line(
                session,
                f"a2-{tag}",
                "assistant",
                [{"type": "text", "text": f"The {label} channel reads 0.41."}],
                f"u2-{tag}",
            ),
        )
    )


def _value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    return {"stringValue": value}


def _attrs(mapping: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{"key": k, "value": _value(v)} for k, v in mapping.items()]


def _log_record(event: str, session: str, sequence: int, attributes: Mapping[str, Any]) -> dict:
    nanos = str(time.time_ns())
    return {
        "timeUnixNano": nanos,
        "observedTimeUnixNano": nanos,
        "body": {"stringValue": f"claude_code.{event}"},
        "attributes": _attrs(
            {"event.name": event, "session.id": session, "event.sequence": sequence, **attributes}
        ),
    }


def otlp_request(session: str, tool_use_id: str, label: str) -> dict[str, Any]:
    """Claude Code's OTLP/JSON log export of the turn: its decision and its usage."""
    prompt = str(uuid.uuid5(uuid.NAMESPACE_URL, session))
    decision = _log_record(
        "tool_decision",
        session,
        1,
        {
            "prompt.id": prompt,
            "decision": "accept",
            "source": "config",
            "tool_name": TOOL_NAME,
            "tool_use_id": tool_use_id,
            "tool_source": "mcp",
            "tool_parameters": "{}",
        },
    )
    usage = _log_record(
        "api_request",
        session,
        2,
        {
            "event.timestamp": datetime.now(UTC).isoformat(),
            "prompt.id": prompt,
            "model": "claude-opus-5-5",
            "input_tokens": 3,
            "output_tokens": 5,
            "cost_usd": 0.0021,
            "duration_ms": 900,
            "ttft_ms": 400,
            "speed": "normal",
            "effort": "medium",
        },
    )
    return {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": _attrs(
                        {"service.name": "claude-code", "terminal.session": f"{label}-1"}
                    )
                },
                "scopeLogs": [
                    {
                        "scope": {"name": "com.anthropic.claude_code.events"},
                        "logRecords": [decision, usage],
                    }
                ],
            }
        ]
    }


# --------------------------------------------------------------------------- #
# The deployment
# --------------------------------------------------------------------------- #


@dataclass
class Deployment:
    """The built hello-world repo and what the cases address it by."""

    repo: Path
    osprey: Path
    env: dict[str, str]
    shims: Path
    config: dict[str, Any] = field(default_factory=dict)
    stand_ins: list[subprocess.Popen] = field(default_factory=list)

    @property
    def config_path(self) -> Path:
        return self.repo / "build" / "config.yml"

    @property
    def runtime_log(self) -> Path:
        return self.shims / "runtime.log"

    @property
    def running(self) -> Path:
        return self.shims / "running"

    @property
    def port(self) -> int:
        return int(self.config["agent_record"]["port"])

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def state_dir(self) -> Path:
        return hr.host_state_dir(self.repo, self.config)

    @property
    def host_identity(self) -> str:
        host = ids.host_identity(self.config)
        assert host, "the build recorded no host identity"
        return host

    @property
    def host_projects(self) -> Path:
        return claude_project_dir(self.repo / "build")

    def login(self, identity: str) -> tuple[str, str]:
        login = ids.login_for(identity)
        token = creds.read_login(self.repo / creds.login_file(login), login)
        assert token, f"the build minted no {login} login"
        return login, token

    def host_client(self) -> RecordClient:
        return RecordClient(self.url, *self.login(self.host_identity))

    def operator_client(self) -> RecordClient:
        return RecordClient(self.url, *self.login(ids.OPERATOR_IDENTITY))

    def reload(self) -> None:
        self.config = yaml.safe_load(self.config_path.read_text())


def _osprey_bin() -> Path:
    candidate = Path(sys.executable).parent / "osprey"
    if candidate.exists():
        return candidate
    found = shutil.which("osprey")
    if found:
        return Path(found)
    raise RuntimeError("Could not locate the 'osprey' console script.")


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o755)


def _shims(base: Path) -> Path:
    shims = base / "shims"
    (shims / "running").mkdir(parents=True)
    runtime = _RUNTIME.format(log=shims / "runtime.log", running=shims / "running")
    for name in ("docker", "podman"):
        _write_executable(shims / name, runtime)
    tests_root = str(Path(__file__).resolve().parents[2])
    _write_executable(
        shims / "claude",
        _CLAUDE.format(python=sys.executable, tests_root=tests_root, version=CLAUDE_VERSION),
    )
    return shims


def _run(
    cmd: list[str], cwd: Path, env: Mapping[str, str], timeout: float = VERB_TIMEOUT_S
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=dict(env),
        stdin=subprocess.DEVNULL,
    )


def _ok(result: subprocess.CompletedProcess, what: str) -> subprocess.CompletedProcess:
    if result.returncode != 0:
        pytest.fail(
            f"{what} failed (rc={result.returncode}):\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    return result


def _deploy(
    base: Path, name: str, env: dict[str, str], shims: Path, port: int, **edits: Any
) -> Deployment:
    """``osprey init --preset hello-world`` and ``osprey build``, nothing more.

    *port* is the record's: the host-local process binds it long after the
    config names it, so the caller holds it reserved for the deployment's life.
    """
    from tests.e2e.profile_edits import set_pairs

    osprey = _osprey_bin()
    repo = base / name
    pins = {"port_base": PORT_BASE, "agent_record": {"port": port, **edits}}
    _ok(
        _run(
            [str(osprey), "init", str(repo), "--preset", "hello-world", "--no-git"]
            + set_pairs(pins),
            cwd=base,
            env=env,
        ),
        "osprey init",
    )
    _ok(
        _run([str(osprey), "build", "--skip-deps", "--skip-lifecycle"], cwd=repo, env=env),
        "osprey build",
    )
    dep = Deployment(repo=repo, osprey=osprey, env=env, shims=shims)
    dep.reload()
    return dep


@pytest.fixture(scope="module")
def module_env(tmp_path_factory) -> Iterator[tuple[Path, dict[str, str], Path]]:
    """The module's Claude Code state, home and ``PATH``, in this process and its verbs."""
    base = tmp_path_factory.mktemp("hostrec")
    config_dir = base / "claude-config"
    home = base / "home"
    config_dir.mkdir()
    home.mkdir()
    shims = _shims(base)
    env = {k: v for k, v in os.environ.items() if k not in _SCRUBBED_ENV}
    env[CLAUDE_CONFIG_DIR_ENV] = str(config_dir)
    env["HOME"] = str(home)
    env["PATH"] = f"{shims}{os.pathsep}{env.get('PATH', '')}"
    # A web launch opens no browser.
    env["BROWSER"] = "true"
    with pytest.MonkeyPatch.context() as mp:
        for key in _SCRUBBED_ENV:
            mp.delenv(key, raising=False)
        for key in (CLAUDE_CONFIG_DIR_ENV, "PATH", "BROWSER"):
            mp.setenv(key, env[key])
        yield base, env, shims


@pytest.fixture(scope="module")
def hello_world(module_env) -> Iterator[Deployment]:
    base, env, shims = module_env
    # Imported here: the claude shim imports this module where the harness
    # package is not on the path.
    from osprey_connectors_tests._harness._ports import reserved_port

    with reserved_port() as port:
        dep = _deploy(base, "hello", env, shims, port)
        assert dep.config["agent_record"]["backend"] == "embedded"
        try:
            yield dep
        finally:
            _stop_everything(dep)


@pytest.fixture(scope="module")
def clickhouse(module_env) -> Iterator[Deployment]:
    base, env, shims = module_env
    # Imported here: the claude shim imports this module where the harness
    # package is not on the path.
    from osprey_connectors_tests._harness._ports import reserved_port

    with reserved_port() as port:
        dep = _deploy(base, "clickhouse", env, shims, port, backend="clickhouse")
        assert dep.config["agent_record"]["backend"] == "clickhouse"
        try:
            yield dep
        finally:
            _stop_everything(dep)


def _stop_everything(dep: Deployment) -> None:
    """Whatever a case did, leave no process and no declared container behind."""
    for proc in dep.stand_ins:
        _terminate(proc)
    dep.stand_ins.clear()
    for path in dep.running.iterdir():
        path.unlink()
    state_dir = dep.state_dir
    if state_dir.exists():
        with contextlib.suppress(hr.HostRecordRefused):
            with hr.host_record_lock(state_dir, wait_s=30):
                (state_dir / hr.MARKER_FILE_NAME).unlink(missing_ok=True)
                hr.stop_host_process(state_dir, wait_s=10)


@pytest.fixture
def web_port() -> Iterator[int]:
    """The port a case's ``osprey web`` is told to bind: named on its command
    line before the server exists, so reserved for the case."""
    # Imported here: the claude shim imports this module where the harness
    # package is not on the path.
    from osprey_connectors_tests._harness._ports import reserved_port

    with reserved_port() as port:
        yield port


@pytest.fixture
def dep(hello_world) -> Iterator[Deployment]:
    """The hello-world deployment, with no record running before or after the case."""
    _stop_everything(hello_world)
    assert _record_down(hello_world), "a record answers before the case started"
    yield hello_world
    _stop_everything(hello_world)


# --------------------------------------------------------------------------- #
# Processes and the record
# --------------------------------------------------------------------------- #


def _terminate(proc: subprocess.Popen, wait_s: float = 30.0) -> None:
    if proc.poll() is None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=wait_s)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)


def _await(check: Callable[[], Any], timeout: float, what: str) -> Any:
    deadline = time.monotonic() + timeout
    last: Any = None
    while time.monotonic() < deadline:
        last = check()
        if last:
            return last
        time.sleep(0.5)
    pytest.fail(f"{what} within {timeout:.0f} s; last: {last!r}")


def _record_down(dep: Deployment) -> bool:
    try:
        httpx.get(f"{dep.url}/health", timeout=2, trust_env=False)
    except httpx.ConnectError:
        return True
    return False


def _assert_no_record_left(dep: Deployment) -> None:
    """No host-local process, no lease, nothing on the port."""
    assert hr._running_process(dep.state_dir) is None, "a host-local record is still running"
    assert hr.live_leases(dep.state_dir / hr.LEASE_DIR_NAME) == []
    _await(lambda: _record_down(dep), 30, "the record port to close")


def _assert_no_container_started(dep: Deployment) -> None:
    calls = dep.runtime_log.read_text().splitlines() if dep.runtime_log.exists() else []
    starting = {"run", "start", "create", "up", "restart"}
    starts = [
        call
        for call in calls
        if (words := call.split())
        and (words[0] in starting or (words[0] == "compose" and starting & set(words)))
    ]
    assert starts == [], f"a container runtime was asked to start something: {starts}"


def _session_read(client: RecordClient, session: str) -> session_record.SessionRead:
    return session_record.read_timeline(session, client=client)


def _assert_session_in_record(dep: Deployment, client: RecordClient, session: str, label: str):
    """The session log, the session list, the chat's thinking, and OTLP."""

    def _tool_call() -> list[dict]:
        read = _session_read(client, session)
        events = session_record.session_events(read.nodes, session)
        return [e for e in events if e["type"] == "tool_call" and e["tool"] == "channel_read"]

    calls = _await(_tool_call, READ_TIMEOUT_S, f"the {label} session's tool call in its log")
    assert len(calls) == 1, calls
    assert calls[0]["server"] == "controls"

    rows, answer = session_record.own_sessions(dep.host_identity, client=client)
    assert answer.available, answer.unavailable
    assert session in [row["session_id"] for row in rows], rows

    read = _session_read(client, session)
    assert read.record == session_record.RECORD_COMPLETE, read.reason
    turns = session_routes._chat_turns(read.nodes)
    thinking = [
        event["content"]
        for turn in turns
        for event in turn.get("events") or []
        if event.get("type") == "thinking"
    ]
    assert thinking == [thinking_text(label)], turns

    _assert_otlp_in_record(dep, session)


def _session_entries(dep: Deployment, kind: str, session: str) -> list[dict]:
    answer = dep.operator_client().entries(kind, wait=Drain())
    if not answer.available or not isinstance(answer.data, list):
        return []
    return [e for e in answer.data if isinstance(e, dict) and e.get("session_id") == session]


def _assert_otlp_in_record(dep: Deployment, session: str) -> None:
    for kind in (KIND_DECISION, KIND_USAGE):
        found = _await(
            lambda kind=kind: _session_entries(dep, kind, session),
            READ_TIMEOUT_S,
            f"the session's OTLP {kind} event in the record",
        )
        assert len(found) == 1, found


# --------------------------------------------------------------------------- #
# A host session holder, and the stack's record
# --------------------------------------------------------------------------- #


class Holder:
    """A host session (``host_record_launch``) in its own process."""

    def __init__(self, dep: Deployment):
        self.proc = subprocess.Popen(
            [sys.executable, "-c", _HOLDER, str(dep.repo)],
            cwd=str(dep.repo),
            env=dep.env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
        self.dep = dep
        self.hello = self._read()

    def _read(self, timeout: float = 240.0) -> dict:
        assert self.proc.stdout is not None
        ready, _, _ = select.select([self.proc.stdout], [], [], timeout)
        assert ready, "the host session did not answer"
        line = self.proc.stdout.readline()
        assert line, f"the host session ended (rc={self.proc.poll()})"
        return json.loads(line)

    def step(self, session: str, message: str) -> str:
        """A spool step from the session; its record key."""
        assert self.proc.stdin is not None
        self.proc.stdin.write(f"step {session} {message}\n")
        self.proc.stdin.flush()
        entry = self._read()["entry"]
        assert entry, "the writer stored nothing"
        return spool_key(self.dep.host_identity, entry)

    def end(self) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write("end\n")
        self.proc.stdin.flush()
        assert self._read() == {"released": True}
        assert self.proc.wait(timeout=60) == 0

    def kill(self) -> None:
        _terminate(self.proc)


@contextlib.contextmanager
def _holders(dep: Deployment, count: int = 1) -> Iterator[list[Holder]]:
    started: list[Holder] = []
    try:
        for _ in range(count):
            started.append(Holder(dep))
        yield started
    finally:
        for holder in started:
            holder.kill()


def _stand_in(dep: Deployment) -> subprocess.Popen:
    """The stack's agent-record: the same service over the same store, with no lease
    watch and, like the container, no read of the host's transcripts."""
    from osprey.agent_record import service

    env = hr._child_env(dep.repo, dep.config, dep.state_dir, dep.config_path)
    env.pop(service.HOST_PROJECTS_ENV)
    log = (dep.repo / "var" / "stack-agent-record.log").open("ab")
    proc = subprocess.Popen(
        [sys.executable, "-m", "osprey.cli.main", "agent-record", "serve", "--port", str(dep.port)],
        cwd=str(dep.repo),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log.close()
    dep.stand_ins.append(proc)
    return proc


def _container_name(dep: Deployment) -> str:
    from osprey.deployment.container_lifecycle import _record_container_name

    return _record_container_name(dep.config)


def osprey_up(dep: Deployment) -> subprocess.Popen:
    """What ``osprey up`` does to the record: take it over from the host, start the stack's."""
    from osprey.deployment.container_lifecycle import _stack_takes_over_host_record

    with _stack_takes_over_host_record(dep.config, dep.repo):
        proc = _stand_in(dep)
        (dep.running / _container_name(dep)).touch()
    assert hr.marker_is_live(dep.state_dir)
    return proc


def osprey_down(dep: Deployment, stack: subprocess.Popen) -> None:
    """What ``osprey down`` does to the record: stop the stack's, hand the record back."""
    from osprey.deployment.container_lifecycle import _hand_record_back_to_host

    _terminate(stack)
    (dep.running / _container_name(dep)).unlink()
    _hand_record_back_to_host(dep.config, dep.repo)
    assert not (dep.state_dir / hr.MARKER_FILE_NAME).exists()


def _listener_pid(port: int) -> int | None:
    out = subprocess.run(
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    return int(out[0]) if out else None


def _keys_in(dep: Deployment, session: str) -> set[str]:
    read = _session_read(dep.operator_client(), session)
    if read.record != session_record.RECORD_COMPLETE:
        return set()
    keys: set[str] = set()
    stack = list(read.nodes)
    while stack:
        node = stack.pop()
        if node.get("key"):
            keys.add(node["key"])
        for joined in ("result", "decision"):
            if isinstance(node.get(joined), dict) and node[joined].get("key"):
                keys.add(node[joined]["key"])
    return keys


# --------------------------------------------------------------------------- #
# No stack: hello-world's host sessions
# --------------------------------------------------------------------------- #


def _web_login(stdout_path: Path) -> str:
    def _token() -> str | None:
        match = re.search(r"\?token=([A-Za-z0-9_\-]+)", stdout_path.read_text())
        return match.group(1) if match else None

    return _await(_token, 120, "osprey web to print its login URL")


def test_a_host_web_session_records_without_a_container(dep, tmp_path, web_port):
    port = web_port
    out = tmp_path / "web.out"
    with out.open("w") as sink:
        web = subprocess.Popen(
            [str(dep.osprey), "web", "--port", str(port), "--shell", "cat", "--skip-preflight"],
            cwd=str(dep.repo),
            env=dep.env,
            stdin=subprocess.DEVNULL,
            stdout=sink,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        token = _web_login(out)
        base = f"http://127.0.0.1:{port}"
        with httpx.Client(base_url=base, trust_env=False, timeout=30) as browser:

            def _signed_in() -> bool:
                try:
                    return browser.get(f"/?token={token}").status_code in (200, 303)
                except httpx.ConnectError:
                    return False

            _await(_signed_in, 120, "the web login")
            assert hr._running_process(dep.state_dir) is not None, "web started no record"

            # Claude Code in the web terminal: its transcript in build/'s project
            # directory, its OTLP export to the record with the host login.
            session = str(uuid.uuid4())
            tool_use_id = f"toolu_{uuid.uuid4().hex[:20]}"
            dep.host_projects.mkdir(parents=True, exist_ok=True)
            (dep.host_projects / f"{session}.jsonl").write_text(
                transcript(session, tool_use_id, "web")
            )
            login, secret = dep.login(dep.host_identity)
            exported = httpx.post(
                f"{dep.url}/v1/logs",
                json=otlp_request(session, tool_use_id, "web"),
                auth=(login, secret),
                trust_env=False,
            )
            assert exported.status_code == 200, exported.text

            def _logged() -> list[dict]:
                body = browser.get("/api/session-log").json()
                return [e for e in body.get("events", []) if e.get("tool") == "channel_read"]

            calls = _await(_logged, READ_TIMEOUT_S, "the web session log to show the tool call")
            assert len(calls) == 1 and calls[0]["session_id"] == session, calls

            listed = browser.get("/api/sessions").json()
            assert listed.get("record") == "complete", listed
            assert session in [s["session_id"] for s in listed["sessions"]], listed

            chat = browser.get(
                "/api/session-chat", params={"session_id": session, "events": "1"}
            ).json()
            assert chat.get("record") == "complete", chat
            thinking = [
                e["content"]
                for turn in chat["turns"]
                for e in turn.get("events") or []
                if e.get("type") == "thinking"
            ]
            assert thinking == [thinking_text("web")], chat

            _assert_otlp_in_record(dep, session)
    finally:
        _terminate(web)
    assert web.returncode is not None
    _assert_no_record_left(dep)
    _assert_no_container_started(dep)


def test_a_host_chat_session_records_without_a_container(dep, tmp_path):
    report = tmp_path / "claude-report.json"
    release = tmp_path / "claude-release"
    env = {**dep.env, "HSA_CLAUDE_REPORT": str(report), "HSA_CLAUDE_RELEASE": str(release)}
    chat = subprocess.Popen(
        [str(dep.osprey), "chat", "--repo", str(dep.repo), "--no-pin"],
        cwd=str(dep.repo),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        _await(report.exists, 240, "the chat's agent to take its turn")
        seen = json.loads(report.read_text())
        assert seen["otlp_status"] == 200, seen
        assert seen["env"]["OSPREY_AGENT_RECORD_URL"] == dep.url
        assert seen["env"]["OTEL_EXPORTER_OTLP_LOGS_ENDPOINT"] == f"{dep.url}/v1/logs"
        assert hr._running_process(dep.state_dir) is not None, "chat started no record"
        _assert_session_in_record(dep, dep.host_client(), seen["session"], "chat")
        release.touch()
        output, _ = chat.communicate(timeout=120)
        assert chat.returncode == 0, output
    finally:
        release.touch()
        _terminate(chat)
    _assert_no_record_left(dep)
    _assert_no_container_started(dep)


def test_a_host_query_records_without_a_container(dep, monkeypatch):
    from osprey.cli.query_cmd import query

    seen: dict[str, Any] = {}

    async def _agent_turn(*_args, **_kwargs):
        # The query's agent: Claude Code writes its transcript and exports
        # OTLP with the environment the launcher published.
        session = str(uuid.uuid4())
        tool_use_id = f"toolu_{uuid.uuid4().hex[:20]}"
        dep.host_projects.mkdir(parents=True, exist_ok=True)
        (dep.host_projects / f"{session}.jsonl").write_text(
            transcript(session, tool_use_id, "query")
        )
        header = os.environ["OTEL_EXPORTER_OTLP_HEADERS"].removeprefix("Authorization=")
        exported = httpx.post(
            os.environ["OTEL_EXPORTER_OTLP_LOGS_ENDPOINT"],
            json=otlp_request(session, tool_use_id, "query"),
            headers={"Authorization": header},
            trust_env=False,
        )
        seen.update(session=session, otlp=exported.status_code)
        client = RecordClient.from_env(os.environ)
        _assert_session_in_record(dep, client, session, "query")
        seen["checked"] = True

        class _Result:
            text_blocks = ["The query channel reads 0.41."]
            is_error = False
            mcp_servers: list = []
            tool_traces: list = []

        return _Result()

    monkeypatch.chdir(dep.repo)
    with (
        patch("osprey.cli.query_cmd.run_query", side_effect=_agent_turn),
        patch("osprey.cli.query_cmd.read_only_disallowed_tools", return_value=[]),
        patch("osprey.cli.query_cmd.expected_mcp_servers", return_value=set()),
        patch("osprey.cli.query_cmd.evaluate_verdict", return_value=0),
    ):
        result = CliRunner().invoke(query, ["--repo", str(dep.repo), "What does it read?"])

    assert result.exit_code == 0, (result.output, result.exception)
    assert seen.get("checked"), seen
    assert seen["otlp"] == 200
    assert URL_ENV not in os.environ, "the query left its record in this process's environment"
    _assert_no_record_left(dep)
    _assert_no_container_started(dep)


def test_feedback_list_answers_and_leaves_no_process(dep):
    _ok(_run([str(dep.osprey), "feedback", "list"], cwd=dep.repo, env=dep.env), "feedback list")
    _assert_no_record_left(dep)
    _assert_no_container_started(dep)


# --------------------------------------------------------------------------- #
# ClickHouse with no stack
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "verb",
    [
        pytest.param(["web", "--skip-preflight", "--shell", "cat"], id="web"),
        pytest.param(["chat", "--no-pin"], id="chat"),
        pytest.param(["query", "hello"], id="query"),
    ],
)
def test_a_clickhouse_project_with_no_stack_refuses_with_the_fix(clickhouse, verb, web_port):
    args = list(verb)
    if args[0] == "web":
        args += ["--port", str(web_port)]
    else:
        args[1:1] = ["--repo", str(clickhouse.repo)]
    result = _run([str(clickhouse.osprey), *args], cwd=clickhouse.repo, env=clickhouse.env)

    assert result.returncode != 0, result.stdout
    said = " ".join((result.stdout + result.stderr).split())
    assert "No agent record" in said, said
    assert "osprey up" in said and "agent_record.backend: embedded" in said, said
    assert list(clickhouse.repo.rglob("record.duckdb")) == []
    assert hr._running_process(clickhouse.state_dir) is None
    _assert_no_container_started(clickhouse)


# --------------------------------------------------------------------------- #
# With a stack
# --------------------------------------------------------------------------- #


def test_a_launcher_uses_the_stacks_record(dep):
    stack = osprey_up(dep)
    with _holders(dep) as (session,):
        assert session.hello == {"url": dep.url, "started": False}
        assert hr._running_process(dep.state_dir) is None, "a host-local record started"
        assert _listener_pid(dep.port) == stack.pid
        sid = str(uuid.uuid4())
        key = session.step(sid, "on-the-stack")
        _await(lambda: key in _keys_in(dep, sid), READ_TIMEOUT_S, "the step in the stack's record")
        session.end()
    assert stack.poll() is None, "the last host session stopped the stack's record"
    osprey_down(dep, stack)
    _assert_no_record_left(dep)


def test_a_host_chat_under_the_stack_reaches_its_record_through_the_spool(dep, tmp_path):
    stack = osprey_up(dep)
    report = tmp_path / "claude-report.json"
    release = tmp_path / "claude-release"
    env = {**dep.env, "HSA_CLAUDE_REPORT": str(report), "HSA_CLAUDE_RELEASE": str(release)}
    chat = subprocess.Popen(
        [str(dep.osprey), "chat", "--repo", str(dep.repo), "--no-pin"],
        cwd=str(dep.repo),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        _await(report.exists, 240, "the chat's agent to take its turn")
        seen = json.loads(report.read_text())
        assert seen["env"]["OSPREY_AGENT_RECORD_URL"] == dep.url
        assert hr._running_process(dep.state_dir) is None, "chat started a host-local record"
        # The stack's record reads no host transcript; the chat forwards it.
        sources = dep.host_client().health_sources()
        assert sources.available, sources.unavailable
        assert "host" not in sources.data["transcripts"], sources.data
        _assert_session_in_record(dep, dep.host_client(), seen["session"], "chat")
        assert hr.FORWARD_STATE_FILE_NAME in os.listdir(dep.state_dir)
        release.touch()
        output, _ = chat.communicate(timeout=120)
        assert chat.returncode == 0, output
    finally:
        release.touch()
        _terminate(chat)
    assert stack.poll() is None, "the chat's end stopped the stack's record"
    osprey_down(dep, stack)
    _assert_no_record_left(dep)
    _assert_no_container_started(dep)


def test_a_host_session_keeps_its_record_through_up_and_down(dep):
    sid = str(uuid.uuid4())
    with _holders(dep) as (session,):
        assert session.hello == {"url": dep.url, "started": True}
        host_local = hr._running_process(dep.state_dir)
        assert host_local is not None
        before = session.step(sid, "before-up")
        _await(lambda: before in _keys_in(dep, sid), READ_TIMEOUT_S, "the first step")

        stack = osprey_up(dep)
        assert hr._running_process(dep.state_dir) is None, "the host-local record kept running"
        assert not hr._alive(*host_local)
        assert _listener_pid(dep.port) == stack.pid
        during = session.step(sid, "after-up")
        _await(
            lambda: {before, during} <= _keys_in(dep, sid),
            READ_TIMEOUT_S,
            "the step taken after `osprey up` in the stack's record",
        )
        # Never spawned under the stack, though the session's supervisor ran.
        time.sleep(hr.SUPERVISE_INTERVAL_S * 2)
        assert hr._running_process(dep.state_dir) is None
        assert _listener_pid(dep.port) == stack.pid

        osprey_down(dep, stack)
        # The session's supervisor brings the host-local record back, and the
        # next read answers with what the stack committed.
        back = _await(
            lambda: hr._running_process(dep.state_dir),
            RESTART_TIMEOUT_S,
            "the host-local record to come back after `osprey down`",
        )
        assert back != host_local
        _await(
            lambda: {before, during} <= _keys_in(dep, sid),
            RESTART_TIMEOUT_S,
            "the next read after `osprey down` to answer with both steps",
        )
        session.end()
    _assert_no_record_left(dep)
    _assert_no_container_started(dep)


def test_two_concurrent_host_sessions_share_one_record_and_leave_none(dep):
    with _holders(dep, 2) as (first, second):
        assert first.hello["url"] == second.hello["url"] == dep.url
        assert sorted([first.hello["started"], second.hello["started"]]) == [False, True]
        running = hr._running_process(dep.state_dir)
        assert running is not None
        assert _listener_pid(dep.port) == running[0]
        assert len(hr.live_leases(dep.state_dir / hr.LEASE_DIR_NAME)) == 2

        sid = str(uuid.uuid4())
        keys = {first.step(sid, "first"), second.step(sid, "second")}
        _await(lambda: keys <= _keys_in(dep, sid), READ_TIMEOUT_S, "both sessions' steps")

        first.end()
        assert hr._running_process(dep.state_dir) == running, "a live session lost its record"
        second.end()
    _assert_no_record_left(dep)
    _assert_no_container_started(dep)
