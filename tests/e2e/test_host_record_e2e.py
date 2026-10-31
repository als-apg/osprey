"""The host-local agent record, live: hello-world's host sessions, then a stack.

Reaches a model (the e2e provider) and runs real containers, so it carries no
``model_free`` marker and runs only under the host e2e lock.

One deployment: ``osprey init --preset hello-world`` and ``osprey build
--dev``, with ``CLAUDE_CONFIG_DIR`` and ``HOME`` pointed into the module's temp
tree, ``deployed_services`` pinned empty and a stated
``deployment.host_identity``. The cases run in file order:

1. No ``osprey up``. A live ``osprey query`` and a live ``osprey chat --print``
   each leave their session in the record: a tool call in the session log, the
   session in the session list, thinking in the Simple view's turns, and their
   OTLP decision and usage events. A host ``osprey web`` reads the same
   sessions through its own routes. ``osprey feedback list`` answers. No
   container is started and no record process is left after any of them.
2. A host session (``host_record_launch``, in its own process) starts first.
   ``osprey up`` takes the record over: the host-local process stops, the
   stack's container answers, and the session's next step is committed by
   it. A live ``osprey query`` uses the stack's record and starts none, and
   its session reaches that record through the spool: the stack's container
   reads no host transcript, the host session forwards it. A second host
   session shares the record. ``osprey down`` mid-session: the session's
   supervisor starts the host-local record again and the next read answers
   with everything the stack committed. When the last session ends no
   record process is left.

The model-free variant, with every launcher and the handover played without
containers, is ``tests/agent_record/test_host_record_integration.py``.

CONTAINER SAFETY: every docker invocation names an exact container or volume
of this project. Teardown is ``osprey down`` followed by the project-scoped
volume sweep (``tests/e2e/_volumes.py``); never a prune, never a wildcard.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import pwd
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import httpx
import pytest

from osprey.agent_record import credentials as creds
from osprey.agent_record import host as hr
from osprey.agent_record.client import URL_ENV, RecordClient
from osprey.agent_record.contract import KIND_DECISION, KIND_USAGE, SPOOL_DIR_ENV, Drain
from osprey.agent_runner.project_paths import CLAUDE_CONFIG_DIR_ENV
from osprey.interfaces.web_terminal.routes import session as session_routes
from osprey.mcp_server import session as session_record
from osprey_connectors_tests._harness._ports import reserved_port
from tests.agent_record.test_host_record_integration import (
    Deployment,
    Holder,
    _await,
    _holders,
    _keys_in,
    _record_down,
    _terminate,
    _web_login,
)
from tests.e2e._facility_select import repo_view, seeded_readback
from tests.e2e._volumes import remove_project_volumes
from tests.e2e.profile_edits import set_pairs
from tests.e2e.provider import build_model, build_provider, e2e_provider

pytestmark = [
    pytest.mark.harness_benchmark,
    pytest.mark.slow,
    pytest.mark.requires_e2e_provider,
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available"),
    pytest.mark.skipif(shutil.which("claude") is None, reason="claude CLI not available"),
]

#: This module's compose project. A compose project is addressed by name, so
#: no other e2e module may deploy under it (``tests/deployment/test_e2e_project_names.py``).
PROJECT_NAME = "osprey-e2e-host-record"
#: This module's port block (``deployment.port_base``); no other e2e module uses it.
PORT_BASE = 25600
#: One deployment of this module per host at a time: an advisory ``flock`` the
#: operating system releases when its holder dies.
STACK_LOCK = Path(tempfile.gettempdir()) / f"osprey-{PROJECT_NAME}.lock"
HOST_IDENTITY = "e2e-host"
RECORD_CONTAINER = f"{PROJECT_NAME}-agent-record"
RECORD_STORE_CONTAINER = f"{PROJECT_NAME}-agent-record-clickhouse"

#: ``osprey up`` builds the project image on its first run.
VERB_TIMEOUT_S = 1800
#: One live agent turn.
AGENT_TIMEOUT_S = 420
#: From a fact being written to it being readable.
READ_TIMEOUT_S = 120.0
#: The supervisor probes every few seconds; a restart then waits out the
#: host-local process's first pass over the transcripts.
RESTART_TIMEOUT_S = 180.0

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
)


# --------------------------------------------------------------------------- #
# The deployment
# --------------------------------------------------------------------------- #


def _osprey_bin() -> Path:
    candidate = Path(sys.executable).parent / "osprey"
    if candidate.exists():
        return candidate
    found = shutil.which("osprey")
    if found:
        return Path(found)
    raise RuntimeError("Could not locate the 'osprey' console script.")


def _real_home() -> Path:
    return Path(pwd.getpwuid(os.getuid()).pw_dir)


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


def _verb(
    dep: Deployment, *args: str, timeout: float = VERB_TIMEOUT_S
) -> subprocess.CompletedProcess:
    """Run an ``osprey`` verb in the repo; ``pytest.fail`` on a non-zero exit."""
    result = _run([str(dep.osprey), *args], cwd=dep.repo, env=dep.env, timeout=timeout)
    if result.returncode != 0:
        pytest.fail(
            f"osprey {' '.join(args)} failed (rc={result.returncode}):\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    return result


def _docker(*args: str, timeout: int = 120, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=timeout, check=False
    )
    if check and result.returncode != 0:
        raise AssertionError(f"docker {' '.join(args)} failed: {result.stderr.strip()}")
    return result


def _remove_project_containers() -> None:
    """Remove this project's own containers by exact name, should a run leave any."""
    for name in (RECORD_CONTAINER, RECORD_STORE_CONTAINER):
        _docker("rm", "-f", name, check=False)


def _project_containers() -> list[str]:
    """Every container, running or not, whose name this project's compose would give it."""
    out = _docker("ps", "-a", "--filter", f"name=^{PROJECT_NAME}-", "--format", "{{.Names}}").stdout
    return sorted(line for line in out.splitlines() if line.strip())


def _container_running(name: str) -> bool:
    out = _docker("inspect", "-f", "{{.State.Running}}", name, check=False)
    return out.returncode == 0 and out.stdout.strip() == "true"


@contextlib.contextmanager
def _module_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@pytest.fixture(scope="module")
def deployment(tmp_path_factory) -> Iterator[Deployment]:
    base = tmp_path_factory.mktemp("hostrec-e2e")
    config_dir = base / "claude-config"
    home = base / "home"
    shims = base / "unused-shims"
    for path in (config_dir, home, shims / "running"):
        path.mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if k not in _SCRUBBED_ENV}
    env.setdefault("DOCKER_CONFIG", str(_real_home() / ".docker"))
    env[CLAUDE_CONFIG_DIR_ENV] = str(config_dir)
    env["HOME"] = str(home)
    env["BROWSER"] = "true"
    osprey = _osprey_bin()
    repo = base / PROJECT_NAME
    edits: dict[str, Any] = {
        "provider": build_provider(e2e_provider()),
        "model": build_model(None),
        "port_base": PORT_BASE,
        "agent_record": {"backend": "embedded"},
        "config": {
            "deployed_services": [],
            "deployment": {"host_identity": HOST_IDENTITY},
        },
    }
    with _module_lock(STACK_LOCK), pytest.MonkeyPatch.context() as mp:
        for key in _SCRUBBED_ENV:
            mp.delenv(key, raising=False)
        mp.setenv(CLAUDE_CONFIG_DIR_ENV, str(config_dir))
        _remove_project_containers()
        remove_project_volumes(PROJECT_NAME)
        init = _run(
            [str(osprey), "init", str(repo), "--preset", "hello-world", "--no-git"]
            + set_pairs(edits),
            cwd=base,
            env=env,
        )
        if init.returncode != 0:
            pytest.fail(f"osprey init failed (rc={init.returncode}):\n{init.stdout}\n{init.stderr}")
        dep = Deployment(repo=repo, osprey=osprey, env=env, shims=shims)
        try:
            _verb(dep, "build", "--skip-deps", "--skip-lifecycle", "--dev")
            dep.reload()
            assert dep.host_identity == HOST_IDENTITY
            assert dep.config["agent_record"]["backend"] == "embedded"
            yield dep
        finally:
            down = _run([str(osprey), "down"], cwd=repo, env=env, timeout=300)
            if down.returncode != 0:
                print(f"osprey down rc={down.returncode}\n{down.stdout}\n{down.stderr}")
            _remove_project_containers()
            remove_project_volumes(PROJECT_NAME)
            if dep.state_dir.exists():
                with contextlib.suppress(hr.HostRecordRefused):
                    with hr.host_record_lock(dep.state_dir, wait_s=30):
                        hr.stop_host_process(dep.state_dir, wait_s=10)
            shutil.rmtree(repo / "build" / ".venv", ignore_errors=True)


# --------------------------------------------------------------------------- #
# Reading the record
# --------------------------------------------------------------------------- #


def _assert_no_record_left(dep: Deployment) -> None:
    assert hr._running_process(dep.state_dir) is None, "a host-local record is still running"
    assert hr.live_leases(dep.state_dir / hr.LEASE_DIR_NAME) == []
    _await(lambda: _record_down(dep), 30, "the record port to close")


@contextlib.contextmanager
def _reader(dep: Deployment) -> Iterator[RecordClient]:
    """The operator's client, on whatever record answers (a one-shot host-local one if none)."""
    with hr.ensure_host_record(dep.repo, dep.config, config_path=dep.config_path):
        yield dep.operator_client()


def _session_ids(client: RecordClient) -> set[str]:
    answer = client.list_sessions(wait=Drain())
    assert answer.available, answer.unavailable
    return {row["session_id"] for row in answer.data or [] if isinstance(row, dict)}


def _kind_entries(client: RecordClient, kind: str, session: str) -> list[dict]:
    answer = client.entries(kind, wait=Drain())
    if not answer.available or not isinstance(answer.data, list):
        return []
    return [e for e in answer.data if isinstance(e, dict) and e.get("session_id") == session]


def _transcript_ingest_state(client: RecordClient) -> str:
    """What the record says of its transcript roots, and what the stack's container sees there."""
    from osprey.agent_record.contract import HEALTH_SOURCES_PATH

    try:
        health = httpx.get(
            f"{client.url}{HEALTH_SOURCES_PATH}",
            auth=(client.login, client._password),
            timeout=10,
            trust_env=False,
        ).text[:6000]
    except httpx.HTTPError as exc:
        health = f"unreadable: {exc}"
    seen = ""
    if _container_running(RECORD_CONTAINER):
        listed = _docker(
            "exec",
            RECORD_CONTAINER,
            "sh",
            "-c",
            'id; ls -lnaR "$OSPREY_AGENT_RECORD_TRANSCRIPTS_DIR" | head -80',
            check=False,
        )
        seen = f"{listed.stdout}{listed.stderr}"
    return f"--- record health ---\n{health}\n--- {RECORD_CONTAINER} ---\n{seen}"


def _assert_session_recorded(client: RecordClient, session: str) -> None:
    """A tool call in the session log, the session listed, thinking, decisions and usage."""

    def _calls() -> list[dict]:
        read = session_record.read_timeline(session, client=client)
        events = session_record.session_events(read.nodes, session)
        return [e for e in events if e["type"] == "tool_call"]

    try:
        calls = _await(_calls, READ_TIMEOUT_S, f"session {session}'s tool calls in its log")
    except pytest.fail.Exception as exc:
        pytest.fail(f"{exc}\n{_transcript_ingest_state(client)}")
    assert all(e["server"] == "controls" for e in calls), calls

    rows, answer = session_record.own_sessions(HOST_IDENTITY, client=client)
    assert answer.available, answer.unavailable
    assert session in [row["session_id"] for row in rows], rows

    read = session_record.read_timeline(session, client=client)
    assert read.record == session_record.RECORD_COMPLETE, read.reason
    thinking = [
        e["content"]
        for turn in session_routes._chat_turns(read.nodes)
        for e in turn.get("events") or []
        if e.get("type") == "thinking"
    ]
    assert any(text.strip() for text in thinking), "no thinking in the Simple view's turns"

    for kind in (KIND_DECISION, KIND_USAGE):
        _await(
            lambda kind=kind: _kind_entries(client, kind, session),
            READ_TIMEOUT_S,
            f"session {session}'s OTLP {kind} events in the record",
        )


def _prompt(dep: Deployment) -> str:
    channel = seeded_readback(repo_view(dep.repo))
    return (
        "Think it through step by step first. Then read the current value of "
        f"{channel} with the controls tools and report it in one sentence."
    )


def _query(dep: Deployment) -> None:
    result = _verb(dep, "query", "--repo", str(dep.repo), "--json", _prompt(dep), timeout=600)
    payload = json.loads(result.stdout[result.stdout.find("{") :])
    assert payload["exit_code"] == 0, payload


def _chat(dep: Deployment) -> None:
    _verb(dep, "chat", "--repo", str(dep.repo), "--print", _prompt(dep), timeout=AGENT_TIMEOUT_S)


# --------------------------------------------------------------------------- #
# The cases, in order
# --------------------------------------------------------------------------- #


# The module fixture's ``osprey build --dev`` runs inside this case's budget,
# ahead of one live query, one live chat and the web login.
@pytest.fixture
def web_port() -> Iterator[int]:
    """The port the case's ``osprey web`` binds: named on its command line before
    the server exists, so reserved for the case."""
    with reserved_port() as port:
        yield port


@pytest.mark.timeout(3600)
def test_host_sessions_record_without_a_container(deployment, web_port):
    dep = deployment
    with _reader(dep) as client:
        before = _session_ids(client)
    _assert_no_record_left(dep)

    _query(dep)
    _assert_no_record_left(dep)
    _chat(dep)
    _assert_no_record_left(dep)
    assert _project_containers() == [], "a host session started a container"

    with _reader(dep) as client:
        new = _await(
            lambda: found if len(found := _session_ids(client) - before) >= 2 else None,
            READ_TIMEOUT_S,
            "the query's and the chat's sessions in the record",
        )
        assert len(new) == 2, new
        for session in new:
            _assert_session_recorded(client, session)
    _assert_no_record_left(dep)

    # A host web session reads the same sessions through its own routes.
    port = web_port
    out = dep.repo / "var" / "web-e2e.out"
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
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=60) as b:

            def _signed_in() -> bool:
                try:
                    return b.get(f"/?token={token}").status_code in (200, 303)
                except httpx.ConnectError:
                    return False

            _await(_signed_in, 120, "the web login")
            listed = b.get("/api/sessions").json()
            assert listed.get("record") == "complete", listed
            assert new <= {s["session_id"] for s in listed["sessions"]}, listed
            for session in new:
                chat = b.get(
                    "/api/session-chat", params={"session_id": session, "events": "1"}
                ).json()
                assert chat.get("record") == "complete", chat
                assert any(
                    e.get("type") == "thinking"
                    for turn in chat["turns"]
                    for e in turn.get("events") or []
                ), chat
            log = b.get("/api/session-log").json()
            assert log.get("record") == "complete", log
            assert log["events"], log
    finally:
        _terminate(web)
    _assert_no_record_left(dep)

    _verb(dep, "feedback", "list")
    _assert_no_record_left(dep)
    assert _project_containers() == [], "a host command started a container"


@pytest.mark.timeout(3600)
def test_the_stack_takes_the_record_over_and_hands_it_back(deployment):
    dep = deployment
    sid = str(uuid.uuid4())
    with _holders(dep) as (session,):
        assert session.hello == {"url": dep.url, "started": True}
        host_local = hr._running_process(dep.state_dir)
        assert host_local is not None
        first = session.step(sid, "before-up")
        _await(lambda: first in _keys_in(dep, sid), READ_TIMEOUT_S, "the first step")

        _verb(dep, "up", "-d", "--dev")
        assert _container_running(RECORD_CONTAINER)
        assert hr.marker_is_live(dep.state_dir)
        assert hr._running_process(dep.state_dir) is None, "the host-local record kept running"
        assert not hr._alive(*host_local)

        second = session.step(sid, "after-up")
        _await(
            lambda: {first, second} <= _keys_in(dep, sid),
            READ_TIMEOUT_S,
            "the step taken after `osprey up` in the stack's record",
        )

        # A live launcher uses the stack's record and starts none.
        before = _session_ids(dep.operator_client())
        _query(dep)
        assert hr._running_process(dep.state_dir) is None, "the query started a host record"
        stacked = _await(
            lambda: _session_ids(dep.operator_client()) - before,
            READ_TIMEOUT_S,
            "the query's session in the stack's record",
        )
        (query_session,) = stacked
        _assert_session_recorded(dep.operator_client(), query_session)

        # A second host session shares the stack's record.
        other = Holder(dep)
        try:
            assert other.hello == {"url": dep.url, "started": False}
            third = other.step(sid, "second-session")
            _await(lambda: third in _keys_in(dep, sid), READ_TIMEOUT_S, "the second's step")
        finally:
            other.end()
        assert _container_running(RECORD_CONTAINER), "a session's end stopped the stack"

        # `osprey down` mid-session: the session's supervisor starts the
        # host-local record again, and the next read answers.
        _verb(dep, "down")
        assert not _container_running(RECORD_CONTAINER)
        assert not (dep.state_dir / hr.MARKER_FILE_NAME).exists()
        back = _await(
            lambda: hr._running_process(dep.state_dir),
            RESTART_TIMEOUT_S,
            "the host-local record to come back after `osprey down`",
        )
        assert back != host_local
        _await(
            lambda: {first, second, third} <= _keys_in(dep, sid),
            RESTART_TIMEOUT_S,
            "the next read after `osprey down` to answer with what the stack committed",
        )
        assert query_session in _session_ids(dep.operator_client())
        session.end()
    _assert_no_record_left(dep)
    _remove_project_containers()
    remove_project_volumes(PROJECT_NAME)
    assert _project_containers() == []
