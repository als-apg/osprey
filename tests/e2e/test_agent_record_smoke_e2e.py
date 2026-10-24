"""The agent record end to end: one deployment, its record, every fact once.

Runs in the model-free lane: no test here names a provider or reaches a model.
The transcripts it reads are synthetic Claude Code 2.1.300-shaped files written
by the test itself.

Each case deploys the hello-world preset under this module's own project name
and port block, with no old telemetry store (``deployed_services`` pinned
empty), a stated ``deployment.host_identity`` and one ``agent_record.backend``,
and runs the real ``osprey build`` and ``osprey up``. Both verbs run with
``CLAUDE_CONFIG_DIR`` and ``HOME`` pointed into the case's temp tree; the host
build's Claude Code project directory is created there, holding a transcript,
BEFORE the build, so the render mounts it into agent-record. The developer's
own ``~/.claude`` gains nothing for this deployment.

The facts reach the record by every route a deployment has: the host
transcript (thinking, a tool call and a subagent file), one spool line written
through :func:`osprey.agent_record.writer.write`, and, in the embedded case, a
transcript a one-shot container writes into the dispatch workers' transcripts
volume. Everything is read back through the read API with the operator
credential, and every key must appear exactly once — before and after a
restart of agent-record. The two cases run one after the other, never side by
side: each tears its deployment down before it returns.

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
import time
import uuid
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from osprey.agent_record import writer
from osprey.agent_record.config import BACKEND_CLICKHOUSE, BACKEND_EMBEDDED
from osprey.agent_record.contract import (
    KIND_DEBUG,
    KIND_SUBAGENT,
    KIND_THINKING,
    KIND_TOOL_CALL,
    SPOOL_DIR_ENV,
    spool_key,
    transcript_key,
)
from osprey.agent_record.credentials import login_file, read_login
from osprey.agent_record.identities import (
    DISPATCH_TRANSCRIPT_VOLUME,
    OPERATOR_IDENTITY,
    transcript_roots,
)
from osprey.agent_runner.project_paths import (
    CLAUDE_CONFIG_DIR_ENV,
    claude_project_dir,
    encode_claude_project_path,
)
from osprey.deployment.compose_generator import resolve_project_name
from osprey.port_layout import default_port
from osprey_connectors.workspace import SPOOL_DIR_RELPATH
from tests.e2e._volumes import remove_project_volumes
from tests.e2e.profile_edits import set_pairs

pytestmark = [
    pytest.mark.model_free,
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available"),
]

#: This module's compose project. A compose project is addressed by name, so
#: no other e2e module may deploy under it (``tests/deployment/test_e2e_project_names.py``).
PROJECT_NAME = "osprey-e2e-agent-record"
#: This module's port block (``deployment.port_base``); no other e2e module uses it.
PORT_BASE = 19500
#: One deployment of this module per host at a time: an advisory ``flock`` the
#: operating system releases when its holder dies.
STACK_LOCK = Path(tempfile.gettempdir()) / f"osprey-{PROJECT_NAME}.lock"
#: The identity the host build's transcripts and spool are recorded under.
HOST_IDENTITY = "e2e-builder"

RECORD_CONTAINER = f"{PROJECT_NAME}-agent-record"
RECORD_STORE_CONTAINER = f"{PROJECT_NAME}-agent-record-clickhouse"
#: The old telemetry store's containers, which no deployment here may start.
OLD_STORE_CONTAINERS = (
    f"{PROJECT_NAME}-telemetry-store",
    f"{PROJECT_NAME}-telemetry-gateway",
)

#: ``osprey up`` builds the project image on its first run.
VERB_TIMEOUT_S = 1800
#: From agent-record answering ``/health`` to every fact being readable.
READ_TIMEOUT_S = 60.0
#: After the expected keys are all read, how long a case keeps watching for a
#: duplicate before it calls each count exact.
SETTLE_S = 10.0
HEALTH_TIMEOUT_S = 180.0

#: Variables a developer shell or an earlier test may carry that would change
#: what a spawned ``osprey`` verb resolves.
_SCRUBBED_ENV = (
    "CLAUDECODE",
    SPOOL_DIR_ENV,
    "OSPREY_TERMINAL_SESSION",
    "OSPREY_TERMINAL_USER",
)

_CLAUDE_VERSION = "2.1.300"
_AGENT_ID = "a5m0ke"
_DISPATCH_PROJECT = "-data-workspace"


# --------------------------------------------------------------------------- #
# Synthetic transcripts
# --------------------------------------------------------------------------- #


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _line(session: str, uid: str, kind: str, content: Any, **extra: Any) -> str:
    role = "user" if kind == "user" else "assistant"
    message: dict[str, Any] = {"role": role, "content": content}
    if role == "assistant":
        message["model"] = "claude-opus-5-5"
        message["usage"] = {"input_tokens": 3, "output_tokens": 5}
    obj = {
        "parentUuid": extra.pop("parent", None),
        "isSidechain": "agentId" in extra,
        "userType": "external",
        "cwd": "/data/workspace",
        "sessionId": session,
        "version": _CLAUDE_VERSION,
        "type": kind,
        "message": message,
        "uuid": uid,
        "timestamp": _now(),
        **extra,
    }
    return json.dumps(obj) + "\n"


@dataclass(frozen=True)
class Transcript:
    """One session's main transcript, its subagent file, and the keys they make."""

    session: str
    main: str
    subagent: str
    #: (uuid, block index) of every entry, in file order: main, then subagent.
    blocks: tuple[tuple[str, int], ...]

    def keys(self, identity: str) -> set[str]:
        return {transcript_key(identity, self.session, uid, i) for uid, i in self.blocks}


def _transcript(session: str) -> Transcript:
    """A session that thinks, calls a subagent through a tool, and answers."""
    tool_use_id = f"toolu_{uuid.uuid4().hex[:20]}"
    main = "".join(
        (
            _line(session, "u-1", "user", "List the agent record's sources."),
            _line(
                session,
                "a-1",
                "assistant",
                [
                    {"type": "thinking", "thinking": "A subagent can look.", "signature": "c2ln"},
                    {
                        "type": "tool_use",
                        "id": tool_use_id,
                        "name": "Agent",
                        "input": {"prompt": "Name the sources."},
                    },
                ],
                parent="u-1",
            ),
            _line(
                session,
                "u-2",
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": f"Spool and transcripts.\nagentId: {_AGENT_ID}",
                    }
                ],
                parent="a-1",
            ),
            _line(
                session,
                "a-2",
                "assistant",
                [{"type": "text", "text": "Spool and transcripts."}],
                parent="u-2",
            ),
        )
    )
    subagent = "".join(
        (
            _line(session, "s-1", "user", "Name the sources.", agentId=_AGENT_ID),
            _line(
                session,
                "s-2",
                "assistant",
                [{"type": "text", "text": "Spool and transcripts."}],
                parent="s-1",
                agentId=_AGENT_ID,
            ),
        )
    )
    blocks = (("u-1", 0), ("a-1", 0), ("a-1", 1), ("u-2", 0), ("a-2", 0), ("s-1", 0), ("s-2", 0))
    return Transcript(session=session, main=main, subagent=subagent, blocks=blocks)


def _write_host_transcript(project_dir: Path, transcript: Transcript) -> None:
    """Write *transcript* where Claude Code keeps it, readable by agent-record."""
    subagents = project_dir / transcript.session / "subagents"
    subagents.mkdir(parents=True)
    (project_dir / f"{transcript.session}.jsonl").write_text(transcript.main)
    (subagents / f"agent-{_AGENT_ID}.jsonl").write_text(transcript.subagent)
    # The bind mount is read by agent-record's own (unprivileged) user.
    for path in (project_dir, *project_dir.rglob("*")):
        path.chmod(0o755 if path.is_dir() else 0o644)


# --------------------------------------------------------------------------- #
# The deployment
# --------------------------------------------------------------------------- #


@dataclass
class Stack:
    """The deployed repo and what the cases address it by."""

    repo: Path
    osprey: Path
    env: dict[str, str]
    host_project_dir: Path
    config: dict[str, Any] = field(default_factory=dict)

    def reload(self) -> None:
        self.config = yaml.safe_load((self.repo / "build" / "config.yml").read_text())

    @property
    def record_port(self) -> int:
        block = self.config.get("agent_record") or {}
        return int(block.get("port") or default_port("agent_record", base=PORT_BASE))

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.record_port}"

    def operator_token(self) -> str:
        token = read_login(self.repo / login_file(OPERATOR_IDENTITY), OPERATOR_IDENTITY)
        assert token, "no operator credential at the repo root"
        return token


def _osprey_bin() -> Path:
    candidate = Path(sys.executable).parent / "osprey"
    if candidate.exists():
        return candidate
    found = shutil.which("osprey")
    if found:
        return Path(found)
    raise RuntimeError("Could not locate the 'osprey' console script.")


def _real_home() -> Path:
    """The account's home, whatever ``HOME`` says."""
    return Path(pwd.getpwuid(os.getuid()).pw_dir)


def _verb_env(config_dir: Path, home: Path) -> dict[str, str]:
    """The verbs' environment: the case's own Claude Code state and home.

    The docker CLI keeps its contexts and plugins under ``~/.docker``; with
    ``HOME`` moved it is pointed back at the account's own.
    """
    env = {k: v for k, v in os.environ.items() if k not in _SCRUBBED_ENV}
    env.setdefault("DOCKER_CONFIG", str(_real_home() / ".docker"))
    env[CLAUDE_CONFIG_DIR_ENV] = str(config_dir)
    env["HOME"] = str(home)
    return env


def _run(
    cmd: list[str], cwd: Path, env: Mapping[str, str], timeout: int = VERB_TIMEOUT_S
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout, env=dict(env)
    )


def _verb(stack: Stack, *args: str, timeout: int = VERB_TIMEOUT_S) -> subprocess.CompletedProcess:
    """Run an ``osprey`` verb in the repo; ``pytest.fail`` on a non-zero exit."""
    result = _run([str(stack.osprey), *args], cwd=stack.repo, env=stack.env, timeout=timeout)
    if result.returncode != 0:
        pytest.fail(
            f"osprey {' '.join(args)} failed (rc={result.returncode}):\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    return result


def _docker(
    *args: str, timeout: int = 120, check: bool = True, input: str | None = None
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        input=input,
    )
    if check and result.returncode != 0:
        raise AssertionError(f"docker {' '.join(args)} failed: {result.stderr.strip()}")
    return result


def _remove_project_containers() -> None:
    """Remove this project's own containers by exact name, should a run leave any."""
    for name in (RECORD_CONTAINER, RECORD_STORE_CONTAINER, *OLD_STORE_CONTAINERS):
        _docker("rm", "-f", name, check=False)


def _containers_named(name: str) -> list[str]:
    """The containers, running or not, called exactly *name*."""
    out = _docker("ps", "-a", "--filter", f"name=^{name}$", "--format", "{{.Names}}").stdout
    return [line for line in out.splitlines() if line.strip()]


def _real_claude_entries(repo: Path) -> list[str]:
    """What the account's own ``~/.claude/projects`` holds for *repo* or its build."""
    projects = _real_home() / ".claude" / "projects"
    prefix = encode_claude_project_path(repo)
    if not projects.is_dir():
        return []
    return sorted(p.name for p in projects.iterdir() if p.name.startswith(prefix))


@contextlib.contextmanager
def _host_stack_lock(path: Path) -> Iterator[None]:
    """Hold this module's host-wide stack lock for the duration of the block."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextlib.contextmanager
def _deployed(
    base: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    backend: str,
    transcript: Transcript,
    dispatch: bool,
) -> Iterator[Stack]:
    """``osprey init`` + ``build`` + ``up`` one backend under the host stack
    lock; tear it down on exit."""
    with _host_stack_lock(STACK_LOCK):
        yield from _deploy(
            base, monkeypatch, backend=backend, transcript=transcript, dispatch=dispatch
        )


def _deploy(
    base: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    backend: str,
    transcript: Transcript,
    dispatch: bool,
) -> Iterator[Stack]:
    osprey = _osprey_bin()
    repo = base / PROJECT_NAME
    config_dir = base / "claude-config"
    home = base / "home"
    config_dir.mkdir()
    home.mkdir()
    env = _verb_env(config_dir, home)
    # In-process path helpers resolve the same Claude Code state the verbs do.
    monkeypatch.setenv(CLAUDE_CONFIG_DIR_ENV, str(config_dir))

    _remove_project_containers()
    remove_project_volumes(PROJECT_NAME)

    edits: dict[str, Any] = {
        "port_base": PORT_BASE,
        "agent_record": {"backend": backend},
        "config": {
            "deployed_services": [],
            "deployment": {"host_identity": HOST_IDENTITY},
            # The OTLP exporter dials the old gateway, which is not deployed.
            "claude_code": {"telemetry": {"enabled": False}},
        },
    }
    if dispatch:
        edits["dispatch"] = {
            "triggers": "tutorial_triggers.yml",
            "worker_count": 1,
            "workspace_mode": "shared",
        }
    init = _run(
        [str(osprey), "init", str(repo), "--preset", "hello-world", "--no-git", *set_pairs(edits)],
        cwd=base,
        env=env,
    )
    if init.returncode != 0:
        pytest.fail(f"osprey init failed (rc={init.returncode}):\n{init.stdout}\n{init.stderr}")

    # Created before the build: the render mounts the host project directory
    # only when it exists, and never creates it.
    host_project_dir = claude_project_dir(repo / "build")
    assert host_project_dir.is_relative_to(config_dir), host_project_dir
    _write_host_transcript(host_project_dir, transcript)
    # `osprey up` refuses a repo without a .env; the dispatch services read
    # their tokens from it.
    (repo / ".env").write_text("EVENT_DISPATCHER_TOKEN=x\nDISPATCH_WORKER_TOKEN=y\n")

    stack = Stack(repo=repo, osprey=osprey, env=env, host_project_dir=host_project_dir)
    try:
        _verb(stack, "build", "--skip-deps", "--skip-lifecycle", "--dev")
        stack.reload()
        services = set(stack.config.get("deployed_services") or [])
        assert not services & {"telemetry_store", "telemetry_gateway"}, services
        rendered = (repo / "build" / "services" / "agent_record" / "docker-compose.yml").read_text()
        assert host_project_dir.as_posix() in rendered, (
            "the render does not mount the host build's project directory"
        )
        _verb(stack, "up", "-d", "--dev")
        _wait_healthy(stack)
        yield stack
    finally:
        down = _run([str(osprey), "down"], cwd=repo, env=env, timeout=300)
        if down.returncode != 0:
            print(f"osprey down rc={down.returncode}\n{down.stdout}\n{down.stderr}")
        _remove_project_containers()
        remove_project_volumes(PROJECT_NAME)
        shutil.rmtree(repo / "build" / ".venv", ignore_errors=True)


def _wait_healthy(stack: Stack, timeout: float = HEALTH_TIMEOUT_S) -> None:
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{stack.url}/health", timeout=3)
            if response.status_code == 200:
                return
            last = f"{response.status_code} {response.text}"
        except httpx.HTTPError as exc:
            last = repr(exc)
        time.sleep(2)
    logs = _docker("logs", "--tail", "80", RECORD_CONTAINER, check=False)
    pytest.fail(f"agent-record never answered /health: {last}\n{logs.stdout}\n{logs.stderr}")


# --------------------------------------------------------------------------- #
# Reading the record
# --------------------------------------------------------------------------- #


def _timeline(stack: Stack, session: str) -> dict[str, Any]:
    response = httpx.get(
        f"{stack.url}/v1/read/timeline",
        params={"session_id": session},
        auth=(OPERATOR_IDENTITY, stack.operator_token()),
        timeout=10,
    )
    assert response.status_code in (200, 503), f"{response.status_code} {response.text}"
    return response.json()


def _entries(nodes: list[Mapping[str, Any]]) -> Iterator[Mapping[str, Any]]:
    """Every entry a timeline holds: its nodes, their joined results and
    decisions, and every nested subagent stream."""
    for node in nodes:
        if node.get("key"):
            yield node
        for joined in ("result", "decision"):
            if node.get(joined):
                yield node[joined]
        sub = node.get("subagent")
        if sub:
            yield from _entries(sub.get("timeline") or [])


def _read_each_once(stack: Stack, session: str, expected: set[str]) -> dict[str, str]:
    """Read *session* until every *expected* key is there, then watch for duplicates.

    Returns each key's kind. Fails when a key is missing after
    :data:`READ_TIMEOUT_S`, or when any key of the session is read more than once.
    """
    deadline = time.monotonic() + READ_TIMEOUT_S
    last: Any = None
    counts: Counter[str] = Counter()
    kinds: dict[str, str] = {}
    while True:
        answer = _timeline(stack, session)
        last = answer
        if answer.get("status") == "ok":
            entries = list(_entries(answer.get("data") or []))
            counts = Counter(e["key"] for e in entries)
            kinds = {e["key"]: e["kind"] for e in entries}
            if expected <= set(counts):
                break
        if time.monotonic() > deadline:
            pytest.fail(
                f"session {session}: keys missing after {READ_TIMEOUT_S:.0f} s: "
                f"{sorted(expected - set(counts))}\nlast answer: {json.dumps(last)[:4000]}"
            )
        time.sleep(1)

    settle = time.monotonic() + SETTLE_S
    while True:
        duplicates = {k: n for k, n in counts.items() if n != 1}
        assert duplicates == {}, f"session {session}: facts read more than once: {duplicates}"
        if time.monotonic() > settle:
            break
        time.sleep(2)
        answer = _timeline(stack, session)
        assert answer.get("status") == "ok", answer
        counts = Counter(e["key"] for e in _entries(answer.get("data") or []))
    assert set(counts) == set(kinds), f"session {session}: the key set moved while settling"
    return kinds


def _write_spool_line(stack: Stack, monkeypatch: pytest.MonkeyPatch, session: str) -> str:
    """One spool line from the host identity, through the writer; its record key."""
    monkeypatch.setenv(SPOOL_DIR_ENV, str(stack.repo / SPOOL_DIR_RELPATH / HOST_IDENTITY))
    entry_id = writer.write(KIND_DEBUG, {"message": "agent-record smoke"}, session_id=session)
    writer.seal_all()
    assert entry_id, "writer.write stored nothing"
    return spool_key(HOST_IDENTITY, entry_id)


#: Run in a one-shot container: writes ``{relative path: text}`` from stdin
#: under ``/projects``, readable by agent-record's unprivileged user.
_VOLUME_WRITER = """
import json, os, sys
for rel, text in json.load(sys.stdin).items():
    path = os.path.join("/projects", rel)
    os.makedirs(os.path.dirname(path), mode=0o755, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)
for top, dirs, names in os.walk("/projects"):
    os.chmod(top, 0o755)
    for name in names:
        os.chmod(os.path.join(top, name), 0o644)
"""


def _write_dispatch_transcript(stack: Stack, transcript: Transcript) -> str:
    """Write *transcript* into the dispatch workers' transcripts volume; its identity."""
    roots = transcript_roots(stack.config)
    identity = roots.get(DISPATCH_TRANSCRIPT_VOLUME)
    assert identity, f"the render names no shared dispatch transcript root: {roots}"
    volume = f"{resolve_project_name(stack.config)}_{DISPATCH_TRANSCRIPT_VOLUME}"
    image = _docker("inspect", "--format", "{{.Config.Image}}", RECORD_CONTAINER).stdout.strip()
    files = {
        f"{_DISPATCH_PROJECT}/{transcript.session}.jsonl": transcript.main,
        f"{_DISPATCH_PROJECT}/{transcript.session}/subagents/agent-{_AGENT_ID}.jsonl": (
            transcript.subagent
        ),
    }
    _docker(
        "run",
        "--rm",
        "-i",
        "--user",
        "0",
        "--entrypoint",
        "python3",
        "-v",
        f"{volume}:/projects",
        image,
        "-c",
        _VOLUME_WRITER,
        input=json.dumps(files),
        timeout=180,
    )
    return identity


def _restart_record(stack: Stack) -> None:
    _docker("restart", RECORD_CONTAINER, timeout=180)
    _wait_healthy(stack)


# --------------------------------------------------------------------------- #
# The cases
# --------------------------------------------------------------------------- #


def test_embedded_record_holds_every_fact_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Host transcript, spool line and dispatch transcript, each fact once,
    through a restart; no store container; the real ``~/.claude`` untouched."""
    host = _transcript(str(uuid.uuid4()))
    dispatched = _transcript(str(uuid.uuid4()))
    repo = tmp_path / PROJECT_NAME
    before = _real_claude_entries(repo)

    with _deployed(
        tmp_path, monkeypatch, backend=BACKEND_EMBEDDED, transcript=host, dispatch=True
    ) as stack:
        spooled = _write_spool_line(stack, monkeypatch, host.session)
        dispatch_identity = _write_dispatch_transcript(stack, dispatched)

        host_expected = host.keys(HOST_IDENTITY) | {spooled}
        dispatch_expected = dispatched.keys(dispatch_identity)
        host_kinds = _read_each_once(stack, host.session, host_expected)
        _read_each_once(stack, dispatched.session, dispatch_expected)
        assert {KIND_THINKING, KIND_TOOL_CALL, KIND_SUBAGENT, KIND_DEBUG} <= set(
            host_kinds.values()
        ), host_kinds

        _restart_record(stack)
        assert set(_read_each_once(stack, host.session, host_expected)) == set(host_kinds)
        _read_each_once(stack, dispatched.session, dispatch_expected)

        stores = {
            name: found
            for name in (*OLD_STORE_CONTAINERS, RECORD_STORE_CONTAINER)
            if (found := _containers_named(name))
        }
        assert stores == {}, f"an embedded deployment started a store container: {stores}"

    assert _real_claude_entries(repo) == before, "the deployment wrote into the real ~/.claude"


def test_clickhouse_record_holds_every_fact_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same read against ``backend: clickhouse``: every fact once."""
    host = _transcript(str(uuid.uuid4()))
    repo = tmp_path / PROJECT_NAME
    before = _real_claude_entries(repo)

    with _deployed(
        tmp_path, monkeypatch, backend=BACKEND_CLICKHOUSE, transcript=host, dispatch=False
    ) as stack:
        assert _containers_named(RECORD_STORE_CONTAINER), "no record store container"
        spooled = _write_spool_line(stack, monkeypatch, host.session)
        expected = host.keys(HOST_IDENTITY) | {spooled}
        kinds = _read_each_once(stack, host.session, expected)
        assert {KIND_THINKING, KIND_TOOL_CALL, KIND_SUBAGENT, KIND_DEBUG} <= set(kinds.values())
        assert not any(_containers_named(name) for name in OLD_STORE_CONTAINERS)

    assert _real_claude_entries(repo) == before, "the deployment wrote into the real ~/.claude"
