"""The agent record end to end: one deployment per backend, every fact once.

Runs in its own model-free lane (``agent-record-e2e``): no test here names a
provider or reaches a model. The transcripts it reads are the captured Claude
Code sessions under ``tests/fixtures/agent_record/``, replayed into the host
build's project directory, beside a synthetic 2.1.300-shaped session the test
writes itself, and forwarded to the stack's record the way a host session
forwards them.

Two cases, one per backend and preset: ``hello-world`` on the embedded record
and ``control-assistant`` on the ClickHouse record. Each deploys under this
module's own project name and port block with a stated
``deployment.host_identity``, and runs the real ``osprey build`` and
``osprey up``. Both verbs run with ``CLAUDE_CONFIG_DIR`` and ``HOME`` pointed
into the case's temp tree; the host build's Claude Code project directory is
created there, holding the transcripts. The render mounts nothing of it: once
the stack is up, the case forwards it every supervisor round as a host session
does (``host.forward_host_transcripts``), until the reset. The developer's own
``~/.claude`` gains nothing.

What each case proves, read back through the read API:

* every fact once, with its source identity and person: host transcripts,
  every captured fixture, spool lines from the host, a sidecar line on behalf
  of a person, a dispatch run with a requester and a scheduled run with none;
* a transcript step appended to a live session is readable within 60 s;
* counts kept across a restart of agent-record and an in-place rewrite of
  every captured transcript under a new inode;
* no loss across a record outage, and on ClickHouse the backlog it leaves and
  the stopped source it makes, on ``/health/sources``;
* a refused spool line, and (embedded) a stalled writer, on health within the
  five minutes the record allows a source;
* an oversized tool result and a large spool body read back byte-identical;
* every read route, and a scoped reader login, on hello-world; the history
  page behind its own origin on control-assistant;
* the day's Parquet file and manifest mark on the host once the jobs clock
  closes the day, exported exactly once;
* ``osprey reset`` keeping the record and its archive, ``--purge-audit``
  removing both and keeping the rest of the archive;
* (hello-world) a profile carrying retired keys refused by ``osprey build``
  naming each replacement, then built and brought up once edited as directed.

The two cases run one after the other, never side by side: each tears its
deployment down before it returns.

CONTAINER SAFETY: every docker invocation names an exact container or volume
of this project. Teardown is ``osprey down`` followed by the project-scoped
volume sweep (``tests/e2e/_volumes.py``); never a prune, never a wildcard.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import pwd
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import pytest
import yaml

from osprey.agent_record import host as host_record
from osprey.agent_record import writer
from osprey.agent_record.archive import EXPORT_FILE, EXPORT_SUBDIR, exported_days
from osprey.agent_record.attribution import REQUESTER_FIELD
from osprey.agent_record.config import (
    BACKEND_CLICKHOUSE,
    BACKEND_EMBEDDED,
    RETIRED_CONFIG_KEYS,
)
from osprey.agent_record.contract import (
    KIND_AUDIT,
    KIND_DEBUG,
    KIND_RUN,
    KIND_SUBAGENT,
    KIND_THINKING,
    KIND_TOOL_CALL,
    RUN_ACCEPT,
    RUN_PHASE_FIELD,
    SPOOL_DIR_ENV,
    run_key,
    segment_name,
    spool_key,
    transcript_key,
)
from osprey.agent_record.credentials import login_file, read_login, reader_file
from osprey.agent_record.identities import (
    DISPATCH_TRANSCRIPT_VOLUME,
    OPERATOR_IDENTITY,
    READER_LOGIN_PREFIX,
    SHARED_DISPATCH_IDENTITY,
    SIDECAR_IDENTITY,
    transcript_roots,
)
from osprey.agent_record.ingest.transcripts import parse_line
from osprey.agent_record.service import JOBS_CLOCK_ENV, READ_METHODS, STOPPED_AFTER
from osprey.agent_runner.project_paths import (
    CLAUDE_CONFIG_DIR_ENV,
    claude_project_dir,
    encode_claude_project_path,
)
from osprey.deployment.compose_generator import ensure_spool_dir, resolve_project_name
from osprey.port_layout import default_port
from osprey.services.archive.manifest import MANIFEST_NAME
from osprey.utils.workspace import ARCHIVE_DIR_RELPATH, SPOOL_DIR_RELPATH
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
#: The identity the host build's transcripts and spool are recorded under; the
#: captured fixtures were recorded under it too.
HOST_IDENTITY = "e2e-builder"
#: The person the on-behalf and requested runs name.
PERSON = "alice"
#: control-assistant's password roster: the preset's ``env.defaults``.
PASSWORDS = {"alice": "alice"}
#: The scoped reader hello-world states, and the one kind it may read.
READER_NAME = "e2e"
READER_KINDS = (KIND_DEBUG,)

RECORD_CONTAINER = f"{PROJECT_NAME}-agent-record"
RECORD_STORE_CONTAINER = f"{PROJECT_NAME}-agent-record-clickhouse"
RECORD_STORE_VOLUME = f"{PROJECT_NAME}_agent_record_clickhouse_data"
#: Where the embedded record keeps its file, relative to the repo.
EMBEDDED_RECORD_RELPATH = "var/agent_record"

#: ``osprey up`` builds the project image on its first run.
VERB_TIMEOUT_S = 1800
#: From a fact being written (or agent-record answering ``/health``) to it
#: being readable.
READ_TIMEOUT_S = 60.0
#: After the expected keys are all read, how long a case keeps watching for a
#: duplicate before it calls each count exact.
SETTLE_S = 10.0
HEALTH_TIMEOUT_S = 180.0
#: How long health may take to name a refusal or a stopped source: the
#: record's own five minutes, plus one ingest pass and one poll.
HEALTH_SIGNAL_TIMEOUT_S = STOPPED_AFTER.total_seconds() + 60.0
#: A body at least this large is an oversized result.
LARGE_BYTES = 64 * 1024
#: The size of the large spool body the cases write.
LARGE_SPOOL_BYTES = 1024 * 1024

#: Variables a developer shell or an earlier test may carry that would change
#: what a spawned ``osprey`` verb resolves.
_SCRUBBED_ENV = (
    "CLAUDECODE",
    SPOOL_DIR_ENV,
    JOBS_CLOCK_ENV,
    "OSPREY_AGENT_RECORD_URL",
    "OSPREY_TERMINAL_SESSION",
    "OSPREY_TERMINAL_USER",
)

#: The captured Claude Code sessions, one directory per launch shape, each with
#: a ``transcripts/`` tree laid out as Claude Code's ``projects/<project>/``.
FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "agent_record"

_CLAUDE_VERSION = "2.1.300"
_AGENT_ID = "a5m0ke"
_DISPATCH_PROJECT = "-data-workspace"

#: The retired keys the FR6 profile carries: one a removed setting, one a
#: renamed history-page key.
FR6_RETIRED = {
    "claude_code.telemetry.enabled": True,
    "modules.web_terminals.reader_port": 19999,
}


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
    _open_tree(project_dir)


def _open_tree(root: Path) -> None:
    """Make *root* readable by agent-record's own (unprivileged) user."""
    for path in (root, *root.rglob("*")):
        path.chmod(0o755 if path.is_dir() else 0o644)


def _fixture_dirs() -> list[Path]:
    dirs = sorted(d for d in FIXTURE_ROOT.iterdir() if (d / "transcripts").is_dir())
    assert dirs, f"no captured fixture under {FIXTURE_ROOT}"
    return dirs


@dataclass(frozen=True)
class Replay:
    """What the captured fixtures put in the record."""

    #: Each replayed session's expected keys.
    sessions: dict[str, set[str]]
    #: Every oversized body the fixtures hold, by key.
    large: dict[str, dict[str, Any]]


def _replay_fixtures(project_dir: Path) -> Replay:
    """Copy every captured fixture's transcripts into *project_dir*.

    Each replayed session's expected keys, and its oversized bodies, are what
    the record's own parser derives from every line of the session's main and
    subagent files.
    """
    sessions: dict[str, set[str]] = {}
    large: dict[str, dict[str, Any]] = {}
    for fixture in _fixture_dirs():
        shutil.copytree(fixture / "transcripts", project_dir, dirs_exist_ok=True)
        for path in sorted((fixture / "transcripts").rglob("*.jsonl")):
            relpath = f"{project_dir.name}/{path.relative_to(fixture / 'transcripts').as_posix()}"
            entries = [
                entry
                for line in path.read_bytes().splitlines()
                for entry in parse_line(HOST_IDENTITY, relpath, line)
            ]
            session = path.stem if path.parent == fixture / "transcripts" else path.parts[-3]
            sessions.setdefault(session, set()).update(e.key for e in entries)
            large.update(
                (e.key, dict(e.body))
                for e in entries
                if len(json.dumps(dict(e.body)).encode()) >= LARGE_BYTES
            )
    assert all(sessions.values()), f"a captured fixture yields no key: {sessions}"
    assert large, "no captured fixture holds an oversized result"
    _open_tree(project_dir)
    return Replay(sessions=sessions, large=large)


def _rewrite_transcript(project_dir: Path, session: str) -> None:
    """Rewrite *session*'s main and subagent transcripts the way an editor or
    Claude Code's own rewrite does: the same bytes, a new file under the same name."""
    targets = [
        project_dir / f"{session}.jsonl",
        *sorted((project_dir / session / "subagents").glob("*.jsonl")),
    ]
    for target in targets:
        staged = target.with_name(f".{target.name}.rewrite")
        staged.write_bytes(target.read_bytes())
        staged.chmod(0o644)
        os.replace(staged, target)


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
    backend: str
    replay: Replay
    config: dict[str, Any] = field(default_factory=dict)
    host_session: HostSession | None = None

    def reload(self) -> None:
        self.config = yaml.safe_load((self.repo / "build" / "config.yml").read_text())

    def stop_host_session(self) -> None:
        if self.host_session is not None:
            self.host_session.stop()
            self.host_session = None

    @property
    def record_port(self) -> int:
        block = self.config.get("agent_record") or {}
        return int(block.get("port") or default_port("agent_record", base=PORT_BASE))

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.record_port}"

    @property
    def archive_root(self) -> Path:
        return self.repo / ARCHIVE_DIR_RELPATH

    def operator_auth(self) -> tuple[str, str]:
        token = read_login(self.repo / login_file(OPERATOR_IDENTITY), OPERATOR_IDENTITY)
        assert token, "no operator credential at the repo root"
        return OPERATOR_IDENTITY, token


class HostSession:
    """The host session's forwarding of its transcripts, as its supervisor runs it.

    The stack's agent-record reads no host transcript: Claude Code writes them
    for the host account alone, and a host session forwards them through the
    spool every supervisor round (``host.forward_host_transcripts``).
    """

    def __init__(self, repo: Path) -> None:
        self._repo = repo
        self._config = yaml.safe_load((repo / "build" / "config.yml").read_text())
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, name="host-forward", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stopped.wait(host_record.SUPERVISE_INTERVAL_S):
            host_record.forward_host_transcripts(self._repo, self._config)

    def stop(self) -> None:
        self._stopped.set()
        self._thread.join(timeout=host_record.FORWARD_WAIT_S)


@dataclass(frozen=True)
class Preset:
    """What one case deploys: a preset, its pins and the ``.env`` ``up`` needs."""

    name: str
    backend: str
    edits: dict[str, Any]
    seed_env: Callable[[Path], None]


#: The dispatch services read their tokens from the repo's ``.env``.
_DISPATCH_TOKENS = "EVENT_DISPATCHER_TOKEN=x\nDISPATCH_WORKER_TOKEN=y\n"


def _hello_world_env(repo: Path) -> None:
    # `osprey up` refuses a repo without a .env.
    (repo / ".env").write_text(_DISPATCH_TOKENS)


def _example_env(repo: Path) -> None:
    # `osprey init` writes a .env only when the shell exports a provider key,
    # which this lane never does: the `cp .env.example .env` the CLI
    # recommends, with the dispatch tokens it leaves commented out.
    text = (repo / ".env.example").read_text()
    (repo / ".env").write_text(f"{text.rstrip()}\n{_DISPATCH_TOKENS}")


def _common_config() -> dict[str, Any]:
    return {
        "deployment": {"host_identity": HOST_IDENTITY},
        # The jobs clock reaches agent-record only when `osprey up` passes it
        # through; it is set only when a case closes a day.
        "services.agent_record.env": [JOBS_CLOCK_ENV],
    }


HELLO_WORLD = Preset(
    name="hello-world",
    backend=BACKEND_EMBEDDED,
    edits={
        "port_base": PORT_BASE,
        "agent_record": {
            "backend": BACKEND_EMBEDDED,
            "readers": [{"name": READER_NAME, "kinds": list(READER_KINDS)}],
        },
        "config": {**_common_config(), "deployed_services": []},
        "dispatch": {
            "triggers": "tutorial_triggers.yml",
            "worker_count": 1,
            "workspace_mode": "shared",
        },
    },
    seed_env=_hello_world_env,
)

#: control-assistant as shipped, with a CI-sized archiver store. Its
#: ``deployed_services`` stay, since its ARIEL and graph consumers dial them;
#: its dispatch stack stays, since its web panels name the events page; and
#: its web terminals stay, since they deploy the history page.
CONTROL_ASSISTANT = Preset(
    name="control-assistant",
    backend=BACKEND_CLICKHOUSE,
    edits={
        "port_base": PORT_BASE,
        "agent_record": {"backend": BACKEND_CLICKHOUSE},
        "config": {**_common_config(), "control_system.type": "virtual_accelerator"},
        "va_archiver": {"retention_days": 2, "hot_span_hours": 2},
    },
    seed_env=_example_env,
)


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


def _build(stack: Stack) -> None:
    _verb(stack, "build", "--skip-deps", "--skip-lifecycle", "--dev")
    stack.reload()


def _up(stack: Stack) -> None:
    _verb(stack, "up", "-d", "--dev")
    _wait_healthy(stack)


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
    for name in (RECORD_CONTAINER, RECORD_STORE_CONTAINER):
        _docker("rm", "-f", name, check=False)


def _containers_named(name: str) -> list[str]:
    """The containers, running or not, called exactly *name*."""
    out = _docker("ps", "-a", "--filter", f"name=^{name}$", "--format", "{{.Names}}").stdout
    return [line for line in out.splitlines() if line.strip()]


def _volume_exists(name: str) -> bool:
    return _docker("volume", "inspect", name, check=False).returncode == 0


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
    preset: Preset,
    transcript: Transcript,
    retired_keys_first: bool = False,
) -> Iterator[Stack]:
    """``osprey init`` + ``build`` + ``up`` one preset under the host stack
    lock; tear it down on exit."""
    with _host_stack_lock(STACK_LOCK):
        yield from _deploy(
            base,
            monkeypatch,
            preset=preset,
            transcript=transcript,
            retired_keys_first=retired_keys_first,
        )


def _deploy(
    base: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    preset: Preset,
    transcript: Transcript,
    retired_keys_first: bool,
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

    init = _run(
        [
            str(osprey),
            "init",
            str(repo),
            "--preset",
            preset.name,
            "--no-git",
            *set_pairs(preset.edits),
        ],
        cwd=base,
        env=env,
    )
    if init.returncode != 0:
        pytest.fail(f"osprey init failed (rc={init.returncode}):\n{init.stdout}\n{init.stderr}")

    # The host session's transcripts, which the stack's agent-record never
    # mounts: the host session forwards them once the stack owns the record.
    host_project_dir = claude_project_dir(repo / "build")
    assert host_project_dir.is_relative_to(config_dir), host_project_dir
    _write_host_transcript(host_project_dir, transcript)
    replay = _replay_fixtures(host_project_dir)
    preset.seed_env(repo)

    stack = Stack(
        repo=repo,
        osprey=osprey,
        env=env,
        host_project_dir=host_project_dir,
        backend=preset.backend,
        replay=replay,
    )
    try:
        if retired_keys_first:
            _refuse_then_build_as_directed(stack)
        _build(stack)
        _provision_test_writers(repo)
        rendered = (repo / "build" / "services" / "agent_record" / "docker-compose.yml").read_text()
        assert host_project_dir.as_posix() not in rendered, (
            "the render mounts the host build's project directory"
        )
        _up(stack)
        stack.host_session = HostSession(repo)
        yield stack
    finally:
        stack.stop_host_session()
        down = _run([str(osprey), "down"], cwd=repo, env=env, timeout=300)
        if down.returncode != 0:
            print(f"osprey down rc={down.returncode}\n{down.stdout}\n{down.stderr}")
        _remove_project_containers()
        remove_project_volumes(PROJECT_NAME)
        shutil.rmtree(repo / "build" / ".venv", ignore_errors=True)


def _provision_test_writers(repo: Path) -> None:
    """Provision the spool directories this module writes as, the way the
    deployment provisions its own writers' (setgid, group-shared), before
    agent-record starts: a preset that deploys no such writer leaves them out,
    and a directory the test process created would be the runner's alone,
    which agent-record's unprivileged user can neither read nor clean."""
    for identity in (SIDECAR_IDENTITY, SHARED_DISPATCH_IDENTITY):
        assert ensure_spool_dir(repo, identity) is not None, f"no spool dir for {identity}"


def _wait_healthy(stack: Stack, timeout: float = HEALTH_TIMEOUT_S) -> None:
    """Wait for agent-record's ``/health``, start-up jobs included."""
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{stack.url}/health", timeout=3)
            if response.status_code == 200 and response.json().get("jobs_ready"):
                return
            last = f"{response.status_code} {response.text}"
        except httpx.HTTPError as exc:
            last = repr(exc)
        time.sleep(2)
    logs = _docker("logs", "--tail", "80", RECORD_CONTAINER, check=False)
    pytest.fail(f"agent-record never answered /health: {last}\n{logs.stdout}\n{logs.stderr}")


# --------------------------------------------------------------------------- #
# FR6: a profile carrying retired keys
# --------------------------------------------------------------------------- #


def _refuse_then_build_as_directed(stack: Stack) -> None:
    """A profile carrying retired keys fails ``osprey build`` naming each
    replacement; the operator removes them, and the case's own build and up
    then succeed."""
    profile = stack.repo / "profile.yml"
    original = profile.read_text()
    data = yaml.safe_load(original)
    data.setdefault("config", {}).update(FR6_RETIRED)
    profile.write_text(yaml.safe_dump(data, sort_keys=False))

    refused = _run(
        [str(stack.osprey), "build", "--skip-deps", "--skip-lifecycle", "--dev"],
        cwd=stack.repo,
        env=stack.env,
    )
    said = f"{refused.stdout}\n{refused.stderr}"
    assert refused.returncode != 0, f"a profile with retired keys built:\n{said}"
    for key in FR6_RETIRED:
        retired = next(r for r in RETIRED_CONFIG_KEYS if key == r or key.startswith(f"{r}."))
        assert key in said, f"the refusal does not name {key}:\n{said}"
        assert RETIRED_CONFIG_KEYS[retired] in " ".join(said.split()), (
            f"the refusal does not name {key}'s replacement:\n{said}"
        )
    assert not (stack.repo / "build" / "config.yml").exists(), "a refused build rendered"

    # The edit the refusal directs: both keys removed, nothing else touched.
    profile.write_text(original)


# --------------------------------------------------------------------------- #
# Reading the record
# --------------------------------------------------------------------------- #


def _read(
    stack: Stack, name: str, auth: tuple[str, str] | None = None, **params: str
) -> dict[str, Any]:
    response = httpx.get(
        f"{stack.url}/v1/read/{name}",
        params=params,
        auth=auth or stack.operator_auth(),
        timeout=30,
    )
    assert response.status_code in (200, 503), f"{name}: {response.status_code} {response.text}"
    return response.json()


def _timeline(stack: Stack, session: str) -> dict[str, Any]:
    return _read(stack, "timeline", session_id=session)


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


def _read_each_once(
    stack: Stack, session: str, expected: set[str], timeout: float = READ_TIMEOUT_S
) -> dict[str, Mapping[str, Any]]:
    """Read *session* until every *expected* key is there, then watch for duplicates.

    Returns each key's entry. Fails when a key is missing after *timeout*, or
    when any key of the session is read more than once.
    """
    deadline = time.monotonic() + timeout
    last: Any = None
    counts: Counter[str] = Counter()
    found: dict[str, Mapping[str, Any]] = {}
    while True:
        answer = _timeline(stack, session)
        last = answer
        if answer.get("status") == "ok":
            entries = list(_entries(answer.get("data") or []))
            counts = Counter(e["key"] for e in entries)
            found = {e["key"]: e for e in entries}
            if expected <= set(counts):
                break
        if time.monotonic() > deadline:
            pytest.fail(
                f"session {session}: keys missing after {timeout:.0f} s: "
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
    assert set(counts) == set(found), f"session {session}: the key set moved while settling"
    return found


def _read_replay_once(stack: Stack) -> None:
    """Every captured fixture's facts, once each, under the host identity and person."""
    for session, keys in stack.replay.sessions.items():
        found = _read_each_once(stack, session, keys)
        _assert_attributed(found, keys, identity=HOST_IDENTITY, person=HOST_IDENTITY)


def _assert_attributed(
    found: Mapping[str, Mapping[str, Any]],
    keys: set[str],
    *,
    identity: str,
    person: str | None,
) -> None:
    wrong = {
        k: (found[k].get("identity"), found[k].get("person"))
        for k in keys
        if (found[k].get("identity"), found[k].get("person")) != (identity, person)
    }
    assert wrong == {}, f"attributed to other than ({identity!r}, {person!r}): {wrong}"


def _full_entry(stack: Stack, key: str) -> Mapping[str, Any]:
    deadline = time.monotonic() + READ_TIMEOUT_S
    while True:
        answer = _read(stack, "full_entry", key=key)
        if answer.get("status") == "ok" and answer.get("data"):
            return answer["data"]
        if time.monotonic() > deadline:
            pytest.fail(f"full_entry {key}: {json.dumps(answer)[:2000]}")
        time.sleep(1)


def _canonical(body: Any) -> bytes:
    return json.dumps(body, sort_keys=True, ensure_ascii=False).encode()


# --------------------------------------------------------------------------- #
# Writing the record
# --------------------------------------------------------------------------- #


def _spool_dir(stack: Stack, identity: str) -> Path:
    return stack.repo / SPOOL_DIR_RELPATH / identity


def _spool(
    stack: Stack,
    monkeypatch: pytest.MonkeyPatch,
    identity: str,
    kind: str,
    body: Mapping[str, Any],
    **ids: Any,
) -> str:
    """One sealed spool line from *identity*, through the writer; its entry id."""
    monkeypatch.setenv(SPOOL_DIR_ENV, str(_spool_dir(stack, identity)))
    entry_id = writer.write(kind, body, **ids)
    writer.seal_all()
    assert entry_id, f"writer.write stored nothing for {identity}"
    return entry_id


def _write_spool_line(stack: Stack, monkeypatch: pytest.MonkeyPatch, session: str) -> str:
    """One spool line from the host identity, through the writer; its record key."""
    entry_id = _spool(
        stack,
        monkeypatch,
        HOST_IDENTITY,
        KIND_DEBUG,
        {"message": "agent-record e2e"},
        session_id=session,
    )
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
# Health
# --------------------------------------------------------------------------- #


def _health_sources(stack: Stack) -> dict[str, Any]:
    response = httpx.get(f"{stack.url}/health/sources", auth=stack.operator_auth(), timeout=30)
    assert response.status_code == 200, f"{response.status_code} {response.text}"
    return response.json()


def _spool_row(health: Mapping[str, Any], identity: str) -> Mapping[str, Any] | None:
    sources = (health.get("spool") or {}).get("sources") or {}
    return next((row for row in sources.values() if row.get("identity") == identity), None)


def _wait_health(
    stack: Stack,
    identity: str,
    holds: Callable[[Mapping[str, Any]], bool],
    what: str,
    timeout: float = HEALTH_SIGNAL_TIMEOUT_S,
) -> Mapping[str, Any]:
    """Poll ``/health/sources`` until *identity*'s spool row *holds*; that row."""
    deadline = time.monotonic() + timeout
    row: Mapping[str, Any] | None = None
    while True:
        row = _spool_row(_health_sources(stack), identity)
        if row is not None and holds(row):
            return row
        if time.monotonic() > deadline:
            pytest.fail(f"health never showed {what} for {identity} in {timeout:.0f} s: {row}")
        time.sleep(5)


# --------------------------------------------------------------------------- #
# Steps shared by both cases
# --------------------------------------------------------------------------- #


def _step_readable_within_a_minute(stack: Stack, transcript: Transcript) -> str:
    """Append one step to the live host session; its key, read within 60 s."""
    path = stack.host_project_dir / f"{transcript.session}.jsonl"
    with path.open("a") as fh:
        fh.write(
            _line(
                transcript.session,
                "a-3",
                "assistant",
                [{"type": "text", "text": "One more step."}],
                parent="a-2",
            )
        )
    key = transcript_key(HOST_IDENTITY, transcript.session, "a-3", 0)
    _read_each_once(stack, transcript.session, {key}, timeout=READ_TIMEOUT_S)
    return key


def _large_results_identical(stack: Stack, monkeypatch: pytest.MonkeyPatch) -> None:
    """The fixtures' oversized results, and a large spool body, read back whole."""
    for key, body in stack.replay.large.items():
        data = _full_entry(stack, key)
        assert _canonical(data["body"]) == _canonical(body), f"{key} did not read back whole"

    session = str(uuid.uuid4())
    body = {"message": os.urandom(LARGE_SPOOL_BYTES // 2).hex()}
    entry_id = _spool(stack, monkeypatch, HOST_IDENTITY, KIND_DEBUG, body, session_id=session)
    data = _full_entry(stack, spool_key(HOST_IDENTITY, entry_id))
    assert _canonical(data["body"]) == _canonical(body), "the large spool body changed"


def _refusal_on_health(stack: Stack, monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``run`` line from the host identity is refused, and health names it."""
    entry_id = _spool(
        stack,
        monkeypatch,
        HOST_IDENTITY,
        KIND_RUN,
        {RUN_PHASE_FIELD: RUN_ACCEPT},
        run_id=f"refused-{uuid.uuid4().hex[:8]}",
        session_id=str(uuid.uuid4()),
    )
    row = _wait_health(
        stack,
        HOST_IDENTITY,
        lambda r: (
            (r.get("refusals") or 0) >= 1
            and entry_id in {x.get("entry_id") for x in r.get("refused") or []}
        ),
        "the refused run line",
    )
    assert row["spool_errors"] >= 1, row


def _attributed_runs(stack: Stack, monkeypatch: pytest.MonkeyPatch) -> None:
    """On-behalf, requested and scheduled runs, each fact with its person."""
    # The sidecar is the one source whose `on_behalf_of` names a person.
    behalf_session = str(uuid.uuid4())
    behalf = _spool(
        stack,
        monkeypatch,
        SIDECAR_IDENTITY,
        KIND_AUDIT,
        {"surface": "web_terminal", "event": "agent-record e2e"},
        session_id=behalf_session,
        on_behalf_of=PERSON,
    )
    found = _read_each_once(stack, behalf_session, {spool_key(SIDECAR_IDENTITY, behalf)})
    _assert_attributed(found, set(found), identity=SIDECAR_IDENTITY, person=PERSON)

    # A dispatch run: its session belongs to the requester of its accept; a
    # scheduled run names no requester and so no person.
    for requester in (PERSON, None):
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        session = str(uuid.uuid4())
        accept_body: dict[str, Any] = {RUN_PHASE_FIELD: RUN_ACCEPT}
        if requester is not None:
            accept_body[REQUESTER_FIELD] = requester
        _spool(
            stack,
            monkeypatch,
            SHARED_DISPATCH_IDENTITY,
            KIND_RUN,
            accept_body,
            run_id=run_id,
            session_id=session,
        )
        step = _spool(
            stack,
            monkeypatch,
            SHARED_DISPATCH_IDENTITY,
            KIND_DEBUG,
            {"message": "a dispatched step"},
            session_id=session,
            run_id=run_id,
        )
        keys = {run_key(run_id, RUN_ACCEPT), spool_key(SHARED_DISPATCH_IDENTITY, step)}
        found = _read_each_once(stack, session, keys)
        _assert_attributed(found, keys, identity=SHARED_DISPATCH_IDENTITY, person=requester)
        run = _read(stack, "run", run_id=run_id)
        assert run.get("status") == "ok", run
        assert (run["data"] or {}).get("requester") == requester, run


def _restart_and_rewrite_keep_counts(stack: Stack, checks: list[tuple[str, set[str]]]) -> None:
    """Every *checks* session and every fixture, once each, across a restart
    of agent-record and a rewrite of every captured transcript."""
    _restart_record(stack)
    for session, keys in checks:
        _read_each_once(stack, session, keys)
    _read_replay_once(stack)

    for session in stack.replay.sessions:
        _rewrite_transcript(stack.host_project_dir, session)
    _read_replay_once(stack)


def _day_exported_on_host(stack: Stack, today: date) -> date:
    """Close *today*, the day the case started writing, with the jobs clock;
    its Parquet file and mark on the host, once."""
    if datetime.now(UTC).date() != today:
        pytest.fail(
            f"the case crossed UTC midnight (started {today}): its facts span two"
            " arrival days, so the one-day export cannot be judged; re-run it"
        )
    stack.env[JOBS_CLOCK_ENV] = (today + timedelta(days=1)).isoformat()
    # `up` recreates agent-record with the clock passed through; its start-up
    # jobs export every closed day before health reports them ready.
    _up(stack)

    health = _health_sources(stack)
    assert health.get("jobs_clock"), f"the jobs clock override is not reported: {health}"
    export = health["export"]
    assert export.get("last_error") is None, export
    assert export.get("lag_days") == 0, export
    assert export.get("last_exported_day") == today.isoformat(), export
    assert today in exported_days(stack.archive_root)

    day_dir = stack.archive_root / today.isoformat()
    parquet = day_dir / EXPORT_SUBDIR / EXPORT_FILE
    try:
        exported = parquet.is_file()
    except PermissionError:
        exported = False
    assert exported, (
        f"no Parquet file for {today} the host can read: {parquet}\n{_listing(day_dir)}"
    )
    lines = [json.loads(x) for x in (day_dir / MANIFEST_NAME).read_text().splitlines() if x]
    marks = [x for x in lines if x.get("kind") == "agent_record_day"]
    assert len(marks) == 1 and marks[0]["day"] == today.isoformat(), lines
    files = [
        x for x in lines if x.get("kind") == "file" and x.get("path", "").endswith(EXPORT_FILE)
    ]
    assert len(files) == 1, lines
    assert files[0]["sha256"] == hashlib.sha256(parquet.read_bytes()).hexdigest(), files
    assert marks[0]["rows"] == files[0]["rows"] > 0, lines
    assert lines.index(marks[0]) > lines.index(files[0]), "the day mark is not last"

    rerun = httpx.post(f"{stack.url}/jobs/export", auth=stack.operator_auth(), timeout=120)
    assert rerun.status_code == 200, rerun.text
    assert rerun.json().get("exported") == [], f"a day was exported twice: {rerun.json()}"
    return today


def _listing(directory: Path) -> str:
    """``ls -lnaR`` of *directory* and the identity reading it, for a failure message."""
    listed = subprocess.run(
        ["ls", "-lnaR", str(directory)], capture_output=True, text=True, timeout=30
    )
    return f"read as uid {os.getuid()} groups {os.getgroups()}:\n{listed.stdout}{listed.stderr}"


def _reset_keeps_then_purges(stack: Stack, day: date, *, reread: tuple[str, set[str]]) -> None:
    """Plain reset keeps the record and its archive; ``--purge-audit`` removes
    both and keeps the rest of the archive."""
    day_dir = stack.archive_root / day.isoformat()
    beside = day_dir / "e2e-artifact.txt"
    beside.write_text("an archived artifact beside the record's day\n")
    parquet = day_dir / EXPORT_SUBDIR / EXPORT_FILE
    embedded = stack.repo / EMBEDDED_RECORD_RELPATH

    def record_present() -> bool:
        if stack.backend == BACKEND_EMBEDDED:
            # A reset leaves the state directory behind empty, the shape a
            # fresh `osprey init` gives it: the record is its contents.
            return embedded.is_dir() and any(embedded.iterdir())
        return _volume_exists(RECORD_STORE_VOLUME)

    assert record_present(), "no record before the reset"
    # The host session ends before the reset takes the spool and state away.
    stack.stop_host_session()
    _verb(stack, "reset", "-y")
    assert record_present(), "a plain reset removed the record"
    assert parquet.is_file(), "a plain reset removed the record's archive"
    assert day in exported_days(stack.archive_root), "a plain reset cleared the day's mark"
    assert beside.is_file()

    if stack.backend == BACKEND_EMBEDDED:
        # The kept record is the one the redeployed stack reads.
        stack.env.pop(JOBS_CLOCK_ENV, None)
        _build(stack)
        _up(stack)
        _read_each_once(stack, *reread)

    _verb(stack, "reset", "-y", "--purge-audit")
    assert not record_present(), "--purge-audit kept the record"
    assert not (day_dir / EXPORT_SUBDIR).exists(), "--purge-audit kept the record's archive"
    assert day not in exported_days(stack.archive_root), "the purge left the day marked"
    assert beside.is_file(), "--purge-audit removed the rest of the archive"


# --------------------------------------------------------------------------- #
# Steps one case owns
# --------------------------------------------------------------------------- #


def _stall_a_writer(stack: Stack, identity: str) -> Path:
    """An open segment no writer appends to: a live source that never commits."""
    directory = _spool_dir(stack, identity)
    directory.mkdir(parents=True)
    segment = directory / segment_name("e2e-host", 4242, time.time_ns())
    segment.touch()
    _open_tree(directory)
    return segment


def _every_reader(stack: Stack, host: Transcript, debug_key: str) -> None:
    """Every read route answers the operator; the scoped reader sees its kind only."""
    run_id = next(iter(_read(stack, "entries", kind=KIND_RUN).get("data") or []), {}).get("run_id")
    assert run_id, "no run to read"
    arguments: dict[str, dict[str, str]] = {
        "list_sessions": {},
        "session_users": {},
        "timeline": {"session_id": host.session},
        "agent_timeline": {"session_id": host.session, "agent_id": _AGENT_ID},
        "full_entry": {"key": debug_key},
        "run": {"run_id": run_id},
        "run_timeline": {"run_id": run_id},
        "recent_audit": {},
        "bridge_history": {},
        "entries": {"kind": KIND_DEBUG},
    }
    assert set(arguments) == set(READ_METHODS), "a read route this case does not read"
    for name, params in arguments.items():
        answer = _read(stack, name, **params)
        assert answer.get("status") == "ok", f"{name}: {json.dumps(answer)[:1000]}"

    sessions = _read(stack, "list_sessions").get("data") or []
    assert host.session in json.dumps(sessions), "list_sessions omits the host session"

    login = f"{READER_LOGIN_PREFIX}{READER_NAME}"
    token = read_login(stack.repo / reader_file(READER_NAME), login)
    assert token, "no credential for the stated reader"
    scoped = _read(stack, "timeline", auth=(login, token), session_id=host.session)
    assert scoped.get("status") == "ok", scoped
    kinds = {e["kind"] for e in _entries(scoped.get("data") or [])}
    assert kinds and kinds <= set(READER_KINDS), f"the reader read {kinds}"

    health = _run(
        [str(stack.osprey), "agent-record", "health"],
        cwd=stack.repo,
        env={**stack.env, "OSPREY_AGENT_RECORD_URL": stack.url},
        timeout=300,
    )
    said = f"{health.stdout}\n{health.stderr}"
    assert "Traceback" not in said, said
    assert "Backend:" in health.stdout and HOST_IDENTITY in health.stdout, said


def _record_outage_loses_nothing(
    stack: Stack, monkeypatch: pytest.MonkeyPatch, session: str
) -> None:
    """Lines written while agent-record is down wait on the host, then land once."""
    _docker("stop", RECORD_CONTAINER, timeout=180)
    try:
        written = [
            _spool(stack, monkeypatch, HOST_IDENTITY, KIND_DEBUG, {"n": n}, session_id=session)
            for n in range(5)
        ]
    finally:
        _docker("start", RECORD_CONTAINER, timeout=180)
    _wait_healthy(stack)
    _read_each_once(stack, session, {spool_key(HOST_IDENTITY, e) for e in written})


def _store_outage_on_health(stack: Stack, monkeypatch: pytest.MonkeyPatch, session: str) -> None:
    """With the ClickHouse store down, health shows the backlog and, after five
    minutes without a commit, the stopped source; nothing is lost."""
    _docker("stop", RECORD_STORE_CONTAINER, timeout=180)
    try:
        written = [
            _spool(stack, monkeypatch, HOST_IDENTITY, KIND_DEBUG, {"n": n}, session_id=session)
            for n in range(5)
        ]
        _wait_health(
            stack,
            HOST_IDENTITY,
            lambda r: (r.get("backlog_lines") or 0) >= len(written),
            "the outage's backlog",
            timeout=READ_TIMEOUT_S,
        )
        _wait_health(stack, HOST_IDENTITY, lambda r: bool(r.get("stopped")), "a stopped source")
    finally:
        _docker("start", RECORD_STORE_CONTAINER, timeout=180)
    _read_each_once(
        stack,
        session,
        {spool_key(HOST_IDENTITY, e) for e in written},
        timeout=HEALTH_TIMEOUT_S,
    )
    _wait_health(
        stack,
        HOST_IDENTITY,
        lambda r: not r.get("stopped") and not r.get("backlog_lines"),
        "the drained backlog",
        timeout=HEALTH_TIMEOUT_S,
    )


def _history_page(stack: Stack) -> None:
    """The history page answers on its own host origin, behind the sign-in."""
    web = (stack.config.get("modules") or {}).get("web_terminals") or {}
    port = web.get("history_port") or default_port("history_origin", base=PORT_BASE)
    history = f"http://127.0.0.1:{port}/history/"
    # A person navigating: a program gets a bare 401, a page asks for the sign-in.
    navigation = {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}
    with httpx.Client(timeout=30) as anonymous:
        refused = anonymous.get(history, headers=navigation)
    assert refused.status_code == 302, f"{refused.status_code} {refused.text[:400]}"
    location = refused.headers.get("location", "")
    assert "/auth/enter" in location, location
    target = urlsplit(location)
    terminal_origin = (
        f"{target.scheme}://{target.netloc}"
        if target.netloc
        else f"http://127.0.0.1:{default_port('nginx', base=PORT_BASE)}"
    )

    with httpx.Client(timeout=30) as browser:
        login = browser.post(
            f"{terminal_origin}/auth/login",
            data={"user": PERSON, "password": PASSWORDS[PERSON]},
            headers={"Origin": terminal_origin, "Accept": "text/html"},
        )
        assert login.status_code == 303, f"{login.status_code} {login.text[:400]}"
        page = browser.get(history, headers=navigation, follow_redirects=True)
    assert page.status_code == 200, f"{page.status_code} {page.text[:400]}"
    assert "text/html" in page.headers.get("content-type", ""), page.headers


# --------------------------------------------------------------------------- #
# The cases
# --------------------------------------------------------------------------- #


def test_embedded_record_on_hello_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """hello-world on the embedded record: every fact once with its person,
    through a restart, a rewrite and an outage; every reader; health; the
    archived day; reset; and a profile with retired keys built as directed."""
    started_on = datetime.now(UTC).date()
    host = _transcript(str(uuid.uuid4()))
    dispatched = _transcript(str(uuid.uuid4()))
    repo = tmp_path / PROJECT_NAME
    before = _real_claude_entries(repo)

    with _deployed(
        tmp_path,
        monkeypatch,
        preset=HELLO_WORLD,
        transcript=host,
        retired_keys_first=True,
    ) as stack:
        # Started first: health may take the full five minutes to call it stopped.
        stalled_identity = "e2e-stalled-writer"
        stalled = _stall_a_writer(stack, stalled_identity)

        spooled = _write_spool_line(stack, monkeypatch, host.session)
        dispatch_identity = _write_dispatch_transcript(stack, dispatched)

        host_expected = host.keys(HOST_IDENTITY) | {spooled}
        dispatch_expected = dispatched.keys(dispatch_identity)
        found = _read_each_once(stack, host.session, host_expected)
        _assert_attributed(found, host_expected, identity=HOST_IDENTITY, person=HOST_IDENTITY)
        assert {KIND_THINKING, KIND_TOOL_CALL, KIND_SUBAGENT, KIND_DEBUG} <= {
            e["kind"] for e in found.values()
        }, found
        _read_each_once(stack, dispatched.session, dispatch_expected)
        _read_replay_once(stack)
        _attributed_runs(stack, monkeypatch)

        host_expected.add(_step_readable_within_a_minute(stack, host))
        _large_results_identical(stack, monkeypatch)
        _every_reader(stack, host, spooled)
        _refusal_on_health(stack, monkeypatch)
        _wait_health(stack, stalled_identity, lambda r: bool(r.get("stopped")), "a stalled writer")
        stalled.unlink()

        _restart_and_rewrite_keep_counts(
            stack, [(host.session, host_expected), (dispatched.session, dispatch_expected)]
        )
        outage_session = str(uuid.uuid4())
        _record_outage_loses_nothing(stack, monkeypatch, outage_session)

        started = _containers_named(RECORD_STORE_CONTAINER)
        assert started == [], f"an embedded deployment started a store container: {started}"

        day = _day_exported_on_host(stack, started_on)
        _reset_keeps_then_purges(stack, day, reread=(host.session, host_expected))

    assert _real_claude_entries(repo) == before, "the deployment wrote into the real ~/.claude"


def test_clickhouse_record_on_control_assistant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """control-assistant on the ClickHouse record: every fact once with its
    person, on-behalf and dispatched runs included, through a store outage on health, a restart and a rewrite; the history page; the
    archived day; reset."""
    started_on = datetime.now(UTC).date()
    host = _transcript(str(uuid.uuid4()))
    repo = tmp_path / PROJECT_NAME
    before = _real_claude_entries(repo)

    with _deployed(tmp_path, monkeypatch, preset=CONTROL_ASSISTANT, transcript=host) as stack:
        assert _containers_named(RECORD_STORE_CONTAINER), "no record store container"
        spooled = _write_spool_line(stack, monkeypatch, host.session)
        expected = host.keys(HOST_IDENTITY) | {spooled}
        found = _read_each_once(stack, host.session, expected)
        _assert_attributed(found, expected, identity=HOST_IDENTITY, person=HOST_IDENTITY)
        assert {KIND_THINKING, KIND_TOOL_CALL, KIND_SUBAGENT, KIND_DEBUG} <= {
            e["kind"] for e in found.values()
        }, found
        _read_replay_once(stack)
        _attributed_runs(stack, monkeypatch)

        expected.add(_step_readable_within_a_minute(stack, host))
        _large_results_identical(stack, monkeypatch)
        _refusal_on_health(stack, monkeypatch)
        _store_outage_on_health(stack, monkeypatch, str(uuid.uuid4()))
        _restart_and_rewrite_keep_counts(stack, [(host.session, expected)])
        _history_page(stack)

        day = _day_exported_on_host(stack, started_on)
        _reset_keeps_then_purges(stack, day, reread=(host.session, expected))

    assert _real_claude_entries(repo) == before, "the deployment wrote into the real ~/.claude"
