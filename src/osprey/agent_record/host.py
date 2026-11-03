"""The host-local agent record: the one agent-record service, launched on the host.

A host session (``osprey web``, ``chat``, ``query``, and every host command
that reads the record) needs a record when no stack is up. It gets the same
service the stack runs, ``osprey agent-record serve``, started as a local
process on the embedded backend over the project's own ``var/agent_record``,
``var/spool`` and the host build's Claude Code project directory, on the
same loopback port the stack publishes. One service, two launchers (compose
and this module), one owner of the DuckDB file at a time.

:func:`ensure_host_record` is the one entry. Its invariants:

- Everything happens under one exclusive lock file in the host identity's
  state directory (``var/state/<host identity>/``), the lock ``osprey up``
  takes for its handover too.
- It never spawns over a live port. A record that answers healthy and
  authenticates the host login is reused, whoever runs it. A port that
  answers but is unhealthy, not ready yet, or refuses the login is waited on
  up to the health bound, then refused. The one exception is the host-local
  process itself, still alive and answering not ready during its first
  ingest pass: it is waited on up to the first-pass bound, never killed for
  being slow.
- It spawns only when the connection is refused, no live ``stack-owner``
  marker is in the state directory, no process it started earlier is still
  coming up, and the backend is ``embedded``. A marker whose container is not
  running, or on a host with no container runtime, is stale and removed.
- With ``backend: clickhouse`` and nothing answering, it refuses before
  creating any file.
- Each holder writes a lease (its pid and process start time) and removes it
  on exit; the service, given the lease directory, stops itself once no
  lease names a live process. Whoever releases the last live lease, with no
  live marker, stops the process named by the pid-and-start-time file the
  spawner wrote, synchronously (the service seals, commits and closes on
  SIGTERM), so a one-shot command leaves no process behind.
- A process is identified by its pid AND its start time, never its pid
  alone: a recycled pid is a different process. The start time is read in
  UTC, so every process formats the same instant the same way.
- A long-lived holder (``osprey web``, ``chat``, ``query``) asks for a
  supervisor: while it holds its lease, a thread probes the record every few
  seconds and, when nothing answers and no live marker says the stack owns
  the record, starts the host-local process again under the same lock and
  the same rules. The holder's lease is the one it already holds, so the
  record comes back after ``osprey down`` or a crash without a second lease.
- The stack's agent-record never reads the host's transcripts: Claude Code
  writes them for the account that runs the session alone. While a
  ``stack-owner`` marker is in the state directory, the host session forwards
  them (:func:`forward_host_transcripts`): the supervisor every round, every
  holder as it ends, a one-shot holder as it starts, and a drained read from
  a host launch's children before it asks (:func:`forward_for_drain`). One pass
  runs at a time, under the forward lock; a pass that finds it busy skips its
  round.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from osprey.agent_record import credentials as creds
from osprey.agent_record import identities as ids
from osprey.agent_record.contract import HEALTH_SOURCES_PATH
from osprey_connectors.safety_file import atomic_replace

if TYPE_CHECKING:  # pragma: no cover - typing only
    from osprey.agent_record.config import AgentRecordConfig
    from osprey.agent_record.ingest.tailer import PollReport

logger = logging.getLogger(__name__)

__all__ = [
    "FIRST_PASS_WAIT_S",
    "FORWARD_CONFIG_ENV",
    "FORWARD_LOCK_FILE_NAME",
    "FORWARD_STATE_FILE_NAME",
    "FORWARD_WAIT_S",
    "HEALTH_WAIT_S",
    "LEASE_DIR_NAME",
    "LOCK_FILE_NAME",
    "LOG_FILE_NAME",
    "MARKER_FILE_NAME",
    "PROCESS_FILE_NAME",
    "SUPERVISE_INTERVAL_S",
    "HostRecord",
    "HostRecordRefused",
    "Lease",
    "any_live_lease",
    "claim_idle_stop",
    "ensure_host_record",
    "forward_for_drain",
    "forward_host_transcripts",
    "host_record_lock",
    "host_record_refusal",
    "host_state_dir",
    "live_leases",
    "marker_is_live",
    "running_record_pid",
    "stack_owns_record",
    "stop_host_process",
    "stop_unheld_record",
    "write_stack_marker",
]

#: The ensure lock, in the host identity's state directory.
LOCK_FILE_NAME = "agent-record.lock"
#: Written by ``osprey up`` while the stack's agent-record owns the store; its
#: first line names the container.
MARKER_FILE_NAME = "stack-owner"
#: The pid and start time of the host-local process, written by its spawner.
PROCESS_FILE_NAME = "agent-record.process"
#: One lease file per holder.
LEASE_DIR_NAME = "agent-record.leases"
#: The host-local process's output.
LOG_FILE_NAME = "agent-record.log"
#: The forward lock: one pass forwarding the host's transcripts at a time.
FORWARD_LOCK_FILE_NAME = "transcripts.forward.lock"
#: The positions the forwarder committed, per transcript file.
FORWARD_STATE_FILE_NAME = "transcripts.forwarded.json"
#: Set by a host launch for its children: the rendered config whose host
#: transcripts a drained read forwards first (:func:`forward_for_drain`).
FORWARD_CONFIG_ENV = "OSPREY_AGENT_RECORD_FORWARD_CONFIG"
#: Mode of every file in the state directory: the host identity's group
#: shares the lock, the marker, the process file and the leases.
_STATE_FILE_MODE = 0o664

#: Seconds an ensure waits for a record to answer healthy before it refuses.
HEALTH_WAIT_S = 60.0
#: Seconds an ensure waits for the host-local process's first ingest pass
#: (alive, answering not ready) before it refuses.
FIRST_PASS_WAIT_S = 1800.0
#: Seconds an ensure waits for the lock.
LOCK_WAIT_S = 180.0
#: Seconds a stop waits for the process to end before it kills it.
STOP_WAIT_S = 30.0
#: Seconds between two supervisor probes of a held record.
SUPERVISE_INTERVAL_S = 5.0
#: Seconds a forward a read or a holder's end asks for waits for one already running.
FORWARD_WAIT_S = 30.0
#: Seconds between two probes while waiting.
_PROBE_INTERVAL_S = 0.2
#: Seconds one health request may take.
_PROBE_TIMEOUT_S = 2.0

_LEASE_SUFFIX = ".lease"


class HostRecordRefused(Exception):
    """No record can be had for the host; the message names the fix."""


@dataclass(frozen=True)
class HostRecord:
    """A record a host session may use: its loopback URL and the host login."""

    url: str
    login: str
    token: str
    #: Whether this ensure started the process.
    started: bool = False

    def env(self) -> dict[str, str]:
        """The variables a child reads the record with (``RecordClient.from_env``)."""
        from osprey.agent_record.client import URL_ENV

        return {URL_ENV: self.url, creds.LOGIN_ENV: f"{self.login}:{self.token}"}

    def __repr__(self) -> str:  # the token never reaches a log line
        return f"HostRecord(url={self.url!r}, login={self.login!r}, started={self.started})"


@dataclass(frozen=True)
class Lease:
    """A process holding the host-local record, as a lease file names it."""

    pid: int
    start: str
    path: Path


# ---------------------------------------------------------------------------
# Process identity
# ---------------------------------------------------------------------------


def _process_start(pid: int) -> str | None:
    """The start time of live process *pid*, or ``None`` when none runs (a zombie neither)."""
    if pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        pass  # exists, owned by another uid
    try:
        out = subprocess.run(
            ["ps", "-o", "stat=,lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "LC_ALL": "C", "TZ": "UTC"},
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not out:
        return None
    status, _, start = out.partition(" ")
    if status.startswith("Z"):
        return None
    return " ".join(start.split())


def _alive(pid: int, start: str) -> bool:
    return bool(start) and _process_start(pid) == start


def _read_identity(path: Path) -> tuple[int, str] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        pid, start = data["pid"], data["start"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not isinstance(pid, int) or isinstance(pid, bool) or not isinstance(start, str):
        return None
    return pid, start


def _write_identity(path: Path, pid: int, start: str) -> None:
    """Write *path* whole: a reader sees the old file, the new one, or none."""
    data = json.dumps({"pid": pid, "start": start}).encode("utf-8")
    atomic_replace(path, data, mode=_STATE_FILE_MODE)


# ---------------------------------------------------------------------------
# Leases
# ---------------------------------------------------------------------------


def _scan_live_leases(lease_dir: Path) -> Iterator[Lease]:
    """The leases in *lease_dir* whose process still runs; nothing when it does not exist.

    Raises:
        OSError: If *lease_dir* exists but cannot be listed.
    """
    try:
        paths = sorted(p for p in Path(lease_dir).iterdir() if p.name.endswith(_LEASE_SUFFIX))
    except FileNotFoundError:
        return
    for path in paths:
        identity = _read_identity(path)
        if identity is not None and _alive(*identity):
            yield Lease(identity[0], identity[1], path)


def live_leases(lease_dir: Path) -> list[Lease]:
    """The leases in *lease_dir* whose process still runs."""
    try:
        return list(_scan_live_leases(lease_dir))
    except OSError:
        return []


def any_live_lease(lease_dir: Path) -> bool:
    """Whether any lease in *lease_dir* names a live process."""
    return next(_scan_live_leases(lease_dir), None) is not None


def _open_lock(state_dir: Path) -> int:
    """A descriptor on the ensure lock file of *state_dir*, created when missing."""
    return os.open(
        state_dir / LOCK_FILE_NAME, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, _STATE_FILE_MODE
    )


#: Ensure locks this process keeps until it exits (see :func:`claim_idle_stop`).
_HELD_UNTIL_EXIT: list[int] = []


def claim_idle_stop(lease_dir: Path) -> bool:
    """Whether the host-local service may stop now, decided under the ensure lock.

    Takes the ensure lock of ``lease_dir.parent`` without waiting; when it is
    busy (an ensure or a release is deciding right now) the answer is no.
    With the lock held it checks the leases again. When none names a live
    process, it keeps the lock until this process exits, so no ensure can
    probe and reuse a record that is about to go away, and answers yes.
    """
    state_dir = Path(lease_dir).parent
    fd = _open_lock(state_dir)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return False
    try:
        idle = not any_live_lease(lease_dir)
    except BaseException:
        os.close(fd)
        raise
    if not idle:
        os.close(fd)
        return False
    _HELD_UNTIL_EXIT.append(fd)
    return True


def _prune_leases(lease_dir: Path) -> None:
    """Remove the lease files whose process is gone."""
    live = {lease.path for lease in live_leases(lease_dir)}
    with contextlib.suppress(OSError):
        for path in Path(lease_dir).iterdir():
            if path.name.endswith(_LEASE_SUFFIX) and path not in live:
                with contextlib.suppress(OSError):
                    path.unlink()


def _write_lease(lease_dir: Path) -> Path:
    pid = os.getpid()
    start = _process_start(pid)
    if start is None:
        raise HostRecordRefused(
            "this process's start time cannot be read (`ps`), so it holds no lease"
        )
    lease_dir.mkdir(parents=True, exist_ok=True)
    path = lease_dir / f"{pid}-{uuid.uuid4().hex}{_LEASE_SUFFIX}"
    _write_identity(path, pid, start)
    return path


# ---------------------------------------------------------------------------
# Where things are, the lock and the marker
# ---------------------------------------------------------------------------


def host_state_dir(repo_root: Path | str, config: Mapping[str, Any]) -> Path:
    """``<repo>/var/state/<host identity>``, where the lock, marker, leases and pid file live.

    Raises:
        HostRecordRefused: If the deployment records no host sessions
            (``deployment.host_identity`` is not set) or the identity is not
            a valid directory name.
    """
    from osprey.deployment.compose_generator import state_identity_dir

    host = ids.host_identity(config)
    if host is None:
        raise HostRecordRefused(
            "this deployment records no host sessions: set `deployment.host_identity` "
            "and run `osprey build`"
        )
    try:
        return Path(state_identity_dir(repo_root, host))
    except ValueError as exc:
        raise HostRecordRefused(str(exc)) from exc


@contextlib.contextmanager
def host_record_lock(state_dir: Path, *, wait_s: float = LOCK_WAIT_S) -> Iterator[None]:
    """Hold the exclusive ensure lock of *state_dir*, waiting up to *wait_s* for it.

    Raises:
        HostRecordRefused: If the lock is not had within *wait_s*.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    fd = _open_lock(state_dir)
    try:
        deadline = time.monotonic() + wait_s
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise HostRecordRefused(
                        f"the agent-record lock {state_dir / LOCK_FILE_NAME} was not free "
                        f"within {wait_s:.0f} s; another `osprey` command holds it"
                    ) from None
                time.sleep(_PROBE_INTERVAL_S)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _container_running(name: str) -> bool:
    """Whether a container runtime on this host runs the container *name*."""
    runtime = shutil.which("docker") or shutil.which("podman")
    if runtime is None or not name:
        return False
    try:
        out = subprocess.run(
            [runtime, "inspect", "-f", "{{.State.Running}}", name],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0 and out.stdout.strip() == "true"


def write_stack_marker(state_dir: Path, container: str) -> None:
    """Write the ``stack-owner`` marker of *state_dir* whole, naming *container*.

    Call it under :func:`host_record_lock`; :func:`marker_is_live` reads it.
    """
    data = f"{container}\n".encode()
    atomic_replace(state_dir / MARKER_FILE_NAME, data, mode=_STATE_FILE_MODE)


def marker_is_live(state_dir: Path) -> bool:
    """Whether the ``stack-owner`` marker of *state_dir* names a running container.

    A stale marker (its container is not running, or there is no container
    runtime here) is removed. Call it under :func:`host_record_lock`.
    """
    path = state_dir / MARKER_FILE_NAME
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    except OSError:
        return True  # present but unreadable: never spawn under it
    name = text.strip().splitlines()[0].strip() if text.strip() else ""
    if _container_running(name):
        return True
    logger.info("removing stale agent-record marker %s (container %r not running)", path, name)
    with contextlib.suppress(FileNotFoundError):
        path.unlink()
    return False


def stack_owns_record(state_dir: Path) -> bool:
    """Whether a ``stack-owner`` marker is in *state_dir*; read only, never removed.

    The forwarder's question, asked without the ensure lock: a marker left by
    a stack that is gone only has the host's transcripts forwarded to a spool
    the next record ingests, which drops what it already holds.
    """
    try:
        return (state_dir / MARKER_FILE_NAME).exists()
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Forwarding the host's transcripts while the stack owns the record
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _forward_lock(state_dir: Path, wait_s: float) -> Iterator[bool]:
    """Hold the forward lock of *state_dir*; yields ``False`` when it stays busy past *wait_s*."""
    state_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(
        state_dir / FORWARD_LOCK_FILE_NAME,
        os.O_RDWR | os.O_CREAT | os.O_CLOEXEC,
        _STATE_FILE_MODE,
    )
    try:
        deadline = time.monotonic() + wait_s
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    yield False
                    return
                time.sleep(_PROBE_INTERVAL_S)
        try:
            yield True
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _forward_secrets(repo_root: Path) -> tuple[str, ...]:
    """The secret set scrubbed before a forwarded entry reaches the spool; none without one."""
    value = creds.read_env_value(repo_root / creds.SERVICE_FILE, creds.SECRETS_ENV)
    if value is None:
        return ()
    try:
        return tuple(creds.decode_secret_set(value))
    except ValueError:
        return ()


def forward_host_transcripts(
    repo_root: Path | str, config: Mapping[str, Any], *, wait_s: float = 0.0
) -> PollReport | None:
    """Forward the host build's transcripts to the stack's record, while the stack owns it.

    One pass of the transcript tailer over the host build's Claude Code project
    directory, read as the ``host`` root exactly as the host-local record reads
    it, into the host identity's own spool directory
    (:mod:`osprey.agent_record.ingest.forward`). Run by the account the session
    runs as, the one account that can read those files.

    Args:
        repo_root: The deployment repo root.
        config: The rendered config.
        wait_s: How long to wait for a pass another process is running.

    Returns:
        The pass's report, or ``None`` when no pass ran: the deployment records
        no host sessions, no ``stack-owner`` marker is there, another pass held
        the lock past *wait_s*, or the pass failed (logged). Never raises.
    """
    from osprey.agent_record.config import parse_agent_record_block
    from osprey.agent_record.ingest.forward import ForwardState, SpoolForward
    from osprey.agent_record.ingest.tailer import TranscriptTailer
    from osprey.deployment.compose_generator import (
        agent_record_host_projects_dir,
        ensure_spool_dir,
        spool_identity_dir,
    )
    from osprey.port_layout import resolve_port_base

    root = Path(repo_root)
    host = ids.host_identity(config)
    if host is None:
        return None
    try:
        state_dir = host_state_dir(root, config)
        if not stack_owns_record(state_dir):
            return None
        projects = agent_record_host_projects_dir(dict(config), root)
        assert projects is not None
        with _forward_lock(state_dir, wait_s) as had:
            if not had:
                return None
            ensure_spool_dir(root, host)
            sink = SpoolForward(
                spool_identity_dir(root, host), ForwardState(state_dir / FORWARD_STATE_FILE_NAME)
            )
            settings = parse_agent_record_block(config, base=resolve_port_base(config))
            tailer = TranscriptTailer(
                sink,
                {ids.HOST_TRANSCRIPT_ROOT: projects},
                config,
                secret_set=_forward_secrets(root),
                retention_days=settings.retention_days,
                prefixes={ids.HOST_TRANSCRIPT_ROOT: f"{projects.name}/"},
            )
            report = tailer.poll_once()
    except Exception as exc:  # the next pass retries
        logger.warning("the host transcripts were not forwarded to the stack's record: %s", exc)
        return None
    for where, why in report.errors.items():
        logger.warning("forwarding the host transcripts: %s: %s", where, why)
    return report


def forward_for_drain(config_path: Path | str) -> None:
    """Forward the host's transcripts before a drained read, for the config at *config_path*.

    A drain covers what the record's spool holds when it is asked, and the
    stack's record reads no host transcript itself: a host launch's children
    (:data:`FORWARD_CONFIG_ENV`) forward first, so a read of the session they
    run in finds its latest lines. Never raises.
    """
    import yaml

    from osprey.deployment.compose_generator import resolve_repo_root

    path = Path(config_path)
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("the host transcripts were not forwarded: %s is unreadable: %s", path, exc)
        return
    if not isinstance(config, dict):
        return
    forward_host_transcripts(
        resolve_repo_root(config, config_path=path), config, wait_s=FORWARD_WAIT_S
    )


# ---------------------------------------------------------------------------
# The process
# ---------------------------------------------------------------------------

#: Processes this process spawned, so a stop can reap its own child.
_CHILDREN: dict[int, subprocess.Popen[bytes]] = {}


def _running_process(state_dir: Path) -> tuple[int, str] | None:
    identity = _read_identity(state_dir / PROCESS_FILE_NAME)
    if identity is not None and _alive(*identity):
        return identity
    return None


def running_record_pid(state_dir: Path) -> int | None:
    """The pid of the live host-local record process of *state_dir*, else ``None``.

    Read-only: the process file is read and the pid's start time checked, so a
    stale file left by a process that is gone, or by a reused pid, names nothing.
    """
    running = _running_process(state_dir)
    return None if running is None else running[0]


def _wait_gone(pid: int, start: str, wait_s: float) -> bool:
    child = _CHILDREN.get(pid)
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if child is not None and child.poll() is not None:
            return True
        if not _alive(pid, start):
            return True
        time.sleep(0.05)
    return False


def stop_host_process(state_dir: Path, *, wait_s: float = STOP_WAIT_S) -> bool:
    """Stop the host-local process *state_dir*'s pid file names, and wait until it is gone.

    SIGTERM to its process group first (it seals, commits and closes the
    store), SIGKILL after *wait_s*. Call it under :func:`host_record_lock`.

    Returns:
        Whether a running process was stopped.
    """
    running = _running_process(state_dir)
    stopped = False
    if running is not None:
        pid, start = running
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pid, signal.SIGTERM)
        if not _wait_gone(pid, start, wait_s):
            logger.warning("agent-record (pid %d) did not stop in %.0f s; killing it", pid, wait_s)
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pid, signal.SIGKILL)
            _wait_gone(pid, start, 5.0)
        stopped = True
    for pid in [p for p, child in _CHILDREN.items() if child.poll() is not None]:
        _CHILDREN.pop(pid, None)
    with contextlib.suppress(FileNotFoundError):
        (state_dir / PROCESS_FILE_NAME).unlink()
    return stopped


def _service_secrets(repo_root: Path) -> tuple[str, str]:
    """The verifier and secrets the host-local service starts with, from the service file.

    Raises:
        HostRecordRefused: If the service file is missing or lacks either.
    """
    service_file = repo_root / creds.SERVICE_FILE
    verifier = creds.read_env_value(service_file, creds.VERIFIER_ENV)
    secrets = creds.read_env_value(service_file, creds.SECRETS_ENV)
    if verifier is None or secrets is None:
        raise HostRecordRefused(
            f"{service_file} is missing or incomplete, so agent-record cannot start: "
            f"run `osprey build`, or, in a container, mount the build's service file at "
            f"{service_file}"
        )
    return verifier, secrets


def _child_env(
    repo_root: Path, config: Mapping[str, Any], state_dir: Path, config_path: Path
) -> dict[str, str]:
    from osprey.agent_record import service
    from osprey.deployment.compose_generator import agent_record_host_projects_dir
    from osprey.utils.workspace import RECORD_DIR_RELPATH, SPOOL_DIR_RELPATH

    verifier, secrets = _service_secrets(repo_root)
    host_projects = agent_record_host_projects_dir(dict(config), repo_root)
    env = dict(os.environ)
    for key in (service.CLICKHOUSE_URL_ENV, creds.LOGIN_ENV):
        env.pop(key, None)
    env.update(
        {
            creds.VERIFIER_ENV: verifier,
            creds.SECRETS_ENV: secrets,
            "OSPREY_CONFIG": str(config_path),
            service.SPOOL_ROOT_ENV: str(repo_root / SPOOL_DIR_RELPATH),
            service.RECORD_DIR_ENV: str(repo_root / RECORD_DIR_RELPATH),
            # No container volume is mounted on the host: every root but the
            # host's own is absent, and absent roots are skipped.
            service.TRANSCRIPTS_DIR_ENV: str(state_dir / "no-container-transcripts"),
            service.HOST_PROJECTS_ENV: str(host_projects),
        }
    )
    return env


def _spawn(
    repo_root: Path, config: Mapping[str, Any], state_dir: Path, port: int, config_path: Path
) -> subprocess.Popen[bytes]:
    env = _child_env(repo_root, config, state_dir, config_path)
    command = [
        sys.executable,
        "-m",
        "osprey.cli.main",
        "agent-record",
        "serve",
        "--port",
        str(port),
        "--lease-dir",
        str(state_dir / LEASE_DIR_NAME),
    ]
    with open(state_dir / LOG_FILE_NAME, "ab") as log:
        child = subprocess.Popen(
            command,
            cwd=repo_root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    start = None
    for _ in range(50):
        start = _process_start(child.pid)
        if start is not None or child.poll() is not None:
            break
        time.sleep(0.02)
    if start is None:
        child.wait()
        raise HostRecordRefused(f"agent-record exited at start; see {state_dir / LOG_FILE_NAME}")
    _CHILDREN[child.pid] = child
    _write_identity(state_dir / PROCESS_FILE_NAME, child.pid, start)
    logger.info("started the host-local agent-record (pid %d) on port %d", child.pid, port)
    return child


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

_DOWN = "down"
_READY = "ready"
_STARTING = "starting"
_UNHEALTHY = "unhealthy"


def _probe(url: str, login: str, token: str) -> tuple[str, str]:
    """``(state, why)``: down (refused), ready, starting (not ready yet), or unhealthy."""
    import httpx

    try:
        with httpx.Client(timeout=_PROBE_TIMEOUT_S, trust_env=False) as client:
            health = client.get(f"{url}/health")
            if health.status_code != 200:
                return _UNHEALTHY, f"/health answered {health.status_code}"
            try:
                body = health.json()
            except ValueError:
                return _UNHEALTHY, "/health answered no JSON"
            if isinstance(body, dict) and body.get("ready") is False:
                return _STARTING, "it is not ready yet"
            sources = client.get(f"{url}{HEALTH_SOURCES_PATH}", auth=(login, token))
            if sources.status_code != 200:
                return _UNHEALTHY, (
                    f"it refused the host login {login!r} ({sources.status_code}); "
                    "another project's record may hold the port"
                )
            return _READY, ""
    except httpx.ConnectError:
        return _DOWN, "nothing answers"
    except httpx.HTTPError as exc:
        return _UNHEALTHY, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Ensure
# ---------------------------------------------------------------------------


def _clickhouse_refusal(port: int) -> HostRecordRefused:
    return HostRecordRefused(
        f"no agent-record answers on 127.0.0.1:{port}, and this project's record is "
        "ClickHouse, which runs only in the stack: start it with `osprey up`, or set "
        "`agent_record.backend: embedded`"
    )


def _acquire(
    repo_root: Path,
    config: Mapping[str, Any],
    state_dir: Path,
    record: HostRecord,
    port: int,
    embedded: bool,
    wait_s: float,
    config_path: Path,
) -> tuple[Path, bool]:
    """Under the lock: reuse, wait or spawn; returns ``(lease path, started)``."""
    lease_dir = state_dir / LEASE_DIR_NAME
    _prune_leases(lease_dir)
    lease: Path | None = None
    child: subprocess.Popen[bytes] | None = None
    begun = time.monotonic()
    why = ""
    try:
        while True:
            state, why = _probe(record.url, record.login, record.token)
            if state == _READY:
                break
            bound = wait_s
            if state == _STARTING and (
                (child is not None and child.poll() is None)
                or (child is None and _running_process(state_dir) is not None)
            ):
                # The host-local process's first ingest pass: slow, not broken.
                bound = max(wait_s, FIRST_PASS_WAIT_S)
            if state == _DOWN and child is not None and child.poll() is not None:
                raise HostRecordRefused(
                    f"agent-record exited ({child.returncode}) before it answered; "
                    f"see {state_dir / LOG_FILE_NAME}"
                )
            if state == _DOWN and child is None:
                if marker_is_live(state_dir):
                    why = "the stack owns the record (`osprey up`) but it does not answer"
                elif _running_process(state_dir) is not None:
                    why = "the host-local agent-record started earlier does not answer yet"
                elif not embedded:
                    raise _clickhouse_refusal(port)
                else:
                    lease = _write_lease(lease_dir)
                    child = _spawn(repo_root, config, state_dir, port, config_path)
                    # The wait for a spawned child counts from its start, never
                    # from a probe that saw the port empty before it existed.
                    begun = time.monotonic()
            if time.monotonic() - begun >= bound:
                where = f"; see {state_dir / LOG_FILE_NAME}" if child is not None else ""
                raise HostRecordRefused(
                    f"the agent-record on {record.url} is not usable after {bound:.0f} s: "
                    f"{why}{where}"
                )
            time.sleep(_PROBE_INTERVAL_S)
        if lease is None:
            lease = _write_lease(lease_dir)
        return lease, child is not None
    except BaseException:
        if lease is not None:
            with contextlib.suppress(FileNotFoundError):
                lease.unlink()
        if child is not None:
            stop_host_process(state_dir)
        raise


def _release(state_dir: Path, lease: Path) -> None:
    """Under the lock: drop *lease*; the last live lease stops the host-local process."""
    with contextlib.suppress(FileNotFoundError):
        lease.unlink()
    if any_live_lease(state_dir / LEASE_DIR_NAME):
        return
    if marker_is_live(state_dir):
        return
    stop_host_process(state_dir)


def stop_unheld_record(state_dir: Path, *, wait_s: float = LOCK_WAIT_S) -> bool:
    """Stop the host-local process when no live lease holds it and no live marker exists.

    For a command that ended a holder from outside (``osprey web stop``): the
    holder's own release normally stops the process, and this makes sure of it
    when the holder died before it could.

    Returns:
        Whether a running process was stopped.

    Raises:
        HostRecordRefused: If the lock is not had within *wait_s*.
    """
    with host_record_lock(state_dir, wait_s=wait_s):
        if any_live_lease(state_dir / LEASE_DIR_NAME) or marker_is_live(state_dir):
            return False
        return stop_host_process(state_dir)


class _Supervisor:
    """Brings the host-local record back while its holder holds its lease.

    Every *interval* seconds it probes the record. When nothing answers it
    takes the ensure lock (skipping the round when the lock is busy), probes
    again, and starts the process only under the rules an ensure spawns by:
    no live marker, no process of ours still coming up, the embedded backend.
    It never waits for the new process to answer; the next rounds see it.
    Every round then forwards the host's transcripts while the stack owns the
    record (:func:`forward_host_transcripts`), skipping a busy round.
    """

    def __init__(
        self,
        repo_root: Path,
        config: Mapping[str, Any],
        state_dir: Path,
        record: HostRecord,
        port: int,
        embedded: bool,
        lease: Path,
        config_path: Path,
        interval: float,
    ) -> None:
        self._repo_root = repo_root
        self._config = config
        self._state_dir = state_dir
        self._record = record
        self._port = port
        self._embedded = embedded
        self._lease = lease
        self._config_path = config_path
        self._interval = interval
        self._stopped = threading.Event()
        self._warned_clickhouse = False
        self._thread = threading.Thread(
            target=self._run, name="agent-record-supervisor", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        self._thread.join(timeout=LOCK_WAIT_S)

    def _run(self) -> None:
        while not self._stopped.wait(self._interval):
            try:
                self.tick()
            except Exception as exc:  # a round that fails is retried next round
                logger.warning("the agent-record supervisor round failed: %s", exc)
            forward_host_transcripts(self._repo_root, self._config)

    def tick(self) -> bool:
        """One round; returns whether it started the process."""
        record = self._record
        if _probe(record.url, record.login, record.token)[0] != _DOWN:
            return False
        try:
            with host_record_lock(self._state_dir, wait_s=self._interval):
                if self._stopped.is_set():
                    return False
                if _probe(record.url, record.login, record.token)[0] != _DOWN:
                    return False
                if marker_is_live(self._state_dir) or _running_process(self._state_dir):
                    return False
                if not self._embedded:
                    if not self._warned_clickhouse:
                        self._warned_clickhouse = True
                        logger.warning("%s", _clickhouse_refusal(self._port))
                    return False
                self._keep_lease()
                _spawn(
                    self._repo_root, self._config, self._state_dir, self._port, self._config_path
                )
                return True
        except HostRecordRefused as exc:
            logger.warning("the agent-record supervisor skipped a round: %s", exc)
            return False

    def _keep_lease(self) -> None:
        """Write the holder's lease again if something removed it."""
        if self._lease.exists():
            return
        pid = os.getpid()
        start = _process_start(pid)
        if start is not None:
            self._lease.parent.mkdir(parents=True, exist_ok=True)
            _write_identity(self._lease, pid, start)


class _HostSite(NamedTuple):
    """Where a project's host record lives and who reaches it."""

    state_dir: Path
    url: str
    login: str
    token: str
    settings: AgentRecordConfig


def _host_site(root: Path, config: Mapping[str, Any]) -> _HostSite:
    """The host record's state dir, loopback URL, host login and settings.

    Raises:
        HostRecordRefused: If the deployment records no host sessions, the
            host login file is missing, or the backend is ClickHouse and no
            record answers.
    """
    from osprey.agent_record.config import BACKEND_EMBEDDED, parse_agent_record_block
    from osprey.port_layout import resolve_port_base

    state_dir = host_state_dir(root, config)
    host = ids.host_identity(config)
    assert host is not None
    login = ids.login_for(host)
    token = creds.read_login(root / creds.login_file(login), login)
    if token is None:
        path = root / creds.login_file(login)
        raise HostRecordRefused(
            f"the host login file {path} is missing: run `osprey build`, or, in a container, "
            f"mount the build's login file at {path}"
        )
    settings = parse_agent_record_block(config, base=resolve_port_base(config))
    url = f"http://127.0.0.1:{settings.port}"
    if settings.backend != BACKEND_EMBEDDED and _probe(url, login, token)[0] == _DOWN:
        raise _clickhouse_refusal(settings.port)
    return _HostSite(state_dir, url, login, token, settings)


def host_record_refusal(repo_root: Path | str, config: Mapping[str, Any]) -> str | None:
    """Why :func:`ensure_host_record` would refuse right now, or ``None``; starts nothing.

    For a launcher that hands the ensure to a process it starts in the
    background (``osprey web --detach``) and must refuse in the foreground,
    where the operator reads it. Checks the host identity, the host login,
    the service file, and, for ClickHouse, whether a record answers.
    """
    from osprey.agent_record.config import BACKEND_EMBEDDED

    root = Path(repo_root)
    try:
        if _host_site(root, config).settings.backend == BACKEND_EMBEDDED:
            _service_secrets(root)
    except HostRecordRefused as exc:
        return str(exc)
    return None


@contextlib.contextmanager
def ensure_host_record(
    repo_root: Path | str,
    config: Mapping[str, Any],
    *,
    config_path: Path | str | None = None,
    wait_s: float = HEALTH_WAIT_S,
    supervise_s: float | None = None,
) -> Iterator[HostRecord]:
    """Hold a usable agent record for a host session, starting one when none runs.

    Call it only when ``OSPREY_AGENT_RECORD_URL`` is not already set (a
    container is handed its record by its render). On exit the lease is
    released, and the last holder stops the host-local process. While the
    stack owns the record, the host's transcripts are forwarded to it as the
    holder starts (one-shot), every supervisor round (long-lived) and as it
    ends (:func:`forward_host_transcripts`).

    Args:
        repo_root: The deployment repo root.
        config: The rendered config.
        config_path: The rendered config file the spawned service loads;
            ``<repo>/build/config.yml`` when ``None``.
        wait_s: How long to wait for a record to answer healthy.
        supervise_s: When given, a supervisor probes the record every
            *supervise_s* seconds while the lease is held and starts the
            host-local process again when nothing answers and no live marker
            exists. A long-lived launcher asks for it; a one-shot command
            does not.

    Yields:
        The record's loopback URL and the host login.

    Raises:
        HostRecordRefused: With the fix, when no record can be had: the
            deployment records no host sessions, the host login or the
            service file is missing (`osprey build`), the backend is
            ClickHouse and no stack answers, or the port stays unhealthy or
            refuses the login.
    """
    from osprey.agent_record.config import BACKEND_EMBEDDED
    from osprey_connectors.workspace import RENDERED_CONFIG_RELPATH

    root = Path(repo_root)
    state_dir, url, login, token, settings = _host_site(root, config)
    embedded = settings.backend == BACKEND_EMBEDDED
    path = Path(config_path) if config_path is not None else root / RENDERED_CONFIG_RELPATH
    handle = HostRecord(url, login, token)
    with host_record_lock(state_dir):
        lease, started = _acquire(
            root, config, state_dir, handle, settings.port, embedded, wait_s, path
        )
    supervisor = None
    if supervise_s is not None:
        supervisor = _Supervisor(
            root, config, state_dir, handle, settings.port, embedded, lease, path, supervise_s
        )
        supervisor.start()
    else:
        # A one-shot holder reads what the host's sessions left, not what it runs.
        forward_host_transcripts(root, config, wait_s=FORWARD_WAIT_S)
    try:
        yield HostRecord(url, login, token, started=started)
    finally:
        if supervisor is not None:
            supervisor.stop()
        # What the holder's own session wrote last reaches the stack's record.
        forward_host_transcripts(root, config, wait_s=FORWARD_WAIT_S)
        try:
            with host_record_lock(state_dir):
                _release(state_dir, lease)
        except HostRecordRefused as exc:
            # Never mask the caller's own error; the lease dies with this process.
            logger.warning("the agent-record lease %s was not released: %s", lease, exc)
