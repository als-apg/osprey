"""Capture a child process's output to a spool file instead of the terminal.

The lifecycle verbs report progress as a handful of phase lines (see
:mod:`osprey.cli.phase_reporter`). That reporting is only readable if the
subprocesses underneath it — image builds, compose invocations — stop writing
their own thousands of lines to the same terminal. :func:`run_captured` is the
one seam that makes that true: the child's stdout and stderr go to
``<repo>/var/logs/<spool_name>-<timestamp>.log`` as a single merged stream, and
the file is named on the failure path so nothing is lost.

A captured child owns no terminal: its stdin is the null device, so a prompt it
prints reads EOF and takes its own default instead of waiting for an answer
nobody can see. Compose does this for a named volume whose labels no longer
match the compose file: it keeps the volume and exits 0. Because that success
is silent, every captured run's spool is scanned for it and a warning names the
volume.

Under the global ``--verbose`` flag the helper is a pass-through: the child
inherits this process's stdio and streams straight to the terminal, with no
redirection kwargs at all. Verbosity is read from the reporter's module-level
singleton rather than threaded through every caller, so the two surfaces cannot
disagree about which mode a run is in.

The default reporter is a quiet ``NullReporter(verbose=False)``, so calls made
before a verb installs a reporter still capture. That is deliberate: capturing
is the safe default, and the spool path is reported on failure either way.
"""

from __future__ import annotations

import contextlib
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, NamedTuple, TypeVar

from osprey.cli.phase_reporter import current_reporter, is_verbose
from osprey.deployment.errors import CapturedProcessError, CapturedProcessTimeout
from osprey_connectors.logger import get_logger

logger = get_logger("deployment.capture")

#: Spool directory, relative to the deployment repo root.
SPOOL_DIR = Path("var") / "logs"

#: How many spool files survive a prune. Older ones are removed at entry, so a
#: long-lived deployment repo keeps the recent runs without growing unbounded.
SPOOL_RETENTION = 20

#: What compose prints for a named volume whose labels differ from the compose
#: file. With stdin at EOF it answers its own "Recreate?" prompt No.
_VOLUME_MISMATCH_MARKER = "exists but doesn't match configuration"
_QUOTED = re.compile(r'"([^"]+)"')

#: How long a stopped child's process group gets to exit on SIGTERM before it
#: is killed outright.
_STOP_GRACE_S = 10.0

#: How long a process group gets to be gone after SIGKILL. A process still there
#: after this is stuck in the kernel, not slow.
_KILL_WAIT_S = 5.0

#: How often a stopping process group is checked for members still running.
_GROUP_POLL_S = 0.05

#: How long a liveness probe waits for another thread's reap of a child to set
#: its ``returncode``. A reap that does not land in this window was made by
#: something other than the child's stop.
_REAP_SETTLE_S = 1.0

#: Called while a bounded run is still going: the seconds it has run, and the
#: spool its output is going to (``None`` under ``--verbose``).
Watch = Callable[[float, Path | None], None]


class CapturedProcess(subprocess.CompletedProcess[bytes]):
    """A finished captured run, naming the file its output went to.

    A plain :class:`subprocess.CompletedProcess` plus ``spool_path``: the spool
    this run's output was written to, or ``None`` in verbose mode where it went
    straight to the terminal. Declaring it as a type rather than setting the
    attribute dynamically is what lets a caller read ``result.spool_path``
    without a type checker rejecting it.

    ``stdout``/``stderr`` stay ``None`` for a captured run — that output is in
    the spool file, not in memory — but the constructor accepts them so this
    remains substitutable for its base class.
    """

    def __init__(
        self,
        args: Sequence[str] | str,
        returncode: int,
        stdout: bytes | None = None,
        stderr: bytes | None = None,
        *,
        spool_path: Path | None = None,
    ) -> None:
        super().__init__(args, returncode, stdout, stderr)
        self.spool_path = spool_path


def _prune_spools(spool_dir: Path, keep: int = SPOOL_RETENTION) -> None:
    """Delete all but the newest ``keep`` spool files, best-effort.

    Retention is a housekeeping nicety; a directory that cannot be read or an
    entry that cannot be removed must never take down the deploy that was about
    to run, so every failure here is a debug line and nothing more.
    """
    try:
        existing = sorted(spool_dir.glob("*.log"), key=lambda p: p.stat().st_mtime)
    except OSError as exc:  # pragma: no cover - depends on filesystem state
        logger.debug("Could not list spool directory %s: %s", spool_dir, exc)
        return
    for stale in existing[: max(0, len(existing) - keep)]:
        try:
            stale.unlink()
        except OSError as exc:  # pragma: no cover - depends on filesystem state
            logger.debug("Could not prune spool %s: %s", stale, exc)


def _spool_path(spool_dir: Path, spool_name: str) -> Path:
    """Return an unused spool path for ``spool_name`` in ``spool_dir``.

    The timestamp sorts lexicographically, which is what makes the directory
    listing readable. Two runs can still land in the same second — a compose
    ``rm`` followed immediately by a build — so the name takes a numeric suffix
    until it is free rather than overwriting the earlier run's output.
    """
    stamp = time.strftime("%Y%m%d-%H%M%S")
    candidate = spool_dir / f"{spool_name}-{stamp}.log"
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = spool_dir / f"{spool_name}-{stamp}-{suffix}.log"
    return candidate


def _mismatched_volumes(text: str) -> list[str]:
    """Names of the volumes compose reported as not matching the compose file.

    Each name appears once, in the order compose first reported it. Only lines
    carrying the mismatch marker are read, so other quoted text in the output
    never counts.
    """
    names: list[str] = []
    for line in text.splitlines():
        if _VOLUME_MISMATCH_MARKER not in line:
            continue
        for name in _QUOTED.findall(line):
            if name not in names:
                names.append(name)
    return names


def _warn_on_kept_volumes(spool_path: Path) -> None:
    """Warn once when a run's spool shows compose keeping a mismatched volume.

    Compose answers its own "Recreate?" prompt No when stdin is at EOF, keeps
    the volume and exits 0, so neither the exit code nor any failure diagnoser
    would otherwise surface it. Best-effort: an unreadable spool is skipped.
    """
    try:
        text = spool_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:  # pragma: no cover - depends on filesystem state
        logger.debug("Could not scan spool %s: %s", spool_path, exc)
        return
    volumes = _mismatched_volumes(text)
    if not volumes:
        return
    logger.warning(
        "Compose kept the existing volume(s) %s unchanged although they do not match "
        "the compose file (see %s). This is expected once after an OSPREY upgrade "
        "that changed volume labels; if it persists, `osprey status` shows the "
        "deployment's volumes.",
        ", ".join(volumes),
        spool_path,
    )


def run_captured(
    cmd: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    spool_name: str,
    repo_root: Path | str | None = None,
    check: bool = True,
    on_line: Callable[[str], None] | None = None,
    timeout: float | None = None,
    watch: Watch | None = None,
    watch_interval: float = 30.0,
) -> CapturedProcess:
    """Run ``cmd``, spooling its merged output to ``<repo>/var/logs/``.

    :param cmd: The argv to run. Passed through unchanged in both modes.
    :param env: Environment for the child, as for :func:`subprocess.run`.
    :param cwd: Working directory for the child, as for :func:`subprocess.run`.
        Unrelated to ``repo_root``: a scaffolded script is run from the project
        root so its own relative paths resolve, while its output still spools
        under the deployment repo. Forwarded in both modes, so a run behaves
        the same way under ``--verbose``.
    :param spool_name: Descriptive stem for the spool file (``"build-worker"``).
    :param repo_root: Deployment repo the ``var/logs/`` directory hangs off.
        Callers already hold this (``resolve_repo_root(config)``); ``None``
        means the current directory, the same default the compose ``--env-file``
        helpers use. No repo discovery happens here.
    :param check: Raise :class:`CapturedProcessError` on a non-zero exit.
    :param on_line: Watch the child's merged stream, line by line, as decoded
        text without its newline. The spool stays byte-for-byte complete either
        way — the callback is a tee, not a diversion — and it is cosmetic by
        contract: an exception it raises is logged at debug and changes nothing
        about the run's outcome. Ignored under ``--verbose``, where the child's
        own lines are already on the terminal and derived progress would only
        duplicate them.
    :param timeout: Seconds the child may run. Past it the child is stopped and
        :class:`CapturedProcessTimeout` is raised, whatever ``check`` says: a
        run that never finished has no exit status to judge. ``None`` waits for
        as long as the child runs.
    :param watch: Called every ``watch_interval`` seconds while the child is
        still running, with the seconds it has run and its spool path. Cosmetic
        by contract, like ``on_line``: an exception it raises is logged at debug
        and changes nothing about the run.
    :param watch_interval: Seconds between ``watch`` calls.
    :returns: A :class:`CapturedProcess`, whose ``spool_path`` is the file this
        run's output went to, or ``None`` in verbose mode.
        ``stdout``/``stderr`` are ``None`` — the output is in that file (or on
        the terminal). A ``check=False`` caller that wants to point at the
        output reads it from there; the reporter's copy only covers runs made
        while a phase is open.
    :raises CapturedProcessError: ``check`` is set and the child exited
        non-zero. The exception carries the spool path.

    Under ``--verbose`` the child inherits this process's stdio: same argv, no
    redirection at all, so long-running builds stream live.

    A bounded run (``timeout`` or ``watch`` given) starts the child in a process
    group of its own, so that stopping it stops everything it started: ``podman
    compose`` hands the work to a separate provider process, and stopping only
    the first would leave the second running. The same group keeps Ctrl-C from
    reaching the child directly, so an interrupt stops the group here before it
    propagates.
    """
    if timeout is not None or watch is not None:
        if on_line is not None:
            raise TypeError("run_captured: on_line cannot be combined with timeout or watch")
        return _run_bounded(
            cmd,
            env=env,
            cwd=cwd,
            spool_name=spool_name,
            repo_root=repo_root,
            check=check,
            timeout=timeout,
            watch=watch,
            watch_interval=watch_interval,
        )

    if is_verbose():
        # Spool-less in both senses: no file was written, and the result says so
        # explicitly, so a caller can read ``spool_path`` unconditionally.
        completed = subprocess.run(cmd, env=env, cwd=cwd)
        if check and completed.returncode != 0:
            raise CapturedProcessError(cmd, completed.returncode)
        return CapturedProcess(cmd, completed.returncode)

    spool_dir = Path(repo_root or Path.cwd()) / SPOOL_DIR
    spool_dir.mkdir(parents=True, exist_ok=True)
    _prune_spools(spool_dir)
    spool_path = _spool_path(spool_dir, spool_name)

    logger.debug("Capturing %s output to %s", spool_name, spool_path)
    with open(spool_path, "wb") as spool:
        # The reporter keeps the path for the whole phase, not just this call:
        # an interrupt during the child needs to name the partial output.
        current_reporter().note_spool(spool_path)
        if on_line is None:
            # stderr into the same handle, not a second file — a build's errors
            # are only legible interleaved with the step they interrupted.
            completed = subprocess.run(
                cmd,
                env=env,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=spool,
                stderr=subprocess.STDOUT,
            )
        else:
            completed = _run_teeing(cmd, env=env, cwd=cwd, spool=spool, on_line=on_line)

    _warn_on_kept_volumes(spool_path)

    if check and completed.returncode != 0:
        raise CapturedProcessError(cmd, completed.returncode, spool_path)

    # The spool goes on the result, not only on the reporter: an advisory caller
    # that does not raise still has to be able to name the file its output went
    # to, and it may be running with no phase open at all.
    return CapturedProcess(cmd, completed.returncode, spool_path=spool_path)


def _run_teeing(
    cmd: Sequence[str],
    *,
    env: Mapping[str, str] | None,
    cwd: Path | str | None,
    spool,
    on_line: Callable[[str], None],
) -> subprocess.CompletedProcess:
    """The watched variant of the capture: every line to the spool AND the callback.

    The direct-to-file ``subprocess.run`` cannot offer a per-line hook, so this
    path reads the child's merged stream through a pipe and writes it on. The
    bytes reaching the spool are the child's own; only the callback gets a
    decoded copy, with U+FFFD standing in for anything that is not UTF-8 —
    a build log's occasional garbage must not take down the build.

    The callback is fenced per line rather than per run: one line the parser
    chokes on costs that line's step at debug level, not every step after it.
    """
    with subprocess.Popen(
        cmd,
        env=env,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ) as process:
        assert process.stdout is not None  # PIPE above guarantees it
        for raw in process.stdout:
            spool.write(raw)
            try:
                on_line(raw.decode("utf-8", errors="replace").rstrip("\r\n"))
            except Exception as exc:
                logger.debug("on_line callback failed on %r: %s", raw[:200], exc)
        returncode = process.wait()
    return subprocess.CompletedProcess(list(cmd), returncode)


def _run_bounded(
    cmd: Sequence[str],
    *,
    env: Mapping[str, str] | None,
    cwd: Path | str | None,
    spool_name: str,
    repo_root: Path | str | None,
    check: bool,
    timeout: float | None,
    watch: Watch | None,
    watch_interval: float,
) -> CapturedProcess:
    """:func:`run_captured` for a run with a time limit or a watch on it.

    A ``Popen`` rather than ``subprocess.run``, because stopping the run has to
    reach the child's whole process group and only ``Popen`` hands back the pid
    that names it. Waited on in slices, with the watch called between them.
    """
    if is_verbose():
        # In this process's own group: the child shares the terminal and may ask
        # it a question, and a child in a background group that reads the
        # terminal is stopped by the kernel, which would be a hang of its own.
        process = subprocess.Popen(cmd, env=env, cwd=cwd)
        returncode = _wait_bounded(
            process, cmd, None, timeout, watch, watch_interval, own_group=False
        )
        if check and returncode != 0:
            raise CapturedProcessError(cmd, returncode)
        return CapturedProcess(cmd, returncode)

    spool_dir = Path(repo_root or Path.cwd()) / SPOOL_DIR
    spool_dir.mkdir(parents=True, exist_ok=True)
    _prune_spools(spool_dir)
    spool_path = _spool_path(spool_dir, spool_name)

    logger.debug("Capturing %s output to %s", spool_name, spool_path)
    with open(spool_path, "wb") as spool:
        current_reporter().note_spool(spool_path)
        process = subprocess.Popen(
            cmd,
            env=env,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=spool,
            stderr=subprocess.STDOUT,
            process_group=0,
        )
        returncode = _wait_bounded(
            process, cmd, spool_path, timeout, watch, watch_interval, own_group=True
        )

    _warn_on_kept_volumes(spool_path)

    if check and returncode != 0:
        raise CapturedProcessError(cmd, returncode, spool_path)
    return CapturedProcess(cmd, returncode, spool_path=spool_path)


def _wait_bounded(
    process: subprocess.Popen,
    cmd: Sequence[str],
    spool_path: Path | None,
    timeout: float | None,
    watch: Watch | None,
    watch_interval: float,
    *,
    own_group: bool,
) -> int:
    """Wait for ``process``, calling ``watch`` between slices; stop it past ``timeout``.

    Anything that leaves this early — the time limit, Ctrl-C, an error — stops
    the child first (its whole process group when it has one of its own), so no
    part of it outlives the run that started it.
    """
    start = time.monotonic()
    next_watch = start + watch_interval
    try:
        while True:
            now = time.monotonic()
            slice_s = next_watch - now if watch is not None else None
            if timeout is not None:
                remaining = start + timeout - now
                slice_s = remaining if slice_s is None else min(slice_s, remaining)
            try:
                return process.wait(timeout=None if slice_s is None else max(slice_s, 0.0))
            except subprocess.TimeoutExpired:
                pass
            now = time.monotonic()
            if timeout is not None and now - start >= timeout:
                raise CapturedProcessTimeout(cmd, timeout, spool_path)
            if watch is not None and now >= next_watch:
                next_watch += watch_interval
                try:
                    watch(now - start, spool_path)
                except Exception as exc:
                    logger.debug("watch callback failed: %s", exc)
    except BaseException:
        _stop(process, own_group=own_group)
        raise


_T = TypeVar("_T")


def _pid_name(pid: object) -> str:
    return f"pid {pid}"


def _group_name(pgid: object) -> str:
    return f"group {pgid}"


@dataclass(frozen=True)
class StopOutcome(Generic[_T]):
    """What a stop proved about the things it ended.

    ``ended`` holds what the stop signalled; ``survivors`` what a read of the
    process table still found running after SIGKILL; ``unknown`` what could not
    be re-identified because the process table could not be read. Unknown is
    never folded into survivors, and a thing in ``unknown`` was never signalled
    after the read that failed.

    An instance is always true: it defines neither ``__bool__`` nor
    ``__len__``, so a truth test can never read an outcome as gone. Ask
    :attr:`gone`, :attr:`survivors` or :attr:`unknown`.
    """

    ended: tuple[_T, ...] = ()
    survivors: tuple[_T, ...] = ()
    unknown: tuple[_T, ...] = ()

    @property
    def gone(self) -> bool:
        """Whether everything the stop ended is proven gone."""
        return not self.survivors and not self.unknown

    def describe(
        self,
        *,
        survivor: Callable[[_T], str] = _pid_name,
        unknown: Callable[[_T], str] = _group_name,
    ) -> str:
        """The outcome in the words every log line uses, one renderer per set."""
        if self.gone:
            return "gone"
        clauses = []
        if self.survivors:
            clauses.append(
                "still running after SIGKILL: " + ", ".join(survivor(x) for x in self.survivors)
            )
        if self.unknown:
            clauses.append(
                "process table unreadable: "
                + ", ".join(unknown(x) for x in self.unknown)
                + " may still be running"
            )
        return "; ".join(clauses)


def _stop(process: subprocess.Popen, *, own_group: bool) -> None:
    """Stop ``process`` — and its process group, if it leads one: SIGTERM, then SIGKILL.

    The whole group is waited out, not just its leader: the leader of a
    ``podman compose`` is the dispatcher, and the provider it started can take
    longer to go. A member still running after SIGKILL is named in a warning,
    and the outcome of the stop is logged in its own words when the group is
    not proven gone.
    Only ever called on a run that is being abandoned; a child that exits on its
    own leaves its group alone.
    """
    if own_group:
        outcome = stop_child(process, grace=_STOP_GRACE_S)
        if not outcome.gone:
            logger.warning("%s", outcome.describe())
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if process.poll() is not None:
            return
        with contextlib.suppress(ProcessLookupError, PermissionError):
            process.send_signal(sig)
        try:
            process.wait(timeout=_STOP_GRACE_S)
            return
        except subprocess.TimeoutExpired:
            continue
        except KeyboardInterrupt:
            # A second Ctrl-C during the grace period asks for it to be cut
            # short, not for the child to be left running.
            continue


class _ChildState(NamedTuple):
    """What ``waitid(2)`` reports about a child that changed state."""

    si_pid: int
    si_code: int
    si_status: int


def _waitid(pid: int, flags: int) -> _ChildState | None:
    """``waitid(P_PID, pid, flags)``: the child's state, or ``None`` while it runs.

    ``flags`` are the ``os.W*`` constants; with ``os.WNOWAIT`` the child is left
    unreaped. Raises :class:`ChildProcessError` when ``pid`` is not an unreaped
    child of this process.
    """
    waitid = getattr(os, "waitid", None)
    if waitid is None:
        # Interim until the Python floor is 3.13: macOS CPython binds no os.waitid before it.
        return _libc_waitid(pid, flags)
    result = waitid(os.P_PID, pid, flags)
    if result is None:
        return None
    return _ChildState(result.si_pid, result.si_code, result.si_status)


def _libc_waitid(pid: int, flags: int) -> _ChildState | None:
    """``waitid(2)`` called in the C library, for an interpreter that has no ``os.waitid``.

    Interim: CPython binds ``os.waitid`` on macOS only from 3.13, so on 3.11 and
    3.12 the call goes through ctypes with Darwin's ``siginfo_t`` layout. Removed,
    and ``os.waitid`` used everywhere, once the supported Python floor is 3.13.
    """
    import ctypes
    import errno

    if sys.platform != "darwin":
        raise OSError(errno.ENOSYS, f"waitid is not available on {sys.platform}")

    class _SigInfo(ctypes.Structure):
        # Darwin's siginfo_t (sys/signal.h).
        _fields_ = [
            ("si_signo", ctypes.c_int),
            ("si_errno", ctypes.c_int),
            ("si_code", ctypes.c_int),
            ("si_pid", ctypes.c_int),
            ("si_uid", ctypes.c_uint),
            ("si_status", ctypes.c_int),
            ("si_addr", ctypes.c_void_p),
            ("si_value", ctypes.c_void_p),
            ("si_band", ctypes.c_long),
            ("_pad", ctypes.c_ulong * 7),
        ]

    waitid = ctypes.CDLL(None, use_errno=True).waitid
    waitid.argtypes = [ctypes.c_int, ctypes.c_uint, ctypes.POINTER(_SigInfo), ctypes.c_int]
    waitid.restype = ctypes.c_int
    info = _SigInfo()
    while waitid(os.P_PID, pid, ctypes.byref(info), flags) != 0:
        err = ctypes.get_errno()
        if err != errno.EINTR:
            # OSError picks the subclass from the errno: ECHILD is ChildProcessError.
            raise OSError(err, os.strerror(err))
    if info.si_pid == 0:
        return None
    return _ChildState(info.si_pid, info.si_code, info.si_status)


def observe_exit(process: subprocess.Popen, *, within: float | None = 0.0) -> int | None:
    """The exit status of a child this process owns, or ``None`` while it runs.

    The public liveness probe for an owned child. The status follows
    ``Popen``'s convention: the exit code, or the negated signal number for a
    child a signal ended. ``within=0`` gives one look; a positive ``within``
    looks until the child has exited or that many seconds have passed;
    ``within=None`` blocks until it exits.

    It never reaps: the reap is left to :func:`stop_child`, so the group id the
    child leads stays pinned by its unreaped leader while a stop runs. The rule
    is one of correctness, not tidiness: ``Popen.wait()`` and ``poll()`` report
    exit status 0 for a child something else reaped, so a second reaper turns a
    real exit status into a false one.

    Raises:
        RuntimeError: the child was reaped by something other than its stop.
    """
    if process.returncode is not None:
        return int(process.returncode)
    flags = os.WEXITED | os.WNOWAIT
    if within is not None:
        flags |= os.WNOHANG
    deadline = None if within is None else time.monotonic() + within
    while True:
        try:
            result = _waitid(process.pid, flags)
        except ChildProcessError:
            return _settled_returncode(process)
        if result is not None:
            if result.si_code == os.CLD_EXITED:
                return int(result.si_status)
            return -int(result.si_status)
        if deadline is None or time.monotonic() >= deadline:
            return None
        time.sleep(_GROUP_POLL_S)


def _settled_returncode(process: subprocess.Popen) -> int:
    """The status a reap in flight on another thread is about to record.

    Between ``waitpid`` returning and ``Popen`` storing ``returncode`` there is
    a short window in which the child is reaped but its status not yet set.
    """
    deadline = time.monotonic() + _REAP_SETTLE_S
    while process.returncode is None:
        if time.monotonic() >= deadline:
            raise RuntimeError(f"pid {process.pid} was reaped by something other than its stop")
        time.sleep(_GROUP_POLL_S)
    return int(process.returncode)


def group_is_gone(pgid: int, *, within: float = 0.0) -> bool:
    """Whether no process of the process group ``pgid`` is still running.

    The group is named by its leader's pid. The group is looked at until it is
    gone or ``within`` seconds have passed; ``within=0`` gives one look. An
    exited process nobody has reaped (a zombie) counts as gone. A member this
    user may not signal counts as present.

    Raises:
        ProcessTableUnreadable: the process table could not be read by the
            deadline (at once when the failure is not retryable).
    """
    from osprey.deployment.process_table import snapshot_within

    deadline = time.monotonic() + within
    while True:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            # Either a member this user may not signal, or (on macOS) a group whose
            # only members are zombies: the process table tells the two apart.
            pass
        rows = snapshot_within(max(deadline - time.monotonic(), 0.0))
        if not any(row.pgid == pgid and not row.exited for row in rows.values()):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(_GROUP_POLL_S)


def _signal_group(pgid: int, sig: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, sig)


def _wait_group_gone(pgid: int, seconds: float) -> bool:
    """Wait up to ``seconds`` for the group to be gone; a Ctrl-C ends the wait.

    A process table that cannot be read is read again until the deadline, so a
    group whose state is unknown gets the whole wait.
    """
    from osprey.deployment.process_table import ProcessTableUnreadable

    deadline = time.monotonic() + seconds
    try:
        while True:
            try:
                return group_is_gone(pgid, within=max(deadline - time.monotonic(), 0.0))
            except ProcessTableUnreadable:
                if time.monotonic() >= deadline:
                    return False
                time.sleep(_GROUP_POLL_S)
    except KeyboardInterrupt:
        return False


def _stop_group(pgid: int, *, dump: float | None = None, grace: float) -> list[int] | None:
    """Stop the process group ``pgid`` and return the pids still running after SIGKILL.

    The engine of :func:`stop_child`, which checks the group's leader first and
    reaps it after. The group is named by its leader's pid. When ``dump`` is
    given the group is first sent SIGABRT, which makes every process with
    ``faulthandler`` enabled write each thread's stack before it dies, and gets
    ``dump`` seconds to be gone. Then SIGTERM and ``grace`` seconds, then
    SIGKILL and :data:`_KILL_WAIT_S`. Every wait is on the whole group, not
    only its leader. A Ctrl-C during a wait ends that wait and moves on to the
    next signal; it is not re-raised.

    An exited process nobody has reaped counts as gone; a member this user may
    not signal counts as present.

    Returns:
        ``[]`` when the group is gone; the pids of members still running after
        SIGKILL; ``None`` when the process table could not be read to tell.
    """
    if dump is not None:
        _signal_group(pgid, signal.SIGABRT)
        if _wait_group_gone(pgid, dump):
            return []
    _signal_group(pgid, signal.SIGTERM)
    if _wait_group_gone(pgid, grace):
        return []
    _signal_group(pgid, signal.SIGKILL)
    if _wait_group_gone(pgid, _KILL_WAIT_S):
        return []
    from osprey.deployment.process_table import ProcessTableUnreadable, snapshot_within

    try:
        rows = snapshot_within(_KILL_WAIT_S)
    except ProcessTableUnreadable:
        return None
    return sorted(row.pid for row in rows.values() if row.pgid == pgid and not row.exited)


def stop_child(
    process: subprocess.Popen, *, dump: float | None = None, grace: float
) -> StopOutcome[int]:
    """Stop the child ``process`` and its process group, reap it, and say what is left.

    ``process`` leads its own process group (every caller starts it with
    ``process_group=0``, ``start_new_session=True`` or ``setsid()``), so the
    group is named by its pid. The signal ladder is SIGABRT and ``dump``
    seconds when ``dump`` is given, SIGTERM and ``grace`` seconds, then
    SIGKILL; every wait is on the whole group.

    The leader must be unreaped when the stop starts: an unreaped leader holds
    the group's id, so nothing else can take it while the stop signals and
    reads it. A leader that already exited and nobody reaped is a zombie that
    still holds the id, and is stopped like a running one. The stop reaps the
    leader once it has exited; a leader still running after SIGKILL is stuck in
    the kernel, is not waited on, and is named among the survivors.

    Two concurrent calls on one child are harmless: both pass the check, and the
    second ``wait()`` returns the status the first stored under ``Popen``'s lock.
    A call made after another call has reaped the child is refused.

    Returns:
        An outcome whose ``ended`` is the group. ``survivors`` names the pids
        a read of the process table found running after SIGKILL; ``unknown``
        names the group when the table could not be read, and then the leader
        is a survivor if it was seen running.

    Raises:
        ValueError: the leader was already reaped, or it does not lead its own
            process group.
    """
    pid = process.pid
    _check_leader(process)
    pids = _stop_group(pid, dump=dump, grace=grace)
    leader_running = observe_exit(process) is None
    if not leader_running:
        process.wait()
    if pids is None:
        return StopOutcome(ended=(pid,), survivors=(pid,) if leader_running else (), unknown=(pid,))
    return StopOutcome(ended=(pid,), survivors=tuple(pids))


def _check_leader(process: subprocess.Popen) -> None:
    """Refuse a leader that is reaped or that does not lead its own group."""
    pid = process.pid
    reaped = ValueError(f"the leader {pid} is already reaped; its group id is no longer pinned")
    if process.returncode is not None:
        raise reaped
    while True:
        try:
            if _waitid(pid, os.WEXITED | os.WNOWAIT | os.WNOHANG) is not None:
                return  # a zombie: its group id is still pinned
        except ChildProcessError:
            raise reaped from None
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            continue  # it exited between the two looks
        if pgid != pid:
            raise ValueError(f"pid {pid} does not lead its own process group (group {pgid})")
        return


def diagnose_captured_failure(exc: BaseException) -> str | None:
    """Translate a captured child's failure into a remedy, or ``None``.

    A :class:`~osprey.deployment.errors.CapturedProcessError` names the file its
    child's output went to, and some of those failures have a cause OSPREY can
    recognize and answer in words the raw text never uses. This is the one seam
    that reads that spool and asks the diagnosers, so every verb that runs a
    build gets the same translation rather than each growing its own.

    Total by construction. No spool (a ``--verbose`` run, where the output went
    to the terminal instead, or an exception of some other type), a spool that
    cannot be read, or output that nothing recognizes all yield ``None`` — and
    the caller then renders the failure exactly as it did before.

    :param exc: The exception a verb is about to fail on.
    :returns: Operator-facing remedy text, or ``None`` when there is nothing
        honest to add.
    """
    from osprey.deployment.runtime_helper import diagnose_build_failure

    spool_path = getattr(exc, "spool_path", None)
    if spool_path is None:
        return None
    try:
        output = Path(spool_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return diagnose_build_failure(output)


def captured_failure_detail(
    exc: BaseException, *, config: Mapping[str, object] | None = None
) -> str | None:
    """What the failed child's output points at but does not contain, or ``None``.

    The sibling of :func:`diagnose_captured_failure` for the other half of a
    failure report -- its cause rather than its remedy. Today it answers one
    shape: ``compose up`` aborting on a container whose healthcheck never
    passed, where the spool names the container and the container's own log
    holds the reason. The log is read here, through the runtime the deployment
    resolved, and returned as text the verb appends beneath the exception's own
    line.

    Total by construction, like its sibling: no spool, an unreadable one, a
    runtime that cannot be resolved, or output naming no container all yield
    ``None`` or a shorter answer, never a second failure.

    :param exc: The exception a verb is about to fail on.
    :param config: The deployment's config, for runtime selection only.
        ``None`` resolves the runtime from the environment and the host.
    :returns: Operator-facing detail, or ``None`` when there is nothing to add.
    """
    from osprey.deployment.runtime_helper import (
        explain_unhealthy_containers,
        get_runtime_command,
        unhealthy_containers_in,
    )

    # A stopped run's caller looked at the stall while it was still there to see;
    # that account is the detail, and the spool of a run that never ended holds
    # no "unhealthy" verdict to read.
    found: str | None = getattr(exc, "detail", None)
    if found:
        return found
    spool_path = getattr(exc, "spool_path", None)
    if spool_path is None:
        return None
    try:
        output = Path(spool_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if not unhealthy_containers_in(output):
        return None
    try:
        runtime: str | None = get_runtime_command(config)[0]
    except (RuntimeError, FileNotFoundError):
        runtime = None
    return explain_unhealthy_containers(output, runtime)


__all__ = [
    "SPOOL_DIR",
    "SPOOL_RETENTION",
    "CapturedProcess",
    "StopOutcome",
    "captured_failure_detail",
    "diagnose_captured_failure",
    "group_is_gone",
    "observe_exit",
    "run_captured",
    "stop_child",
]
