"""The run lock: one read-write run per control target.

A read-write run holds its control target's run lock for the whole run, so two
runs never drive one machine at once, from any process and any container of
the deployment. The lock is a non-blocking ``flock`` on
``<repo root>/var/guarded_run/<target>/run.lock``. That only works if every
process of the deployment resolves the SAME file for a target, so the
directory is deployment-wide and lives in the repo's state zone, never under a
per-container agent-data root. It holds no file but ``run.lock``.

The deploy provisions ``var/guarded_run/<target>/`` on the host before compose
runs and binds ``var/guarded_run`` read-write at the same place under the
container repo root of every container that runs the agent, so the path this
module resolves inside a container names the one host directory. A process run
on the host itself (a local ``osprey chat``) has no mount to rely on, and the
directory is created here when it is missing.

The directory is keyed by the target NAME (``live``, ``va`` or ``standin``),
so two runs aimed at one machine contend for one lock however each was
launched. A process carries the target it was stamped with; an unstamped
process, and the executor's ``baseline`` stand-in for "no recorded target", are
on the deployment's own baseline target, and resolve to its name.

:func:`lock` takes the target's lock without blocking and records its holder
(pid, start time) in the lock file, emptied on release. A second run on the
target, from this or any other process or container, raises
:class:`OspreyRunBusy` naming the holder before it starts. A readonly run
raises :class:`OspreyWriteRefused` before the lock is taken.

The lock belongs to the process, not to the thread that took it. A process
holds one run lock at a time, kept in :mod:`osprey.runtime._run_state`; a
thread that asks again is busy like any second run, and a caller that means to
run under the process's hold says so (``lock(target, join=True)``).

A forked child never holds the run lock, at any moment, from any thread. A
lock entry, from the open of the directory to publishing the hold, and a
release run under :data:`osprey.runtime._run_state.LOCK`, and a fork takes that
lock before it forks, so it waits at most for a lock entry's system calls;
the child then closes every descriptor of the lock the run state names,
without unlocking the parent's. A child started with ``subprocess`` or
``exec`` never inherits the lock, because its descriptors are close-on-exec.

Nothing here records or restores a setpoint: a run that is killed or fails
leaves the machine where it stopped.

This module is stdlib-only at import time; everything it reads from the
deployment is imported when a directory is asked for.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import json
import os
import stat
import threading
import time
from collections.abc import Iterator
from pathlib import Path

from osprey.runtime import _run_state

__all__ = [
    "GUARDED_RUN_DIR",
    "GUARDED_RUN_DIR_MODE",
    "LOCK_FILE_NAME",
    "GuardedRunDirError",
    "OspreyRunBusy",
    "OspreyWriteRefused",
    "guarded_run_dir",
    "guarded_run_target",
    "lock",
]

#: The directory under the repo's state zone (``var/``) holding one
#: subdirectory per control target.
GUARDED_RUN_DIR = "guarded_run"

#: The lock a read-write run holds on its target for the whole run.
LOCK_FILE_NAME = "run.lock"

#: The mode a directory created here gets: setgid, so files created under it
#: take the directory's group, and group-writable, so every process of the
#: deployment sharing that group can take the lock.
GUARDED_RUN_DIR_MODE = 0o2775

#: Why a guarded run refuses to start in a readonly run. Carries the readonly
#: marker every readonly refusal shares.
_READONLY_REASON = "readonly execution mode: a guarded run needs execution_mode='readwrite'"

#: The executor's spelling of "no recorded control target". It names no
#: machine, so it is never a directory; it resolves to the baseline's name.
_BASELINE_STAND_IN = "baseline"

#: What every refusal of a name in a guarded-run directory ends with.
_FILES_ONLY = "the guarded-run directory holds files only"

#: The mode a lock file created here gets, whatever the umask.
_FILE_MODE = 0o664


class OspreyWriteRefused(Exception):
    """A write was refused and no value of the call was written.

    Treat it as a stop, not a retry.

    Attributes:
        reason: One-line reason taken from the refusal.
        channel_address: The refused channel, when the refusal names one.
    """

    def __init__(self, reason: str, channel_address: str | None = None) -> None:
        self.reason = reason
        self.channel_address = channel_address
        where = f" to '{channel_address}'" if channel_address else ""
        super().__init__(f"OSPREY refused the write{where}: {reason}")


class GuardedRunDirError(RuntimeError):
    """The guarded-run directory for a target cannot be resolved or written."""


class OspreyRunBusy(Exception):
    """Another guarded run holds the target's run lock; this run did not start.

    Attributes:
        root: The guarded-run directory holding the lock.
        pid: The holder's process id as its record names it, ``None`` when the
            record could not be read.
        started: When the holder took the lock (ISO-8601), ``None`` when unknown.
    """

    def __init__(self, root: str, pid: int | None = None, started: str | None = None) -> None:
        self.root = root
        self.pid = pid
        self.started = started
        who = "unknown" if pid is None else str(pid)
        since = "unknown" if started is None else started
        super().__init__(f"another guarded run is live under {root} (pid {who}, since {since})")


#: Whether the fork hooks (:func:`_register_fork_hook`) are registered in this process.
_FORK_HOOK_REGISTERED = False
_FORK_HOOK_GUARD = threading.Lock()


def _open_dir(name: str | Path, *, dir_fd: int | None = None, where: Path) -> int:
    """Open the directory ``name`` for reading, never through a link at ``name``.

    Args:
        name: The directory, relative to ``dir_fd`` when one is given.
        dir_fd: The open directory ``name`` is looked up in.
        where: The path a refusal names.

    Returns:
        The directory's descriptor, close-on-exec.

    Raises:
        GuardedRunDirError: ``name`` is a link or not a directory.
        FileNotFoundError: There is no ``name``.
        OSError: Any other failure to open, unchanged.
    """
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        return os.open(name, flags, dir_fd=dir_fd)
    except FileNotFoundError:
        raise
    except OSError as exc:
        # The open has failed; this only words the refusal.
        is_link = False
        with contextlib.suppress(OSError):
            is_link = stat.S_ISLNK(os.lstat(name, dir_fd=dir_fd).st_mode)
        if is_link:
            raise GuardedRunDirError(
                f"{where} is a link; a guarded-run directory is never a link"
            ) from None
        if exc.errno == errno.ENOTDIR:
            raise GuardedRunDirError(f"{where} is not a directory") from None
        raise


def _open_regular(
    name: str, flags: int, *, dir_fd: int, where: Path, create: bool = False
) -> tuple[int, bool]:
    """Open the lock file ``name`` in the open directory ``dir_fd``.

    The one way the lock file is opened. A link at ``name`` is not
    followed, the open never waits for the other end of a FIFO, and the
    descriptor is accepted only when it is a regular file with exactly one
    name; that is checked on the descriptor before anything reads, writes,
    truncates or changes the mode of it. Whether the file was created is the
    exclusive create's answer, never a look at the name.

    Args:
        name: The file's name in ``dir_fd``.
        flags: The access mode, with ``O_APPEND`` when wanted.
        dir_fd: The open guarded-run directory.
        where: The path a refusal names.
        create: Create the file when it is missing, at mode ``0o664``.

    Returns:
        ``(fd, created)``: the blocking, close-on-exec descriptor and whether
        this call created the file.

    Raises:
        GuardedRunDirError: ``name`` is a link, has more than one hard link,
            is not a regular file, or was replaced while it was opened;
            nothing was read, written or created through it.
        FileNotFoundError: There is no ``name`` and ``create`` is not set.
        OSError: Any other failure to open, unchanged.
    """
    flags |= os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    replaced = False
    while True:
        created = False
        try:
            if create:
                for last in (False, True):
                    try:
                        fd = os.open(
                            name, flags | os.O_CREAT | os.O_EXCL, _FILE_MODE, dir_fd=dir_fd
                        )
                        created = True
                        break
                    except FileExistsError:
                        pass
                    try:
                        fd = os.open(name, flags, dir_fd=dir_fd)
                        break
                    except FileNotFoundError:
                        if last:
                            raise
            else:
                fd = os.open(name, flags, dir_fd=dir_fd)
        except OSError as exc:
            if exc.errno in (errno.ELOOP, errno.EMLINK):
                raise GuardedRunDirError(f"{where} is a link; {_FILES_ONLY}") from None
            if exc.errno in (errno.EISDIR, errno.ENXIO):
                raise GuardedRunDirError(f"{where} is not a regular file; {_FILES_ONLY}") from None
            raise
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise GuardedRunDirError(f"{where} is not a regular file; {_FILES_ONLY}")
            if st.st_nlink > 1:
                raise GuardedRunDirError(f"{where} has {st.st_nlink} hard links; {_FILES_ONLY}")
            if st.st_nlink == 0 and replaced:
                raise GuardedRunDirError(f"{where} was replaced while it was opened")
            if st.st_nlink == 1:
                os.set_blocking(fd, True)
                if created:
                    os.fchmod(fd, _FILE_MODE)
                return fd, created
        except BaseException:
            os.close(fd)
            raise
        # No name holds this file any more: a rename replaced it between the
        # open and the check. The name is opened once more.
        os.close(fd)
        replaced = True


def guarded_run_target(target: str | None) -> str:
    """The control-target name a guarded run's directory is keyed by.

    Args:
        target: The target the run is for. ``None`` reads the process's target
            stamp (:data:`osprey.runtime.ENV_CONTROL_TARGET`). An absent stamp
            and the executor's ``baseline`` both mean the deployment's own
            baseline target, and resolve to the name
            :func:`osprey_connectors.types.baseline_target` gives the resolved
            config's ``control_system`` section.

    Returns:
        ``live``, ``va`` or ``standin``.

    Raises:
        GuardedRunDirError: If the name is not one of the control targets,
            which would otherwise become a directory of its own.
    """
    from osprey_connectors.types import CONTROL_TARGETS, baseline_target

    if target is None:
        from osprey.runtime import ENV_CONTROL_TARGET

        target = os.environ.get(ENV_CONTROL_TARGET, "").strip() or None
    if target is None or target == _BASELINE_STAND_IN:
        from osprey_connectors.workspace import load_osprey_config

        section = load_osprey_config().get("control_system")
        target = baseline_target(section if isinstance(section, dict) else {})
    if target not in CONTROL_TARGETS:
        raise GuardedRunDirError(
            f"Unknown control target {target!r} for a guarded run. "
            f"Valid targets are {', '.join(CONTROL_TARGETS)}."
        )
    return target


def _repo_root() -> Path:
    """The deployment repo root this process belongs to.

    Resolved the way every runtime path is anchored
    (:func:`osprey_connectors.workspace.resolve_project_root`), and accepted
    only when it holds the repo's ``profile.yml`` or a rendered config: that
    resolver ends in the working directory when nothing else answers, and a
    guarded-run directory planted wherever a process was started would be one
    no other process of the deployment finds.
    """
    from osprey_connectors.workspace import (
        PROFILE_FILENAME,
        load_osprey_config,
        rendered_config_path,
        resolve_project_root,
    )

    root = resolve_project_root(load_osprey_config())
    markers = (root / PROFILE_FILENAME, rendered_config_path(root), root / "config.yml")
    if not any(marker.is_file() for marker in markers):
        raise GuardedRunDirError(
            f"guarded runs need var/{GUARDED_RUN_DIR}: no deployment repo root "
            f"found (resolved {root}, which holds no {PROFILE_FILENAME} or config.yml)."
        )
    return root


def _open_levels(directory: Path, *, create: bool) -> int:
    """Open *directory*, a ``var/guarded_run/<target>/``, never through a link at its two levels.

    The ``var/`` state zone is opened as named: only the deployment's owner
    places it. ``guarded_run`` and the target's directory are shared by every
    identity that runs on the target, so each is opened relative to its
    parent's descriptor and refused when it is a link. With *create* a missing
    ``var/`` is created with the process's default mode and each missing
    guarded-run level at :data:`GUARDED_RUN_DIR_MODE`, set on the descriptor of
    the directory this call created; one that already exists keeps the mode it
    has, because inside a container it is the deploy's, provisioned for the
    deployment's shared group, and a concurrent process may have created it.

    Returns:
        The open descriptor of *directory*.

    Raises:
        GuardedRunDirError: A guarded-run level is a link or not a directory.
        FileNotFoundError: A level is missing and *create* is not set.
        OSError: A level cannot be created or opened.
    """
    state = directory.parent.parent
    if create:
        state.mkdir(parents=True, exist_ok=True)
    parent = os.open(state, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    for level in (directory.parent, directory):
        try:
            made = False
            if create:
                try:
                    os.mkdir(level.name, dir_fd=parent)
                    made = True
                except FileExistsError:
                    pass
            fd = _open_dir(level.name, dir_fd=parent, where=level)
        finally:
            os.close(parent)
        if made:
            try:
                os.fchmod(fd, GUARDED_RUN_DIR_MODE)
            except BaseException:
                os.close(fd)
                raise
        parent = fd
    return parent


def _run_dir_path(name: str) -> Path:
    """The guarded-run directory of the target named *name*, not opened.

    Raises:
        GuardedRunDirError: The repo root cannot be resolved.
    """
    from osprey_connectors.workspace import STATE_DIR_NAME

    return _repo_root() / STATE_DIR_NAME / GUARDED_RUN_DIR / name


def _open_run_dir(directory: Path, *, create: bool = True) -> int:
    """Open *directory*, a target's guarded-run directory (:func:`_run_dir_path`).

    Only system calls run here, so a caller may hold :data:`_run_state.LOCK`.

    Returns:
        The directory's open descriptor, which the caller closes.

    Raises:
        GuardedRunDirError: ``guarded_run`` or the target's directory is a
            link, or the directory cannot be created or is not writable.
        FileNotFoundError: A level is missing and *create* is not set.
    """
    try:
        fd = _open_levels(directory, create=create)
    except OSError as exc:
        if not create:
            raise
        raise GuardedRunDirError(
            f"guarded runs need var/{GUARDED_RUN_DIR}: cannot create {directory}: {exc}"
        ) from exc
    try:
        writable = os.access(".", os.W_OK | os.X_OK, dir_fd=fd)
    except BaseException:
        os.close(fd)
        raise
    if not writable:
        os.close(fd)
        raise GuardedRunDirError(
            f"guarded runs need var/{GUARDED_RUN_DIR}: {directory} is not writable."
        )
    return fd


def guarded_run_dir(target: str | None) -> Path:
    """The directory holding the run lock of *target*.

    ``<repo root>/var/guarded_run/<target name>/``. Inside a container the deploy
    has provisioned and mounted it; anywhere else it is created when missing,
    each created directory at :data:`GUARDED_RUN_DIR_MODE`.

    Args:
        target: The target the run is for, or ``None`` for the process's own
            stamp (see :func:`guarded_run_target`).

    Returns:
        The absolute directory, existing and writable.

    Raises:
        GuardedRunDirError: If the repo root cannot be resolved, or the
            directory cannot be created or is not writable, or ``guarded_run``
            or the target's directory is a link.
    """
    directory = _run_dir_path(guarded_run_target(target))
    fd = _open_run_dir(directory)
    os.close(fd)
    return directory


def _utc_now() -> str:
    """The current UTC time as ISO-8601 to the second."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_holder(fd: int) -> str:
    """Record this process as the holder in the lock file ``fd`` it has locked.

    Returns:
        The ``started`` time the record holds.
    """
    started = _utc_now()
    holder = json.dumps({"pid": os.getpid(), "started": started}).encode("utf-8")
    os.ftruncate(fd, 0)
    os.pwrite(fd, holder, 0)
    return started


def _read_holder(fd: int) -> tuple[int | None, str | None]:
    """The ``(pid, started)`` the lock file ``fd`` records, ``None`` for what is unknown."""
    try:
        holder = json.loads(os.pread(fd, 4096, 0) or b"null")
    except (OSError, ValueError):
        return None, None
    if not isinstance(holder, dict):
        return None, None
    pid = holder.get("pid")
    started = holder.get("started")
    return (
        pid if isinstance(pid, int) and not isinstance(pid, bool) else None,
        started if isinstance(started, str) else None,
    )


def _acquire_for_fork() -> None:
    """Take :data:`_run_state.LOCK` before this process forks.

    A lock entry and a release run under that lock, so a fork waits until the
    run state names every descriptor of the lock this process holds.
    """
    _run_state.LOCK.acquire()


def _release_after_fork() -> None:
    """Release, in the parent, the :data:`_run_state.LOCK` the fork took."""
    _run_state.LOCK.release()


def _drop_held_locks_in_child() -> None:
    """Close, in a forked child, the run lock and its directory the parent holds.

    A forked child shares the lock's open file description, so without this a
    child that outlives a killed holder would keep the target busy under a
    dead pid. At any fork :data:`_run_state.STATE` names every descriptor of
    the lock, because a lock entry publishes its hold and a release unlocks
    before it forgets the hold, both under :data:`_run_state.LOCK`, and a fork
    takes that lock first (:func:`_acquire_for_fork`). The child closes its
    copies and never unlocks, which would release the parent's lock, then
    starts from a run state that holds nothing and a lock of its own.
    """
    state = _run_state.STATE
    if state.hold is not None:
        with contextlib.suppress(OSError):
            os.close(state.hold.fd)
        with contextlib.suppress(OSError):
            os.close(state.hold.dir_fd)
    _run_state.reset_in_child()


def _register_fork_hook() -> None:
    """Order this process's forks against the run lock, once per process."""
    global _FORK_HOOK_REGISTERED
    with _FORK_HOOK_GUARD:
        if not _FORK_HOOK_REGISTERED:
            os.register_at_fork(
                before=_acquire_for_fork,
                after_in_parent=_release_after_fork,
                after_in_child=_drop_held_locks_in_child,
            )
            _FORK_HOOK_REGISTERED = True


def _release(hold: _run_state.Hold) -> None:
    """Release *hold*.

    Does nothing when *hold* is not the one this process holds: a forked child
    closed its copy at the fork.
    """
    with _run_state.LOCK:
        state = _run_state.STATE
        if state.hold is not hold:
            return
        # The shared description is unlocked while the run state still names
        # its descriptor, and forgotten only after: a child forked before the
        # unlock closes its copy, and one forked after holds a copy that locks
        # nothing.
        try:
            with contextlib.suppress(OSError):
                os.ftruncate(hold.fd, 0)
            fcntl.flock(hold.fd, fcntl.LOCK_UN)
        finally:
            state.hold = None
            try:
                os.close(hold.fd)
            finally:
                os.close(hold.dir_fd)


@contextlib.contextmanager
def lock(target: str | None, *, join: bool = False) -> Iterator[Path]:
    """Hold *target*'s run lock for the body.

    The lock is taken without blocking. Its holder record (pid, started) is
    written into the lock file and emptied on release. The lock is held by this
    process, not by the thread that took it: a thread or task the body starts
    runs under it. A process holds one run lock at a time; asking again, from
    any thread, is busy like any second run, and asking for another target is
    refused. With *join* a caller runs under the hold this process already
    has on *target*, taking and releasing nothing, and takes the lock as
    without it when the process holds none. A forked child holds nothing; a
    child started with ``subprocess`` or ``exec`` never inherits the lock,
    because the lock's descriptor is not inheritable (PEP 446).

    Args:
        target: The target the run is for, or ``None`` for the process's own
            stamp (see :func:`guarded_run_target`).
        join: Run under this process's hold on *target* when it has one.

    Yields:
        The target's guarded-run directory.

    Raises:
        OspreyWriteRefused: This is a readonly run; no lock was taken.
        RuntimeError: This process holds the run lock on another target.
        GuardedRunDirError: The guarded-run directory cannot be resolved, a
            directory on the way is a link, or the lock is a link, has more
            than one hard link or is not a regular file.
        OspreyRunBusy: Another run holds the target's lock.
        OSError: The lock could not be opened.
    """
    from osprey_connectors.control_system.base import is_readonly_run

    if is_readonly_run():
        raise OspreyWriteRefused(_READONLY_REASON)
    name = guarded_run_target(target)
    directory = _run_dir_path(name)
    where = directory / LOCK_FILE_NAME
    _register_fork_hook()
    with _run_state.LOCK:
        held = _run_state.STATE.hold
        if held is not None and held.target != name:
            raise RuntimeError(
                f"this process holds the run lock on {held.target}; it cannot take one on {name}"
            )
        joined = join and held is not None
        if not joined:
            dir_fd = _open_run_dir(directory)
            try:
                fd, _created = _open_regular(
                    LOCK_FILE_NAME, os.O_RDWR, dir_fd=dir_fd, where=where, create=True
                )
                try:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        pid, started = _read_holder(fd)
                        raise OspreyRunBusy(str(directory), pid, started) from None
                    try:
                        started = _write_holder(fd)
                    except BaseException:
                        with contextlib.suppress(OSError):
                            os.ftruncate(fd, 0)
                        raise
                except BaseException:
                    os.close(fd)
                    raise
            except BaseException:
                os.close(dir_fd)
                raise
            hold = _run_state.Hold(
                target=name, directory=directory, fd=fd, started=started, dir_fd=dir_fd
            )
            _run_state.STATE.hold = hold
    if joined:
        yield directory
        return
    try:
        yield directory
    finally:
        _release(hold)
