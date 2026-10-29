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
(pid, start time, owner, target and control identity) in the lock file,
emptied on release. The hold keeps the run's :class:`ActionContext`: whose run
it is and on what target it was admitted. A second run on the
target, from this or any other process or container, raises
:class:`OspreyRunBusy` naming the holder before it starts. A readonly run
raises :class:`OspreyWriteRefused` before the lock is taken.

:func:`command_write` is how a run that holds the lock with a context fires
commands: a batch the run's owner, its target, its limits posture, the
rendered facility definition and the connector's limits validator, in its own
mode, all admit, refused whole before anything is written when any of them
does not. Every call files one audit record naming the commands that fired.

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
import logging
import os
import stat
import threading
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from osprey.runtime import _run_state
from osprey.runtime._run_state import ActionContext

__all__ = [
    "ENV_CONTROL_IDENTITY",
    "GUARDED_RUN_DIR",
    "GUARDED_RUN_DIR_MODE",
    "LOCK_FILE_NAME",
    "SURFACE_COMMAND_WRITE",
    "ActionContext",
    "GuardedRunDirError",
    "OspreyRunBusy",
    "OspreyWriteRefused",
    "command_write",
    "guarded_run_dir",
    "guarded_run_target",
    "lock",
    "target_limits_off",
]

#: The directory under the repo's state zone (``var/``) holding one
#: subdirectory per control target.
GUARDED_RUN_DIR = "guarded_run"

#: The lock a read-write run holds on its target for the whole run.
LOCK_FILE_NAME = "run.lock"

#: The environment variable naming the control identity a process writes as;
#: the holder record names it, so a busy run says who holds the target.
ENV_CONTROL_IDENTITY = "OSPREY_CONTROL_IDENTITY"

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

#: The audit surface every :func:`command_write` call files one record on.
SURFACE_COMMAND_WRITE = "command_write"

#: The trailing word of a vocabulary signal that makes a setpoint a command
#: (the write word :data:`osprey.facility.pairing.WRITE_WORDS` pairs no
#: readback with, ``*_command`` in the facility vocabulary).
_COMMAND_WORD = "command"


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

    Every holder field is ``None`` when the record does not name it.

    Attributes:
        root: The guarded-run directory holding the lock.
        pid: The holder's process id as its record names it, ``None`` when the
            record could not be read.
        started: When the holder took the lock (ISO-8601), ``None`` when unknown.
        owner: The account the holder's run acts for.
        target: The control target the holder's run drives.
        identity: The control identity the holder writes as.
    """

    def __init__(
        self,
        root: str,
        pid: int | None = None,
        started: str | None = None,
        owner: str | None = None,
        target: str | None = None,
        identity: str | None = None,
    ) -> None:
        self.root = root
        self.pid = pid
        self.started = started
        self.owner = owner
        self.target = target
        self.identity = identity

        def known(value: object) -> str:
            return "unknown" if value is None else str(value)

        super().__init__(
            f"another guarded run is live under {root} (pid {known(pid)}, "
            f"since {known(started)}, owner {known(owner)}, target {known(target)}, "
            f"identity {known(identity)})"
        )


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
    from osprey.utils.workspace import PROFILE_FILENAME
    from osprey_connectors.workspace import (
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
    from osprey.utils.workspace import STATE_DIR_NAME

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


def target_limits_off(target: str) -> bool:
    """Whether limits checking is explicitly off for the control target *target*.

    The target's limits posture
    (:func:`osprey_connectors.types.target_limits_posture`, the one reader the
    hook, the executor and the tool layer share) over the resolved config's
    ``control_system`` section: off only when the posture is complete and
    states ``enabled: false``. Silence and an unreadable block are not off.

    Args:
        target: A control-target name, as :func:`guarded_run_target` resolves it.

    Returns:
        ``True`` only for a complete posture with ``enabled`` ``False``.
    """
    from osprey_connectors.types import target_limits_posture
    from osprey_connectors.workspace import load_osprey_config

    section = load_osprey_config().get("control_system")
    posture = target_limits_posture(section if isinstance(section, dict) else {}, target)
    return not posture.incomplete and posture.enabled is False


def _utc_now() -> str:
    """The current UTC time as ISO-8601 to the second."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_holder(fd: int, owner: str | None, target: str, identity: str | None) -> str:
    """Record this process as the holder in the lock file ``fd`` it has locked.

    Args:
        fd: The locked lock file.
        owner: The account the run acts for, ``None`` for nobody.
        target: The control target the run drives.
        identity: The control identity this process writes as, ``None`` when
            it names none.

    Returns:
        The ``started`` time the record holds.
    """
    started = _utc_now()
    record = {
        "pid": os.getpid(),
        "started": started,
        "owner": owner,
        "target": target,
        "identity": identity,
    }
    holder = json.dumps(record).encode("utf-8")
    os.ftruncate(fd, 0)
    os.pwrite(fd, holder, 0)
    return started


#: What :func:`_read_holder` answers when the record cannot be read.
_UNKNOWN_HOLDER: tuple[None, None, None, None, None] = (None, None, None, None, None)


def _read_holder(
    fd: int,
) -> tuple[int | None, str | None, str | None, str | None, str | None]:
    """The ``(pid, started, owner, target, identity)`` the lock file ``fd`` records.

    A field the record does not hold, or holds with the wrong type, is ``None``.
    """
    try:
        holder = json.loads(os.pread(fd, 4096, 0) or b"null")
    except (OSError, ValueError):
        return _UNKNOWN_HOLDER
    if not isinstance(holder, dict):
        return _UNKNOWN_HOLDER
    pid = holder.get("pid")
    texts = [holder.get(key) for key in ("started", "owner", "target", "identity")]
    started, owner, target, identity = [v if isinstance(v, str) else None for v in texts]
    return (
        pid if isinstance(pid, int) and not isinstance(pid, bool) else None,
        started,
        owner,
        target,
        identity,
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
def lock(
    target: str | None, *, join: bool = False, context: ActionContext | None = None
) -> Iterator[Path]:
    """Hold *target*'s run lock for the body.

    The lock is taken without blocking. Its holder record (pid, started, the
    context's owner, the target and the process's control identity from
    :data:`ENV_CONTROL_IDENTITY`) is written into the lock file and emptied on
    release. The hold keeps *context*; a context whose ``target`` or
    ``approved_target`` is not the resolved target, or whose ``limits_off`` is
    not :func:`target_limits_off` of it, is refused before anything is opened.
    The lock is held by this
    process, not by the thread that took it: a thread or task the body starts
    runs under it. A process holds one run lock at a time; asking again, from
    any thread, is busy like any second run, and asking for another target is
    refused. With *join* a caller runs under the hold this process already
    has on *target*, taking and releasing nothing, and takes the lock as
    without it when the process holds none. A join without a context runs
    under the held context unchanged; a join with a context other than the
    held one is refused. A forked child holds nothing; a
    child started with ``subprocess`` or ``exec`` never inherits the lock,
    because the lock's descriptor is not inheritable (PEP 446).

    Args:
        target: The target the run is for, or ``None`` for the process's own
            stamp (see :func:`guarded_run_target`).
        join: Run under this process's hold on *target* when it has one.
        context: Whose run this is and on what target it was admitted.

    Yields:
        The target's guarded-run directory.

    Raises:
        OspreyWriteRefused: This is a readonly run; no lock was taken.
        RuntimeError: This process holds the run lock on another target;
            *context* names another target than the resolved one, or says
            limits checking is off when the target's posture does not (or the
            reverse); or a join names a context other than the held one.
        GuardedRunDirError: The guarded-run directory cannot be resolved, a
            directory on the way is a link, or the lock is a link, has more
            than one hard link or is not a regular file.
        OspreyRunBusy: Another run holds the target's lock.
        OSError: The lock could not be opened.
    """
    from osprey_connectors.posture_env import is_readonly_run

    if is_readonly_run():
        raise OspreyWriteRefused(_READONLY_REASON)
    name = guarded_run_target(target)
    if context is not None and (context.target != name or context.approved_target != name):
        raise RuntimeError(
            f"a run on {name} cannot hold a context for target {context.target} "
            f"admitted on {context.approved_target}"
        )
    if context is not None and context.limits_off != target_limits_off(name):
        said = "off" if context.limits_off else "on"
        raise RuntimeError(
            f"a run on {name} cannot hold a context that says limits checking is {said}: "
            f"the target's limits posture says otherwise"
        )
    identity = os.environ.get(ENV_CONTROL_IDENTITY) or None
    owner = None if context is None else context.owner
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
        if held is not None and joined and context is not None and context != held.context:
            raise RuntimeError(
                f"this process holds the run lock on {name} under another context; "
                "a join runs under the held one"
            )
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
                        raise OspreyRunBusy(str(directory), *_read_holder(fd)) from None
                    try:
                        started = _write_holder(fd, owner, name, identity)
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
                target=name,
                directory=directory,
                fd=fd,
                started=started,
                dir_fd=dir_fd,
                context=context,
            )
            _run_state.STATE.hold = hold
    if joined:
        yield directory
        return
    try:
        yield directory
    finally:
        _release(hold)


def _facility_commands() -> frozenset[str]:
    """The command channels of the rendered facility definition.

    ``facility.json`` of the render this process reads its config from: every
    channel whose role is ``setpoint`` and whose vocabulary signal ends in the
    word ``command``. A channel with no role is a readback and never a command.

    Raises:
        OspreyWriteRefused: There is no rendered facility definition, or it
            cannot be read.
    """
    from osprey.channel_roster.sources import facility_file_path
    from osprey_connectors.workspace import load_osprey_config

    path = facility_file_path(load_osprey_config())
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise OspreyWriteRefused(
            f"no command fires without the rendered facility definition ({path} is missing)"
        ) from None
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise OspreyWriteRefused(
            f"the rendered facility definition {path} cannot be read: {exc}"
        ) from None
    channels = document.get("channels") if isinstance(document, dict) else None
    commands: set[str] = set()
    for channel in channels if isinstance(channels, list) else ():
        if not isinstance(channel, dict) or channel.get("role") != "setpoint":
            continue
        signal = channel.get("signal")
        if isinstance(signal, str) and signal.rpartition("_")[2] == _COMMAND_WORD:
            commands.add(str(channel.get("id")))
    return frozenset(commands)


def _admit(connector: Any, batch: list[str], value: Any, context: ActionContext | None) -> None:
    """Refuse the batch unless every condition :func:`command_write` names holds.

    Raises:
        OspreyWriteRefused: A condition does not hold.
        ChannelLimitsViolationError: An address fails the connector's validator.
    """
    from osprey_connectors.posture_store import NO_OWNER, current_owner

    if not batch:
        raise OspreyWriteRefused("a command write names no command")
    seen: set[str] = set()
    for address in batch:
        if address in seen:
            raise OspreyWriteRefused("a command write names each command once", address)
        seen.add(address)
    if context is None:
        raise OspreyWriteRefused(
            "a command write needs this process to hold a run lock with a context"
        )
    owner = current_owner()
    if context.owner is None or owner is NO_OWNER or owner != context.owner:
        named = "nobody" if owner is NO_OWNER else repr(owner)
        runs_for = "nobody" if context.owner is None else repr(context.owner)
        raise OspreyWriteRefused(
            f"a command write acts for {named}, not for the run's owner {runs_for}"
        )
    stamped = getattr(connector, "_control_target", None)
    if stamped is None:
        raise OspreyWriteRefused("the connector names no control target")
    try:
        target = guarded_run_target(stamped)
    except GuardedRunDirError as exc:
        raise OspreyWriteRefused(f"the connector's control target is unusable: {exc}") from None
    if target != context.target:
        raise OspreyWriteRefused(
            f"the connector drives {target}; the run was admitted on {context.target}"
        )
    if context.limits_off != target_limits_off(target):
        raise OspreyWriteRefused(
            f"the run's limits_off is not what {target}'s limits posture says now"
        )
    validator = getattr(connector, "_limits_validator", None)
    if validator is None and not context.limits_off:
        raise OspreyWriteRefused(
            "the connector carries no enforcing limits validator and limits checking "
            f"is not off for {target}"
        )
    if validator is not None and getattr(validator, "failsafe_reason", None) is not None:
        raise OspreyWriteRefused(
            f"the connector carries no enforcing limits validator: {validator.failsafe_reason}"
        )
    commands = _facility_commands()
    for address in batch:
        if address not in commands:
            raise OspreyWriteRefused("not a command of the facility definition", address)
    if validator is not None:
        read_current = connector.current_value_reader()
        for address in batch:
            validator.validate(address, value, read_current=read_current)


#: The audit reason of a command write whose batch call raised: the batch was
#: admitted and handed to the connector, and what reached hardware is unknown.
_OUTCOME_UNKNOWN = "outcome_unknown"


def _audit(
    context: ActionContext | None,
    fired: list[str],
    refusal: BaseException | None,
    *,
    attempted: list[str] | None = None,
) -> None:
    """File the one audit record of a command write; never raises.

    The record names, in ``detail``, ``commands_fired=<address>,…``: the
    commands that went out, in order, empty when none did. When the batch
    call itself raised (*attempted* given), what went out is unknown: the
    reason is ``outcome_unknown`` and ``detail`` is
    ``commands_attempted=<address>,…``, the whole admitted batch.
    """
    try:
        from osprey.audit import posture, writer
        from osprey.audit.envelope import DECISION_ALLOWED, DECISION_REFUSED

        writer.record(
            surface=SURFACE_COMMAND_WRITE,
            posture=posture.posture(),
            posture_source=posture.posture_source(),
            session=None,
            on_behalf_of=None if context is None else context.owner,
            subject="unknown" if context is None else context.target,
            decision=DECISION_ALLOWED if refusal is None else DECISION_REFUSED,
            reason=(
                "fired"
                if refusal is None
                else _OUTCOME_UNKNOWN
                if attempted is not None
                else type(refusal).__name__
            ),
            detail=(
                "commands_fired=" + ",".join(fired)
                if attempted is None
                else "commands_attempted=" + ",".join(attempted)
            ),
        )
    except Exception:  # the audit trail degrades; the write's outcome does not
        logging.getLogger(__name__).warning("Could not record a command write", exc_info=True)


async def command_write(connector: Any, addresses: Sequence[str], value: Any) -> list[str]:
    """Fire *value* at every command in *addresses* on *connector*, in one batch.

    A command is a setpoint whose write acts and holds no setting (arming a
    monitor, requesting a shot). Command-ness is the facility definition's,
    never the caller's: a command is a channel of the rendered facility
    definition whose role is ``setpoint`` and whose signal ends in
    ``command``. The batch is refused whole, before anything is written,
    unless every one of these holds:

    * this process holds a run lock with an :class:`ActionContext`;
    * the batch names each command once;
    * the owner the write path reads
      (:func:`osprey_connectors.posture_store.current_owner`) is that
      context's owner, and the context names one;
    * the connector's control target, its ``baseline`` resolved as
      :func:`guarded_run_target` resolves it, is the context's target;
    * the context's ``limits_off`` is still :func:`target_limits_off` of it;
    * the connector's limits validator is not the fail-safe one that stands
      in for a database that could not be loaded, and the connector carries
      a validator at all unless limits checking is off for the target;
    * every address is a command of the facility definition and passes the
      connector's validator in the validator's own mode, read with the
      connector's own reader: under ``optional`` a command with no limits
      record is written with no limits, under ``exclusive`` it is refused,
      and a record's bounds always apply.

    The approval of the call that leads here is its caller's to have checked;
    no approval field is read here. The connector's own write path judges the
    write posture of every command, as it does every write. A command holds no
    setting, so there is nothing to re-read: the batch goes out as one
    ``write_multiple_channels`` call with ``confirm=False``, and every result
    goes through :func:`~osprey_connectors.control_system.raise_for_write_result`.
    The connector is the caller's own; nothing here reaches the sandbox's
    connector, target pin or validator.

    Every call, refused or not, files one record on the audit surface
    :data:`SURFACE_COMMAND_WRITE` naming the commands that fired. When the
    batch call itself raises, some commands may have gone out before it did:
    the record names the whole batch as attempted, outcome unknown, and the
    connector's exception propagates. Each command's phases are reported to
    :func:`osprey.runtime.record_writes` as a ``write_channels`` batch reports
    its own. The commands fired are this call's own: read from the batch, its
    results and a refusal raised partway, never from a recorder that other
    writers in the process also feed.

    Args:
        connector: The connector the run writes through.
        addresses: The commands to fire, in order.
        value: The value written to each of them.

    Returns:
        The addresses fired, in order.

    Raises:
        OspreyWriteRefused: A condition above does not hold; nothing was
            written.
        ChannelLimitsViolationError: An address fails the validator; nothing
            was written.
        ChannelWriteBlockedError: A result came back refused.
        ChannelWriteFailedError: A result came back failed, mismatched or
            unconfirmed. A :class:`~osprey_connectors.errors.ChannelError`
            raised by the batch, or for one of its results, carries
            ``commands_fired``: the addresses of the batch, in order, that
            were not refused, the commands that went out or may have.
    """
    from osprey.runtime import (
        _WRITE_PHASE_ATTEMPT,
        _WRITE_PHASE_REFUSED,
        _notify_outcome,
        _notify_write,
        _phase_for_outcome,
    )
    from osprey_connectors.control_system import raise_for_write_result
    from osprey_connectors.errors import (
        ChannelError,
        ChannelLimitsViolationError,
        ChannelWriteBlockedError,
    )

    batch = list(addresses)
    with _run_state.LOCK:
        hold = _run_state.STATE.hold
    context = None if hold is None else hold.context
    try:
        _admit(connector, batch, value, context)
    except BaseException as exc:
        _audit(context, [], exc)
        raise
    for address in batch:
        _notify_write(address, _WRITE_PHASE_ATTEMPT)
    try:
        results = await connector.write_multiple_channels(
            [(address, value) for address in batch], confirm=False
        )
    except BaseException as exc:
        # A refusal raised partway names the one command it refused; every
        # other command's outcome is unknown and may have gone out.
        refused = (
            exc.channel_address
            if isinstance(exc, ChannelLimitsViolationError | ChannelWriteBlockedError)
            and exc.channel_address in batch
            else None
        )
        if refused is not None:
            _notify_write(refused, _WRITE_PHASE_REFUSED)
        fired = [address for address in batch if address != refused]
        if isinstance(exc, ChannelError):
            exc.commands_fired = fired
        _audit(context, fired, exc, attempted=batch)
        raise
    for result in results:
        _notify_outcome(result.channel_address, result.outcome)
    # The results come back in batch order, one per command; a command with no
    # result has an unknown outcome and may have gone out.
    outcomes = [result.outcome for result in results]
    outcomes += [None] * (len(batch) - len(outcomes))
    fired = [
        address
        for address, outcome in zip(batch, outcomes, strict=False)
        if _phase_for_outcome(outcome) != _WRITE_PHASE_REFUSED
    ]
    try:
        for result in results:
            try:
                raise_for_write_result(result)
            except ChannelError as exc:
                exc.commands_fired = fired
                raise
    except BaseException as exc:
        _audit(context, fired, exc)
        raise
    _audit(context, fired, None)
    return fired
