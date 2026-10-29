"""The spool writer client: one call appends one line to the caller's spool.

Every OSPREY component that records a fact hands it to :func:`write` and gets
back the line's ``entry_id``, or ``None``. agent-record's spool consumer is the
only reader of what lands here; the spool is transport, never history.

**Where a line lands.** ``OSPREY_SPOOL_DIR`` names the acting identity's *own*
spool directory, and the writer writes straight into it: a container binds only
its own directory, so there is nothing above it to resolve. When the variable is unset (the host), the
directory is ``<project root>/var/spool/<acting identity>``, with the identity
from :func:`osprey_connectors.identity.acting_identity`. Both are read on every
call, never at import, because compose and the entrypoint set them per process.
The identity is never written into the line; the consumer takes it from the
directory.

**Segment protocol.** A process appends only to a segment it created itself,
named ``<host>-<pid>-<start ns>.open``. Before each append it opens that segment
by name with ``O_APPEND`` (never ``O_CREAT``), so a segment the consumer has
already removed is noticed and a new one is started, rather than an orphan being
recreated under the old name. A line is one ``os.write`` on that descriptor. A
segment is sealed by renaming it to ``….sealed``:

* before a write, when it has reached :data:`SEGMENT_MAX_BYTES` or is
  :data:`SEGMENT_MAX_AGE_S` old;
* right after the write that takes it to :data:`SEGMENT_MAX_BYTES`;
* by an idle timer, :data:`SEGMENT_IDLE_S` after the last write;
* at interpreter exit.

A forked child never touches its parent's segment: the segment records the pid
that created it, and a write from any other pid starts a segment of its own.

**Receiver mode** is agent-record's OTLP receiver writing an accepted batch into
the receiver-owned ``<spool root>/.otlp/<login>/``. The spool root is the one
agent-record is given, never ``OSPREY_SPOOL_DIR``, and every line carries the
caller's key. A batch is written as one sealed segment of its own, fsynced
before and after the seal rename, so the receiver answers 2xx only for a batch
that is on disk.

**Batch mode** is the same sealed, fsynced segment written into a source's own
spool directory, for a sender that commits its own position only once a batch
is on disk: a host session forwarding its transcripts (:func:`write_batch`).

**Off switch.** With ``OSPREY_AGENT_RECORD_OFF=1`` in the environment,
:func:`write` answers ``None`` before resolving a directory: nothing is created,
appended or logged to ``.errors``. Receiver mode is agent-record's own and does
not consult it.

**Never raises.** A failure is logged at error level, recorded as one JSON line
in the directory's ``.errors`` file for the consumer to report on health, and
answered with ``None``. Callers test the result for truthiness.

The module is stdlib-only (beside :mod:`osprey.agent_record.contract`, which is
stdlib-only too, the connectors' identity and workspace helpers and the repo
layout's names, imported lazily), so the hook emitter can mirror it under a
pinning test.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import socket
import threading
import time
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from osprey.agent_record.contract import (
    OTLP_SPOOL_DIR,
    RECORD_OFF_ENV,
    RECORD_OFF_VALUE,
    SEGMENT_IDLE_S,
    SEGMENT_MAX_AGE_S,
    SEGMENT_MAX_BYTES,
    SPOOL_DIR_ENV,
    WRITER_ERRORS_FILE,
    SpoolLine,
    WriteResult,
    check_identity,
    sealed_name,
    segment_name,
)

__all__ = [
    "SEGMENT_FILE_MODE",
    "SPOOL_DIR_MODE",
    "ReceivedLine",
    "record_off",
    "seal_all",
    "spool_dir",
    "write",
    "write_batch",
    "write_received",
]

logger = logging.getLogger("osprey.agent_record.writer")

#: A segment is readable by the spool directory's group (the consumer), and
#: written only by the process that created it.
SEGMENT_FILE_MODE: int = 0o640

#: Mode of a spool directory the writer has to create itself. Group-write and
#: setgid let the consumer, in another container, remove sealed segments.
SPOOL_DIR_MODE: int = 0o2770

#: The id fields a caller may pass to :func:`write`.
_ID_FIELDS: frozenset[str] = frozenset(
    {"entry_id", "ts", "session_id", "terminal_session", "run_id", "tool_use_id", "on_behalf_of"}
)

_APPEND_FLAGS: int = os.O_WRONLY | os.O_APPEND
_CREATE_FLAGS: int = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_EXCL


def _monotonic() -> float:
    """The clock segment age and idleness are measured on."""
    return time.monotonic()


def _pid() -> int:
    return os.getpid()


def _host() -> str:
    """This host's name, made safe as the first part of a segment file name."""
    try:
        name = socket.gethostname()
    except OSError:
        name = ""
    for ch in ("/", "\\", "\0"):
        name = name.replace(ch, "_")
    return name if name not in ("", ".", "..") else "host"


def record_off() -> bool:
    """Whether the record off switch is set for this call.

    True only when ``$OSPREY_AGENT_RECORD_OFF`` is exactly ``1``; read on every
    call, never at import.
    """
    return os.environ.get(RECORD_OFF_ENV) == RECORD_OFF_VALUE


def spool_dir() -> Path:
    """The acting identity's own spool directory.

    ``$OSPREY_SPOOL_DIR`` when it is set and non-empty; otherwise
    ``<project root>/var/spool/<acting identity>``.
    """
    configured = os.environ.get(SPOOL_DIR_ENV, "").strip()
    if configured:
        return Path(configured)

    from osprey.utils.workspace import SPOOL_DIR_RELPATH
    from osprey_connectors.identity import acting_identity
    from osprey_connectors.workspace import load_osprey_config, resolve_project_root

    identity = check_identity(acting_identity())
    return resolve_project_root(load_osprey_config()) / SPOOL_DIR_RELPATH / identity


# ---------------------------------------------------------------------------
# Segment state
# ---------------------------------------------------------------------------


@dataclass
class _Segment:
    """The one open segment this process writes in one directory."""

    directory: Path
    name: str
    pid: int
    started: float
    last_write: float
    size: int = 0
    timer: threading.Timer | None = field(default=None, repr=False)

    @property
    def path(self) -> Path:
        return self.directory / self.name


_lock = threading.Lock()
_segments: dict[Path, _Segment] = {}
_atexit_registered = False


def _reset_after_fork() -> None:
    """A child starts with no segments: its parent's are not its own."""
    global _lock, _segments
    _lock = threading.Lock()
    _segments = {}


os.register_at_fork(after_in_child=_reset_after_fork)


def _seal(segment: _Segment) -> None:
    """Rename *segment* to its sealed name. Caller holds the lock.

    A segment that is already gone was taken by the consumer or removed by
    hand; there is nothing left to seal.
    """
    if segment.timer is not None:
        segment.timer.cancel()
        segment.timer = None
    if _segments.get(segment.directory) is segment:
        del _segments[segment.directory]
    sealed = segment.directory / sealed_name(segment.name)
    try:
        os.rename(segment.path, sealed)
    except FileNotFoundError:
        pass
    except OSError:
        logger.error("Could not seal spool segment %s", segment.path, exc_info=True)


def seal_all() -> None:
    """Seal every segment this process has open. Runs at interpreter exit."""
    with _lock:
        for segment in list(_segments.values()):
            if segment.pid == _pid():
                _seal(segment)
        _segments.clear()


def _idle_check(segment: _Segment) -> None:
    """The idle timer: seal *segment* once it has gone unwritten long enough."""
    with _lock:
        if _segments.get(segment.directory) is not segment:
            return
        idle = _monotonic() - segment.last_write
        if idle >= SEGMENT_IDLE_S:
            _seal(segment)
        else:
            _arm_timer(segment, SEGMENT_IDLE_S - idle)


def _arm_timer(segment: _Segment, delay: float) -> None:
    timer = threading.Timer(max(delay, 0.0), _idle_check, args=(segment,))
    timer.daemon = True
    segment.timer = timer
    timer.start()


def _ensure_dir(directory: Path) -> None:
    """Create *directory* at :data:`SPOOL_DIR_MODE`; the mode is best-effort."""
    directory.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(directory, SPOOL_DIR_MODE)
    except OSError:
        logger.debug("Could not set %o on the spool directory %s", SPOOL_DIR_MODE, directory)


def _open_creating(path: Path, flags: int) -> int:
    """``os.open`` *path* at :data:`SEGMENT_FILE_MODE`, creating its directory if missing."""
    try:
        return os.open(path, flags, SEGMENT_FILE_MODE)
    except FileNotFoundError:
        _ensure_dir(path.parent)
        return os.open(path, flags, SEGMENT_FILE_MODE)


def _new_segment(directory: Path) -> tuple[_Segment, int]:
    """Create a fresh ``.open`` segment in *directory* and return it opened."""
    global _atexit_registered
    now = _monotonic()
    name = segment_name(_host(), _pid(), time.time_ns())
    fd = _open_creating(directory / name, _CREATE_FLAGS)
    segment = _Segment(directory=directory, name=name, pid=_pid(), started=now, last_write=now)
    _segments[directory] = segment
    if not _atexit_registered:
        atexit.register(seal_all)
        _atexit_registered = True
    return segment, fd


def _open_for_append(directory: Path) -> tuple[_Segment, int]:
    """The segment the next line goes to, opened by name. Caller holds the lock."""
    segment = _segments.get(directory)
    if segment is not None and segment.pid != _pid():
        # Inherited from a parent process: never write to it, never seal it.
        del _segments[directory]
        segment = None
    if segment is not None:
        age = _monotonic() - segment.started
        idle = _monotonic() - segment.last_write
        if segment.size >= SEGMENT_MAX_BYTES or age >= SEGMENT_MAX_AGE_S or idle >= SEGMENT_IDLE_S:
            _seal(segment)
            segment = None
    if segment is not None:
        try:
            return segment, os.open(segment.path, _APPEND_FLAGS)
        except FileNotFoundError:
            if segment.timer is not None:
                segment.timer.cancel()
            del _segments[directory]
    return _new_segment(directory)


def _write_all(fd: int, data: bytes, path: Path) -> None:
    """One ``os.write`` of *data*; a short write is terminated and raised."""
    written = os.write(fd, data)
    if written != len(data):
        if written > 0 and not data[:written].endswith(b"\n"):
            # Terminate the fragment so the next line starts on its own.
            with contextlib.suppress(OSError):
                os.write(fd, b"\n")
        raise OSError(f"short append to {path}: {written} of {len(data)} bytes")


def _append(directory: Path, data: bytes) -> None:
    with _lock:
        segment, fd = _open_for_append(directory)
        try:
            _write_all(fd, data, segment.path)
        finally:
            os.close(fd)
        segment.size += len(data)
        segment.last_write = _monotonic()
        if segment.size >= SEGMENT_MAX_BYTES:
            _seal(segment)
        elif segment.timer is None:
            _arm_timer(segment, SEGMENT_IDLE_S)


# ---------------------------------------------------------------------------
# Lines and failures
# ---------------------------------------------------------------------------


def _new_entry_id() -> str:
    return uuid.uuid4().hex


def _ts_text(value: Any) -> str:
    if value is None:
        return datetime.now(UTC).isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _encode(line: SpoolLine) -> bytes:
    return (line.to_json() + "\n").encode("utf-8")


def _record_failure(directory: Path | None, kind: Any, entry_id: Any, exc: BaseException) -> None:
    """Log *exc* loudly and append it to *directory*'s ``.errors``, best-effort."""
    logger.error(
        "Spool write failed in %s (kind=%r, entry_id=%r): %s",
        directory if directory is not None else "<unresolved spool directory>",
        kind,
        entry_id,
        exc,
    )
    if directory is None:
        return
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "kind": kind if isinstance(kind, str) else repr(kind),
        "entry_id": entry_id if isinstance(entry_id, str) else repr(entry_id),
        "error": f"{type(exc).__name__}: {exc}",
    }
    data = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    try:
        fd = _open_creating(directory / WRITER_ERRORS_FILE, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    except Exception:
        logger.error("Could not record the spool failure in %s", directory, exc_info=True)
        return
    try:
        os.write(fd, data)
    except Exception:
        logger.error("Could not record the spool failure in %s", directory, exc_info=True)
    finally:
        os.close(fd)


def _build_line(kind: Any, body: Any, ids: Mapping[str, Any], *, key: str | None) -> SpoolLine:
    unknown = sorted(set(ids) - _ID_FIELDS)
    if unknown:
        raise TypeError(f"unknown spool line fields {unknown}; the fields are {sorted(_ID_FIELDS)}")
    entry_id = ids.get("entry_id") or _new_entry_id()
    return SpoolLine(
        entry_id=entry_id,
        kind=kind,
        ts=_ts_text(ids.get("ts")),
        body=body,
        key=key,
        session_id=ids.get("session_id"),
        terminal_session=ids.get("terminal_session"),
        run_id=ids.get("run_id"),
        tool_use_id=ids.get("tool_use_id"),
        on_behalf_of=ids.get("on_behalf_of"),
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def write(kind: str, body: Mapping[str, Any], **ids: Any) -> WriteResult:
    """Append one spool line to the acting identity's spool directory.

    Args:
        kind: The entry kind (one of the contract's kinds).
        body: The entry body, a JSON-serialisable mapping.
        **ids: Any of ``entry_id`` (a fresh one is minted when absent), ``ts``
            (an ISO-8601 string or an aware ``datetime``; now when absent),
            ``session_id``, ``terminal_session``, ``run_id``, ``tool_use_id``
            and ``on_behalf_of``.

    Returns:
        The line's ``entry_id``, which is also the caller's wait mark, or
        ``None`` when the line was not stored. Never raises. With the record
        off switch set (:func:`record_off`) nothing is resolved or written and
        the answer is ``None``.
    """
    if record_off():
        return None
    directory: Path | None = None
    entry_id = ids.get("entry_id")
    try:
        directory = spool_dir()
        line = _build_line(kind, body, ids, key=None)
        entry_id = line.entry_id
        _append(directory, _encode(line))
        return line.entry_id
    except Exception as exc:
        _record_failure(directory, kind, entry_id, exc)
        return None


@dataclass(frozen=True)
class ReceivedLine:
    """One line of a receiver-mode batch: a kind, a body, its key, its ids."""

    kind: str
    body: Mapping[str, Any]
    key: str
    ids: Mapping[str, Any] = field(default_factory=dict)


def _receiver_dir(spool_root: Path | str, login: str) -> Path:
    login = check_identity(login)
    return Path(spool_root) / OTLP_SPOOL_DIR / login


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_received(
    spool_root: Path | str, login: str, lines: Iterable[ReceivedLine]
) -> list[str] | None:
    """Write one accepted receiver batch as a sealed, fsynced segment.

    The batch lands in ``<spool_root>/.otlp/<login>/`` as a single segment that
    is written, fsynced, sealed by rename, and fsynced again through its
    directory. A batch is stored whole or not at all: on failure the partial
    segment is removed.

    Args:
        spool_root: The spool root agent-record is given.
        login: The sender's login, the receiver-owned directory's name.
        lines: The batch; every line carries its own key.

    Returns:
        The lines' ``entry_id`` values in order, or ``None`` on failure. An
        empty batch writes nothing and returns ``[]``. Never raises.
    """
    directory: Path | None = None
    try:
        directory = _receiver_dir(spool_root, login)
        built = []
        for item in lines:
            if not isinstance(item.key, str) or not item.key:
                raise ValueError(f"a receiver line needs a key; got {item.key!r}")
            built.append(_build_line(item.kind, item.body, item.ids, key=item.key))
    except Exception as exc:
        _record_failure(directory, "batch", None, exc)
        return None
    return write_batch(directory, built)


def write_batch(directory: Path | str, lines: Iterable[SpoolLine]) -> list[str] | None:
    """Write *lines* into *directory* as one sealed, fsynced segment.

    The segment is written, fsynced, sealed by rename and fsynced again through
    its directory, so the batch is stored whole or not at all: on failure the
    partial segment is removed. The off switch is not consulted; the caller
    that sends a batch decides.

    Args:
        directory: An existing spool directory: a source's own, or a
            receiver-owned one.
        lines: The batch, in order.

    Returns:
        The lines' ``entry_id`` values in order, or ``None`` on failure. An
        empty batch writes nothing and returns ``[]``. Never raises.
    """
    directory = Path(directory)
    path: Path | None = None
    fd: int | None = None
    try:
        built = list(lines)
        if not built:
            return []
        data = b"".join(_encode(line) for line in built)
        name = segment_name(_host(), _pid(), time.time_ns())
        path = directory / name
        fd = _open_creating(path, _CREATE_FLAGS)
        _write_all(fd, data, path)
        os.fsync(fd)
        os.close(fd)
        fd = None
        os.rename(path, directory / sealed_name(name))
        path = None
        _fsync_dir(directory)
        return [line.entry_id for line in built]
    except Exception as exc:
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
        if path is not None:
            with contextlib.suppress(OSError):
                os.unlink(path)
        _record_failure(directory, "batch", None, exc)
        return None
