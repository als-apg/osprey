"""The spool consumer: every spool line becomes one record entry, or is refused.

agent-record is the only reader of the spool root. :class:`SpoolConsumer`
walks it one pass at a time (:meth:`SpoolConsumer.consume`), following the
contract's segment protocol:

**Identity comes from the directory.** ``<spool root>/<directory>/`` holds the
lines of the identity :func:`~osprey.agent_record.identities.spool_source_identity`
maps the directory name to, and ``<spool root>/.otlp/<login>/`` (receiver-owned,
written only by agent-record's OTLP receiver) the lines of the login's identity.
An identity a line names is never read. A line's ``key`` is honoured only under
``.otlp/``, and on a line that forwards an entry of the directory identity's
own transcripts (a host session's, see
:func:`~osprey.agent_record.contract.forwarded_transcript`), which keeps the
transcript key and is ingested under the transcript route; everywhere else the
key is ``sp:<identity>:<entry_id>`` (or the run key of a ``run`` line).

**What is refused.** A kind the source may not send over its route (the
per-source allow-list: ``run`` only from dispatch identities, ``bridge_turn``
only from bridges), a forwarded transcript entry from a source that may not
forward one, a line that is not a spool line, and a run accept naming a
session an earlier accept from the same identity already named (first accept
wins). A refused line is recorded under
:func:`~osprey.agent_record.contract.refused_state_name`, counts as consumed
(a wait mark naming it passes) and is shown on :meth:`SpoolConsumer.health`.
``on_behalf_of`` is kept only where the trust list allows it, else dropped.
Every body is scrubbed against the build's secret set before it is appended.

**Checkpoints.** After each batch is appended, the committed byte offset is put
under ``spool:<identity>:<segment name>:<inode>``. A ``.sealed`` segment whose
lines are all committed is deleted; an ``.open`` segment is read up to its last
complete line and never sealed or deleted here, so no liveness check across
containers is needed. A sealed segment starts from the offset its ``.open``
name committed under the same inode, so sealing never re-reads a segment. An
``.open`` segment unwritten for :data:`~osprey.agent_record.contract.ABANDONED_OPEN_S`
is reported as an abandoned writer.

**Outages.** When the backend refuses or is down, nothing is deleted: the
segments stay, and health shows each source's backlog (lines and the age of the
oldest pending segment) next to the writers' own ``.errors``.

**Hostile files.** A symlink, FIFO, socket or device in the spool tree, or a
non-regular entry carrying a segment or ``.errors`` name, is renamed aside as
``<name>.refused-<UTC stamp>`` and kept as evidence. A FIFO is first opened for
writing and closed, so nothing stays blocked on it. Segments are opened with
``O_NOFOLLOW | O_NONBLOCK`` and checked again on the descriptor.
"""

from __future__ import annotations

import contextlib
import errno
import json
import logging
import os
import stat
import threading
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from osprey.agent_record.contract import (
    ABANDONED_OPEN_S,
    KIND_RUN,
    OPEN_SUFFIX,
    OTLP_SPOOL_DIR,
    ROUTE_OTLP,
    ROUTE_SPOOL,
    ROUTE_TRANSCRIPT,
    RUN_ACCEPT,
    RUN_KEY_PREFIX,
    RUN_PHASE_FIELD,
    SEALED_SUFFIX,
    SEGMENT_MAX_BYTES,
    SURFACE_FIELD,
    TRANSCRIPT_FORWARDING_CLASSES,
    WRITER_ERRORS_FILE,
    Entry,
    EntryInvalid,
    Query,
    RecordBackend,
    RecordBackendError,
    ScopeFilter,
    SpoolCheckpoint,
    SpoolLine,
    SpoolLineInvalid,
    accept_session,
    arrival_order,
    entry_from_spool_line,
    forwarded_transcript,
    is_token,
    kind_allowed,
    parse_segment_name,
    parse_spool_line,
    refused_state_name,
    refused_state_value,
    scrub,
    sealed_name,
    spool_state_name,
)
from osprey.agent_record.identities import (
    IdentityClasses,
    keep_on_behalf_of,
    login_source_identity,
    spool_source_identity,
    trusted_on_behalf,
)

__all__ = [
    "QUARANTINE_SUFFIX",
    "ConsumeReport",
    "RefusedLine",
    "SegmentMark",
    "SourceHealth",
    "SpoolConsumer",
    "SpoolHealth",
]

logger = logging.getLogger(__name__)

#: The suffix a hostile path is renamed with, before its UTC stamp.
QUARANTINE_SUFFIX = ".refused-"
#: Lines appended per backend call; the checkpoint is committed after each.
BATCH_LINES = 500
#: A line longer than this is refused unread (a writer seals at this size).
MAX_LINE_BYTES = SEGMENT_MAX_BYTES
#: How many refused lines and writer errors health keeps per source.
SHOWN_PER_SOURCE = 50
#: How much of an open segment one backward read takes while looking for its last line.
_TAIL_CHUNK = 64 * 1024
#: How much of a writer's ``.errors`` file health reads, from its end.
ERRORS_TAIL_BYTES = 64 * 1024


_READ_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)


# ---------------------------------------------------------------------------
# Health and reports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RefusedLine:
    """One spool line consumed but not stored, and why."""

    segment: str
    entry_id: str
    kind: str
    reason: str


@dataclass(frozen=True)
class SourceHealth:
    """One spool directory as the last pass saw it."""

    identity: str
    directory: str
    receiver_owned: bool
    #: Complete lines not yet committed.
    backlog_lines: int = 0
    #: Seconds since the oldest segment holding uncommitted lines was started.
    oldest_backlog_age_s: float | None = None
    #: ``.open`` segments unwritten for longer than ``ABANDONED_OPEN_S``.
    abandoned: tuple[str, ...] = ()
    #: The last lines of the writers' ``.errors`` file.
    writer_errors: tuple[str, ...] = ()
    #: The most recent refused lines, oldest first.
    refused: tuple[RefusedLine, ...] = ()
    refused_total: int = 0
    #: The newest segment write any pass has seen in this directory.
    last_seen: datetime | None = None
    #: An ``.open`` segment not yet abandoned: a writer may still append to it.
    live: bool = False
    #: When this process last committed a line from this directory.
    last_commit: datetime | None = None


@dataclass(frozen=True)
class SpoolHealth:
    """What :meth:`SpoolConsumer.health` reports."""

    #: The backend failure that stopped the last pass, or ``""``.
    backend_error: str = ""
    sources: Mapping[str, SourceHealth] = field(default_factory=dict)
    #: ``(path, moved to)`` of every hostile path set aside so far.
    quarantined: tuple[tuple[str, str], ...] = ()
    #: Paths that could not be set aside or understood, with the reason.
    problems: tuple[str, ...] = ()

    @property
    def backlog_lines(self) -> int:
        return sum(s.backlog_lines for s in self.sources.values())


@dataclass(frozen=True)
class ConsumeReport:
    """The outcome of one :meth:`SpoolConsumer.consume` pass."""

    appended: int = 0
    duplicates: int = 0
    refused: int = 0
    deleted: tuple[str, ...] = ()
    backend_error: str = ""


@dataclass
class _Pass:
    appended: int = 0
    duplicates: int = 0
    refused: int = 0
    deleted: list[str] = field(default_factory=list)
    backend_error: str = ""


@dataclass
class _SourceState:
    refused: deque[RefusedLine] = field(default_factory=lambda: deque(maxlen=SHOWN_PER_SOURCE))
    refused_total: int = 0
    last_seen: datetime | None = None
    last_commit: datetime | None = None


@dataclass(frozen=True)
class SegmentMark:
    """Where one segment ended when a drain was asked for: ``offset`` bytes at ``inode``.

    ``offset`` is the end of the last complete line of an ``.open`` segment and
    the size of a ``.sealed`` one: what one pass would commit.
    """

    identity: str
    directory: Path
    name: str
    inode: int
    offset: int


@dataclass(frozen=True)
class _Directory:
    path: Path
    identity: str
    receiver_owned: bool


# ---------------------------------------------------------------------------
# Quarantine (the rename-aside of hostile paths)
# ---------------------------------------------------------------------------


def _release_fifo(path: str) -> None:
    """Open a FIFO for writing and close it, so a reader blocked in ``open()`` sees EOF."""
    try:
        fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
    except OSError:
        return
    os.close(fd)


def _quarantine(path: Path, now: datetime) -> Path:
    """Rename the hostile *path* aside as ``<name>.refused-<UTC stamp>``.

    A FIFO is released first. Raises :class:`OSError` when it cannot be moved.
    """
    mode = os.lstat(path).st_mode
    if stat.S_ISFIFO(mode):
        _release_fifo(str(path))
    target = path.with_name(f"{path.name}{QUARANTINE_SUFFIX}{now.strftime('%Y%m%dT%H%M%SZ')}")
    os.rename(path, target)
    return target


def _is_quarantined(name: str) -> bool:
    return QUARANTINE_SUFFIX in name


# ---------------------------------------------------------------------------
# The consumer
# ---------------------------------------------------------------------------


class SpoolConsumer:
    """Ingest a spool root into a :class:`~osprey.agent_record.contract.RecordBackend`.

    Args:
        backend: Where entries, checkpoints and refusals go.
        spool_root: The spool root agent-record is given.
        config: The rendered deployment config, for the identity map, the
            per-source allow-list and the trusted on-behalf list.
        secrets: The build's secret set; every body is scrubbed against it.
        now: The wall clock (an aware ``datetime``); defaults to UTC now.

    Passes are serialized: :meth:`consume` may be called from any thread.
    """

    def __init__(
        self,
        backend: RecordBackend,
        spool_root: Path | str,
        *,
        config: Mapping[str, Any] | None = None,
        secrets: Iterable[str] = (),
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.backend = backend
        self.spool_root = Path(spool_root)
        self.config = dict(config or {})
        self._secrets = tuple(secrets)
        self._now = now or (lambda: datetime.now(UTC))
        self._trusted = trusted_on_behalf(self.config)
        self._class = IdentityClasses(self.config)
        self._lock = threading.Lock()
        #: Committed offsets known to this process, used while the backend is down.
        self._offsets: dict[str, int] = {}
        #: identity -> {session -> key of the run accept that won it}; loaded lazily.
        self._accepts: dict[str, dict[str, str]] = {}
        self._sources: dict[str, _SourceState] = {}
        self._health = SpoolHealth()
        self._quarantined: list[tuple[str, str]] = []

    # ------------------------------------------------------------------ public

    def consume(self) -> ConsumeReport:
        """Ingest everything the spool holds now, and refresh :meth:`health`."""
        with self._lock:
            run = _Pass()
            problems: list[str] = []
            now = self._now()
            sources: dict[str, SourceHealth] = {}
            for directory in self._directories(now, problems):
                health = self._consume_directory(directory, run, now, problems)
                sources[str(directory.path)] = health
            self._health = SpoolHealth(
                backend_error=run.backend_error,
                sources=sources,
                quarantined=tuple(self._quarantined),
                problems=tuple(problems),
            )
            return ConsumeReport(
                appended=run.appended,
                duplicates=run.duplicates,
                refused=run.refused,
                deleted=tuple(run.deleted),
                backend_error=run.backend_error,
            )

    def health(self) -> SpoolHealth:
        """Backlog, abandoned writers, writer errors and refusals per source, as of the last pass."""
        return self._health

    def snapshot(self, visible: Callable[[str], bool] | None = None) -> tuple[SegmentMark, ...]:
        """The end of every segment now in the directories whose identity is *visible*.

        *visible* takes a directory's mapped source identity; ``None`` takes
        every directory. Nothing is moved aside or read past what is needed:
        this runs beside a pass, from any thread. A path that cannot be read
        is left out (the pass reports it on :meth:`health`).
        """
        marks: list[SegmentMark] = []
        for directory in self._listed_directories():
            if visible is not None and not visible(directory.identity):
                continue
            for entry in self._scan(directory.path, []):
                parsed = parse_segment_name(entry.name)
                if parsed is None:
                    continue
                try:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    info = entry.stat(follow_symlinks=False)
                    offset = (
                        info.st_size
                        if parsed.sealed
                        else _last_line_end(Path(entry.path), info.st_size)
                    )
                except OSError:
                    continue
                if offset:
                    marks.append(
                        SegmentMark(
                            directory.identity, directory.path, entry.name, info.st_ino, offset
                        )
                    )
        return tuple(marks)

    def drained(self, marks: Iterable[SegmentMark]) -> bool:
        """Whether every segment of *marks* is committed up to its mark.

        A segment committed under its ``.sealed`` name counts for the ``.open``
        name it had, and a segment deleted since (every line committed) counts
        as passed. Refused lines count as consumed: their offsets are committed.

        Raises:
            RecordBackendError: If the backend cannot be read.
        """
        for mark in marks:
            names = [mark.name]
            if mark.name.endswith(OPEN_SUFFIX):
                names.append(sealed_name(mark.name))
            committed = 0
            for name in names:
                value = self.backend.get_state(spool_state_name(mark.identity, name, mark.inode))
                if value is not None:
                    with contextlib.suppress(ValueError):
                        committed = max(committed, SpoolCheckpoint.from_state(value).offset)
            if committed >= mark.offset:
                continue
            if any(_same_inode(mark.directory / name, mark.inode) for name in names):
                return False
        return True

    # ------------------------------------------------------------------ walking

    def _directories(self, now: datetime, problems: list[str]) -> list[_Directory]:
        found: list[_Directory] = []
        for entry in self._scan(self.spool_root, problems):
            if entry.name == OTLP_SPOOL_DIR:
                for login in self._scan(Path(entry.path), problems):
                    if self._is_directory(login, now, problems):
                        try:
                            identity = login_source_identity(self.config, login.name)
                        except ValueError as exc:
                            problems.append(f"{login.path}: {exc}")
                            continue
                        found.append(_Directory(Path(login.path), identity, True))
                continue
            if entry.name.startswith(".") or not self._is_directory(entry, now, problems):
                continue
            try:
                identity = spool_source_identity(self.config, entry.name)
            except ValueError as exc:
                problems.append(f"{entry.path}: {exc}")
                continue
            found.append(_Directory(Path(entry.path), identity, False))
        return found

    def _listed_directories(self) -> list[_Directory]:
        """The spool directories :meth:`_directories` would walk, with nothing set aside."""
        found: list[_Directory] = []
        for entry in self._scan(self.spool_root, []):
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                continue
            if not is_dir:
                continue
            if entry.name == OTLP_SPOOL_DIR:
                for login in self._scan(Path(entry.path), []):
                    try:
                        if not login.is_dir(follow_symlinks=False):
                            continue
                        identity = login_source_identity(self.config, login.name)
                    except (OSError, ValueError):
                        continue
                    found.append(_Directory(Path(login.path), identity, True))
                continue
            if entry.name.startswith("."):
                continue
            try:
                identity = spool_source_identity(self.config, entry.name)
            except ValueError:
                continue
            found.append(_Directory(Path(entry.path), identity, False))
        return found

    def _scan(self, path: Path, problems: list[str]) -> list[os.DirEntry[str]]:
        """The entries of *path* sorted by name, without those already set aside."""
        try:
            with os.scandir(path) as it:
                entries = sorted(it, key=lambda e: e.name)
        except FileNotFoundError:
            return []
        except OSError as exc:
            problems.append(f"{path}: cannot be listed ({exc.strerror or exc})")
            return []
        return [e for e in entries if not _is_quarantined(e.name)]

    def _is_directory(self, entry: os.DirEntry[str], now: datetime, problems: list[str]) -> bool:
        """Whether *entry* is a real directory; a symlink or special file is set aside."""
        try:
            if entry.is_dir(follow_symlinks=False):
                return True
            if entry.is_file(follow_symlinks=False):
                return False
        except OSError:
            return False
        self._set_aside(Path(entry.path), now, problems)
        return False

    def _set_aside(self, path: Path, now: datetime, problems: list[str]) -> None:
        try:
            target = _quarantine(path, now)
        except FileNotFoundError:
            return
        except OSError as exc:
            message = (
                f"{path} is not a regular file and could not be moved aside "
                f"({exc.strerror or exc}); remove it by hand"
            )
            logger.error("%s", message)
            problems.append(message)
            return
        logger.warning("Spool path %s is not a regular file; moved aside to %s", path, target)
        self._quarantined.append((str(path), str(target)))

    # ------------------------------------------------------------------ one directory

    def _source(self, directory: _Directory) -> _SourceState:
        key = str(directory.path)
        state = self._sources.get(key)
        if state is None:
            state = self._sources[key] = _SourceState()
        return state

    def _consume_directory(
        self, directory: _Directory, run: _Pass, now: datetime, problems: list[str]
    ) -> SourceHealth:
        source = self._source(directory)
        segments: list[tuple[str, bool]] = []
        errors: tuple[str, ...] = ()
        newest_mtime: float | None = None
        for entry in self._scan(directory.path, problems):
            name = entry.name
            parsed = parse_segment_name(name)
            try:
                regular = entry.is_file(follow_symlinks=False)
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                continue
            if not regular:
                # A directory with an ordinary name is left alone; every other
                # non-regular entry, and anything posing as a segment or as
                # .errors, is hostile.
                if not is_dir or parsed is not None or name == WRITER_ERRORS_FILE:
                    self._set_aside(Path(entry.path), now, problems)
                continue
            if name == WRITER_ERRORS_FILE:
                errors = self._writer_errors(Path(entry.path))
            elif parsed is not None:
                segments.append((name, parsed.sealed))
                with contextlib.suppress(OSError):
                    mtime = entry.stat(follow_symlinks=False).st_mtime
                    newest_mtime = mtime if newest_mtime is None else max(newest_mtime, mtime)
        if newest_mtime is not None:
            written = datetime.fromtimestamp(newest_mtime, UTC)
            if source.last_seen is None or written > source.last_seen:
                source.last_seen = written
        backlog = 0
        oldest_ns: int | None = None
        abandoned: list[str] = []
        live = False
        # Sealed before open, oldest first: a writer's earlier lines land first.
        segments.sort(key=lambda item: (parse_segment_name(item[0]).start_ns, not item[1]))  # type: ignore[union-attr]
        for name, sealed in segments:
            outcome = self._consume_segment(directory, source, name, sealed, run, now, problems)
            if outcome is None:
                continue
            pending, idle_s = outcome
            if pending:
                backlog += pending
                start_ns = parse_segment_name(name).start_ns  # type: ignore[union-attr]
                oldest_ns = start_ns if oldest_ns is None else min(oldest_ns, start_ns)
            if not sealed:
                if idle_s > ABANDONED_OPEN_S:
                    abandoned.append(name)
                else:
                    live = True
        oldest_age = None
        if oldest_ns is not None:
            oldest_age = max(0.0, now.timestamp() - oldest_ns / 1e9)
        return SourceHealth(
            identity=directory.identity,
            directory=str(directory.path),
            receiver_owned=directory.receiver_owned,
            backlog_lines=backlog,
            oldest_backlog_age_s=oldest_age,
            abandoned=tuple(abandoned),
            writer_errors=errors,
            refused=tuple(source.refused),
            refused_total=source.refused_total,
            last_seen=source.last_seen,
            live=live,
            last_commit=source.last_commit,
        )

    def _writer_errors(self, path: Path) -> tuple[str, ...]:
        try:
            fd = os.open(path, _READ_FLAGS)
        except OSError:
            return ()
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                return ()
            start = max(0, info.st_size - ERRORS_TAIL_BYTES)
            os.lseek(fd, start, os.SEEK_SET)
            data = b""
            while chunk := os.read(fd, ERRORS_TAIL_BYTES):
                data += chunk
                if len(data) >= ERRORS_TAIL_BYTES:
                    break
        except OSError:
            return ()
        finally:
            os.close(fd)
        lines = data.decode("utf-8", errors="replace").splitlines()
        if start > 0 and lines:
            lines = lines[1:]
        return tuple(line for line in lines if line.strip())[-SHOWN_PER_SOURCE:]

    # ------------------------------------------------------------------ one segment

    def _committed(self, identity: str, name: str, inode: int) -> int | None:
        """The committed offset of *name* at *inode*, from the backend or this process."""
        state_name = spool_state_name(identity, name, inode)
        try:
            value = self.backend.get_state(state_name)
        except RecordBackendError:
            return self._offsets.get(state_name)
        if value is None:
            return self._offsets.get(state_name)
        try:
            return SpoolCheckpoint.from_state(value).offset
        except ValueError:
            return None

    def _consume_segment(
        self,
        directory: _Directory,
        source: _SourceState,
        name: str,
        sealed: bool,
        run: _Pass,
        now: datetime,
        problems: list[str],
    ) -> tuple[int, float] | None:
        """Ingest one segment; ``(uncommitted complete lines, idle seconds)`` or ``None``."""
        path = directory.path / name
        try:
            fd = os.open(path, _READ_FLAGS)
        except FileNotFoundError:
            return None  # sealed or removed since the listing; the next pass sees it
        except OSError as exc:
            if exc.errno == errno.ELOOP or path.is_symlink():
                self._set_aside(path, now, problems)
            else:
                problems.append(f"{path}: cannot be opened ({exc.strerror or exc})")
            return None
        with os.fdopen(fd, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode):
                handle.close()
                self._set_aside(path, now, problems)
                return None
            inode = info.st_ino
            idle_s = now.timestamp() - info.st_mtime
            identity = directory.identity
            offset = self._committed(identity, name, inode)
            if offset is None and sealed:
                # Sealing is a rename: the inode keeps what its open name committed.
                open_name = name[: -len(SEALED_SUFFIX)] + OPEN_SUFFIX
                offset = self._committed(identity, open_name, inode)
            offset = offset or 0
            handle.seek(offset)
            state_name = spool_state_name(identity, name, inode)
            if offset:
                self._offsets.setdefault(state_name, offset)
            batch: list[tuple[bytes, int]] = []
            while True:
                raw = handle.readline(MAX_LINE_BYTES + 1)
                if not raw:
                    break
                if not raw.endswith(b"\n"):
                    if len(raw) > MAX_LINE_BYTES:
                        # Overlong: skip to the end of the line and refuse it.
                        while raw and not raw.endswith(b"\n"):
                            raw = handle.readline(MAX_LINE_BYTES + 1)
                        end = handle.tell()
                        batch.append((b"", end))
                        continue
                    if not sealed:
                        break  # a writer is mid-line; read it once it is complete
                batch.append((raw, handle.tell()))
                if len(batch) >= BATCH_LINES:
                    if run.backend_error or not self._commit(
                        directory, source, name, state_name, batch, run
                    ):
                        return self._pending(handle, offset, sealed), idle_s
                    offset = batch[-1][1]
                    batch = []
            if batch:
                if run.backend_error or not self._commit(
                    directory, source, name, state_name, batch, run
                ):
                    return self._pending(handle, offset, sealed), idle_s
                offset = batch[-1][1]
            if sealed and offset >= os.fstat(handle.fileno()).st_size:
                self._delete(path, inode, state_name, run, problems)
            return 0, idle_s

    @staticmethod
    def _pending(handle: Any, offset: int, sealed: bool) -> int:
        """Count the complete lines from *offset* on, for the backlog."""
        handle.seek(offset)
        count = 0
        while True:
            raw = handle.readline(MAX_LINE_BYTES + 1)
            if not raw:
                break
            if raw.endswith(b"\n") or sealed:
                count += 1
            if not raw.endswith(b"\n") and len(raw) <= MAX_LINE_BYTES:
                break
        return count

    def _delete(
        self, path: Path, inode: int, state_name: str, run: _Pass, problems: list[str]
    ) -> None:
        try:
            if os.lstat(path).st_ino != inode:
                return
            os.unlink(path)
        except FileNotFoundError:
            return
        except OSError as exc:
            problems.append(f"{path}: committed but cannot be deleted ({exc.strerror or exc})")
            return
        self._offsets.pop(state_name, None)
        run.deleted.append(str(path))

    # ------------------------------------------------------------------ one batch

    def _commit(
        self,
        directory: _Directory,
        source: _SourceState,
        segment: str,
        state_name: str,
        batch: list[tuple[bytes, int]],
        run: _Pass,
    ) -> bool:
        """Append *batch*, record its refusals, then commit its end offset.

        Returns ``False``, with the failure on *run*, when the backend fails;
        nothing of the batch is then committed.
        """
        entries: list[Entry] = []
        refusals: list[RefusedLine] = []
        claimed: dict[str, str] = {}
        now = self._now()
        try:
            for raw, _end in batch:
                entry, refusal = self._ingest_line(directory, segment, raw, claimed)
                if entry is not None:
                    entries.append(entry)
                if refusal is not None:
                    refusals.append(refusal)
            result = self.backend.append(entries) if entries else None
            for refusal in refusals:
                if is_token(refusal.entry_id):
                    self.backend.put_state(
                        refused_state_name(directory.identity, refusal.entry_id),
                        refused_state_value(refusal.kind, refusal.reason, now),
                    )
            offset = batch[-1][1]
            self.backend.put_state(state_name, SpoolCheckpoint(offset).to_state())
        except RecordBackendError as exc:
            run.backend_error = str(exc) or type(exc).__name__
            logger.warning("Spool ingest paused, the backend failed: %s", run.backend_error)
            # The claims of this batch were never stored; reload them next time.
            self._accepts.pop(directory.identity, None)
            return False
        self._offsets[state_name] = offset
        source.last_commit = now
        accepted = self._accepts.get(directory.identity)
        if accepted is not None:
            accepted.update(claimed)
        if result is not None:
            run.appended += result.appended
            run.duplicates += result.duplicates
        for refusal in refusals:
            logger.warning(
                "Refused spool line %s from %s (%s): %s",
                refusal.entry_id or "<unnamed>",
                directory.identity,
                refusal.kind or "<no kind>",
                refusal.reason,
            )
            source.refused.append(refusal)
        source.refused_total += len(refusals)
        run.refused += len(refusals)
        return True

    def _ingest_line(
        self, directory: _Directory, segment: str, raw: bytes, claimed: dict[str, str]
    ) -> tuple[Entry | None, RefusedLine | None]:
        """The entry *raw* becomes, or why it is refused (blank lines are neither)."""
        if not raw.strip():
            if raw:
                return None, None
            return None, RefusedLine(segment, "", "", f"line longer than {MAX_LINE_BYTES} bytes")
        try:
            line = parse_spool_line(raw)
        except SpoolLineInvalid as exc:
            entry_id, kind = _salvage(raw)
            return None, RefusedLine(segment, entry_id, kind, str(exc))
        identity = directory.identity
        source_class = self._class(identity)
        if directory.receiver_owned:
            route = ROUTE_OTLP
        elif forwarded_transcript(line, identity):
            route = ROUTE_TRANSCRIPT
            if source_class not in TRANSCRIPT_FORWARDING_CLASSES:
                return None, RefusedLine(
                    segment,
                    line.entry_id,
                    line.kind,
                    f"the {source_class} source {identity!r} may not forward transcript "
                    "entries over the spool",
                )
        else:
            route = ROUTE_SPOOL
        if not kind_allowed(line.kind, route=route, source_class=source_class):
            return None, RefusedLine(
                segment,
                line.entry_id,
                line.kind,
                f"kind {line.kind!r} is not allowed from the {source_class} source "
                f"{identity!r} over the {route} route",
            )
        surface = line.body.get(SURFACE_FIELD)
        trusted = bool(line.on_behalf_of) and keep_on_behalf_of(
            self._trusted, identity, surface if isinstance(surface, str) else None
        )
        try:
            entry = entry_from_spool_line(
                line,
                identity=identity,
                receiver_owned=directory.receiver_owned,
                on_behalf_of_trusted=trusted,
            )
        except (SpoolLineInvalid, EntryInvalid, ValueError) as exc:
            return None, RefusedLine(segment, line.entry_id, line.kind, str(exc))
        refusal = self._first_accept(identity, segment, line, entry, claimed)
        if refusal is not None:
            return None, refusal
        return replace(entry, body=scrub(entry.body, self._secrets)), None

    # ------------------------------------------------------------------ first accept wins

    def _first_accept(
        self,
        identity: str,
        segment: str,
        line: SpoolLine,
        entry: Entry,
        claimed: dict[str, str],
    ) -> RefusedLine | None:
        """Refuse a run accept naming a session an earlier accept already named."""
        if entry.kind != KIND_RUN or entry.body.get(RUN_PHASE_FIELD) != RUN_ACCEPT:
            return None
        session = accept_session(entry)
        if not session:
            return None
        winner = claimed.get(session) or self._accepted(identity).get(session)
        if winner is None:
            claimed[session] = entry.key
            return None
        if winner == entry.key:
            return None  # the same accept again: a duplicate, not a rival
        return RefusedLine(
            segment,
            line.entry_id,
            line.kind,
            f"session {session!r} was already accepted by {winner!r}; first accept wins",
        )

    def _accepted(self, identity: str) -> dict[str, str]:
        """``session -> winning accept key`` for *identity*, loaded once from the backend.

        Raises:
            RecordBackendError: If the backend cannot be read.
        """
        known = self._accepts.get(identity)
        if known is not None:
            return known
        rows = self.backend.select(
            ScopeFilter(identities=frozenset({identity}), kinds=frozenset({KIND_RUN})),
            Query(key_prefix=f"{RUN_KEY_PREFIX}:"),
        )
        rows.sort(key=arrival_order)
        known = {}
        for row in rows:
            if row.chunk.total != 1 or not row.key.endswith(f":{RUN_ACCEPT}"):
                continue
            session = accept_session(row)
            if session and session not in known:
                known[session] = row.key
        self._accepts[identity] = known
        return known


def _last_line_end(path: Path, size: int) -> int:
    """The offset just past the last newline of the first *size* bytes of *path*; 0 when none."""
    fd = os.open(path, _READ_FLAGS)
    try:
        end = size
        while end > 0:
            start = max(0, end - _TAIL_CHUNK)
            data = os.pread(fd, end - start, start)
            if not data:
                return 0
            at = data.rfind(b"\n")
            if at >= 0:
                return start + at + 1
            end = start
        return 0
    finally:
        os.close(fd)


def _same_inode(path: Path, inode: int) -> bool:
    """Whether *path* still names the file *inode*."""
    try:
        return os.lstat(path).st_ino == inode
    except OSError:
        return False


def _salvage(raw: bytes) -> tuple[str, str]:
    """The ``entry_id`` and ``kind`` of an invalid line, where they can be read."""
    with contextlib.suppress(ValueError, UnicodeDecodeError):
        data = json.loads(raw)
        if isinstance(data, dict):
            entry_id = data.get("entry_id")
            kind = data.get("kind")
            return (
                entry_id if is_token(entry_id) else "",
                kind if isinstance(kind, str) else "",
            )
    return "", ""
