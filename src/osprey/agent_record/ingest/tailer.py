"""The transcript tailer: mounted ``projects/**/*.jsonl`` files into the record.

Each transcript root is a directory that plays ``projects/`` for one source
identity (a terminal's or a dispatch worker's projects volume, or the host's
build directory); :func:`osprey.agent_record.identities.transcript_source_identity`
maps the root's name to that identity. :meth:`TranscriptTailer.poll_once`
walks every root, ``subagents/`` included, and appends the entries of every
complete line it has not committed yet.

Invariants:

- The walk opens each directory and file relative to its parent's descriptor
  with ``O_NOFOLLOW``, so a symlink is never followed and nothing outside the
  root is read. Only regular files named ``*.jsonl`` are read.
- The committed position of a file lives in the state namespace under
  :func:`~osprey.agent_record.contract.transcript_state_name` (identity and
  path relative to ``projects/``) as a
  :class:`~osprey.agent_record.contract.TranscriptOffset` plus the file's
  ``inode``. A different inode (the file was replaced) or a size below the
  committed offset (the file was compacted in place) bumps the generation and
  re-reads the file from the start; key idempotence drops what is already
  stored. A file that appears or grows later, a subagent's included, is read
  on the next poll.
- Only complete lines are read; a partial last line stays uncommitted until
  its newline arrives. ``fully_read`` is set when the committed offset is the
  end of the file.
- Every entry's body is scrubbed of the secret set before it is appended, and
  an offset is committed only after its entries were appended. A failed
  append commits nothing; the next poll retries.
- A line whose time is older than the retention floor is skipped and counted.
  The floor is the earlier of ``retention_days`` before now and the oldest
  retained arrival day, so a re-read after retention never re-inserts an
  expired fact while a young record still takes every unexpired backlog line,
  however many polls it takes to commit it. While the record's health cannot
  be read, the last known floor is kept.
- A root or file that cannot be read, a file whose lines the parser refuses,
  and a directory nested too deep are that source's error in
  :attr:`TranscriptTailer.health`, never a quiet source; one bad file never
  stops the others, and a failed poll never stops :meth:`TranscriptTailer.run`.
- A committed offset that no longer follows a newline means the file was
  rewritten in place; that also bumps the generation.
- A root may carry a relpath prefix, the part of the path relative to
  ``projects/`` that lies above the directory read (the host's root is its
  one ``projects/<encoded build dir>/`` directory, read as
  ``<encoded build dir>/``), so its state names are the ones the same files
  get when the whole parent is mounted as the root. A prefixed root that does
  not exist yet holds no file, never an error.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import os
import stat
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Protocol

from osprey.agent_record.config import DEFAULT_RETENTION_DAYS
from osprey.agent_record.contract import (
    AppendResult,
    BackendHealth,
    Entry,
    RecordBackendError,
    StateStore,
    TranscriptOffset,
    check_projects_relpath,
    scrub,
    transcript_passed,
    transcript_state_name,
)
from osprey.agent_record.ingest.transcripts import TRANSCRIPT_SUFFIX, parse_line, split_relpath

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_POLL_INTERVAL_S",
    "POLL_INTERVAL_S",
    "FileMark",
    "PollReport",
    "SourceHealth",
    "TranscriptSink",
    "TranscriptTailer",
]

#: The longest the tailer waits between two polls.
MAX_POLL_INTERVAL_S = 5.0

#: The default wait between two polls.
POLL_INTERVAL_S = 2.0

#: The state field holding the inode of the file a generation was read from.
INODE_FIELD = "inode"

#: The most bytes read from one file in one read; larger backlogs are committed
#: chunk by chunk.
_READ_CHUNK = 4 * 1024 * 1024

#: The deepest directory nesting walked below a root.
_MAX_DEPTH = 8

_DIR_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_DIRECTORY", 0)
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)


class TranscriptSink(StateStore, Protocol):
    """Where the tailer appends what it read and keeps each file's committed position.

    The record's own backend is one: the tailer commits into the record. A host
    session's forwarder is the other: it hands the entries to its spool and
    keeps the positions in its own state file
    (:mod:`osprey.agent_record.ingest.forward`). Every method raises
    :class:`~osprey.agent_record.contract.RecordBackendError` on failure.
    """

    def append(self, entries: Sequence[Entry]) -> AppendResult: ...

    def health(self) -> BackendHealth: ...


@dataclass
class SourceHealth:
    """What the tailer knows about one transcript root.

    ``error`` names why the root could not be read at all; ``file_errors``
    maps a path relative to ``projects/`` to why that file could not be read.
    Both are cleared by a later poll that reads them cleanly. ``last_seen`` is
    the last time new lines from this source were committed;
    ``last_modified`` the newest transcript modification any poll has seen.
    """

    root: str
    identity: str | None
    error: str | None = None
    file_errors: dict[str, str] = field(default_factory=dict)
    last_polled: datetime | None = None
    last_seen: datetime | None = None
    last_modified: datetime | None = None
    files: int = 0
    appended: int = 0
    duplicates: int = 0
    skipped_retention: int = 0

    @property
    def ok(self) -> bool:
        """No root error and no file error."""
        return self.error is None and not self.file_errors


@dataclass(frozen=True)
class PollReport:
    """What one poll did across every root."""

    appended: int = 0
    duplicates: int = 0
    skipped_retention: int = 0
    files_read: int = 0
    errors: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class FileMark:
    """Where one transcript file ended when a drain was asked for.

    ``generation`` is the one the tailer will read the file in (bumped, as a
    poll would bump it, for a replaced, shrunk or rewritten file) and
    ``offset`` the end of its last complete line.
    """

    identity: str
    relpath: str
    generation: int
    offset: int


@dataclass
class _Tally:
    appended: int = 0
    duplicates: int = 0
    skipped: int = 0
    files_read: int = 0


class TranscriptTailer:
    """Tail the transcript roots into *backend*.

    Args:
        backend: Where entries go and offsets are kept: the record's backend,
            or a host session's forwarder.
        roots: Transcript root name to the directory that plays ``projects/``
            (or, with a prefix, the directory at that prefix below it).
        config: The rendered deployment config, used to map root names to
            source identities.
        secret_set: Values scrubbed from every entry body before append.
        retention_days: The retention floor used while the record holds no
            arrival day.
        prefixes: Root name to the relpath prefix of what it holds, one or
            more path segments each ending in ``/``; a root not named here
            has none.

    Raises:
        ValueError: If *retention_days* is not a positive int, or a prefix is
            not a relative path ending in ``/``.
    """

    def __init__(
        self,
        backend: TranscriptSink,
        roots: Mapping[str, Path | str],
        config: Mapping[str, Any] | None,
        *,
        secret_set: Iterable[str] = (),
        retention_days: int = DEFAULT_RETENTION_DAYS,
        prefixes: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(retention_days, int) or retention_days < 1:
            raise ValueError(f"retention_days {retention_days!r} must be a positive int")
        self._prefixes: dict[str, str] = {}
        for name, prefix in (prefixes or {}).items():
            if not isinstance(prefix, str) or not prefix.endswith("/"):
                raise ValueError(f"root {name!r} prefix {prefix!r} must end with '/'")
            check_projects_relpath(prefix[:-1])
            self._prefixes[name] = prefix
        from osprey.agent_record.identities import transcript_source_identity

        self.backend = backend
        self.retention_days = retention_days
        self._secrets = tuple(secret_set)
        self._roots: dict[str, Path] = {name: Path(path) for name, path in roots.items()}
        self.health: dict[str, SourceHealth] = {}
        self._floor: datetime | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._wake: asyncio.Event | None = None
        for name in self._roots:
            try:
                identity: str | None = transcript_source_identity(config, name)
                error = None
            except ValueError as exc:
                identity, error = None, str(exc)
            self.health[name] = SourceHealth(root=name, identity=identity, error=error)

    # -- polling -----------------------------------------------------------

    async def run(self, *, interval: float = POLL_INTERVAL_S) -> None:
        """Poll forever, waiting *interval* seconds (at most five) between polls,
        or less when :meth:`wake` asks for the next poll sooner."""
        if not 0 < interval <= MAX_POLL_INTERVAL_S:
            raise ValueError(f"interval {interval!r} must be in (0, {MAX_POLL_INTERVAL_S}] s")
        wake = asyncio.Event()
        self._loop, self._wake = asyncio.get_running_loop(), wake
        try:
            while True:
                try:
                    await asyncio.to_thread(self.poll_once)
                except Exception as exc:  # one failed poll never ends the loop
                    logger.exception("transcript poll failed")
                    for source in self.health.values():
                        if source.identity is not None:
                            source.error = f"poll failed: {type(exc).__name__}: {exc}"
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(wake.wait(), timeout=interval)
                wake.clear()
        finally:
            self._loop = self._wake = None

    def wake(self) -> None:
        """Start the next poll of :meth:`run` now; callable from any thread.

        Does nothing while :meth:`run` is not running.
        """
        loop, event = self._loop, self._wake
        if loop is None or event is None:
            return
        with contextlib.suppress(RuntimeError):  # the loop closed meanwhile
            loop.call_soon_threadsafe(event.set)

    # -- drains --------------------------------------------------------------

    def snapshot(
        self, session_id: str, visible: Callable[[str], bool] | None = None
    ) -> tuple[FileMark, ...]:
        """Where each transcript file of *session_id* (its subagents' included) ends now,
        under every root whose identity is *visible* (``None``: every root).

        Each file's generation is the one a poll would read it in, so the mark
        is passed only by reading what is in the file now. Unreadable paths
        are left out: a poll reports them on :attr:`health`.

        Raises:
            RecordBackendError: If the committed positions cannot be read.
        """
        marks: list[FileMark] = []
        for name, root in self._roots.items():
            identity = self.health[name].identity
            if identity is None or (visible is not None and not visible(identity)):
                continue
            try:
                root_fd = os.open(root, _DIR_FLAGS)
            except OSError:
                continue
            prefix = self._prefixes.get(name, "")
            try:
                for dir_fd, file_name, relpath in _walk(root_fd, prefix, prefix.count("/"), {}):
                    try:
                        if split_relpath(relpath).session_id != session_id:
                            continue
                        mark = self._file_mark(identity, dir_fd, file_name, relpath)
                    except (OSError, ValueError):
                        continue
                    if mark is not None:
                        marks.append(mark)
            except OSError:
                continue
            finally:
                os.close(root_fd)
        return tuple(marks)

    def _file_mark(self, identity: str, dir_fd: int, name: str, relpath: str) -> FileMark | None:
        fd = os.open(name, _FILE_FLAGS, dir_fd=dir_fd)
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                return None
            stored = self.backend.get_state(transcript_state_name(identity, relpath))
            position = _read_position(fd, stored, _committed(stored), st)
            end = _last_line_end(fd, position.offset, st.st_size)
        finally:
            os.close(fd)
        if stored is None and end == 0:
            return None  # nothing complete to read yet
        return FileMark(identity, relpath, position.generation, end)

    def drained(self, marks: Iterable[FileMark]) -> bool:
        """Whether the tailer has committed every file of *marks* past its mark.

        Raises:
            RecordBackendError: If the committed positions cannot be read.
        """
        return all(
            transcript_passed(
                self.backend, mark.identity, mark.relpath, mark.generation, mark.offset
            )
            for mark in marks
        )

    def poll_once(self, now: datetime | None = None) -> PollReport:
        """Read every root once and commit what was appended."""
        when = now if now is not None else datetime.now(UTC)
        floor = self._retention_floor(when)
        total = _Tally()
        errors: dict[str, str] = {}
        for name, root in self._roots.items():
            source = self.health[name]
            source.last_polled = when
            if source.identity is None:
                errors[name] = source.error or "no source identity"
                continue
            tally = _Tally()
            self._poll_root(source, root, self._prefixes.get(name, ""), floor, when, tally)
            total.appended += tally.appended
            total.duplicates += tally.duplicates
            total.skipped += tally.skipped
            total.files_read += tally.files_read
            if source.error is not None:
                errors[name] = source.error
            for relpath, why in source.file_errors.items():
                errors[f"{name}:{relpath}"] = why
        return PollReport(
            appended=total.appended,
            duplicates=total.duplicates,
            skipped_retention=total.skipped,
            files_read=total.files_read,
            errors=errors,
        )

    def _retention_floor(self, now: datetime) -> datetime:
        floor = now - timedelta(days=self.retention_days)
        try:
            health = self.backend.health()
        except RecordBackendError:
            health = None
        if health is None or not health.ok:
            # Unknown: keep the last known floor rather than widen it.
            return floor if self._floor is None else min(self._floor, floor)
        oldest: date | None = health.oldest_arrival_day
        if oldest is not None:
            floor = min(floor, datetime.combine(oldest, time.min, tzinfo=UTC))
        self._floor = floor
        return floor

    def _poll_root(
        self,
        source: SourceHealth,
        root: Path,
        prefix: str,
        floor: datetime,
        now: datetime,
        tally: _Tally,
    ) -> None:
        try:
            root_fd = os.open(root, _DIR_FLAGS)
        except FileNotFoundError as exc:
            if prefix:  # not created yet: no session has run there
                source.error, source.file_errors, source.files = None, {}, 0
            else:
                source.error = f"transcript root {root} cannot be read: {exc.strerror or exc}"
            return
        except OSError as exc:
            source.error = f"transcript root {root} cannot be read: {exc.strerror or exc}"
            return
        file_errors: dict[str, str] = {}
        files = 0
        try:
            for dir_fd, name, relpath in _walk(root_fd, prefix, prefix.count("/"), file_errors):
                files += 1
                try:
                    self._tail_file(source, dir_fd, name, relpath, floor, now, tally)
                except OSError as exc:
                    file_errors[relpath] = f"cannot be read: {exc.strerror or exc}"
                except RecordBackendError as exc:
                    file_errors[relpath] = f"not committed, the record refused it: {exc}"
                except Exception as exc:  # one bad file never stops the others
                    file_errors[relpath] = f"not committed: {type(exc).__name__}: {exc}"
        except OSError as exc:
            source.error = f"transcript root {root} cannot be read: {exc.strerror or exc}"
            source.file_errors = file_errors
            return
        finally:
            os.close(root_fd)
        source.error = None
        source.file_errors = file_errors
        source.files = files

    def _tail_file(
        self,
        source: SourceHealth,
        dir_fd: int,
        name: str,
        relpath: str,
        floor: datetime,
        now: datetime,
        tally: _Tally,
    ) -> None:
        identity = source.identity
        assert identity is not None
        state_name = transcript_state_name(identity, relpath)
        fd = os.open(name, _FILE_FLAGS, dir_fd=dir_fd)
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise OSError(f"{relpath} is not a regular file")
            size = st.st_size
            modified = datetime.fromtimestamp(st.st_mtime, UTC)
            if source.last_modified is None or modified > source.last_modified:
                source.last_modified = modified
            stored = self.backend.get_state(state_name)
            committed = _committed(stored)
            position = _read_position(fd, stored, committed, st)
            if _unchanged(committed, position, size):
                return
            generation, offset = position.generation, position.offset
            tally.files_read += 1
            while offset < size:
                lines = _complete_lines(fd, offset, size)
                if not lines:
                    break
                entries, skipped = self._entries(identity, relpath, lines, floor, now)
                if entries:
                    result = self.backend.append(entries)
                    tally.appended += result.appended
                    tally.duplicates += result.duplicates
                    source.appended += result.appended
                    source.duplicates += result.duplicates
                tally.skipped += skipped
                source.skipped_retention += skipped
                offset += len(lines)
                self.backend.put_state(state_name, _state(generation, offset, offset == size, st))
                source.last_seen = now
            new_state = _state(generation, offset, offset == size, st)
            if stored != new_state:
                self.backend.put_state(state_name, new_state)
        finally:
            os.close(fd)

    def _entries(
        self, identity: str, relpath: str, lines: bytes, floor: datetime, now: datetime
    ) -> tuple[list[Entry], int]:
        entries: list[Entry] = []
        skipped = 0
        for line in lines.splitlines():
            parsed = parse_line(identity, relpath, line, now=now)
            if not parsed:
                continue
            if parsed[0].ts < floor:
                skipped += 1
                continue
            entries.extend(
                dataclasses.replace(e, body=scrub(e.body, self._secrets)) for e in parsed
            )
        return entries, skipped


def _committed(stored: Mapping[str, Any] | None) -> TranscriptOffset | None:
    """The committed offset in *stored*, or ``None`` when there is none or it is unreadable."""
    if stored is None:
        return None
    try:
        return TranscriptOffset.from_state(stored)
    except ValueError:
        return None


def _read_position(
    fd: int,
    stored: Mapping[str, Any] | None,
    committed: TranscriptOffset | None,
    st: os.stat_result,
) -> TranscriptOffset:
    """Where to read *fd* from: the committed offset, or a new generation from the start
    when the file was replaced or truncated, or the offset is not at a line start."""
    if committed is None:
        return TranscriptOffset(0, 0)
    if stored is None or stored.get(INODE_FIELD) != st.st_ino or st.st_size < committed.offset:
        return TranscriptOffset(committed.generation + 1, 0)
    if committed.offset > 0 and os.pread(fd, 1, committed.offset - 1) != b"\n":
        return TranscriptOffset(committed.generation + 1, 0)
    return TranscriptOffset(committed.generation, committed.offset)


def _unchanged(committed: TranscriptOffset | None, position: TranscriptOffset, size: int) -> bool:
    """Whether the committed state already covers the whole file in the same generation."""
    if committed is None:
        return False
    return (
        committed.fully_read
        and committed.generation == position.generation
        and committed.offset == position.offset == size
    )


def _complete_lines(fd: int, offset: int, size: int) -> bytes:
    """The complete lines of *fd* from *offset*, never past *size*; ``b""`` when none."""
    buf = b""
    while offset + len(buf) < size:
        data = os.pread(fd, min(_READ_CHUNK, size - offset - len(buf)), offset + len(buf))
        if not data:
            break
        buf += data
        end = buf.rfind(b"\n")
        if end >= 0:
            return buf[: end + 1]
    return b""


def _last_line_end(fd: int, start: int, size: int) -> int:
    """The offset just past the last newline of *fd* in ``[start, size)``; *start* when none."""
    end = size
    while end > start:
        lo = max(start, end - _READ_CHUNK)
        data = os.pread(fd, end - lo, lo)
        if not data:
            break
        at = data.rfind(b"\n")
        if at >= 0:
            return lo + at + 1
        end = lo
    return start


def _state(generation: int, offset: int, fully_read: bool, st: os.stat_result) -> dict[str, Any]:
    value = TranscriptOffset(generation, offset, fully_read).to_state()
    value[INODE_FIELD] = st.st_ino
    return value


def _walk(
    dir_fd: int, prefix: str, depth: int, errors: dict[str, str]
) -> Iterator[tuple[int, str, str]]:
    """Yield ``(parent fd, name, relpath)`` for every ``*.jsonl`` regular file below *dir_fd*.

    Directories are opened relative to their parent with ``O_NOFOLLOW``;
    symlinks and anything that is neither a directory nor a regular file are
    skipped.
    """
    with os.scandir(dir_fd) as it:
        children = sorted(it, key=lambda e: e.name)
    for child in children:
        relpath = f"{prefix}{child.name}"
        if child.is_symlink():
            continue
        if child.is_dir(follow_symlinks=False):
            if depth >= _MAX_DEPTH:
                errors[relpath] = f"directory deeper than {_MAX_DEPTH} levels, not read"
                continue
            try:
                sub_fd = os.open(child.name, _DIR_FLAGS, dir_fd=dir_fd)
            except OSError as exc:
                errors[relpath] = f"directory cannot be read: {exc.strerror or exc}"
                continue
            try:
                yield from _walk(sub_fd, f"{relpath}/", depth + 1, errors)
            except OSError as exc:
                errors[relpath] = f"directory cannot be read: {exc.strerror or exc}"
            finally:
                os.close(sub_fd)
            continue
        if not child.name.endswith(TRANSCRIPT_SUFFIX) or not child.is_file(follow_symlinks=False):
            continue
        try:
            check_projects_relpath(relpath)
        except ValueError as exc:
            errors[relpath] = str(exc)
            continue
        yield dir_fd, child.name, relpath
