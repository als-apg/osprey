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
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import stat
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from osprey.agent_record.contract import (
    Entry,
    RecordBackend,
    RecordBackendError,
    TranscriptOffset,
    check_projects_relpath,
    scrub,
    transcript_state_name,
)
from osprey.agent_record.ingest.transcripts import parse_line

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_POLL_INTERVAL_S",
    "POLL_INTERVAL_S",
    "PollReport",
    "SourceHealth",
    "TranscriptTailer",
]

#: The longest the tailer waits between two polls.
MAX_POLL_INTERVAL_S = 5.0

#: The default wait between two polls.
POLL_INTERVAL_S = 2.0

#: The state field holding the inode of the file a generation was read from.
INODE_FIELD = "inode"

#: Only files with this suffix are transcripts.
_TRANSCRIPT_SUFFIX = ".jsonl"

#: The most bytes read from one file in one read; larger backlogs are committed
#: chunk by chunk.
_READ_CHUNK = 4 * 1024 * 1024

#: The deepest directory nesting walked below a root.
_MAX_DEPTH = 8

_DIR_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_DIRECTORY", 0)
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)


@dataclass
class SourceHealth:
    """What the tailer knows about one transcript root.

    ``error`` names why the root could not be read at all; ``file_errors``
    maps a path relative to ``projects/`` to why that file could not be read.
    Both are cleared by a later poll that reads them cleanly. ``last_seen`` is
    the last time new lines from this source were committed.
    """

    root: str
    identity: str | None
    error: str | None = None
    file_errors: dict[str, str] = field(default_factory=dict)
    last_polled: datetime | None = None
    last_seen: datetime | None = None
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


@dataclass
class _Tally:
    appended: int = 0
    duplicates: int = 0
    skipped: int = 0
    files_read: int = 0


class TranscriptTailer:
    """Tail the transcript roots into *backend*.

    Args:
        backend: The record backend; also the state store for offsets.
        roots: Transcript root name to the directory that plays ``projects/``.
        config: The rendered deployment config, used to map root names to
            source identities.
        secret_set: Values scrubbed from every entry body before append.
        retention_days: The retention floor used while the record holds no
            arrival day.
    """

    def __init__(
        self,
        backend: RecordBackend,
        roots: Mapping[str, Path | str],
        config: Mapping[str, Any] | None,
        *,
        secret_set: Iterable[str] = (),
        retention_days: int = 14,
    ) -> None:
        if not isinstance(retention_days, int) or retention_days < 1:
            raise ValueError(f"retention_days {retention_days!r} must be a positive int")
        from osprey.agent_record.identities import transcript_source_identity

        self.backend = backend
        self.retention_days = retention_days
        self._secrets = tuple(secret_set)
        self._roots: dict[str, Path] = {name: Path(path) for name, path in roots.items()}
        self.health: dict[str, SourceHealth] = {}
        self._floor: datetime | None = None
        for name in self._roots:
            try:
                identity: str | None = transcript_source_identity(config, name)
                error = None
            except ValueError as exc:
                identity, error = None, str(exc)
            self.health[name] = SourceHealth(root=name, identity=identity, error=error)

    # -- polling -----------------------------------------------------------

    async def run(self, *, interval: float = POLL_INTERVAL_S) -> None:
        """Poll forever, waiting *interval* seconds (at most five) between polls."""
        if not 0 < interval <= MAX_POLL_INTERVAL_S:
            raise ValueError(f"interval {interval!r} must be in (0, {MAX_POLL_INTERVAL_S}] s")
        while True:
            try:
                await asyncio.to_thread(self.poll_once)
            except Exception as exc:  # one failed poll never ends the loop
                logger.exception("transcript poll failed")
                for source in self.health.values():
                    if source.identity is not None:
                        source.error = f"poll failed: {type(exc).__name__}: {exc}"
            await asyncio.sleep(interval)

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
            self._poll_root(source, root, floor, when, tally)
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
        self, source: SourceHealth, root: Path, floor: datetime, now: datetime, tally: _Tally
    ) -> None:
        try:
            root_fd = os.open(root, _DIR_FLAGS)
        except OSError as exc:
            source.error = f"transcript root {root} cannot be read: {exc.strerror or exc}"
            return
        file_errors: dict[str, str] = {}
        files = 0
        try:
            for dir_fd, name, relpath in _walk(root_fd, "", 0, file_errors):
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
            stored = self.backend.get_state(state_name)
            position = _position(stored, st.st_ino, size)
            if position.offset > 0 and os.pread(fd, 1, position.offset - 1) != b"\n":
                position = TranscriptOffset(position.generation + 1, 0)
            if _unchanged(stored, position, size):
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


def _position(stored: Mapping[str, Any] | None, inode: int, size: int) -> TranscriptOffset:
    """Where to read from: the committed offset, or a new generation from the start."""
    if stored is None:
        return TranscriptOffset(0, 0)
    try:
        committed = TranscriptOffset.from_state(stored)
    except ValueError:
        return TranscriptOffset(0, 0)
    if stored.get(INODE_FIELD) != inode or size < committed.offset:
        return TranscriptOffset(committed.generation + 1, 0)
    return TranscriptOffset(committed.generation, committed.offset)


def _unchanged(stored: Mapping[str, Any] | None, position: TranscriptOffset, size: int) -> bool:
    """Whether the committed state already covers the whole file in the same generation."""
    if stored is None:
        return False
    try:
        committed = TranscriptOffset.from_state(stored)
    except ValueError:
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
        if not child.name.endswith(_TRANSCRIPT_SUFFIX) or not child.is_file(follow_symlinks=False):
            continue
        try:
            check_projects_relpath(relpath)
        except ValueError as exc:
            errors[relpath] = str(exc)
            continue
        yield dir_fd, child.name, relpath
