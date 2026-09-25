"""One pass of the agent-record archive: copy what changed from a sources tree into a day.

The sources tree is ``<sources>/<kind>/<name>/…`` (``audit`` has no ``<name>``
level); :data:`SOURCE_TABLE` is the only thing that decides what is copied from
each kind. The destination is ``<dest>/<YYYY-MM-DD>/<kind>/<name>/<relpath>``,
the date being the UTC date the pass started. Every file is copied into
``<dest>/.incoming/`` first, hashed while it is copied, and moved to its final
path only when whole, so a day directory never holds a torn copy.
``.incoming`` is the only place a pass ever deletes from; nothing is opened for
writing under ``<sources>``, and a symlink in a source is never followed.
"""

from __future__ import annotations

import fcntl
import hashlib
import logging
import os
import re
import stat
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from osprey.agent_runner.artifact_resolve import DISPATCH_CLAUDE_CONFIG_DIRNAME
from osprey.services.archive.manifest import (
    MANIFEST_NAME,
    ArchiveState,
    SourceRecord,
    append_record,
    load_state,
)

if TYPE_CHECKING:
    from osprey.services.archive.telemetry_export import TelemetryExporter

logger = logging.getLogger(__name__)

#: Kind → the include globs, relative to one source of that kind. ``**`` spans
#: directories, ``*`` stays within one path segment.
SOURCE_TABLE: dict[str, tuple[str, ...]] = {
    # A web terminal's `<user>-claude-config` volume: transcripts, subagent
    # transcripts and tool results — never credentials, settings or caches.
    "terminals": ("projects/**",),
    # A web terminal's `<user>-agent-data` volume: the artifact store, files and index.
    "terminal_agent_data": ("artifacts/**",),
    # A dispatch worker's agent-data volume: the agent's transcripts, the run
    # records and the artifact store.
    "dispatch": (
        f"{DISPATCH_CLAUDE_CONFIG_DIRNAME}/projects/**",
        "dispatch/*.json",
        "artifacts/**",
    ),
    # A plan-queue lane's Redis volume, as Redis persistence files.
    "bluesky": ("appendonlydir/**", "dump.rdb"),
    # The audit ledger tree; its own first level is the identity.
    "audit": ("**",),
}

#: Kinds mounted as one tree rather than one directory per name.
UNNAMED_KINDS: frozenset[str] = frozenset({"audit"})

INCOMING_DIRNAME = ".incoming"
_CHUNK = 1024 * 1024


class ArchiveDestinationError(Exception):
    """The destination is missing or cannot be written; nothing was copied."""


@dataclass
class PassResult:
    """What one pass did."""

    day: str
    started_at: str
    completed_at: str = ""
    files: int = 0
    bytes: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    telemetry_days: list[str] = field(default_factory=list)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _glob_regex(pattern: str) -> re.Pattern[str]:
    out: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


def _literal_prefix(pattern: str) -> tuple[str, ...]:
    """The leading path segments of *pattern* that hold no wildcard."""
    segments: list[str] = []
    for segment in pattern.split("/"):
        if any(ch in segment for ch in "*?"):
            break
        segments.append(segment)
    return tuple(segments)


class _Includes:
    """The include globs of one kind: file matching and directory pruning."""

    def __init__(self, patterns: tuple[str, ...]) -> None:
        self._regexes = [_glob_regex(p) for p in patterns]
        self._prefixes = [_literal_prefix(p) for p in patterns]

    def matches(self, relpath: str) -> bool:
        return any(r.match(relpath) for r in self._regexes)

    def may_contain(self, reldir: str) -> bool:
        parts = tuple(PurePosixPath(reldir).parts) if reldir else ()
        for prefix in self._prefixes:
            n = min(len(parts), len(prefix))
            if parts[:n] == prefix[:n]:
                return True
        return False


class ArchiveTree:
    """Writes into the archive root with its modes, its owner, and ``.incoming``.

    Every directory created gets the root's mode, and every file that mode
    without execute bits. Running as root (rootful docker), created paths are
    handed to the root's owner. It holds an exclusive lock on ``.incoming`` from
    construction to :meth:`close`, so one pass at a time writes the archive.
    """

    def __init__(self, dest: Path, started: datetime) -> None:
        try:
            st = os.stat(dest)
        except OSError as exc:
            raise ArchiveDestinationError(f"{dest} does not exist: {exc.strerror}") from exc
        if not stat.S_ISDIR(st.st_mode):
            raise ArchiveDestinationError(f"{dest} is not a directory")
        if not os.access(dest, os.W_OK | os.X_OK):
            raise ArchiveDestinationError(f"{dest} is not writable")
        self.dest = dest
        self.started = started
        self.day = started.astimezone(UTC).strftime("%Y-%m-%d")
        self.dir_mode = stat.S_IMODE(st.st_mode)
        self.file_mode = self.dir_mode & ~0o111
        self._owner = (st.st_uid, st.st_gid) if os.geteuid() == 0 else None
        self.incoming = dest / INCOMING_DIRNAME
        self._lock_fd: int | None = None
        try:
            self._mkdir(self.incoming)
            self._lock_fd = os.open(self.incoming, os.O_RDONLY)
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
            for leftover in self.incoming.iterdir():
                leftover.unlink()
        except OSError as exc:
            self.close()
            raise ArchiveDestinationError(f"{self.incoming} cannot be used: {exc}") from exc

    def close(self) -> None:
        """Release the archive for the next pass."""
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None

    @property
    def day_dir(self) -> Path:
        return self.dest / self.day

    def _own(self, path: Path) -> None:
        if self._owner is not None:
            os.chown(path, *self._owner, follow_symlinks=False)

    def _mkdir(self, path: Path) -> None:
        if path.is_dir():
            return
        os.mkdir(path, self.dir_mode)
        os.chmod(path, self.dir_mode)
        self._own(path)

    def makedirs(self, path: Path) -> None:
        """Create *path* and any missing parents below the archive root."""
        missing: list[Path] = []
        current = path
        while current != self.dest and not current.is_dir():
            missing.append(current)
            current = current.parent
        for directory in reversed(missing):
            self._mkdir(directory)

    def stage(self, chunks: Iterator[bytes]) -> tuple[Path, int, str]:
        """Write *chunks* to a new file in ``.incoming``; return it, its size and sha256."""
        temp = self.incoming / uuid.uuid4().hex
        digest = hashlib.sha256()
        size = 0
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, self.file_mode)
        try:
            os.fchmod(fd, self.file_mode)
            for chunk in chunks:
                digest.update(chunk)
                size += len(chunk)
                view = memoryview(chunk)
                while view:
                    view = view[os.write(fd, view) :]
            os.fsync(fd)
        except BaseException:
            os.close(fd)
            temp.unlink(missing_ok=True)
            raise
        os.close(fd)
        self._own(temp)
        return temp, size, digest.hexdigest()

    def discard(self, temp: Path) -> None:
        temp.unlink(missing_ok=True)

    def place(self, temp: Path, relpath: str) -> str:
        """Move a staged file to ``<day>/<relpath>``, never over an existing file.

        A path already taken this day (the same source changed twice) gets the
        pass's start time before its final suffix: ``abc.jsonl`` becomes
        ``abc.T142233Z.jsonl``. Returns the path used, relative to the root.
        """
        target = self.day_dir / relpath
        self.makedirs(target.parent)
        if target.exists():
            stamp = f"T{self.started.astimezone(UTC):%H%M%S}Z"
            stem, suffix = target.stem, target.suffix
            candidate = target.with_name(f"{stem}.{stamp}{suffix}")
            n = 1
            while candidate.exists():
                candidate = target.with_name(f"{stem}.{stamp}-{n}{suffix}")
                n += 1
            target = candidate
        os.replace(temp, target)
        return target.relative_to(self.dest).as_posix()

    def append(self, record: dict[str, Any]) -> None:
        """Append one line to this day's manifest."""
        self.makedirs(self.day_dir)
        manifest = self.day_dir / MANIFEST_NAME
        if append_record(manifest, record, file_mode=self.file_mode):
            self._own(manifest)


def _read_chunks(fd: int) -> Iterator[bytes]:
    while True:
        chunk = os.read(fd, _CHUNK)
        if not chunk:
            return
        yield chunk


def _sources_of(sources: Path, result: PassResult) -> Iterator[tuple[str, str, Path]]:
    """Every ``(kind, source prefix, directory)`` under *sources*."""
    try:
        kinds = sorted(p for p in sources.iterdir() if not p.name.startswith("."))
    except OSError as exc:
        result.errors.append({"source": ".", "error": f"{sources}: {exc.strerror or exc}"})
        return
    for kind_dir in kinds:
        kind = kind_dir.name
        if kind not in SOURCE_TABLE:
            logger.warning("Skipping %s: not a kind the archive copies", kind_dir)
            continue
        if kind_dir.is_symlink() or not kind_dir.is_dir():
            result.skipped.append(kind)
            continue
        if kind in UNNAMED_KINDS:
            yield kind, kind, kind_dir
            continue
        try:
            names = sorted(kind_dir.iterdir())
        except OSError as exc:
            result.errors.append({"source": kind, "error": str(exc.strerror or exc)})
            continue
        for name_dir in names:
            prefix = f"{kind}/{name_dir.name}"
            if name_dir.is_symlink() or not name_dir.is_dir():
                result.skipped.append(prefix)
                continue
            yield kind, prefix, name_dir


def _files_of(
    root: Path, includes: _Includes, prefix: str, result: PassResult
) -> Iterator[tuple[str, Path]]:
    """Every included regular file under *root*, as ``(relpath, path)``; symlinks skipped."""

    def _onerror(exc: OSError) -> None:
        result.errors.append({"source": prefix, "error": f"{exc.filename}: {exc.strerror}"})

    for dirpath, dirnames, filenames in os.walk(root, onerror=_onerror):
        reldir = Path(dirpath).relative_to(root).as_posix()
        reldir = "" if reldir == "." else reldir
        kept: list[str] = []
        for d in sorted(dirnames):
            rel = f"{reldir}/{d}" if reldir else d
            if not includes.may_contain(rel):
                continue
            if os.path.islink(os.path.join(dirpath, d)):
                result.skipped.append(f"{prefix}/{rel}")
                continue
            kept.append(d)
        dirnames[:] = kept
        for f in sorted(filenames):
            rel = f"{reldir}/{f}" if reldir else f
            if includes.matches(rel):
                yield rel, Path(dirpath) / f


def _copy_one(
    tree: ArchiveTree,
    state: ArchiveState,
    source_key: str,
    path: Path,
    result: PassResult,
) -> None:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        result.skipped.append(source_key)
        return
    last = state.sources.get(source_key)
    if last is not None and (last.size, last.source_mtime_ns) == (info.st_size, info.st_mtime_ns):
        return
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(fd)
        temp, size, sha = tree.stage(_read_chunks(fd))
    finally:
        os.close(fd)
    if last is not None and last.sha256 == sha:
        tree.discard(temp)
        return
    relpath = tree.place(temp, source_key)
    tree.append(
        {
            "kind": "file",
            "source": source_key,
            "path": relpath,
            "size": size,
            "sha256": sha,
            "source_mtime_ns": opened.st_mtime_ns,
            "archived_at": _iso(_utcnow()),
        }
    )
    state.sources[source_key] = SourceRecord(
        size=size, source_mtime_ns=opened.st_mtime_ns, sha256=sha
    )
    result.files += 1
    result.bytes += size


def run_pass(
    sources: Path,
    dest: Path,
    *,
    now: datetime | None = None,
    telemetry: TelemetryExporter | None = None,
) -> PassResult:
    """Copy every changed included file under *sources* into today's day of *dest*.

    A source is skipped when its size and mtime match its last manifest record,
    and re-hashed otherwise; it is copied only when its content changed. A file
    that grew is copied whole again. A per-source ``OSError`` is recorded in the
    pass line and the pass goes on. A pass waits while another holds the archive.

    Raises:
        ArchiveDestinationError: If *dest* is missing or unwritable before
            anything is copied, or the pass line cannot be written.
    """
    started = now or _utcnow()
    tree = ArchiveTree(dest, started)
    try:
        return _run_locked(tree, sources, dest, started, telemetry)
    finally:
        tree.close()


def _run_locked(
    tree: ArchiveTree,
    sources: Path,
    dest: Path,
    started: datetime,
    telemetry: TelemetryExporter | None,
) -> PassResult:
    try:
        state = load_state(dest)
    except OSError as exc:
        raise ArchiveDestinationError(f"the manifests under {dest} cannot be read: {exc}") from exc
    result = PassResult(day=tree.day, started_at=_iso(started))

    for kind, prefix, root in _sources_of(sources, result):
        includes = _Includes(SOURCE_TABLE[kind])
        for relpath, path in _files_of(root, includes, prefix, result):
            source_key = f"{prefix}/{relpath}"
            try:
                _copy_one(tree, state, source_key, path, result)
            except OSError as exc:
                result.errors.append({"source": source_key, "error": str(exc.strerror or exc)})

    if telemetry is not None:
        exported, errors = telemetry.export(tree, state, now=started)
        result.telemetry_days.extend(exported)
        result.errors.extend(errors)

    result.completed_at = _iso(_utcnow())
    try:
        tree.append(
            {
                "kind": "pass",
                "started_at": result.started_at,
                "completed_at": result.completed_at,
                "files": result.files,
                "bytes": result.bytes,
                "errors": result.errors,
                "skipped": result.skipped,
                "telemetry_days": result.telemetry_days,
            }
        )
    except OSError as exc:
        raise ArchiveDestinationError(
            f"the pass line cannot be written under {dest}: {exc}"
        ) from exc
    return result
