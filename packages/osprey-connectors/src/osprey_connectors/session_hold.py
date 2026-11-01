"""The session hold record: one durable file per session that holds its writes.

A *hold* is a set of open flags on one session. While any flag is open the
session is held, and every write execution site refuses the session's writes
(``session_held``). A flag is closed only by an answer that names it. This
module owns the record's format, its reader and its writer; it knows nothing
about who opens a flag — a flag carries an opaque ``source`` and ``reason``.

**The store.** Every call takes ``root``; ``None`` reads
:data:`HOLD_DIR_ENV_VAR` (``OSPREY_SESSION_HOLD_DIR``). An unset or empty root
is *no hold store*: nothing is written and nothing is ever held. Under a root,
one session's record is ``<root>/<session_key>/record.json``.

**Fail closed under a root.** :func:`is_held` answers ``True`` for an open
flag, and equally for a missing session directory, a missing record, a record
that cannot be read or parsed, a record of another schema, a symbolic link at
the session directory or the record, and an unsafe session key. It never
raises. :func:`open_flags` reports an unreadable record as the one synthetic
flag :data:`UNREADABLE_FLAG_ID`; answering that flag rewrites a fresh record
(with the synthetic flag kept, closed, for audit). Nothing else rewrites an
unreadable record: :func:`ensure_session` is create-if-absent and leaves an
existing file alone, readable or not.

**Writes.** Every write is a read-modify-write under an exclusive ``flock`` on
``<session dir>/record.json.lock`` and lands through
:func:`~osprey_connectors.safety_file.atomic_replace`, so a reader sees the
old record or the new one and two writers in two processes never lose each
other's flag. Answered flags stay in the record, closed, with the answer, the
operator and the time.

**Reads** follow no symbolic link: the session directory is checked with
:func:`os.lstat` and the record is opened ``O_NOFOLLOW`` and bounded, the same
reader the posture store uses for its tree.

This module imports only :mod:`~osprey_connectors.safety_file` from the
package, so the posture store can import it.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import json
import logging
import os
import re
import stat
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

from osprey_connectors.safety_file import atomic_replace

__all__ = [
    "EXCERPT_LIMIT",
    "HOLD_DIR_ENV_VAR",
    "RECORD_NAME",
    "SCHEMA_VERSION",
    "UNREADABLE_FLAG_ID",
    "HoldFlag",
    "answer_flags",
    "ensure_session",
    "has_no_record",
    "hold_root",
    "is_held",
    "is_safe_session_key",
    "mark_stop_unconfirmed",
    "open_flag",
    "open_flags",
]

logger = logging.getLogger(__name__)

#: The environment variable naming the hold root when no ``root`` is passed.
HOLD_DIR_ENV_VAR: Final = "OSPREY_SESSION_HOLD_DIR"

#: The record's file name inside a session's directory.
RECORD_NAME: Final = "record.json"

#: The record format. A record of any other schema is unreadable, and holds.
SCHEMA_VERSION: Final = 1

#: The id of the one flag reported for a missing or unreadable record.
UNREADABLE_FLAG_ID: Final = "hold-record-unreadable"

#: The longest prompt excerpt a flag stores; the writer truncates.
EXCERPT_LIMIT: Final = 200

#: The source of the synthetic flag.
_UNREADABLE_SOURCE: Final = "session_hold"
_UNREADABLE_REASON: Final = (
    "The session's hold record is missing or unreadable, so the session is held "
    "until an operator answers."
)

#: A session key is one path segment: a UUID, or a name of the same alphabet.
_SAFE_KEY: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")

#: Records are a few hundred bytes per flag; an over-long file is refused
#: rather than truncated into a parse.
_RECORD_READ_LIMIT: Final = 1024 * 1024

_LOCK_TIMEOUT_S: Final = 5.0
_LOCK_POLL_S: Final = 0.01
_DIR_MODE: Final = 0o750
_RECORD_MODE: Final = 0o640
_LOCK_MODE: Final = 0o640

_ANSWERS: Final = frozenset({"stop", "continue"})
_STR_FIELDS: Final = ("flag_id", "source", "reason", "prompt_ref", "prompt_excerpt", "opened_at")


@dataclass(frozen=True)
class HoldFlag:
    """One open flag on a session, as :func:`open_flags` reports it."""

    flag_id: str
    source: str
    reason: str
    prompt_ref: str
    prompt_excerpt: str
    opened_at: str
    stop_unconfirmed: bool = False
    started: tuple[str, ...] = ()


# -- paths -----------------------------------------------------------------------


def hold_root(root: Path | str | None = None) -> Path | None:
    """The hold root: *root* when given, else :data:`HOLD_DIR_ENV_VAR`; ``None`` is no store."""
    text = str(root) if root is not None else os.environ.get(HOLD_DIR_ENV_VAR, "")
    if not text.strip():
        return None
    return Path(text)


def is_safe_session_key(session_key: object) -> bool:
    """Whether *session_key* is one safe path segment (no separator, no dot-name)."""
    return isinstance(session_key, str) and _SAFE_KEY.fullmatch(session_key) is not None


def _checked_key(session_key: object) -> str:
    if not is_safe_session_key(session_key):
        raise ValueError(f"unsafe session key {session_key!r}: not a safe file name")
    return session_key  # type: ignore[return-value]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


# -- reading ---------------------------------------------------------------------


def _read_no_follow(path: Path) -> str:
    """*path*'s text, bounded, refusing a symbolic link at its last component."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        with os.fdopen(fd, encoding="utf-8", closefd=False) as handle:
            text = handle.read(_RECORD_READ_LIMIT + 1)
    finally:
        os.close(fd)
    if len(text) > _RECORD_READ_LIMIT:
        raise OSError(errno.EFBIG, f"{path} is larger than a hold record may be")
    return text


def _valid_flag(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    if not all(isinstance(entry.get(name), str) for name in _STR_FIELDS):
        return False
    if entry.get("state") not in ("open", "closed"):
        return False
    if not isinstance(entry.get("stop_unconfirmed"), bool):
        return False
    started = entry.get("started")
    return isinstance(started, list) and all(isinstance(s, str) for s in started)


def _read_record(root: Path, session_key: str) -> list[dict[str, Any]] | None:
    """The session's flag entries, or ``None`` when the record is missing or unreadable.

    ``None`` covers every failure: a missing or non-directory session dir, a
    link at the dir or the record, an absent, unreadable, over-long or
    malformed file, another schema, another session's key.
    """
    directory = root / session_key
    try:
        mode = os.lstat(directory).st_mode
    except OSError:
        return None
    if not stat.S_ISDIR(mode):
        return None
    try:
        payload = json.loads(_read_no_follow(directory / RECORD_NAME))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema") != SCHEMA_VERSION or payload.get("session_key") != session_key:
        return None
    flags = payload.get("flags")
    if not isinstance(flags, list) or not all(_valid_flag(entry) for entry in flags):
        return None
    return flags


def _to_flag(entry: dict[str, Any]) -> HoldFlag:
    return HoldFlag(
        flag_id=entry["flag_id"],
        source=entry["source"],
        reason=entry["reason"],
        prompt_ref=entry["prompt_ref"],
        prompt_excerpt=entry["prompt_excerpt"],
        opened_at=entry["opened_at"],
        stop_unconfirmed=entry["stop_unconfirmed"],
        started=tuple(entry["started"]),
    )


_SYNTHETIC: Final = HoldFlag(
    flag_id=UNREADABLE_FLAG_ID,
    source=_UNREADABLE_SOURCE,
    reason=_UNREADABLE_REASON,
    prompt_ref="",
    prompt_excerpt="",
    opened_at="",
)


def is_held(session_key: str, *, root: Path | str | None = None) -> bool:
    """Whether *session_key* is held. Never raises.

    No root: ``False``. Under a root: ``True`` for an open flag and for every
    way the record can fail to be read (see the module doc), including an
    unsafe key.
    """
    try:
        resolved = hold_root(root)
        if resolved is None:
            return False
        if not is_safe_session_key(session_key):
            return True
        flags = _read_record(resolved, session_key)
        if flags is None:
            return True
        return any(entry["state"] == "open" for entry in flags)
    except Exception:  # a write-path reader fails closed
        logger.warning("session hold for %r could not be read; held", session_key, exc_info=True)
        try:
            return hold_root(root) is not None
        except Exception:
            return True


def open_flags(session_key: str, *, root: Path | str | None = None) -> list[HoldFlag]:
    """The session's open flags, in the order they were opened.

    No root: ``[]``. A missing or unreadable record under a root: the one
    synthetic :data:`UNREADABLE_FLAG_ID` flag.

    Raises:
        ValueError: *session_key* is not a safe file name.
    """
    resolved = hold_root(root)
    if resolved is None:
        return []
    key = _checked_key(session_key)
    flags = _read_record(resolved, key)
    if flags is None:
        return [_SYNTHETIC]
    return [_to_flag(entry) for entry in flags if entry["state"] == "open"]


def has_no_record(session_key: str, *, root: Path | str | None = None) -> bool:
    """Whether nothing at all is on disk for this session under a root.

    The one question :func:`is_held` and :func:`open_flags` deliberately do not
    separate from an unreadable record: a caller that knows no session runs
    under this key can tell "never started" from "lost". It changes no answer of
    theirs. A session directory of any kind, even empty or a dangling link, is
    ``False``, and so is a path that cannot be examined.

    Raises:
        ValueError: *session_key* is not a safe file name.
    """
    resolved = hold_root(root)
    key = _checked_key(session_key)
    if resolved is None:
        return False
    try:
        os.lstat(resolved / key)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return False


# -- writing ---------------------------------------------------------------------


def _session_dir(root: Path, session_key: str) -> Path:
    """The session's directory, created when absent; never a symbolic link."""
    root.mkdir(parents=True, exist_ok=True)
    directory = root / session_key
    with contextlib.suppress(FileExistsError):
        os.mkdir(directory, _DIR_MODE)
    mode = os.lstat(directory).st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise OSError(errno.ELOOP, f"{directory} is not a plain directory; nothing was written")
    return directory


@contextlib.contextmanager
def _locked(root: Path, session_key: str) -> Iterator[Path]:
    """Hold the session's record lock; yields the record path."""
    directory = _session_dir(root, session_key)
    lock = directory / (RECORD_NAME + ".lock")
    fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, _LOCK_MODE)
    try:
        deadline = time.monotonic() + _LOCK_TIMEOUT_S
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"The session hold record at {directory} stayed locked for "
                        f"{_LOCK_TIMEOUT_S:g} s; nothing was written"
                    ) from None
                time.sleep(_LOCK_POLL_S)
        try:
            yield directory / RECORD_NAME
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _write(path: Path, session_key: str, flags: list[dict[str, Any]]) -> None:
    payload = {"schema": SCHEMA_VERSION, "session_key": session_key, "flags": flags}
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_replace(path, data, mode=_RECORD_MODE)


def _ids(flag_ids: Iterable[str]) -> frozenset[str]:
    if isinstance(flag_ids, (str, bytes)):
        raise TypeError("flag_ids is a collection of ids, not one string")
    return frozenset(flag_ids)


def _open_entry(
    *,
    flag_id: str,
    source: str,
    reason: str,
    prompt_ref: str,
    prompt_excerpt: str,
    opened_at: str,
) -> dict[str, Any]:
    """A new flag's record entry, with every field :func:`_valid_flag` requires."""
    return {
        "flag_id": flag_id,
        "source": source,
        "reason": reason,
        "prompt_ref": prompt_ref,
        "prompt_excerpt": prompt_excerpt,
        "opened_at": opened_at,
        "state": "open",
        "stop_unconfirmed": False,
        "started": [],
    }


def ensure_session(session_key: str, *, root: Path | str | None = None) -> None:
    """Create the session's empty record when none exists. No root: nothing.

    Never rewrites an existing file, readable or not.

    Raises:
        ValueError: *session_key* is not a safe file name.
        OSError: The session directory is a link or could not be created.
        RuntimeError: The lock was not acquired in time.
    """
    resolved = hold_root(root)
    if resolved is None:
        return
    key = _checked_key(session_key)
    with _locked(resolved, key) as path:
        if os.path.lexists(path):
            return
        _write(path, key, [])


def open_flag(
    session_key: str,
    *,
    flag_id: str,
    source: str,
    reason: str,
    prompt_ref: str,
    prompt_excerpt: str,
    root: Path | str | None = None,
) -> bool:
    """Open one flag; ``True`` when it was opened.

    ``False`` when there is no root, when a flag with the same
    ``(source, prompt_ref)`` or the same ``flag_id`` exists (open or closed),
    or when the record exists and is unreadable — that session is already held
    by the synthetic flag, and only its answer rewrites the record. A missing
    record is created with the flag.

    Raises:
        ValueError: *session_key* is not a safe file name.
        OSError: The session directory is a link, or the write failed.
        RuntimeError: The lock was not acquired in time.
    """
    resolved = hold_root(root)
    if resolved is None:
        return False
    key = _checked_key(session_key)
    if len(prompt_excerpt) > EXCERPT_LIMIT:
        prompt_excerpt = prompt_excerpt[: EXCERPT_LIMIT - 1] + "…"
    with _locked(resolved, key) as path:
        flags = _read_record(resolved, key)
        if flags is None:
            if os.path.lexists(path):
                logger.warning(
                    "session hold record %s is unreadable; flag %s not added", path, flag_id
                )
                return False
            flags = []
        for entry in flags:
            if entry["flag_id"] == flag_id or (
                entry["source"] == source and entry["prompt_ref"] == prompt_ref
            ):
                return False
        flags.append(
            _open_entry(
                flag_id=flag_id,
                source=source,
                reason=reason,
                prompt_ref=prompt_ref,
                prompt_excerpt=prompt_excerpt,
                opened_at=_now(),
            )
        )
        _write(path, key, flags)
    return True


def answer_flags(
    session_key: str,
    flag_ids: Iterable[str],
    *,
    answer: Literal["stop", "continue"],
    operator: str,
    root: Path | str | None = None,
) -> tuple[str, ...]:
    """Close the named open flags with *answer*; the ids released, in record order.

    Unknown and already-closed ids are ignored. Against a missing or unreadable
    record only :data:`UNREADABLE_FLAG_ID` can be answered, and answering it
    writes a fresh record holding that flag, closed. No root: ``()``.

    Raises:
        ValueError: An unsafe *session_key*, or an *answer* that is neither
            ``"stop"`` nor ``"continue"``.
        TypeError: *flag_ids* is a single string.
        OSError: The session directory is a link, or the write failed.
        RuntimeError: The lock was not acquired in time.
    """
    if answer not in _ANSWERS:
        raise ValueError(f"answer must be 'stop' or 'continue', not {answer!r}")
    wanted = _ids(flag_ids)
    resolved = hold_root(root)
    if resolved is None:
        return ()
    key = _checked_key(session_key)
    if not wanted:
        return ()
    closed = {"state": "closed", "answer": answer, "operator": operator}
    with _locked(resolved, key) as path:
        flags = _read_record(resolved, key)
        if flags is None:
            if UNREADABLE_FLAG_ID not in wanted:
                return ()
            entry = _open_entry(
                flag_id=_SYNTHETIC.flag_id,
                source=_SYNTHETIC.source,
                reason=_SYNTHETIC.reason,
                prompt_ref=_SYNTHETIC.prompt_ref,
                prompt_excerpt=_SYNTHETIC.prompt_excerpt,
                opened_at=_SYNTHETIC.opened_at,
            )
            entry.update(closed, answered_at=_now())
            _write(path, key, [entry])
            return (UNREADABLE_FLAG_ID,)
        released: list[str] = []
        stamp = _now()
        for entry in flags:
            if entry["state"] == "open" and entry["flag_id"] in wanted:
                entry.update(closed, answered_at=stamp)
                released.append(str(entry["flag_id"]))
        if released:
            _write(path, key, flags)
    return tuple(released)


def mark_stop_unconfirmed(
    session_key: str,
    flag_ids: Iterable[str],
    *,
    operator: str,
    started: tuple[str, ...],
    root: Path | str | None = None,
) -> tuple[str, ...]:
    """Mark the named open flags as stopped-but-unconfirmed; the ids marked.

    The flags stay open — the turn did not stop — and record *started*, the
    commands still running, with the operator and the time. Closed and unknown
    ids are ignored; an unreadable record marks nothing. No root: ``()``.

    Raises:
        ValueError: *session_key* is not a safe file name.
        TypeError: *flag_ids* is a single string.
        OSError: The session directory is a link, or the write failed.
        RuntimeError: The lock was not acquired in time.
    """
    wanted = _ids(flag_ids)
    resolved = hold_root(root)
    if resolved is None:
        return ()
    key = _checked_key(session_key)
    if not wanted:
        return ()
    with _locked(resolved, key) as path:
        flags = _read_record(resolved, key)
        if flags is None:
            return ()
        marked: list[str] = []
        stamp = _now()
        for entry in flags:
            if entry["state"] == "open" and entry["flag_id"] in wanted:
                entry.update(
                    stop_unconfirmed=True,
                    started=[str(command) for command in started],
                    stop_operator=operator,
                    stop_marked_at=stamp,
                )
                marked.append(entry["flag_id"])
        if marked:
            _write(path, key, flags)
    return tuple(marked)
