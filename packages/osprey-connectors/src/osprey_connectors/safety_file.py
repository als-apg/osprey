"""Replace a file whole: the one way OSPREY rewrites a file in place.

Every writer that rewrites a file calls :func:`atomic_replace`, so a reader
sees the old file or the new one and never a mix, a crash never leaves half a
file, and the file's mode never depends on the process umask.

It also states, once, the shape of a group-shared location: a place written
and read under several uids that reach one another through the setgid group
of its directory. Such a directory is :data:`SHARED_DIR_MODE`, created through
:func:`ensure_shared_dir`; a file in it is :data:`SHARED_FILE_MODE`, and a lock
in it :data:`SHARED_LOCK_MODE`::

    <location>/             2770  setgid: every entry below takes its group
      <identity>/           2770  every level a writer creates, set exactly
        record.json         0640  read by the group, written by its owner
        record.json.lock    0660  taken by every writer in the group

This module uses the standard library only. The Claude Code hooks run under
the system ``python3`` and import nothing from OSPREY, so they ship this file
beside them as ``.claude/hooks/osprey_safety_file.py``, a byte-identical copy
that ``tests/hooks/test_osprey_safety_file.py`` holds equal to this one. A
change is made here and copied there. That hooks catalog row is also the copy
the build stages beside the control-identity script that containers run by
path, so the byte pin keeps that copy equal to this module too.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Literal

__all__ = [
    "SHARED_DIR_MODE",
    "SHARED_FILE_MODE",
    "SHARED_LOCK_MODE",
    "atomic_replace",
    "ensure_shared_dir",
]

#: A directory of a group-shared location. Setgid, so every entry created in it
#: takes its group whatever uid created it; group-write, so a writer under one
#: uid can replace what another wrote; nothing for ``other``, which would let it
#: list the names inside.
SHARED_DIR_MODE = 0o2770

#: A file in a group-shared location: written by its owner, read through the
#: directory's group by the uids that never coordinate with that owner.
SHARED_FILE_MODE = 0o640

#: A lock file in a group-shared location: whichever writer creates it first
#: sets its mode, and every writer in the group has to be able to open it.
SHARED_LOCK_MODE = 0o660


def _fsync_directory(directory: Path) -> None:
    """Sync ``directory`` so a newly created entry in it survives a crash."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        with contextlib.suppress(OSError):
            os.fsync(fd)
    finally:
        os.close(fd)


def atomic_replace(
    path: Path,
    data: bytes,
    *,
    mode: int,
    sync: Literal["required", "best_effort"] = "required",
) -> None:
    """Replace the file at ``path`` with ``data``, whole or not at all.

    ``data`` goes into a temporary file in the same directory, created here
    under a hidden name nobody can predict (``.<name>.<random>.tmp``), so it is
    never an existing file or a link someone planted; it is given ``mode``
    (whatever the umask), synced, and renamed over ``path``, and the directory
    is synced after the rename. A reader sees the old file or the new one,
    never a mix. When anything fails, the temporary file is removed and the
    exception propagates.

    ``sync="required"`` raises when the disk refuses to flush the file;
    ``sync="best_effort"`` carries on without the flush and still replaces
    the file.

    Choose ``sync="required"`` exactly when the caller surfaces the failure to
    someone who can retry (an operator, a CLI, a build); choose
    ``sync="best_effort"`` when the caller swallows the error, because then a
    file replaced without its flush is more durable than the old file kept.
    """
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        try:
            os.fchmod(fd, mode)
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view) :]
            if sync == "required":
                os.fsync(fd)
            else:
                with contextlib.suppress(OSError):
                    os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    _fsync_directory(path.parent)


def _set_shared_mode(level: Path) -> None:
    """Set :data:`SHARED_DIR_MODE` on *level*, never through a symbolic link.

    The parent is group-writable by parties other than this writer, so the
    level just made could be swapped for a link before its mode is set; the
    open refuses a link at the last component (``O_NOFOLLOW``) and anything
    but a directory (``O_DIRECTORY``), and the mode goes on the descriptor.
    """
    fd = os.open(level, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fchmod(fd, SHARED_DIR_MODE)
    finally:
        os.close(fd)


def ensure_shared_dir(
    directory: Path, *, modes: Literal["required", "best_effort"] = "required"
) -> None:
    """Create *directory* and the levels above it at :data:`SHARED_DIR_MODE`.

    Every level this call creates — the levels above the leaf as much as the
    leaf — is set to exactly :data:`SHARED_DIR_MODE` rather than merely widened:
    ``mkdir`` honours the umask and drops the setgid bit, so a level left as it
    made it is 0755 (``other`` lists the names inside) or, under umask 077,
    0700 (the group cannot reach the files inside at all). A level that already
    exists is not this writer's: deployment provisioned it, or another uid made
    it, and it keeps the mode it has. The levels are made one at a time from
    the top, so a level another writer makes first, between the look and the
    make, is left to that writer while the levels this call does make still
    get their mode. A mode is never set through a symbolic link.

    ``modes="required"`` raises when a mode cannot be set;
    ``modes="best_effort"`` keeps the directory as made, for a writer whose
    record is worth more in a narrower directory than not written at all.

    Raises:
        OSError: A level could not be created, or (``modes="required"``) a
            level's mode could not be set, a link included (``errno.ELOOP``).
    """
    missing = []
    probe = directory
    while probe != probe.parent and not probe.exists():
        missing.append(probe)
        probe = probe.parent
    for level in reversed(missing):
        try:
            os.mkdir(level)
        except FileExistsError:
            # Another writer got there first: that level is not this writer's.
            continue
        if modes == "required":
            _set_shared_mode(level)
        else:
            with contextlib.suppress(OSError):
                _set_shared_mode(level)
