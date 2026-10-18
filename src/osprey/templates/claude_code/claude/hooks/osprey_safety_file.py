"""Replace a file whole: the one way OSPREY rewrites a file in place.

Every writer that rewrites a file calls :func:`atomic_replace`, so a reader
sees the old file or the new one and never a mix, a crash never leaves half a
file, and the file's mode never depends on the process umask.

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

__all__ = ["atomic_replace"]


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
