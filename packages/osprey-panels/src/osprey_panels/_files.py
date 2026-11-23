"""Replace a file whole, through a temporary name nobody can predict.

The panel writer's own copy of ``osprey_connectors.safety_file.atomic_replace``
with ``sync="required"``, so the package needs nothing from the framework.
The temporary name shape (``.<name>.<random>.tmp``) is what
``emit._TEMPORARY`` and ``emit._INDEX_TEMPORARY`` match; it may not change.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path


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


def atomic_replace(path: Path, data: bytes, *, mode: int) -> None:
    """Replace the file at ``path`` with ``data``, whole or not at all.

    ``data`` goes into a temporary file in the same directory under a hidden,
    unpredictable name, so it is never an existing file or a planted link; it
    is given ``mode`` (whatever the umask), synced, and renamed over ``path``,
    and the directory is synced after the rename. A reader sees the old file
    or the new one, never a mix. When anything fails, including the flush,
    the temporary file is removed and the exception propagates.
    """
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        try:
            os.fchmod(fd, mode)
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view) :]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    _fsync_directory(path.parent)
