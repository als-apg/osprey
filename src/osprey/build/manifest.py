"""Reproducible-render manifest primitives.

The leaf constants and file-hashing helper for the project render manifest
(``.osprey-manifest.json``). These are the only manifest pieces the low
layers need (e.g. ``agent_runner.build_artifacts.ownership`` computes user-owned
framework hashes), so they live in the build-time kernel. The catalog-aware
manifest generation/validation stays in ``cli.templates.manifest``, which
re-imports these primitives.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

MANIFEST_FILENAME = ".osprey-manifest.json"

#: The digest a writer took of a file it wrote, with the file's size and
#: modification time just after: see :func:`note_sha256`.
_noted: dict[str, tuple[int, int, str]] = {}


def note_sha256(file_path: Path, sha256: str) -> None:
    """Record the digest a writer already holds of the file it just wrote.

    A build writes the facility file and its views into every render and takes
    each one's sha256 as it writes or copies it; the manifest of the same
    render then lists the checksum of every file. Noting the digest here lets
    :func:`noted_sha256` answer for the file without reading it again, which
    for a large facility is hundreds of megabytes per render.

    Args:
        file_path: The file just written.
        sha256: The hex digest of the bytes written.
    """
    stat = os.stat(file_path)
    _noted[os.fspath(file_path)] = (stat.st_size, stat.st_mtime_ns, sha256)


def noted_sha256(file_path: Path) -> str | None:
    """The digest noted for a file, while the file is as it was when noted.

    Args:
        file_path: The file, spelled as it was to :func:`note_sha256`.

    Returns:
        The noted hex digest when the file's size and modification time are
        what they were then; ``None`` for a file never noted, changed since or
        gone, which the caller hashes.
    """
    noted = _noted.get(os.fspath(file_path))
    if noted is None:
        return None
    try:
        stat = os.stat(file_path)
    except OSError:
        return None
    size, modified, sha256 = noted
    return sha256 if (stat.st_size, stat.st_mtime_ns) == (size, modified) else None


def sha256_file(file_path: Path) -> str:
    """Calculate SHA256 hash of a file.

    Args:
        file_path: Path to the file

    Returns:
        Hex-encoded SHA256 hash
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()
