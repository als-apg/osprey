"""Reproducible-render manifest primitives.

The leaf constants and file-hashing helper for the project render manifest
(``.osprey-manifest.json``). These are the only manifest pieces the low
layers need (e.g. ``services.build_artifacts.ownership`` computes user-owned
framework hashes), so they live in the build-time kernel. The catalog-aware
manifest generation/validation stays in ``cli.templates.manifest``, which
re-imports these primitives.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

MANIFEST_FILENAME = ".osprey-manifest.json"


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
