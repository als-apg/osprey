"""Canonical ``ao.json``/``ad.json`` writer and reader.

``osprey mml import`` stores the merged, normalised export as two files under
``data/mml/``, and every later stage reads them back. Their sha256 digests are
stamped into every emitted artifact, so the bytes must be a pure function of the
content:

- Serialised with ``json.dumps(sort_keys=True, indent=2, ensure_ascii=False,
  allow_nan=False)`` plus one trailing ``\\n``, UTF-8, no newline translation.
  Key order is therefore never meaningful; ``_import_order`` is a list and keeps
  the systems' encounter order.
- ``allow_nan=False`` proves the normaliser left no non-finite float: a stray
  ``inf``/``nan`` raises instead of writing a bare ``Infinity``/``NaN`` token.
- ``None`` slots serialise as ``null``.
- Both documents are serialised before either file is touched, so a refusal
  leaves the previous pair intact and never a half-written pair.
- Each file is replaced through a temporary sibling and a rename, and is not
  rewritten at all when its bytes already match, so an identical re-import
  keeps the file (and its mtime). :func:`write_if_changed` is that rule, and
  every emitter that wants a byte-stable artifact writes through it.

The module is pure stdlib.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

__all__ = [
    "AD_FILENAME",
    "AO_FILENAME",
    "read_canonical",
    "sha256_of",
    "write_canonical",
    "write_if_changed",
]

#: File name of the canonical Accelerator Objects document.
AO_FILENAME = "ao.json"

#: File name of the canonical Accelerator Data document.
AD_FILENAME = "ad.json"

#: Read size used by :func:`sha256_of`.
_CHUNK = 1024 * 1024


def write_canonical(ao: dict, ad: dict, out_dir: Path) -> tuple[Path, Path]:
    """Write the canonical AO and AD documents.

    Args:
        ao: The merged AO ``{system: {family: body}}`` with its ``_exports`` and
            ``_import_order`` bookkeeping keys.
        ad: The merged AD ``{system: ad}``; ``{}`` when no input carried one.
        out_dir: Directory to write into, created (with parents) if missing.

    Returns:
        The paths of ``ao.json`` and ``ad.json``, in that order.

    Raises:
        ValueError: Either document holds a non-finite float. Nothing is written.
        TypeError: Either document holds a value JSON cannot represent.
        OSError: A file could not be written; that file is left as it was.
    """
    out_dir = Path(out_dir)
    ao_text = _dumps(ao)
    ad_text = _dumps(ad)
    ao_path = out_dir / AO_FILENAME
    ad_path = out_dir / AD_FILENAME
    write_if_changed(ao_path, ao_text)
    write_if_changed(ad_path, ad_text)
    return ao_path, ad_path


def read_canonical(out_dir: Path) -> tuple[dict, dict]:
    """Read the canonical AO and AD documents back.

    Args:
        out_dir: Directory holding ``ao.json`` and ``ad.json``.

    Returns:
        The AO and AD dictionaries, in that order.

    Raises:
        FileNotFoundError: Either file is missing.
        ValueError: A file is not valid JSON, carries a bare ``NaN``/``Infinity``
            token, or its top level is not an object. The message names the file.
    """
    out_dir = Path(out_dir)
    return _load(out_dir / AO_FILENAME), _load(out_dir / AD_FILENAME)


def sha256_of(path: Path) -> str:
    """Return the lowercase hex sha256 digest of a file's bytes.

    Args:
        path: File to hash.

    Returns:
        The 64-character hex digest.

    Raises:
        OSError: The file could not be read.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _dumps(document: Any) -> str:
    return (
        json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def _refuse_constant(token: str) -> Any:
    raise ValueError(f"non-finite token {token!r} is not canonical JSON")


def _load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"canonical MML file not found: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"), parse_constant=_refuse_constant)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{path}: top level must be a JSON object")
    return document


def write_if_changed(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically, leaving identical bytes untouched.

    Args:
        path: The file to write; its directory is created (with parents) if
            missing.
        text: What to write, as UTF-8 with no newline translation.

    Raises:
        OSError: The directory could not be created, the temporary sibling
            could not be written, or the rename failed. ``path`` is left as it
            was in every case.
    """
    data = text.encode("utf-8")
    if path.is_file() and path.read_bytes() == data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
