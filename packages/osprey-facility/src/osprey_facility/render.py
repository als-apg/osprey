"""What a render carries from the facility file, and how its bytes are made.

A build makes the facility file once (``build.build_facility``) and every render
of that build carries it, byte-equal, as ``<render root>/facility.json``:
:func:`write_facility_file` writes it, serialised by :func:`facility_bytes`,
and copies the first render's file into each later one (:func:`copy_checked`).
The views a render carries beside it are framework, written by
:func:`osprey.facility.views.outputs.render_facility_outputs`.

No output carries timestamps, version or absolute paths, so equal sources give
equal bytes in every render of every build. :func:`json_bytes` is the one JSON
serialisation the facility file and every view share.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from osprey_facility import FACILITY_FILE
from osprey_facility.build import FacilityDocument
from osprey_facility.roots import (
    DEPLOYMENT_CONTENT_DIRS,
    FacilityRoots,
    is_machine_entry,
    is_vcs_metadata,
)

__all__ = [
    "FACILITY_FILE",
    "copy_checked",
    "facility_bytes",
    "facility_digest",
    "facility_sha256",
    "json_bytes",
    "write_facility_file",
]


@dataclass(slots=True)
class _Serialised:
    """The document last serialised: the size and sha256 of its bytes, and where a render wrote them.

    The bytes are not held. A large facility's file runs to hundreds of
    megabytes, and every render after the first copies the first render's
    file, checked against the digest.
    """

    doc: FacilityDocument
    size: int
    sha256: str
    written: Path | None = None


#: Every render of one build hands in the same document; see :class:`_Serialised`.
_serialised: _Serialised | None = None


def facility_digest(facility_dir: Path, *, content_dir: Path | None = None) -> str:
    """Hash a deployment's facility facts: every file's relative path and bytes.

    A single tree hashes every file of ``data/facility``. A mounted deployment
    hashes the machine entries of its machine root (:func:`is_machine_entry`)
    and the content directories of its content root
    (:data:`DEPLOYMENT_CONTENT_DIRS`), each keyed by its path relative to its
    own root. Version-control metadata (``.git`` at any depth) is never hashed,
    so a clone's history moves no digest.

    Args:
        facility_dir: The machine root, the ``data/facility`` directory of a
            single tree; a missing one hashes as an empty tree.
        content_dir: The content root when mounted; ``None``, or the same
            directory, is a single tree.

    Returns:
        The sha256 hex digest, equal for two trees holding the same files with
        the same bytes wherever they sit on disk.
    """
    roots = FacilityRoots(
        machine=facility_dir, content=content_dir if content_dir is not None else facility_dir
    )
    entries = [
        (relative, path)
        for relative, path in _tree_files(roots.machine)
        if is_machine_entry(roots, relative)
    ]
    if roots.mounted:
        entries.extend(
            (relative, path)
            for relative, path in _tree_files(roots.content)
            if relative.parts[0] in DEPLOYMENT_CONTENT_DIRS and not is_vcs_metadata(relative)
        )
    digest = hashlib.sha256()
    for relative, path in sorted(entries):
        name = relative.as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big") + name)
        digest.update(len(content).to_bytes(8, "big") + content)
    return digest.hexdigest()


def _tree_files(root: Path) -> list[tuple[PurePosixPath, Path]]:
    """Every file under ``root``, with its path relative to ``root``; none when missing."""
    if not root.is_dir():
        return []
    return [
        (PurePosixPath(path.relative_to(root).as_posix()), path)
        for path in root.rglob("*")
        if path.is_file()
    ]


def facility_bytes(doc: FacilityDocument) -> bytes:
    """Serialise the facility file: sorted keys, two-space indent, one final newline.

    Args:
        doc: The facility file as ``build_facility`` returned it.

    Returns:
        The UTF-8 bytes of ``facility.json``.
    """
    global _serialised
    raw = json_bytes(doc)
    if _serialised is None or _serialised.doc is not doc:
        _serialised = _Serialised(doc, len(raw), hashlib.sha256(raw).hexdigest())
    return raw


def facility_sha256(doc: FacilityDocument) -> str:
    """The sha256 hex digest of :func:`facility_bytes`, serialising ``doc`` at most once per build."""
    if _serialised is None or _serialised.doc is not doc:
        facility_bytes(doc)
    assert _serialised is not None
    return _serialised.sha256


def json_bytes(document: Any, *, sort_keys: bool = True) -> bytes:
    """``(json.dumps(document, indent=2, ...) + "\\n").encode("utf-8")``, without its copies.

    An indented ``json.dumps`` joins a list of every token it wrote, then the
    text is copied once more into its bytes: for a document of a large facility
    the transient copies are several times the size of the file. Encoding
    into a byte buffer token by token holds only the bytes. The output is the
    same, byte for byte: ``json.dump`` and ``json.dumps`` write one token
    stream.

    Args:
        document: The document.
        sort_keys: Whether keys are written sorted.

    Returns:
        Its UTF-8 bytes: two-space indent, non-ASCII written as itself, NaN
        refused, one final newline.
    """
    import io
    import json

    buffer = io.BytesIO()
    text = io.TextIOWrapper(buffer, encoding="utf-8", newline="")
    json.dump(document, text, indent=2, sort_keys=sort_keys, ensure_ascii=False, allow_nan=False)
    text.write("\n")
    text.flush()
    text.detach()
    return buffer.getvalue()


def write_facility_file(target: Path, doc: FacilityDocument) -> None:
    """Write :func:`facility_bytes` of ``doc`` to ``target``.

    The first render of a build serialises the document; each later one copies
    that render's file a block at a time, held to the digest
    (:func:`copy_checked`), and serialises again only when the file has
    changed since.
    """
    known = _serialised
    if (
        known is not None
        and known.doc is doc
        and known.written is not None
        and copy_checked(known.written, target, known.sha256)
    ):
        return
    target.write_bytes(facility_bytes(doc))
    if _serialised is not None and _serialised.doc is doc:
        _serialised.written = target


def serialised_document(raw: bytes) -> FacilityDocument | None:
    """The document :func:`facility_bytes` last serialised to ``raw``, when it reads back as itself.

    A render reads its ``facility.json`` back for its templates. In a build those
    are the bytes :func:`facility_bytes` just wrote, and their document is
    already in memory; parsing them again would hold a second copy of a
    facility file, which for a large facility is the size of the first. The
    document stands in for the parse only when the parse would give an equal
    document of the same types (:func:`_reads_back_as_itself`): every container a
    dict or a list and every key a string, the shapes JSON holds.

    Args:
        raw: Bytes read from a facility file.

    Returns:
        The document, shared and never to be changed; ``None`` when ``raw`` is
        not what :func:`facility_bytes` last returned, or its document holds a
        shape the parse would not give back.
    """
    known = _serialised
    if known is None or len(raw) != known.size:
        return None
    if hashlib.sha256(raw).hexdigest() != known.sha256:
        return None
    return _shared_document()


def serialised_file_document(path: Path) -> FacilityDocument | None:
    """:func:`serialised_document` for the file at ``path``, read in pieces.

    The file is compared with the bytes by its size and its sha256, taken a
    block at a time, so asking about a large facility file holds none of it.

    Args:
        path: A facility file.

    Returns:
        As :func:`serialised_document`.

    Raises:
        OSError: The file cannot be read; :exc:`FileNotFoundError` when it is
            not there.
    """
    known = _serialised
    with path.open("rb") as handle:
        if known is None or os.fstat(handle.fileno()).st_size != known.size:
            return None
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(_BLOCK), b""):
            digest.update(block)
    if digest.hexdigest() != known.sha256:
        return None
    return _shared_document()


#: How much of a file is read at a time to hash or copy it.
_BLOCK = 1 << 20


def copy_checked(source: Path, target: Path, sha256: str) -> bool:
    """Copy ``source`` to ``target`` a block at a time; whether the bytes copied have ``sha256``.

    A copy whose digest differs is left at ``target`` for the caller to write over.
    """
    copied = hashlib.sha256()
    try:
        with source.open("rb") as reading, target.open("wb") as writing:
            for block in iter(lambda: reading.read(_BLOCK), b""):
                copied.update(block)
                writing.write(block)
    except OSError:
        return False
    return copied.hexdigest() == sha256


def _shared_document() -> FacilityDocument | None:
    """The memoised document when it reads back as itself (see :func:`serialised_document`)."""
    global _reads_back
    if _serialised is None:
        return None
    doc = _serialised.doc
    if _reads_back is None or _reads_back[0] is not doc:
        _reads_back = (doc, _reads_back_as_itself(doc))
    return doc if _reads_back[1] else None


#: The document last asked about and whether it reads back as itself.
_reads_back: tuple[FacilityDocument, bool] | None = None


def _reads_back_as_itself(doc: Any) -> bool:
    """Whether ``json.loads(json.dumps(doc))`` gives ``doc`` with every type unchanged.

    ``facility_bytes`` refuses NaN and what JSON cannot spell, so the values that
    serialise are dicts, lists, tuples, strings, numbers, booleans and ``None``.
    Of those only a tuple (read back as a list) and a key that is not a string
    (read back as one) change on the way.
    """
    pending: list[Any] = [doc]
    while pending:
        value = pending.pop()
        if type(value) is dict:
            if not all(type(key) is str for key in value):
                return False
            pending.extend(value.values())
        elif type(value) is list:
            pending.extend(value)
        elif type(value) not in (str, int, float, bool, type(None)):
            return False
    return True
