"""What a render carries from the facility file.

A build makes the facility file once (``build.build_facility``) and every render
of that build — the deployment's own, each render's, each container image's —
receives the same outputs through :func:`render_facility_outputs`, the only
writer of them::

    <render root>/facility.json    the facility file, byte-equal in every render
    <render root>/data/<view>/     each view of :data:`osprey.facility.views.VIEWS`
                                   the render's config asks for

No output carries timestamps, version or absolute paths, so equal sources and
equal configs give equal bytes in every render of every build. A build hands
every render one record of the views it has written, so a view whose inputs a
render shares with an earlier one is copied from that one's bytes rather than
written again.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from osprey.facility import FACILITY_FILE
from osprey.facility.build import FacilityDocument
from osprey.facility.roots import (
    DEPLOYMENT_CONTENT_DIRS,
    FacilityRoots,
    is_machine_entry,
    is_vcs_metadata,
)

if TYPE_CHECKING:
    from osprey.facility.views import View, ViewInputs

#: What one build's renders have written of each view, keyed on what the view
#: was written from: per file, its path relative to the render root, the file
#: the first render wrote, and the sha256 of its bytes. The bytes themselves are
#: not held: a large facility's views run to hundreds of megabytes, and every
#: render after the first copies them from the first render's files.
WrittenViews = dict[Hashable, list[tuple[str, Path, str]]]

__all__ = [
    "FACILITY_FILE",
    "facility_bytes",
    "facility_digest",
    "facility_sha256",
    "render_facility_outputs",
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
    from osprey.facility.views import json_bytes

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


def write_facility_file(target: Path, doc: FacilityDocument) -> None:
    """Write :func:`facility_bytes` of ``doc`` to ``target``.

    The first render of a build serialises the document; each later one copies
    that render's file a block at a time, held to the digest
    (:func:`_copy_checked`), and serialises again only when the file has
    changed since.
    """
    known = _serialised
    if (
        known is not None
        and known.doc is doc
        and known.written is not None
        and _copy_checked(known.written, target, known.sha256)
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


def _copy_checked(source: Path, target: Path, sha256: str) -> bool:
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


def render_facility_outputs(
    render_dir: Path,
    doc: FacilityDocument,
    rendered_config: Mapping[str, Any],
    facility_dir: Path,
    *,
    omitted_reported: set[str] | None = None,
    built_databases: dict[str, Path] | None = None,
    views_written: WrittenViews | None = None,
    content_dir: Path | None = None,
) -> list[Path]:
    """Write the facility outputs of one render.

    Args:
        render_dir: The render's root, the directory holding its ``config.yml``.
        doc: The build's facility file.
        rendered_config: The render's ``config.yml``, as a mapping.
        facility_dir: The build's machine root, its ``data/facility``
            directory unless mounted, the source of the machine files a view
            copies.
        omitted_reported: The views this build has already named as not
            written, and the view notes it has printed; one in it is not
            printed again and each one printed is added. ``None`` prints every
            one.
        built_databases: The view databases this build has already written,
            keyed by the sha256 of the view bytes each was imported from; a
            view whose bytes match copies the database, and each one imported
            is added. ``None`` imports every one.
        views_written: What this build's earlier renders wrote of each view;
            each view this render writes is added. A view whose writer reads
            what an earlier render's did is copied from its bytes. ``None``
            writes every view. One build's only: it holds neither the facility
            file nor the files a view copies, which a build does not change.
        content_dir: The content root when mounted, the source of the
            deployment's files a view copies; ``None`` means ``facility_dir``.

    Each view whose predicate is false is named on stderr, one line each, once
    per build, unless a selector picks it from among its alternatives: a view
    the selector did not pick is not named.

    Returns:
        The files written, sorted.

    Raises:
        FacilityBuildError: ``profile-invalid`` when the render's
            ``simulation.models`` does not resolve against the facility file.
    """
    from osprey.facility import views
    from osprey.facility.records import FacilityRecords
    from osprey.facility.served import resolve_served

    records = FacilityRecords(doc)
    inputs = views.ViewInputs(
        records=records,
        rendered_config=rendered_config,
        facility_dir=facility_dir,
        served=resolve_served(rendered_config, records),
        reported=omitted_reported,
        built_databases=built_databases,
        content_dir=content_dir,
    )
    target = render_dir / FACILITY_FILE
    write_facility_file(target, doc)
    written = [target]
    for view in views.VIEWS:
        carried, reason = view.written_when(inputs)
        if carried:
            written.extend(_write_view(view, render_dir, inputs, views_written))
        elif view.selected_by is not None:
            continue
        elif omitted_reported is None or view.name not in omitted_reported:
            views.report_omitted(view, reason)
            if omitted_reported is not None:
                omitted_reported.add(view.name)
    return sorted(written)


def _write_view(
    view: View, render_dir: Path, inputs: ViewInputs, views_written: WrittenViews | None
) -> list[Path]:
    """Write one view into a render, or copy it from an earlier render of the build.

    The writer is handed the config keys it reads and nothing else, and a view
    written from the same keys, the same served models and the same engine
    registry is the same bytes; its notes were printed by the render that wrote
    it, and the build prints each once. A copy writes the files, and makes their
    directories, in the order the writer did: a directory lists its entries in
    the order they were made on some filesystems, and the build manifest takes
    its order from that listing.
    """
    from importlib import metadata

    from osprey.facility.views import config_read

    inputs = replace(inputs, rendered_config=config_read(inputs.rendered_config, view.reads))
    root = render_dir / "data" / view.path
    if views_written is None:
        return view.write(root, inputs)
    key = (
        view.name,
        tuple(inputs.served),
        json.dumps(inputs.rendered_config, default=repr),
        metadata.entry_points,
    )
    files = views_written.get(key)
    if files is not None:
        copied = _copy_written(files, render_dir)
        if copied is not None:
            return copied
    paths = view.write(root, inputs)
    views_written[key] = [
        (
            path.relative_to(render_dir).as_posix(),
            path,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in paths
    ]
    return paths


def _copy_written(files: list[tuple[str, Path, str]], render_dir: Path) -> list[Path] | None:
    """Copy a view's files from the render that wrote them; ``None`` when one has changed.

    One file at a time, in the order the view wrote them, a block at a time.
    Each copy is held to the digest taken when the file was written, so a copy
    is the bytes the view wrote: a file changed or removed since sends the
    caller back to writing the view, over whatever was copied.
    """
    paths = []
    for relative, written, digest in files:
        path = render_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not _copy_checked(written, path, digest):
            return None
        paths.append(path)
    return paths
