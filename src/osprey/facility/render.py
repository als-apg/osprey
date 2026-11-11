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
from collections.abc import Hashable, Mapping
from dataclasses import replace
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

#: What one build's renders have written of each view: the files, relative to
#: the render root, and their bytes, keyed on what the view was written from.
WrittenViews = dict[Hashable, list[tuple[str, bytes]]]

__all__ = ["FACILITY_FILE", "facility_bytes", "facility_digest", "render_facility_outputs"]

#: The document last serialised and its bytes: every render of one build hands
#: in the same document, so it is serialised once.
_serialised: tuple[FacilityDocument, bytes] | None = None


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
    if _serialised is None or _serialised[0] is not doc:
        text = json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        _serialised = (doc, (text + "\n").encode("utf-8"))
    return _serialised[1]


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
    target.write_bytes(facility_bytes(doc))
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
    if files is None:
        paths = view.write(root, inputs)
        views_written[key] = [
            (path.relative_to(render_dir).as_posix(), path.read_bytes()) for path in paths
        ]
        return paths
    paths = []
    for relative, content in files:
        path = render_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        paths.append(path)
    return paths
