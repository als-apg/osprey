"""The facility outputs of one render: the facility file and the views beside it.

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

The facility file itself is the description's (:mod:`osprey_facility.render`);
this module is framework, and reaches the description only through it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Hashable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from osprey_facility import FACILITY_FILE
from osprey_facility.build import FacilityDocument
from osprey_facility.render import copy_checked, write_facility_file

if TYPE_CHECKING:
    from osprey.facility.views import View, ViewInputs

#: What one build's renders have written of each view, keyed on what the view
#: was written from: per file, its path relative to the render root, the file
#: the first render wrote, and the sha256 of its bytes. The bytes themselves are
#: not held: a large facility's views run to hundreds of megabytes, and every
#: render after the first copies them from the first render's files.
WrittenViews = dict[Hashable, list[tuple[str, Path, str]]]

__all__ = ["WrittenViews", "render_facility_outputs"]


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
    from osprey_facility.records import FacilityRecords
    from osprey_facility.served import resolve_served

    records = FacilityRecords.of(doc)
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
        if not copy_checked(written, path, digest):
            return None
        paths.append(path)
    return paths
