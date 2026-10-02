"""What a render carries from the facility file.

A build makes the facility file once (``build.build_facility``) and every render
of that build — the deployment's own, each persona's, each container image's —
receives the same outputs through :func:`render_facility_outputs`, the only
writer of them::

    <render root>/facility.json    the facility file, byte-equal in every render
    <render root>/data/<view>/     each view of :data:`osprey.facility.views.VIEWS`
                                   the render's config asks for

No output carries timestamps, version or absolute paths, so equal sources and
equal configs give equal bytes in every render of every build.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from osprey.facility import FACILITY_FILE
from osprey.facility.build import FacilityDocument

__all__ = ["FACILITY_FILE", "facility_bytes", "facility_digest", "render_facility_outputs"]

#: The document last serialised and its bytes: every render of one build hands
#: in the same document, so it is serialised once.
_serialised: tuple[FacilityDocument, bytes] | None = None


def facility_digest(facility_dir: Path) -> str:
    """Hash a ``data/facility`` tree: every file's relative path and bytes.

    Args:
        facility_dir: The ``data/facility`` directory; a missing one hashes as
            an empty tree.

    Returns:
        The sha256 hex digest, equal for two trees holding the same files with
        the same bytes wherever they sit on disk.
    """
    digest = hashlib.sha256()
    if facility_dir.is_dir():
        for path in sorted(facility_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(facility_dir).as_posix().encode("utf-8")
            content = path.read_bytes()
            digest.update(len(relative).to_bytes(8, "big") + relative)
            digest.update(len(content).to_bytes(8, "big") + content)
    return digest.hexdigest()


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
) -> list[Path]:
    """Write the facility outputs of one render.

    Args:
        render_dir: The render's root, the directory holding its ``config.yml``.
        doc: The build's facility file.
        rendered_config: The render's ``config.yml``, as a mapping.
        facility_dir: The build's ``data/facility`` directory, the source of the
            files a view copies.
        omitted_reported: The views this build has already named as not
            written; a view in it is not named again and each view named is
            added. ``None`` names every omitted view.

    Each view whose predicate is false is named on stderr, one line each, once
    per build.

    Returns:
        The files written, sorted.

    Raises:
        FacilityBuildError: ``profile-invalid`` when the render's
            ``simulation.models`` does not resolve against the facility file.
    """
    from osprey.facility import views
    from osprey.facility.served import resolve_served

    inputs = views.ViewInputs(
        doc=doc,
        rendered_config=rendered_config,
        facility_dir=facility_dir,
        served=resolve_served(rendered_config, doc),
    )
    target = render_dir / FACILITY_FILE
    target.write_bytes(facility_bytes(doc))
    written = [target]
    for view in views.VIEWS:
        if view.written_when(inputs):
            written.extend(view.write(render_dir / "data" / view.path, inputs))
        elif omitted_reported is None or view.name not in omitted_reported:
            views.report_omitted(view)
            if omitted_reported is not None:
                omitted_reported.add(view.name)
    return sorted(written)
