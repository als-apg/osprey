"""The facility views: the files each render carries under its ``data/``.

Every view is one row of :data:`VIEWS`: its name, its directory under the
render's ``data/``, the predicate that decides whether a render carries it and
the writer that writes it. :func:`osprey.facility.render.render_facility_outputs`
is the only caller: it asks each view's predicate of every render and writes the
views whose predicate holds.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osprey.facility.build import FacilityDocument

__all__ = ["VIEWS", "View", "ViewInputs", "view_bytes"]


@dataclass(frozen=True)
class ViewInputs:
    """What every view of one render is written from.

    Attributes:
        doc: The build's facility file.
        rendered_config: The render's ``config.yml``, as a nested mapping.
        facility_dir: The build's ``data/facility`` directory, the source of
            the files a view copies.
        served: The models the render serves, as ``resolve_served`` returned
            them.
    """

    doc: FacilityDocument
    rendered_config: Mapping[str, Any]
    facility_dir: Path
    served: list[str]


@dataclass(frozen=True)
class View:
    """One view: where it goes, when a render carries it and how it is written.

    Attributes:
        name: The view's name.
        path: The view's directory, relative to the render's ``data/``.
        written_when: Whether a render carries the view.
        reason: The config key or model fact ``written_when`` reads, named when
            the view is not written.
        write: Writes the view into the directory it is given and returns the
            files written.
    """

    name: str
    path: str
    written_when: Callable[[ViewInputs], bool]
    reason: str
    write: Callable[[Path, ViewInputs], list[Path]]


def view_bytes(document: Mapping[str, Any]) -> bytes:
    """Serialise a view's JSON document: sorted keys, two-space indent, one final newline.

    Args:
        document: The document.

    Returns:
        Its UTF-8 bytes.
    """
    import json

    text = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return (text + "\n").encode("utf-8")


def _always(_inputs: ViewInputs) -> bool:
    return True


def _write_simulator(root: Path, inputs: ViewInputs) -> list[Path]:
    from osprey.facility.views.simulator import write_simulator_view

    return write_simulator_view(root, inputs)


#: Every view, in the order a render writes them.
VIEWS: tuple[View, ...] = (
    View(
        name="simulator",
        path="simulator",
        written_when=_always,
        reason="always written",
        write=_write_simulator,
    ),
)
