"""The facility views: the files each render carries under its ``data/``.

Every view is one row of :data:`VIEWS`: its name, its directory under the
render's ``data/``, the predicate that decides whether a render carries it and
the writer that writes it. :func:`osprey.facility.render.render_facility_outputs`
is the only caller: it asks each view's predicate of every render, writes the
views whose predicate holds and names each other one on stderr, which
``osprey build`` and ``osprey facility validate`` both keep while ``validate``
drops the render's stdout.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osprey.facility.build import FacilityDocument

__all__ = ["VIEWS", "View", "ViewInputs", "report_omitted", "view_bytes"]


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
        path: The view's directory, relative to the render's ``data/``; ``.``
            is ``data/`` itself.
        written_when: Whether a render carries the view.
        reason: The config key or model fact ``written_when`` reads, named when
            the view is not written.
        write: Writes the view into the directory it is given and returns the
            files written.
        selected_by: The config key that picks this view from among its
            alternatives, or ``None``. A render that picks another alternative
            does not carry the view and does not name it either.
    """

    name: str
    path: str
    written_when: Callable[[ViewInputs], bool]
    reason: str
    write: Callable[[Path, ViewInputs], list[Path]]
    selected_by: str | None = None


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


def report_omitted(view: View) -> None:
    """Name a view a render does not carry, and why, on stderr.

    Args:
        view: The view whose predicate was false.
    """
    from osprey.cli.output import warn

    warn(f"view {view.name} not written: {view.reason}")


def _always(_inputs: ViewInputs) -> bool:
    return True


def _write_simulator(root: Path, inputs: ViewInputs) -> list[Path]:
    from osprey.facility.views.simulator import write_simulator_view

    return write_simulator_view(root, inputs)


def _write_limits(root: Path, inputs: ViewInputs) -> list[Path]:
    from osprey.facility.views.limits import write_limits_view

    return write_limits_view(root, inputs)


def _write_facts(root: Path, inputs: ViewInputs) -> list[Path]:
    from osprey.facility.views.facts import write_facts_view

    return write_facts_view(root, inputs)


def _bluesky_configured(inputs: ViewInputs) -> bool:
    from osprey.facility.views.bluesky import bluesky_configured

    return bluesky_configured(inputs)


def _write_bluesky(root: Path, inputs: ViewInputs) -> list[Path]:
    from osprey.facility.views.bluesky import write_bluesky_view

    return write_bluesky_view(root, inputs)


def _in_context_selected(inputs: ViewInputs) -> bool:
    from osprey.facility.views.channel_finder import in_context_selected

    return in_context_selected(inputs)


def _write_in_context(root: Path, inputs: ViewInputs) -> list[Path]:
    from osprey.facility.views.channel_finder import write_in_context

    return write_in_context(root, inputs)


#: Every view, in the order a render writes them.
VIEWS: tuple[View, ...] = (
    View(
        name="simulator",
        path="simulator",
        written_when=_always,
        reason="always written",
        write=_write_simulator,
    ),
    View(
        name="limits",
        path=".",
        written_when=_always,
        reason="always written",
        write=_write_limits,
    ),
    View(
        name="facts",
        path=".",
        written_when=_always,
        reason="always written",
        write=_write_facts,
    ),
    View(
        name="bluesky",
        path=".",
        written_when=_bluesky_configured,
        reason="services.bluesky",
        write=_write_bluesky,
    ),
    View(
        name="in_context",
        path="channel_finder",
        written_when=_in_context_selected,
        reason="channel_finder.pipeline_mode",
        write=_write_in_context,
        selected_by="channel_finder.pipeline_mode",
    ),
)
