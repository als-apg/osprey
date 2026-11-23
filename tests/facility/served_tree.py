"""Build a served tree for a test: facility layers, then the build, then the views.

A test that drives a connector hands it the addresses it reads and writes as a
facility tree, the way a deployment does. :func:`served_tree` writes those
addresses as channel records under ``<root>/data/facility``, builds the
facility file and renders it into ``<root>/build``, and returns that render's
simulator view, the directory a connector serves from. :func:`in_process_config`
names that view in the in-process simulator's ``connect()`` settings.

Writing and building the tree, and the checks on what a view serves, are
framework-free and live in the facility package's suite
(:mod:`osprey_facility_tests._served_tree`); this module adds the render,
which needs the framework's views, and re-exports the rest.

Every setpoint is writable: the render runs the simulated target with
``limits_checking.mode: optional``, so a setpoint without a limits record takes
no limits. A test that checks limits does so through its own validator, never
through the tree.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from osprey_facility_tests._served_tree import (
    IN_PROCESS_RENDER_CONFIG,
    LEAK_VISIBILITY_FACTOR,
    READBACK_FLOOR,
    SIMULATOR_VIEW,
    assert_unmoved,
    assert_write_visible,
    check_served,
    in_process_config,
    served_active,
    served_facility,
    unmoved_band,
)

__all__ = [
    "IN_PROCESS_RENDER_CONFIG",
    "LEAK_VISIBILITY_FACTOR",
    "READBACK_FLOOR",
    "SIMULATOR_VIEW",
    "assert_unmoved",
    "assert_write_visible",
    "in_process_config",
    "served_active",
    "served_tree",
    "unmoved_band",
]

#: The views every render of this process has named as not written; each is
#: named once.
_REPORTED: set[str] = set()


def served_tree(
    root: Path,
    setpoints: Iterable[str] | Mapping[str, str] = (),
    readings: Iterable[str] = (),
    *,
    channels: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path:
    """Build and render a tree serving the given addresses.

    Args:
        root: An empty directory; the tree is written to ``root/data/facility``
            and rendered into ``root/build``.
        setpoints: The addresses a test writes, each a writable setpoint. A
            mapping pairs each setpoint with its readback address, which joins
            the tree as a readback.
        readings: The addresses a test only reads, each a readback.
        channels: Further channel-record fields by address, in the facility
            schema's own spelling (``value_type``, ``options``, ``unit``,
            ``description``, ``simulation``), merged into that address's
            record.

    Returns:
        The render's simulator view, ``root/build/data/simulator``.

    Raises:
        AssertionError: An address of ``channels`` is not among the setpoints,
            their pairs or the readings; the view does not serve every address
            given; or a setpoint given is not writable in it.
    """
    from osprey.facility.views.outputs import render_facility_outputs

    served = served_facility(root, setpoints, readings, channels=channels)
    render_dir = root / "build"
    render_dir.mkdir()
    render_facility_outputs(
        render_dir,
        served.document,
        IN_PROCESS_RENDER_CONFIG,
        served.facility,
        omitted_reported=_REPORTED,
    )

    view = render_dir / "data" / "simulator"
    check_served(view, served.setpoints, served.readbacks)
    return view
