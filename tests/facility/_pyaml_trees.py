"""A small facility with two measured models, and its pyAML view written in memory.

The tree and its two extensions ship with the application SDK's test support
(:mod:`osprey.apps.sdk.testing`); this module names them where OSPREY's own
tests import them, beside the in-memory builders those tests use.
:func:`write_view` builds a tree in memory and writes its pyAML view.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from osprey.apps.sdk.testing import (
    LINE_GROUPS,
    SR_GROUPS,
    measured_tree,
    with_correctors,
    with_rf,
    write_tree,
)
from osprey.facility.records import FacilityRecords

__all__ = [
    "LINE_GROUPS",
    "SR_GROUPS",
    "built_document",
    "measured_tree",
    "view_inputs",
    "with_correctors",
    "with_rf",
    "write_view",
]


def built_document(root: Path, tree: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    """Write ``tree`` under ``root/data/facility`` and build its facility file in memory."""
    from osprey.facility.build import build_facility

    facility = write_tree(root / "data" / "facility", copy.deepcopy(dict(tree)))
    return build_facility(facility, project_name="demo"), facility


def view_inputs(root: Path, tree: Mapping[str, Any], served: list[str] | None = None) -> Any:
    """The view inputs of a render serving ``served`` (every model when ``None``)."""
    from osprey.facility.served import resolve_served
    from osprey.facility.views import ViewInputs

    document, facility = built_document(root, tree)
    config: dict[str, Any] = {} if served is None else {"simulation": {"models": served}}
    return ViewInputs(
        records=FacilityRecords(document),
        rendered_config=config,
        facility_dir=facility,
        served=resolve_served(config, FacilityRecords(document)),
    )


def write_view(
    root: Path, tree: Mapping[str, Any], served: list[str] | None = None
) -> tuple[Path, list[Path]]:
    """Build ``tree`` and write its pyAML view under ``root/render/data/pyaml``.

    Returns:
        The view's directory and the files written.
    """
    from osprey.facility.views.pyaml import write_pyaml_view

    inputs = view_inputs(root, tree, served)
    directory = root / "render" / "data" / "pyaml"
    return directory, write_pyaml_view(directory, inputs)
