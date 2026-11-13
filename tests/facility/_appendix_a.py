"""The design's Appendix A facility, built as a tree.

Appendix A of the relations design extends the example facility around sector
3 of its storage ring: four documents, five new places, eleven new devices
(one of class ``Host``, and ``SR/MAG-Q-0301``, the identity claim's other
end), one group and eighteen relation claims. The packaged example ships all
of it; this module reads those records from the example by id (``ADDED_IDS``)
and writes them, plus stubs of the existing records they reuse, as a
self-contained ``data/facility`` tree.

Helpers:

- ``appendix_a_tree(variant)``: the tree as ``{relpath: data}``. ``no_policy``
  has no ``policy.yaml``; ``located_in_precedence``
  adds the design's ``policy.yaml``, which the example ships, ranking ``rack-layout`` over
  ``cable-schedule`` for ``located_in``.
- ``with_readout_control(tree)``: a copy of the tree with the
  ``controlled_by`` claim the Appendix B trace query lacks (the readout
  electronics controlled by the application), so the BPM-to-host path exists.
- ``build_appendix_a(root, variant)``: write the tree, then build it into its
  facility document.
- ``example_without_appendix_a(root)``: a copy of the packaged example with
  every ``ADDED_IDS`` record removed -- the example as it stood before
  Appendix A.

The tree has no ``imported/`` layer; a test that needs one builds it itself.

The reused records (``SR``, ``SR/SECT3``, ``SR/SECT4``, ``SR/QF05``,
``SR/BPM13``, ``SR/VALVE03``, ``SR/VALVE04``, ``SR/ION-PUMP13``) are stubs
with the minimum fields a build needs: the level or class the example gives
them, and the place it gives them (``SR/VALVE04`` is in ``SR/SECT4``). The
example places ``SR/QF05`` and ``SR/BPM13`` by a span over the ``SR`` model,
and the sectors are span places; the stubs carry no model, so the two devices
author ``SR/SECT3`` as a stand-in for the span (``SPAN_STAND_IN``).
``policy.yaml`` carries the loader's header ``schema:
osprey.facility.policy/1`` beside ``relations: {precedence: {located_in:
[rack-layout, cable-schedule]}}``.
"""

from __future__ import annotations

import copy
import shutil
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import yaml

from tests._preset_data import packaged_facility_dir
from tests.facility._synthetic_trees import write_tree

Variant = Literal["no_policy", "located_in_precedence"]

VARIANTS: tuple[Variant, ...] = ("no_policy", "located_in_precedence")

POLICY: dict[str, Any] = {
    "schema": "osprey.facility.policy/1",
    "relations": {"precedence": {"located_in": ["rack-layout", "cable-schedule"]}},
}

READOUT_CONTROL_CLAIM: dict[str, Any] = {
    "id": "bpm13-el-control",
    "type": "controlled_by",
    "subject": {"device": "SR/BPM13-EL"},
    "object": {"device": "CTRL/sr-mag-03"},
}

# The ids Appendix A adds to the example, by record kind, in the example's order.
ADDED_IDS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "documents": ("PS-LIST", "RACK-ROW3", "CS-RING", "NET-PLAN"),
        "places": (
            "SR/SECT3/G2",
            "HALL",
            "HALL/ROW3",
            "HALL/ROW3/RK07",
            "HALL/ROW3/RK08",
        ),
        "devices": (
            "SR/QF05-PS",
            "HALL/PANEL-3A",
            "SR/BPM13-EL",
            "SR/MPS-C3",
            "SR/TIM-EVR3",
            "SR/TIM-EVR3:OUT4",
            "SR/LCW-3",
            "CTRL/sr-mag-03",
            "CTRL/srioc03",
            "CTRL/SW-3A",
            "SR/MAG-Q-0301",
        ),
        "groups": ("SR/VS03",),
        "claims": (
            "evr3-out4",
            "bpm13-el-rack",
            "qf05-ps-rack-a",
            "qf05-ps-rack-b",
            "qf05-power",
            "qf05-ps-mains",
            "w-03-0117",
            "w-03-0240",
            "bpm13-readout",
            "qf05-control",
            "app-host",
            "bpm13-trigger",
            "qf05-ps-interlock",
            "vs03-upstream",
            "vs03-downstream",
            "srioc03-net",
            "qf05-water",
            "qf05-identity",
        ),
    }
)

CLAIM_IDS: tuple[str, ...] = ADDED_IDS["claims"]

# Where each record kind lives in a facility tree.
_KIND_FILES: Mapping[str, str] = MappingProxyType(
    {
        "documents": "documents.yaml",
        "places": "records/places.yaml",
        "devices": "records/devices.yaml",
        "groups": "records/groups.yaml",
        "claims": "records/relations.yaml",
    }
)

# Devices the example places by a span; their stubs author this place instead.
SPAN_STAND_IN: Mapping[str, str] = MappingProxyType({"SR/QF05": "SR/SECT3", "SR/BPM13": "SR/SECT3"})

_STUB_PLACES = [
    {"id": "SR", "level": "machine"},
    {"id": "SR/SECT3", "level": "sector"},
    {"id": "SR/SECT4", "level": "sector"},
]

_STUB_DEVICES = [
    {"id": "SR/QF05", "class": "Quadrupole", "place": SPAN_STAND_IN["SR/QF05"]},
    {"id": "SR/BPM13", "class": "BeamPositionMonitor", "place": SPAN_STAND_IN["SR/BPM13"]},
    {"id": "SR/VALVE03", "class": "Valve", "place": "SR/SECT3"},
    {"id": "SR/VALVE04", "class": "Valve", "place": "SR/SECT4"},
    {"id": "SR/ION-PUMP13", "class": "Pump", "place": "SR/SECT3"},
]

# The reused records' ids, by record kind.
STUB_IDS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "places": tuple(place["id"] for place in _STUB_PLACES),
        "devices": tuple(device["id"] for device in _STUB_DEVICES),
    }
)


def _load_text(text: str) -> list[dict[str, Any]]:
    return yaml.safe_load(text) or []


def _load(path: Path) -> list[dict[str, Any]]:
    return _load_text(path.read_text(encoding="utf-8"))


@cache
def _example_records() -> dict[str, list[dict[str, Any]]]:
    """The ``ADDED_IDS`` records as the packaged example holds them, by kind."""
    example = packaged_facility_dir()
    records: dict[str, list[dict[str, Any]]] = {}
    for kind, ids in ADDED_IDS.items():
        shipped = {record["id"]: record for record in _load(example / _KIND_FILES[kind])}
        missing = [rid for rid in ids if rid not in shipped]
        if missing:
            raise AssertionError(f"the example lacks Appendix A {kind} {missing}")
        records[kind] = [shipped[rid] for rid in ids]
    return records


def appendix_a_tree(variant: Variant = "no_policy") -> dict[str, Any]:
    """Build the Appendix A tree.

    Args:
        variant: ``no_policy`` or ``located_in_precedence``.

    Returns:
        A fresh ``{relpath: data}`` mapping; callers may edit it freely.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown Appendix A variant {variant!r}")
    added = _example_records()
    tree: dict[str, Any] = {
        "documents.yaml": added["documents"],
        "records/places.yaml": [*_STUB_PLACES, *added["places"]],
        "records/devices.yaml": [*_STUB_DEVICES, *added["devices"]],
        "records/groups.yaml": added["groups"],
        "records/relations.yaml": added["claims"],
    }
    if variant == "located_in_precedence":
        tree["policy.yaml"] = POLICY
    return copy.deepcopy(tree)


def with_readout_control(tree: dict[str, Any]) -> dict[str, Any]:
    """Add the ``controlled_by`` claim the Appendix B trace query lacks.

    Args:
        tree: An Appendix A tree.

    Returns:
        A copy of ``tree`` whose relations also hold ``SR/BPM13-EL``
        controlled by ``CTRL/sr-mag-03``.
    """
    extended = copy.deepcopy(tree)
    extended["records/relations.yaml"] = [
        *extended["records/relations.yaml"],
        copy.deepcopy(READOUT_CONTROL_CLAIM),
    ]
    return extended


def build_appendix_a(
    root: Path, variant: Variant = "no_policy", *, tree: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Write the Appendix A tree under ``root`` and build its facility document.

    Args:
        root: The ``data/facility`` directory to create.
        variant: ``no_policy`` or ``located_in_precedence``; ignored when
            ``tree`` is given.
        tree: A tree to build in place of the variant's (for example one
            returned by ``with_readout_control``).

    Returns:
        The built facility document.
    """
    from osprey.facility.build import LATER_STAGES
    from osprey.facility.validate import run_stages

    write_tree(root, appendix_a_tree(variant) if tree is None else tree)
    report = run_stages(root, project_name="p", later=LATER_STAGES)
    if report.failed is not None:
        raise AssertionError([error.format_message() for error in report.errors])
    document = report.validated.document
    assert document is not None
    return document


def _without(text: str, ids: set[str]) -> str:
    """``text`` (a YAML list of records) minus the records whose id is in ``ids``.

    Works on the text, one top-level ``- `` item at a time, so every other
    record and the file's header comment keep their bytes.
    """
    kept: list[str] = []
    drop = False
    for line in text.splitlines(keepends=True):
        if line.startswith("- "):
            item = yaml.safe_load(line)
            drop = isinstance(item, list) and str(item[0].get("id")) in ids
        if not drop:
            kept.append(line)
    return "".join(kept)


def example_without_appendix_a(root: Path) -> Path:
    """Write the packaged example minus every ``ADDED_IDS`` record under ``root``.

    The example as it stood before Appendix A: no documents, no claims, no
    policy, and none of the places, devices or group Appendix A adds. A file
    left with no record (``documents.yaml``, ``records/relations.yaml``) is
    not written.

    Args:
        root: The ``data/facility`` directory to create.

    Returns:
        ``root``, the written facility directory.
    """
    shutil.copytree(packaged_facility_dir(), root)
    (root / "policy.yaml").unlink()
    for kind, ids in ADDED_IDS.items():
        path = root / _KIND_FILES[kind]
        remaining = _without(path.read_text(encoding="utf-8"), set(ids))
        left = _load_text(remaining)
        if {record["id"] for record in left} & set(ids):
            raise AssertionError(f"{path} still holds Appendix A {kind}")
        if left:
            path.write_text(remaining, encoding="utf-8")
        else:
            path.unlink()
    return root
