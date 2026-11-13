"""The design's Appendix A facility, built as a tree.

Appendix A of the relations design extends an example facility around sector
3 of its storage ring: four documents, five new places, ten new devices (one
of class ``Host``), a device in an ``imported/controls`` layer, one group and
eighteen relation claims. This module writes that fragment, plus the existing
records it reuses, as a self-contained ``data/facility`` tree.

Helpers:

- ``appendix_a_tree(variant)``: the tree as ``{relpath: data}``. ``no_policy``
  has no ``policy.yaml``; ``located_in_precedence`` adds the design's
  ``policy.yaml`` ranking ``rack-layout`` over ``cable-schedule`` for
  ``located_in``.
- ``with_readout_control(tree)``: a copy of the tree with the
  ``controlled_by`` claim the Appendix B trace query lacks (the readout
  electronics controlled by the application), so the BPM-to-host path exists.
- ``build_appendix_a(root, variant)``: write the tree, then build it into its
  facility document.

Corrections to Appendix A, so the fragment builds under the real loader:

- The reused records (``SR``, ``SR/SECT3``, ``SR/QF05``, ``SR/BPM13``,
  ``SR/VALVE03``, ``SR/VALVE04``, ``SR/ION-PUMP13``) come from the example
  facility, which this tree does not include; they are declared here with the
  minimum fields a build needs (classes, levels and places as the example
  facility gives them: ``SR/BPM13`` is a ``BeamPositionMonitor`` and
  ``SR/ION-PUMP13`` a ``Pump``).
- ``policy.yaml`` carries the loader's header ``schema:
  osprey.facility.policy/1`` beside ``relations: {precedence: {located_in:
  [rack-layout, cable-schedule]}}``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal

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

_DOCUMENTS = [
    {
        "id": "PS-LIST",
        "title": "Magnet power supply list",
        "kind": "equipment-list",
        "system": "document-store",
    },
    {
        "id": "RACK-ROW3",
        "title": "Equipment hall row 3 rack layout",
        "kind": "rack-layout",
        "system": "document-store",
    },
    {
        "id": "CS-RING",
        "title": "Storage ring cable schedule",
        "kind": "cable-schedule",
        "system": "document-store",
    },
    {
        "id": "NET-PLAN",
        "title": "Controls network plan",
        "kind": "network-plan",
        "system": "document-store",
    },
]

_PLACES = [
    # Reused from the example facility.
    {"id": "SR", "level": "machine"},
    {"id": "SR/SECT3", "level": "sector"},
    # Added by Appendix A.
    {"id": "SR/SECT3/G2", "level": "girder", "description": "Sector 3, girder 2"},
    {"id": "HALL", "level": "building", "description": "Equipment hall"},
    {"id": "HALL/ROW3", "level": "row"},
    {"id": "HALL/ROW3/RK07", "level": "rack"},
    {"id": "HALL/ROW3/RK08", "level": "rack"},
]

_DEVICES = [
    # Reused from the example facility.
    {"id": "SR/QF05", "class": "Quadrupole", "place": "SR/SECT3"},
    {"id": "SR/BPM13", "class": "BeamPositionMonitor", "place": "SR/SECT3"},
    {"id": "SR/VALVE03", "class": "Valve", "place": "SR/SECT3"},
    {"id": "SR/VALVE04", "class": "Valve", "place": "SR/SECT3"},
    {"id": "SR/ION-PUMP13", "class": "Pump", "place": "SR/SECT3"},
    # Added by Appendix A.
    {
        "id": "SR/QF05-PS",
        "class": "PowerSupply",
        "scheme_names": [
            {"name": "PS-Q-0301", "scheme": "cable-schedule"},
            {
                "name": "QF05 supply",
                "scheme": "drawings",
                "basis": "inferred",
                "rule": "label-match",
            },
        ],
    },
    {"id": "HALL/PANEL-3A", "class": "Circuit"},
    {"id": "SR/BPM13-EL", "class": "ElectronicsChassis"},
    {
        "id": "SR/MPS-C3",
        "class": "ElectronicsChassis",
        "description": "Interlock chassis, sector 3",
    },
    {
        "id": "SR/TIM-EVR3",
        "class": "ElectronicsChassis",
        "description": "Event receiver, sector 3",
    },
    {"id": "SR/TIM-EVR3:OUT4", "class": "TimingChannel"},
    {"id": "SR/LCW-3", "class": "UtilityLoop"},
    {
        "id": "CTRL/sr-mag-03",
        "class": "ControlApplication",
        "attributes": {"repository": "sr-magnets"},
    },
    {"id": "CTRL/srioc03", "class": "Host"},
    {"id": "CTRL/SW-3A", "class": "NetworkSwitch"},
]

_IMPORTED_DEVICES = [
    {
        "id": "SR/MAG-Q-0301",
        "class": "Quadrupole",
        "scheme_names": [{"name": "MAG-Q-0301", "scheme": "controls"}],
    },
]

_GROUPS = [
    {
        "id": "SR/VS03",
        "description": "Vacuum section 3",
        "members": ["SR/ION-PUMP13", "SR/BPM13", "SR/QF05"],
    },
]


def _source(document: str, version: str, state: str, locator: str) -> dict[str, Any]:
    return {"document": document, "version": version, "state": state, "locator": locator}


_RELATIONS = [
    {
        "id": "evr3-out4",
        "type": "part_of",
        "subject": {"device": "SR/TIM-EVR3:OUT4"},
        "object": {"device": "SR/TIM-EVR3"},
    },
    {
        "id": "bpm13-el-rack",
        "type": "located_in",
        "subject": {"device": "SR/BPM13-EL"},
        "object": {"place": "HALL/ROW3/RK08", "unit": 22},
        "source": _source("RACK-ROW3", "B", "released", "sheet 2"),
    },
    {
        "id": "qf05-ps-rack-a",
        "type": "located_in",
        "subject": {"device": "SR/QF05-PS"},
        "object": {"place": "HALL/ROW3/RK07", "unit": 12},
        "source": _source("RACK-ROW3", "B", "released", "sheet 1"),
    },
    {
        "id": "qf05-ps-rack-b",
        "type": "located_in",
        "subject": {"device": "SR/QF05-PS"},
        "object": {"place": "HALL/ROW3/RK08"},
        "source": _source("CS-RING", "D", "in-work", "row 212"),
    },
    {
        "id": "qf05-power",
        "type": "powered_by",
        "subject": {"device": "SR/QF05"},
        "object": {"device": "SR/QF05-PS", "port": "OUT1"},
        "source": _source("PS-LIST", "C", "released", "row 41"),
    },
    {
        "id": "qf05-ps-mains",
        "type": "powered_by",
        "subject": {"device": "SR/QF05-PS"},
        "object": {"device": "HALL/PANEL-3A", "port": "breaker 14"},
        "source": _source("PS-LIST", "C", "released", "row 41"),
    },
    {
        "id": "w-03-0117",
        "type": "cabled_to",
        "subject": {"device": "SR/QF05", "port": "T1"},
        "object": {"device": "SR/QF05-PS", "port": "OUT1"},
        "attributes": {"cable": "W-03-0117", "routed_from": "object", "length_m": 42},
        "source": _source("CS-RING", "D", "in-work", "row 117"),
    },
    {
        "id": "w-03-0240",
        "type": "cabled_to",
        "subject": {"device": "SR/BPM13", "port": "A"},
        "object": {"place": "HALL/ROW3/RK08"},
        "attributes": {"cable": "W-03-0240", "signal": "button A"},
        "source": _source("CS-RING", "D", "in-work", "row 240"),
    },
    {
        "id": "bpm13-readout",
        "type": "read_out_by",
        "subject": {"device": "SR/BPM13"},
        "object": {"device": "SR/BPM13-EL", "port": "CH1"},
        "basis": "inferred",
        "rule": "bpm-electronics-in-cable-destination-rack",
        "source": _source("CS-RING", "D", "in-work", "row 240"),
    },
    {
        "id": "qf05-control",
        "type": "controlled_by",
        "subject": {"device": "SR/QF05-PS"},
        "object": {"device": "CTRL/sr-mag-03"},
    },
    {
        "id": "app-host",
        "type": "runs_on",
        "subject": {"device": "CTRL/sr-mag-03"},
        "object": {"device": "CTRL/srioc03"},
    },
    {
        "id": "bpm13-trigger",
        "type": "triggered_by",
        "subject": {"device": "SR/BPM13-EL", "port": "TRIG"},
        "object": {"device": "SR/TIM-EVR3:OUT4"},
    },
    {
        "id": "qf05-ps-interlock",
        "type": "interlocked_by",
        "subject": {"device": "SR/QF05-PS"},
        "object": {"device": "SR/MPS-C3", "port": "IN7"},
        "attributes": {"signal": "magnet over-temperature"},
    },
    {
        "id": "vs03-upstream",
        "type": "bounded_by",
        "subject": {"group": "SR/VS03"},
        "object": {"device": "SR/VALVE03"},
        "attributes": {"side": "upstream"},
    },
    {
        "id": "vs03-downstream",
        "type": "bounded_by",
        "subject": {"group": "SR/VS03"},
        "object": {"device": "SR/VALVE04"},
        "attributes": {"side": "downstream"},
    },
    {
        "id": "srioc03-net",
        "type": "networked_to",
        "subject": {"device": "CTRL/srioc03", "port": "eth0"},
        "object": {"device": "CTRL/SW-3A", "port": "12"},
        "source": _source("NET-PLAN", "A", "released", "table 3"),
    },
    {
        "id": "qf05-water",
        "type": "supplied_by",
        "subject": {"device": "SR/QF05"},
        "object": {"device": "SR/LCW-3"},
        "attributes": {"medium": "cooling-water"},
    },
    {
        "id": "qf05-identity",
        "type": "same_as",
        "subject": {"device": "SR/QF05"},
        "object": {"device": "SR/MAG-Q-0301"},
        "basis": "inferred",
        "rule": "family-and-ordinal-match",
    },
]

CLAIM_IDS: tuple[str, ...] = tuple(claim["id"] for claim in _RELATIONS)


def appendix_a_tree(variant: Variant = "no_policy") -> dict[str, Any]:
    """Build the Appendix A tree.

    Args:
        variant: ``no_policy`` or ``located_in_precedence``.

    Returns:
        A fresh ``{relpath: data}`` mapping; callers may edit it freely.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown Appendix A variant {variant!r}")
    tree: dict[str, Any] = {
        "documents.yaml": _DOCUMENTS,
        "records/places.yaml": _PLACES,
        "records/devices.yaml": _DEVICES,
        "records/groups.yaml": _GROUPS,
        "records/relations.yaml": _RELATIONS,
        "imported/controls/devices.yaml": _IMPORTED_DEVICES,
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
