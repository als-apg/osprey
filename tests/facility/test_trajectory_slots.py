"""The channel slots a model's ``trajectory`` measurement stands for.

``trajectory_slots`` answers with ``arm`` and ``armed``, the arm command and
the armed flag of each device of the group ``groups.bpm`` names, one address
per device in the group's ``(s, id)`` order; ``injection_request`` and
``injection_busy``, the two instruments the kind always names; and
``bunch_clean_enable``, ``beam_current`` and ``rf_status`` only when the
measurement file names them; then ``tags``, every signal the group's devices
carry mapped to its addresses in the same device order. A model whose file does
not list the kind has no slots. The agent facts carry the same answer per model under
``trajectory_slots``, apart from the pyAML view records.

The example facility's ``SR`` measures the kind: the devices behind its
``arm`` and ``armed`` slots are the devices its trajectory response matrix
rows read, its busy flag reads 0, and its transfer line has no slots.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from osprey.facility.views.facts import (
    FACTS_KEYS,
    FACTS_SCHEMA,
    facts_document,
    zero_source_facts,
)
from osprey.facility.views.pyaml import (
    INSTRUMENT_CLASSES,
    TRAJECTORY_RM_FILE,
    trajectory_slots,
)
from tests.facility._pyaml_trees import built_document, measured_tree, with_correctors
from tests.facility._synthetic_trees import model, trajectory_channels, with_trajectory

if TYPE_CHECKING:
    from tests.facility.conftest import BuiltProject

#: The slots every answer carries, and the ones it carries only when named.
REQUIRED = ("arm", "armed", "injection_request", "injection_busy")
OPTIONAL = ("bunch_clean_enable", "beam_current", "rf_status")

#: The key of the signal-to-addresses map, after every slot.
TAGS = "tags"


def _tags(*stems: str) -> dict[str, list[str]]:
    """The tags of monitor devices whose five channels sit under ``stems``, in that order."""
    suffixes = {
        "bpm_arm_command": "ARM",
        "bpm_armed_status": "ARMED",
        "bpm_tbt_sum": "TBT:SUM",
        "bpm_tbt_x": "TBT:X",
        "bpm_tbt_y": "TBT:Y",
    }
    return {signal: [f"{stem}:{suffix}" for stem in stems] for signal, suffix in suffixes.items()}


#: The slots of the periodic tree as ``with_trajectory`` leaves it.
SR_SLOTS = {
    "arm": ["SR:BPM:ARM"],
    "armed": ["SR:BPM:ARMED"],
    "injection_request": "SR:INJ:REQ",
    "injection_busy": "SR:INJ:BUSY",
    TAGS: _tags("SR:BPM"),
}


def _wired(tree: dict[str, Any], name: str, stem: str, element: str) -> dict[str, Any]:
    """``tree`` with model ``name`` wiring the five signal channels under ``stem`` on ``element``."""
    model(tree, name)["wiring"] += [
        *(
            {
                "address": f"{stem}:TBT:{axis.upper()}",
                "element": element,
                "engine": {"axis": axis, "series": "turn"},
            }
            for axis in ("x", "y", "sum")
        ),
        {"address": f"{stem}:ARM", "engine": {"attribute": "arm"}},
        {"address": f"{stem}:ARMED", "engine": {"attribute": "armed"}},
    ]
    return tree


def periodic_tree() -> dict[str, Any]:
    """The periodic model SR measuring the kind; its one monitor device is SR/BPM1."""
    tree = with_trajectory(with_correctors(measured_tree()), "SR", "SR/BPM1", marker="M1")
    return _wired(tree, "SR", "SR:BPM", "BPM1")


def with_line(tree: dict[str, Any]) -> dict[str, Any]:
    """``tree`` with the single-pass model LINE measuring the kind on LINE/BPM1."""
    return _wired(with_trajectory(tree, "LINE", "LINE/BPM1"), "LINE", "LINE:BPM", "LBPM")


def _with_instrument(tree: dict[str, Any], slot: str, address: str) -> dict[str, Any]:
    """``tree`` with the channel ``address`` on SR's place, named as instrument ``slot``."""
    tree["records/channels.yaml"].append({"id": address, "on": {"place": "SR"}})
    tree["measurement/SR.yaml"]["instruments"][slot] = address
    return tree


def _with_monitors(tree: dict[str, Any], *devices: str) -> dict[str, Any]:
    """``tree`` with each of ``devices`` a monitor of SR's bpm group, listed before SR/BPM1.

    The deck holds one monitor, so SR wires every device's five channels on it.
    """
    for device in devices:
        stem = device.replace("/", ":")
        tree["records/devices.yaml"].append({"id": device, "class": "BeamPositionMonitor"})
        tree["records/channels.yaml"] += trajectory_channels(device, stem)
        _wired(tree, "SR", stem, "BPM1")
    (group,) = [group for group in tree["records/groups.yaml"] if group["id"] == "SR/BPM"]
    group["members"] = [*devices, *group["members"]]
    return tree


def _place(document: dict[str, Any], positions: dict[str, float | None]) -> None:
    """Give each device of ``positions`` its ``s``; ``None`` leaves it unplaced."""
    for device in document["devices"]:
        if device["id"] in positions:
            device.pop("s", None)
            if positions[device["id"]] is not None:
                device["s"] = positions[device["id"]]


# --- the slots ----------------------------------------------------------------------


def test_the_optional_names_are_the_optional_instrument_class() -> None:
    assert INSTRUMENT_CLASSES["optional"] == OPTIONAL


def test_a_file_naming_no_optional_instrument_has_the_four_required_slots(
    tmp_path: Path,
) -> None:
    document, _ = built_document(tmp_path, periodic_tree())
    slots = trajectory_slots(document, "SR")
    assert slots == SR_SLOTS
    assert tuple(slots) == (*REQUIRED, TAGS)


def test_a_single_pass_model_has_the_same_four_slots(tmp_path: Path) -> None:
    document, _ = built_document(tmp_path, with_line(measured_tree()))
    assert trajectory_slots(document, "LINE") == {
        "arm": ["LINE:BPM:ARM"],
        "armed": ["LINE:BPM:ARMED"],
        "injection_request": "LINE:INJ:REQ",
        "injection_busy": "LINE:INJ:BUSY",
        TAGS: _tags("LINE:BPM"),
    }


@pytest.mark.parametrize("slot", OPTIONAL)
def test_an_optional_slot_is_there_only_when_the_file_names_it(tmp_path: Path, slot: str) -> None:
    tree = _with_instrument(periodic_tree(), slot, "SR:EXTRA")
    document, _ = built_document(tmp_path, tree)
    slots = trajectory_slots(document, "SR")
    assert slots == {**SR_SLOTS, slot: "SR:EXTRA"}
    assert tuple(slots) == (*REQUIRED, slot, TAGS)


def test_every_optional_slot_named_comes_after_the_required_ones_in_one_order(
    tmp_path: Path,
) -> None:
    tree = periodic_tree()
    for slot in reversed(OPTIONAL):
        _with_instrument(tree, slot, f"SR:{slot.upper()}")
    document, _ = built_document(tmp_path, tree)
    slots = trajectory_slots(document, "SR")
    assert tuple(slots) == (*REQUIRED, *OPTIONAL, TAGS)
    assert [slots[slot] for slot in OPTIONAL] == [f"SR:{slot.upper()}" for slot in OPTIONAL]


def test_an_instrument_of_another_kind_is_no_slot(tmp_path: Path) -> None:
    document, _ = built_document(tmp_path, periodic_tree())
    (sr,) = [model for model in document["models"] if model["name"] == "SR"]
    assert sr["measurement"]["instruments"]["tune"] == "SR:TUNE:X"
    assert "tune" not in trajectory_slots(document, "SR")


def test_arm_and_armed_follow_the_group_in_s_then_id_order(tmp_path: Path) -> None:
    tree = _with_monitors(periodic_tree(), "SR/BPMZ", "SR/BPMA", "SR/BPMU")
    document, _ = built_document(tmp_path, tree)
    _place(document, {"SR/BPMZ": 0.5, "SR/BPM1": 1.5, "SR/BPMA": 1.5, "SR/BPMU": None})
    slots = trajectory_slots(document, "SR")
    assert slots["arm"] == ["SR:BPMZ:ARM", "SR:BPM:ARM", "SR:BPMA:ARM", "SR:BPMU:ARM"]
    assert slots["armed"] == [
        "SR:BPMZ:ARMED",
        "SR:BPM:ARMED",
        "SR:BPMA:ARMED",
        "SR:BPMU:ARMED",
    ]
    assert slots[TAGS] == _tags("SR:BPMZ", "SR:BPM", "SR:BPMA", "SR:BPMU")


def test_the_tags_map_holds_every_signal_the_group_devices_carry(tmp_path: Path) -> None:
    document, _ = built_document(tmp_path, _with_monitors(periodic_tree(), "SR/BPM0"))
    document["channels"] += [
        {"id": "SR:BPM0:STATUS", "signal": "bpm_tbt_x", "on": {"device": "SR/BPM0"}},
        {"id": "SR:BPM0:NOTE", "on": {"device": "SR/BPM0"}},
    ]
    _place(document, {"SR/BPM0": 0.5, "SR/BPM1": 1.5})
    tags = trajectory_slots(document, "SR")[TAGS]
    assert list(tags) == sorted(tags)
    assert tags["bpm_tbt_x"] == ["SR:BPM0:STATUS", "SR:BPM0:TBT:X", "SR:BPM:TBT:X"]
    assert "SR:BPM0:NOTE" not in {address for addresses in tags.values() for address in addresses}
    assert tags["bpm_arm_command"] == trajectory_slots(document, "SR")["arm"]


def test_arm_and_armed_hold_one_address_per_device_whatever_the_model_wires(
    tmp_path: Path,
) -> None:
    document, _ = built_document(tmp_path, _with_monitors(periodic_tree(), "SR/BPM0"))
    (sr,) = [model for model in document["models"] if model["name"] == "SR"]
    (group,) = [group for group in document["groups"] if group["id"] == "SR/BPM"]
    slots = trajectory_slots(document, "SR")
    assert len(slots["arm"]) == len(slots["armed"]) == len(group["members"]) == 2
    held = {*slots["arm"], *slots["armed"]}
    assert held <= {entry["address"] for entry in sr["wiring"]}
    sr["wiring"] = [entry for entry in sr["wiring"] if entry["address"] not in held]
    assert trajectory_slots(document, "SR") == slots


def test_a_monitor_device_outside_the_group_gives_no_address(tmp_path: Path) -> None:
    document, _ = built_document(tmp_path, _with_monitors(periodic_tree(), "SR/BPM0"))
    (group,) = [group for group in document["groups"] if group["id"] == "SR/BPM"]
    group["members"] = ["SR/BPM1"]
    assert trajectory_slots(document, "SR") == SR_SLOTS


def test_a_model_whose_file_does_not_list_the_kind_has_no_slots(tmp_path: Path) -> None:
    document, _ = built_document(tmp_path, periodic_tree())
    with pytest.raises(KeyError, match="LINE"):
        trajectory_slots(document, "LINE")


def test_a_model_with_no_measurement_file_has_no_slots(tmp_path: Path) -> None:
    tree = periodic_tree()
    del tree["measurement/LINE.yaml"]
    document, _ = built_document(tmp_path, tree)
    with pytest.raises(KeyError, match="LINE"):
        trajectory_slots(document, "LINE")


def test_an_unknown_model_has_no_slots(tmp_path: Path) -> None:
    document, _ = built_document(tmp_path, periodic_tree())
    with pytest.raises(KeyError, match="NOPE"):
        trajectory_slots(document, "NOPE")


# --- the facts ----------------------------------------------------------------------


def test_the_facts_carry_the_slots_of_each_model_with_the_kind(tmp_path: Path) -> None:
    tree = _with_instrument(periodic_tree(), "beam_current", "SR:DCCT")
    document, _ = built_document(tmp_path, tree)
    facts = facts_document(document, ["LINE", "SR"])
    assert facts["trajectory_slots"] == {"SR": trajectory_slots(document, "SR")}
    assert facts["trajectory_slots"]["SR"] == {**SR_SLOTS, "beam_current": "SR:DCCT"}


def test_the_facts_list_the_models_by_name(tmp_path: Path) -> None:
    document, _ = built_document(tmp_path, with_line(periodic_tree()))
    document["models"].reverse()
    facts = facts_document(document, ["LINE", "SR"])
    assert list(facts["trajectory_slots"]) == ["LINE", "SR"]


def test_the_slots_are_a_key_of_their_own_beside_the_view_records(tmp_path: Path) -> None:
    document, _ = built_document(tmp_path, periodic_tree())
    records = {"SR": {"path": "data/pyaml/SR/configuration.yaml", "sha256": "0", "files": {}}}
    facts = facts_document(document, ["SR"], None, records)
    assert facts["measurement_models"] == records
    assert facts["trajectory_slots"] == {"SR": SR_SLOTS}


def test_a_facility_with_no_model_of_the_kind_has_an_empty_entry(tmp_path: Path) -> None:
    document, _ = built_document(tmp_path, with_correctors(measured_tree()))
    assert facts_document(document, ["LINE", "SR"])["trajectory_slots"] == {}


def test_the_zero_source_facts_carry_the_key_and_the_schema_number() -> None:
    zero = zero_source_facts({})
    assert zero["trajectory_slots"] == {}
    assert "trajectory_slots" in FACTS_KEYS
    assert FACTS_SCHEMA == "osprey.facility.facility_facts/4"


def test_the_facts_and_the_zero_source_facts_carry_the_same_keys(tmp_path: Path) -> None:
    document, _ = built_document(tmp_path, periodic_tree())
    assert set(facts_document(document, ["SR"])) == set(FACTS_KEYS)


# --- the example facility -----------------------------------------------------------


@pytest.mark.slow
def test_the_example_arms_the_devices_its_response_matrix_rows_read(
    built_control_assistant: BuiltProject,
) -> None:
    facility = built_control_assistant.facility
    device = {channel["id"]: channel["on"].get("device") for channel in facility["channels"]}
    (group,) = [group for group in facility["groups"] if group["id"] == "SR/BPM"]
    slots = trajectory_slots(facility, "SR")
    matrix = json.loads(
        (built_control_assistant.build_dir / "data/pyaml/SR" / TRAJECTORY_RM_FILE).read_text(
            encoding="utf-8"
        )
    )
    rows = {device[address] for address in matrix["observable_names"]}
    assert rows == set(group["members"])
    assert len(slots["arm"]) == len(slots["armed"]) == len(rows)
    assert {device[address] for address in slots["arm"]} == rows
    assert {device[address] for address in slots["armed"]} == rows
    assert tuple(slots) == (*REQUIRED, TAGS)
    assert slots[TAGS]["bpm_arm_command"] == slots["arm"]
    assert slots[TAGS]["bpm_armed_status"] == slots["armed"]


@pytest.mark.slow
def test_the_example_busy_flag_reads_zero(built_control_assistant: BuiltProject) -> None:
    pytest.importorskip("at")
    from osprey_connectors.simulation.composite import Composite

    busy = trajectory_slots(built_control_assistant.facility, "SR")["injection_busy"]
    composite = Composite(built_control_assistant.build_dir / "data/simulator", model_log=False)
    assert composite.get([busy]) == {busy: 0}


@pytest.mark.slow
def test_the_example_line_has_no_slots(built_control_assistant: BuiltProject) -> None:
    with pytest.raises(KeyError, match="LINE"):
        trajectory_slots(built_control_assistant.facility, "LINE")
