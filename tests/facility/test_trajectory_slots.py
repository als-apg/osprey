"""The declaration of a model's ``trajectory`` measurement surface.

``trajectory_slots`` reads the measurement file's ``trajectory`` block against
the channel facts and answers, in this key order: ``devices``, the device table
of the group ``groups.bpm`` names in ``(s, id)`` order, each device's
turn-by-turn rows, arm command, armed flag, units and tags; the model-wide
``trigger_sample``; the ``injection_request`` and ``injection_busy``
instruments; ``machine``, the RF-on and stored-current readings the
instruments ``rf_status`` and ``beam_current`` resolve to; ``settings``, each
declared setting with its setpoint, readback pair, settle tolerance and timeout,
per device or once for the model by its scope; ``steps``, the ordered chain,
command steps carrying their write, timeout and optional done reading and
service-owned steps carrying neither. A model whose file does not list the
kind has no declaration. The agent facts carry the same answer per model under
``trajectory_slots``, apart from the pyAML view records.

The example facility's ``SR`` declares the chain arm, inject, clean, readout;
its device table names the devices its trajectory response matrix rows read,
its trigger sample is its engine's, its busy flag reads 0, and its transfer
line has no declaration.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from osprey.facility.views import pyaml
from osprey.facility.views.facts import (
    FACTS_KEYS,
    FACTS_SCHEMA,
    facts_document,
    zero_source_facts,
)
from osprey.facility.views.pyaml import (
    TRAJECTORY_RM_FILE,
    trajectory_slots,
)
from tests.facility._pyaml_trees import built_document, measured_tree, with_correctors
from tests.facility._synthetic_trees import model, trajectory_channels, with_trajectory

if TYPE_CHECKING:
    from tests.facility.conftest import BuiltProject

#: Every key of an answer, in order.
KEYS = (
    "devices",
    "trigger_sample",
    "injection_request",
    "injection_busy",
    "machine",
    "settings",
    "steps",
)

#: Every key of a device-table row, in order.
DEVICE_KEYS = ("x_row", "y_row", "sum_row", "arm", "armed", "units", "tags")

#: Every key of a setting, and of a step without ``done``, in order.
SETTING_KEYS = (
    "id",
    "label",
    "description",
    "scope",
    "setting",
    "readback",
    "tolerance",
    "timeout_s",
)
STEP_KEYS = ("id", "label", "description", "command", "default", "position", "timeout_s")


def _tags(stem: str) -> dict[str, list[str]]:
    """The tags of one monitor device whose five channels sit under ``stem``."""
    return {
        "bpm_arm_command": [f"{stem}:ARM"],
        "bpm_armed_status": [f"{stem}:ARMED"],
        "bpm_tbt_sum": [f"{stem}:TBT:SUM"],
        "bpm_tbt_x": [f"{stem}:TBT:X"],
        "bpm_tbt_y": [f"{stem}:TBT:Y"],
    }


def _row(stem: str, units: dict[str, str | None] | None = None) -> dict[str, Any]:
    """The device-table row of a monitor device whose five channels sit under ``stem``."""
    return {
        "x_row": f"{stem}:TBT:X",
        "y_row": f"{stem}:TBT:Y",
        "sum_row": f"{stem}:TBT:SUM",
        "arm": f"{stem}:ARM",
        "armed": f"{stem}:ARMED",
        "units": units or {"x": None, "y": None, "sum": None},
        "tags": _tags(stem),
    }


def _undeclared(name: str) -> dict[str, Any]:
    """The answer for model ``name`` as ``with_trajectory`` leaves it, no block declared."""
    stem = f"{name}:BPM"
    return {
        "devices": {f"{name}/BPM1": _row(stem)},
        "trigger_sample": None,
        "injection_request": f"{name}:INJ:REQ",
        "injection_busy": f"{name}:INJ:BUSY",
        "machine": {"rf_on": None, "stored_current": None},
        "settings": [],
        "steps": [],
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


def _channels(tree: dict[str, Any], *records: dict[str, Any]) -> dict[str, Any]:
    tree["records/channels.yaml"] += list(records)
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


#: The trajectory block :func:`declared_tree` gives SR.
BLOCK: dict[str, Any] = {
    "trigger_sample": 3,
    "settings": [
        {
            "id": "att",
            "label": "Attenuation",
            "description": "Input attenuation.",
            "scope": "device",
            "signal": "attenuation_setpoint",
            "timeout_s": 2.0,
        },
        {
            "id": "mode",
            "label": "Mode",
            "scope": "device",
            "signal": "bpm_mode_setpoint",
            "timeout_s": 1.5,
        },
        {
            "id": "turns",
            "label": "Turns",
            "description": "Turns recorded.",
            "scope": "global",
            "address": "SR:TURNS:SP",
            "timeout_s": 1.0,
        },
    ],
    "steps": [
        {"id": "arm", "label": "Arm", "description": "Arms.", "default": True},
        {"id": "inject", "label": "Inject", "description": "Fires.", "default": True},
        {
            "id": "clean",
            "label": "Clean",
            "description": "Empties.",
            "default": False,
            "command": {
                "address": "SR:CLEAN:REQ",
                "value": 1,
                "timeout_s": 4.0,
                "done": {"address": "SR:CLEAN:DONE", "value": 0},
            },
        },
        {
            "id": "kick",
            "label": "Kick",
            "description": "Kicks.",
            "default": True,
            "command": {"address": "SR:KICK", "value": "On", "timeout_s": 0.5},
        },
        {"id": "readout", "label": "Readout", "description": "Reads.", "default": True},
    ],
}


def declared_tree() -> dict[str, Any]:
    """The periodic tree with SR declaring :data:`BLOCK` and naming both machine instruments.

    SR/BPM1 carries an attenuation setpoint paired and toleranced, and a mode
    setpoint paired with no tolerance; the turns setpoint, the cleaning request
    and its done flag, the kick, the RF-on setpoint and the stored current sit
    on the place SR.
    """
    on = {"device": "SR/BPM1"}
    place = {"place": "SR"}
    tree = _channels(
        periodic_tree(),
        {
            "id": "SR:BPM:ATT:SP",
            "role": "setpoint",
            "pair": "SR:BPM:ATT:RB",
            "tolerance": {"absolute": 0.5},
            "signal": "attenuation_setpoint",
            "unit": "dB",
            "on": on,
        },
        {"id": "SR:BPM:ATT:RB", "signal": "attenuation_readback", "unit": "dB", "on": on},
        {
            "id": "SR:BPM:MODE:SP",
            "role": "setpoint",
            "pair": "SR:BPM:MODE:RB",
            "signal": "bpm_mode_setpoint",
            "on": on,
        },
        {"id": "SR:BPM:MODE:RB", "signal": "bpm_mode_readback", "on": on},
        {"id": "SR:TURNS:SP", "role": "setpoint", "pair": "SR:TURNS:RB", "on": place},
        {"id": "SR:TURNS:RB", "on": place},
        {"id": "SR:CLEAN:REQ", "role": "setpoint", "value_type": "int", "on": place},
        {"id": "SR:CLEAN:DONE", "value_type": "int", "on": place},
        {"id": "SR:KICK", "role": "setpoint", "on": place},
        {"id": "SR:RF:SP", "role": "setpoint", "pair": "SR:RF:RB", "on": place},
        {"id": "SR:RF:RB", "on": place},
        {"id": "SR:DCCT", "unit": "mA", "on": place},
    )
    measurement = tree["measurement/SR.yaml"]
    measurement["instruments"].update({"rf_status": "SR:RF:SP", "beam_current": "SR:DCCT"})
    measurement["trajectory"] = json.loads(json.dumps(BLOCK))
    return tree


# --- the device table and the instruments -------------------------------------------


def test_a_file_declaring_no_block_answers_the_device_table_and_empty_chain(
    tmp_path: Path,
) -> None:
    document, _ = built_document(tmp_path, periodic_tree())
    slots = trajectory_slots(document, "SR")
    assert slots == _undeclared("SR")
    assert tuple(slots) == KEYS
    assert tuple(slots["devices"]["SR/BPM1"]) == DEVICE_KEYS


def test_a_single_pass_model_has_the_same_shape(tmp_path: Path) -> None:
    document, _ = built_document(tmp_path, with_line(measured_tree()))
    assert trajectory_slots(document, "LINE") == _undeclared("LINE")


def test_the_device_table_follows_the_group_in_s_then_id_order(tmp_path: Path) -> None:
    tree = _with_monitors(periodic_tree(), "SR/BPMZ", "SR/BPMA", "SR/BPMU")
    document, _ = built_document(tmp_path, tree)
    _place(document, {"SR/BPMZ": 0.5, "SR/BPM1": 1.5, "SR/BPMA": 1.5, "SR/BPMU": None})
    slots = trajectory_slots(document, "SR")
    assert list(slots["devices"]) == ["SR/BPMZ", "SR/BPM1", "SR/BPMA", "SR/BPMU"]
    stems = ["SR:BPMZ", "SR:BPM", "SR:BPMA", "SR:BPMU"]
    assert list(slots["devices"].values()) == [_row(stem) for stem in stems]


def test_the_units_are_the_row_channels_units(tmp_path: Path) -> None:
    tree = periodic_tree()
    for channel in tree["records/channels.yaml"]:
        if channel["id"] in ("SR:BPM:TBT:X", "SR:BPM:TBT:Y"):
            channel["unit"] = "m"
        if channel["id"] == "SR:BPM:TBT:SUM":
            channel["unit"] = "counts"
    document, _ = built_document(tmp_path, tree)
    row = trajectory_slots(document, "SR")["devices"]["SR/BPM1"]
    assert row["units"] == {"x": "m", "y": "m", "sum": "counts"}


def test_a_device_carrying_a_signal_twice_names_the_first_address_and_tags_both(
    tmp_path: Path,
) -> None:
    document, _ = built_document(tmp_path, periodic_tree())
    document["channels"] += [
        {"id": "SR:BPM:A:X", "signal": "bpm_tbt_x", "on": {"device": "SR/BPM1"}},
        {"id": "SR:BPM:NOTE", "on": {"device": "SR/BPM1"}},
    ]
    row = trajectory_slots(document, "SR")["devices"]["SR/BPM1"]
    assert row["x_row"] == "SR:BPM:A:X"
    assert row["tags"]["bpm_tbt_x"] == ["SR:BPM:A:X", "SR:BPM:TBT:X"]
    assert list(row["tags"]) == sorted(row["tags"])
    assert "SR:BPM:NOTE" not in {a for addresses in row["tags"].values() for a in addresses}


def test_a_device_without_a_signal_has_none_there(tmp_path: Path) -> None:
    document, _ = built_document(tmp_path, _with_monitors(periodic_tree(), "SR/BPM0"))
    document["channels"] = [
        channel for channel in document["channels"] if channel["id"] != "SR:BPM0:ARM"
    ]
    _place(document, {"SR/BPM0": 0.5, "SR/BPM1": 1.5})
    slots = trajectory_slots(document, "SR")
    assert slots["devices"]["SR/BPM0"]["arm"] is None
    assert "bpm_arm_command" not in slots["devices"]["SR/BPM0"]["tags"]
    assert slots["devices"]["SR/BPM0"]["armed"] == "SR:BPM0:ARMED"


def test_a_monitor_device_outside_the_group_is_no_row(tmp_path: Path) -> None:
    document, _ = built_document(tmp_path, _with_monitors(periodic_tree(), "SR/BPM0"))
    (group,) = [group for group in document["groups"] if group["id"] == "SR/BPM"]
    group["members"] = ["SR/BPM1"]
    assert trajectory_slots(document, "SR") == _undeclared("SR")


def test_the_table_ignores_what_the_model_wires(tmp_path: Path) -> None:
    document, _ = built_document(tmp_path, _with_monitors(periodic_tree(), "SR/BPM0"))
    slots = trajectory_slots(document, "SR")
    (sr,) = [record for record in document["models"] if record["name"] == "SR"]
    sr["wiring"] = []
    assert trajectory_slots(document, "SR") == slots


def test_an_instrument_of_another_kind_is_no_key(tmp_path: Path) -> None:
    document, _ = built_document(tmp_path, periodic_tree())
    (sr,) = [record for record in document["models"] if record["name"] == "SR"]
    assert sr["measurement"]["instruments"]["tune"] == "SR:TUNE:X"
    assert "tune" not in trajectory_slots(document, "SR")
    assert "SR:TUNE:X" not in json.dumps(trajectory_slots(document, "SR"))


# --- the declared block -------------------------------------------------------------


def test_the_machine_readings_resolve_a_setpoint_instrument_to_its_pair(tmp_path: Path) -> None:
    document, _ = built_document(tmp_path, declared_tree())
    assert trajectory_slots(document, "SR")["machine"] == {
        "rf_on": "SR:RF:RB",
        "stored_current": "SR:DCCT",
    }


def test_a_machine_instrument_without_a_pair_is_read_as_named(tmp_path: Path) -> None:
    tree = declared_tree()
    for channel in tree["records/channels.yaml"]:
        if channel["id"] == "SR:RF:SP":
            del channel["pair"]
    document, _ = built_document(tmp_path, tree)
    assert trajectory_slots(document, "SR")["machine"]["rf_on"] == "SR:RF:SP"


def test_the_trigger_sample_is_the_declared_one(tmp_path: Path) -> None:
    document, _ = built_document(tmp_path, declared_tree())
    assert trajectory_slots(document, "SR")["trigger_sample"] == 3


def test_each_setting_resolves_its_pair_and_tolerance_by_scope(tmp_path: Path) -> None:
    document, _ = built_document(tmp_path, declared_tree())
    settings = trajectory_slots(document, "SR")["settings"]
    assert settings == [
        {
            "id": "att",
            "label": "Attenuation",
            "description": "Input attenuation.",
            "scope": "device",
            "setting": {"SR/BPM1": "SR:BPM:ATT:SP"},
            "readback": {"SR/BPM1": "SR:BPM:ATT:RB"},
            "tolerance": {"SR/BPM1": {"absolute": 0.5}},
            "timeout_s": 2.0,
        },
        {
            "id": "mode",
            "label": "Mode",
            "description": None,
            "scope": "device",
            "setting": {"SR/BPM1": "SR:BPM:MODE:SP"},
            "readback": {"SR/BPM1": "SR:BPM:MODE:RB"},
            "tolerance": {"SR/BPM1": None},
            "timeout_s": 1.5,
        },
        {
            "id": "turns",
            "label": "Turns",
            "description": "Turns recorded.",
            "scope": "global",
            "setting": "SR:TURNS:SP",
            "readback": "SR:TURNS:RB",
            "tolerance": None,
            "timeout_s": 1.0,
        },
    ]
    assert all(tuple(setting) == SETTING_KEYS for setting in settings)


def test_a_device_setting_spans_the_devices_carrying_its_setpoint_in_order(
    tmp_path: Path,
) -> None:
    tree = _with_monitors(declared_tree(), "SR/BPM0", "SR/BPM2")
    tree["records/channels.yaml"] += [
        {
            "id": "SR:BPM2:ATT:SP",
            "role": "setpoint",
            "pair": "SR:BPM2:ATT:RB",
            "tolerance": {"absolute": 0.25},
            "signal": "attenuation_setpoint",
            "unit": "dB",
            "on": {"device": "SR/BPM2"},
        },
        {
            "id": "SR:BPM2:ATT:RB",
            "signal": "attenuation_readback",
            "unit": "dB",
            "on": {"device": "SR/BPM2"},
        },
        {"id": "SR:BPM0:ATT:RB", "signal": "attenuation_setpoint", "on": {"device": "SR/BPM0"}},
    ]
    document, _ = built_document(tmp_path, tree)
    _place(document, {"SR/BPM0": 0.5, "SR/BPM1": 1.5, "SR/BPM2": 2.5})
    (att, *_) = trajectory_slots(document, "SR")["settings"]
    assert att["setting"] == {"SR/BPM1": "SR:BPM:ATT:SP", "SR/BPM2": "SR:BPM2:ATT:SP"}
    assert list(att["readback"]) == ["SR/BPM1", "SR/BPM2"]
    assert att["tolerance"] == {"SR/BPM1": {"absolute": 0.5}, "SR/BPM2": {"absolute": 0.25}}


def test_the_steps_are_the_chain_in_order_with_command_steps_timed(tmp_path: Path) -> None:
    document, _ = built_document(tmp_path, declared_tree())
    steps = trajectory_slots(document, "SR")["steps"]
    assert steps == [
        {
            "id": "arm",
            "label": "Arm",
            "description": "Arms.",
            "command": None,
            "default": True,
            "position": 0,
            "timeout_s": None,
        },
        {
            "id": "inject",
            "label": "Inject",
            "description": "Fires.",
            "command": None,
            "default": True,
            "position": 1,
            "timeout_s": None,
        },
        {
            "id": "clean",
            "label": "Clean",
            "description": "Empties.",
            "command": {"address": "SR:CLEAN:REQ", "value": 1},
            "default": False,
            "position": 2,
            "timeout_s": 4.0,
            "done": {"address": "SR:CLEAN:DONE", "value": 0},
        },
        {
            "id": "kick",
            "label": "Kick",
            "description": "Kicks.",
            "command": {"address": "SR:KICK", "value": "On"},
            "default": True,
            "position": 3,
            "timeout_s": 0.5,
        },
        {
            "id": "readout",
            "label": "Readout",
            "description": "Reads.",
            "command": None,
            "default": True,
            "position": 4,
            "timeout_s": None,
        },
    ]
    assert [tuple(step) for step in steps] == [
        STEP_KEYS,
        STEP_KEYS,
        (*STEP_KEYS, "done"),
        STEP_KEYS,
        STEP_KEYS,
    ]


def test_the_declared_answer_keeps_the_key_order(tmp_path: Path) -> None:
    document, _ = built_document(tmp_path, declared_tree())
    assert tuple(trajectory_slots(document, "SR")) == KEYS


def test_the_view_names_no_facility_word() -> None:
    source = "".join(
        inspect.getsource(function)
        for function in (pyaml.trajectory_slots, pyaml._setting, pyaml._step, pyaml._reading_of)
    )
    for word in ("SR:", "SR/", "BPM:", "CLEANING", "RF:", "DCCT", "ring"):
        assert word not in source, word


# --- what has no declaration --------------------------------------------------------


def test_a_model_whose_file_does_not_list_the_kind_has_no_declaration(tmp_path: Path) -> None:
    document, _ = built_document(tmp_path, periodic_tree())
    with pytest.raises(KeyError, match="LINE"):
        trajectory_slots(document, "LINE")


def test_a_model_with_no_measurement_file_has_no_declaration(tmp_path: Path) -> None:
    tree = periodic_tree()
    del tree["measurement/LINE.yaml"]
    document, _ = built_document(tmp_path, tree)
    with pytest.raises(KeyError, match="LINE"):
        trajectory_slots(document, "LINE")


def test_an_unknown_model_has_no_declaration(tmp_path: Path) -> None:
    document, _ = built_document(tmp_path, periodic_tree())
    with pytest.raises(KeyError, match="NOPE"):
        trajectory_slots(document, "NOPE")


# --- the facts ----------------------------------------------------------------------


def test_the_facts_carry_the_declaration_of_each_model_with_the_kind(tmp_path: Path) -> None:
    document, _ = built_document(tmp_path, declared_tree())
    facts = facts_document(document, ["LINE", "SR"])
    assert facts["trajectory_slots"] == {"SR": trajectory_slots(document, "SR")}


def test_the_facts_list_the_models_by_name(tmp_path: Path) -> None:
    document, _ = built_document(tmp_path, with_line(periodic_tree()))
    document["models"].reverse()
    facts = facts_document(document, ["LINE", "SR"])
    assert list(facts["trajectory_slots"]) == ["LINE", "SR"]


def test_the_declaration_is_a_key_of_its_own_beside_the_view_records(tmp_path: Path) -> None:
    document, _ = built_document(tmp_path, periodic_tree())
    records = {"SR": {"path": "data/pyaml/SR/configuration.yaml", "sha256": "0", "files": {}}}
    facts = facts_document(document, ["SR"], None, records)
    assert facts["measurement_models"] == records
    assert facts["trajectory_slots"] == {"SR": _undeclared("SR")}


def test_a_facility_with_no_model_of_the_kind_has_an_empty_entry(tmp_path: Path) -> None:
    document, _ = built_document(tmp_path, with_correctors(measured_tree()))
    assert facts_document(document, ["LINE", "SR"])["trajectory_slots"] == {}


def test_the_zero_source_facts_carry_the_key_and_the_schema_number() -> None:
    zero = zero_source_facts({})
    assert zero["trajectory_slots"] == {}
    assert "trajectory_slots" in FACTS_KEYS
    assert FACTS_SCHEMA == "osprey.facility.facility_facts/5"


def test_the_facts_and_the_zero_source_facts_carry_the_same_keys(tmp_path: Path) -> None:
    document, _ = built_document(tmp_path, periodic_tree())
    assert set(facts_document(document, ["SR"])) == set(FACTS_KEYS)


# --- the example facility -----------------------------------------------------------


def _example(built: BuiltProject) -> dict[str, Any]:
    return trajectory_slots(built.facility, "SR")


@pytest.mark.slow
def test_the_example_device_table_is_the_devices_its_response_matrix_rows_read(
    built_control_assistant: BuiltProject,
) -> None:
    facility = built_control_assistant.facility
    device = {channel["id"]: channel["on"].get("device") for channel in facility["channels"]}
    (group,) = [group for group in facility["groups"] if group["id"] == "SR/BPM"]
    slots = _example(built_control_assistant)
    pyaml_dir = built_control_assistant.build_dir / "data/pyaml/SR"
    matrix = json.loads((pyaml_dir / TRAJECTORY_RM_FILE).read_text(encoding="utf-8"))
    rows = matrix["observable_names"]
    assert {device[address] for address in rows} == set(group["members"])
    table = slots["devices"]
    assert set(table) == set(group["members"])
    half = len(rows) // 2
    by_x = {row["x_row"]: name for name, row in table.items()}
    by_y = {row["y_row"]: name for name, row in table.items()}
    assert [by_x[address] for address in rows[:half]] == [by_y[address] for address in rows[half:]]
    for name, row in table.items():
        assert tuple(row) == DEVICE_KEYS
        assert {device[row[slot]] for slot in DEVICE_KEYS[:5]} == {name}
        assert row["units"] == {"x": "m", "y": "m", "sum": None}


@pytest.mark.slow
def test_the_example_trigger_sample_is_its_engines(built_control_assistant: BuiltProject) -> None:
    from osprey.facility.views.simulator import TRIGGER_SAMPLE_KEY, engine_settings

    record = next(
        record for record in built_control_assistant.facility["models"] if record["name"] == "SR"
    )
    assert TRIGGER_SAMPLE_KEY not in record["settings"]["pyat"]["injection"]
    assert (
        _example(built_control_assistant)["trigger_sample"]
        == (engine_settings(record)["pyat"]["injection"][TRIGGER_SAMPLE_KEY])
    )


@pytest.mark.slow
def test_the_example_declares_arm_inject_clean_readout(
    built_control_assistant: BuiltProject,
) -> None:
    slots = _example(built_control_assistant)
    instruments = next(
        record for record in built_control_assistant.facility["models"] if record["name"] == "SR"
    )["measurement"]["instruments"]
    steps = slots["steps"]
    assert [step["id"] for step in steps] == ["arm", "inject", "clean", "readout"]
    assert [step["position"] for step in steps] == [0, 1, 2, 3]
    assert all(step["default"] is True for step in steps)
    assert all(step["label"] and step["description"] for step in steps)
    arm, inject, clean, readout = steps
    for owned in (arm, inject, readout):
        assert owned["command"] is None
        assert owned["timeout_s"] is None
        assert "done" not in owned
    assert clean["command"] == {"address": instruments["bunch_clean_enable"], "value": 1}
    assert clean["timeout_s"] > 0
    assert slots["injection_request"] == instruments["injection_request"]
    assert slots["injection_busy"] == instruments["injection_busy"]


@pytest.mark.slow
def test_the_example_settings_and_machine_resolve_from_the_channel_facts(
    built_control_assistant: BuiltProject,
) -> None:
    facility = built_control_assistant.facility
    channels = {channel["id"]: channel for channel in facility["channels"]}
    slots = _example(built_control_assistant)
    assert slots["machine"] == {
        "rf_on": "SR:RF:ON:RB",
        "stored_current": "SR:DIAG:STORED_BEAM:CURRENT:RB",
    }
    by_id = {setting["id"]: setting for setting in slots["settings"]}
    assert list(by_id) == ["attenuation", "bpm_mode", "pilot_tone", "turns_tracked"]
    for setting in slots["settings"]:
        assert tuple(setting) == SETTING_KEYS
        assert setting["timeout_s"] > 0
    for name in ("attenuation", "bpm_mode", "pilot_tone"):
        setting = by_id[name]
        assert setting["scope"] == "device"
        assert list(setting["setting"]) == list(slots["devices"])
        for device, address in setting["setting"].items():
            assert channels[address]["on"]["device"] == device
            assert setting["readback"][device] == channels[address]["pair"]
            assert setting["tolerance"][device] == channels[address].get("tolerance")
    assert set(map(json.dumps, by_id["attenuation"]["tolerance"].values())) == {
        json.dumps({"absolute": 0.5})
    }
    assert by_id["turns_tracked"]["scope"] == "global"
    assert by_id["turns_tracked"]["setting"] == "SR:DIAG:TBT:TURNS:SP"
    assert by_id["turns_tracked"]["readback"] == "SR:DIAG:TBT:TURNS:RB"


@pytest.mark.slow
def test_the_example_busy_flag_reads_zero(built_control_assistant: BuiltProject) -> None:
    pytest.importorskip("at")
    from osprey_connectors.simulation.composite import Composite

    busy = _example(built_control_assistant)["injection_busy"]
    composite = Composite(built_control_assistant.build_dir / "data/simulator", model_log=False)
    assert composite.get([busy]) == {busy: 0}


@pytest.mark.slow
def test_the_example_line_has_no_declaration(built_control_assistant: BuiltProject) -> None:
    with pytest.raises(KeyError, match="LINE"):
        trajectory_slots(built_control_assistant.facility, "LINE")
