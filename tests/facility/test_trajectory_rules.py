"""The rules a model's measurement file meets once it allows the ``trajectory`` kind.

Every device of the group ``groups.bpm`` names carries a channel of each of the
three turn-by-turn signals and of the arm command and the armed flag, and no
other device of the model carries a turn-by-turn signal. The file states
``corrector_delta`` beside the kind's own keys. Each instrument is held to its
requirement class: the injection request is named and wired, the busy flag is
named, and the optional ones may be left out. A ``single_pass`` model allows
``orm`` and ``trajectory`` and no other kind; a periodic model states
``settings.pyat.injection`` and a ``single_pass`` model may not.

The kind writes commands to two sorts of channel: each channel of the arm
signal on a device of the model, and the channel the injection request names.
Such a channel is a setpoint of value type int and is no corrector of the
kind, and a limits record naming one states both bounds and ``confirm: false``
and no ``max_step``.

The signals are tied to what the model's engine wires. On a device of the
group, each channel of a turn-by-turn signal or of the armed flag is wired as a
turn series of the signal's plane, and the channel of the arm command as a
command; the injection request is wired as a command too. A record the engine
describes as a turn series or marks a command sits on a channel of a matching
signal.

The file's ``trajectory`` block is held to its schema: a trigger sample of at
least 0, every setting a scope of ``device`` or ``global`` and a timeout, a
chain of steps, and a timeout on every command step and on no other.

Each rule has one stop line, pinned here in full, and a tree that builds.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from osprey.facility.build import LATER_STAGES
from osprey.facility.errors import FacilityBuildError
from osprey.facility.validate import run_stages
from tests.facility._pyaml_trees import built_document, measured_tree, with_correctors
from tests.facility._synthetic_trees import (
    TRAJECTORY_KEYS,
    TURN_SAMPLES,
    model,
    with_trajectory,
    write_tree,
)

#: The signals of a monitor's turn-by-turn readings, and every signal the kind
#: asks of a monitor device.
TURN_SIGNALS = ("bpm_tbt_x", "bpm_tbt_y", "bpm_tbt_sum")
MONITOR_SIGNALS = (*TURN_SIGNALS, "bpm_arm_command", "bpm_armed_status")


#: The plane of the turn series each signal's channel is wired as, and the
#: words a stop line says of that wiring.
SERIES_PLANES = {
    "bpm_tbt_x": "x",
    "bpm_tbt_y": "y",
    "bpm_tbt_sum": None,
    "bpm_armed_status": None,
}
SERIES_WORDS = {
    signal: f"`turn_series` of {f'plane {plane}' if plane else 'no plane'}"
    for signal, plane in SERIES_PLANES.items()
}


def _series(address: str, element: str, axis: str) -> dict[str, Any]:
    """A wiring record reading one turn series of ``element``."""
    return {"address": address, "element": element, "engine": {"axis": axis, "series": "turn"}}


def _monitor_wiring(stem: str, element: str) -> list[dict[str, Any]]:
    """The wiring records of the five signal channels under ``stem``."""
    return [
        _series(f"{stem}:TBT:X", element, "x"),
        _series(f"{stem}:TBT:Y", element, "y"),
        _series(f"{stem}:TBT:SUM", element, "sum"),
        {"address": f"{stem}:ARM", "engine": {"attribute": "arm"}},
        {"address": f"{stem}:ARMED", "engine": {"attribute": "armed"}},
    ]


def _wired(tree: dict[str, Any], name: str, element: str) -> dict[str, Any]:
    """``tree`` with model ``name`` wiring its monitor's five signal channels on ``element``."""
    model(tree, name)["wiring"] += _monitor_wiring(f"{name}:BPM", element)
    return tree


def _unwired(tree: dict[str, Any], *addresses: str) -> dict[str, Any]:
    """``tree`` with no wiring record of SR on any of ``addresses``."""
    record = model(tree, "SR")
    record["wiring"] = [entry for entry in record["wiring"] if entry["address"] not in addresses]
    return tree


def periodic_tree() -> dict[str, Any]:
    """The periodic model SR allowing the kind; its one monitor device is SR/BPM1."""
    tree = with_trajectory(with_correctors(measured_tree()), "SR", "SR/BPM1", marker="M1")
    return _wired(tree, "SR", "BPM1")


def line_tree() -> dict[str, Any]:
    """The single-pass model LINE allowing the kind; its one monitor device is LINE/BPM1."""
    return _wired(with_trajectory(measured_tree(), "LINE", "LINE/BPM1"), "LINE", "LBPM")


def stop_line(tmp_path: Path, tree: dict[str, Any]) -> str:
    """The one line the build of ``tree`` stops with."""
    with pytest.raises(FacilityBuildError) as stopped:
        built_document(tmp_path, tree)
    return str(stopped.value.format_message())


def stop_lines(tmp_path: Path, tree: dict[str, Any]) -> list[str]:
    """Every line of the stage the build of ``tree`` stops in."""
    facility = write_tree(tmp_path / "data" / "facility", copy.deepcopy(tree))
    report = run_stages(facility, project_name="demo", later=LATER_STAGES)
    return [str(error.format_message()) for error in report.errors]


def measurement(document: dict[str, Any], name: str) -> dict[str, Any]:
    """The built measurement block of model ``name``."""
    (record,) = [record for record in document["models"] if record["name"] == name]
    block: dict[str, Any] = record["measurement"]
    return block


def _without_signal(tree: dict[str, Any], signal: str) -> dict[str, Any]:
    """``tree`` without the channels of ``signal`` and the wiring records on them."""
    gone = [c["id"] for c in tree["records/channels.yaml"] if c.get("signal") == signal]
    tree["records/channels.yaml"] = [
        channel for channel in tree["records/channels.yaml"] if channel["id"] not in gone
    ]
    return _unwired(tree, *gone)


def _signal_channel(address: str, signal: str, device: str) -> dict[str, Any]:
    """A channel of ``signal`` on ``device``, shaped as the signal's own kind of value."""
    if signal in TURN_SIGNALS:
        value = {"value_type": "waveform", "shape": [TURN_SAMPLES]}
    elif signal == "bpm_arm_command":
        value = {"value_type": "int", "role": "setpoint"}
    else:
        value = {"value_type": "int"}
    return {"id": address, "signal": signal, **value, "on": {"device": device}}


#: The two command channels of the periodic tree, each with the words a stop
#: line says of it.
COMMAND_CHANNELS = {
    "SR:BPM:ARM": "which carries signal `bpm_arm_command`",
    "SR:INJ:REQ": "which `instruments.injection_request` names",
}

#: A limits record a command channel builds with, less its address.
COMMAND_LIMITS = {"min_value": 0, "max_value": 1, "confirm": False}


def _channel(tree: dict[str, Any], address: str) -> dict[str, Any]:
    """The channel record ``address`` of a tree, for the caller to edit."""
    (record,) = [c for c in tree["records/channels.yaml"] if c["id"] == address]
    channel: dict[str, Any] = record
    return channel


def _limited(address: str, **changes: Any) -> dict[str, Any]:
    """The periodic tree with one limits record on ``address``.

    The record is :data:`COMMAND_LIMITS` with ``changes`` laid over it; a
    change of ``None`` leaves the slot out.
    """
    record = {"address": address, **COMMAND_LIMITS, **changes}
    tree = periodic_tree()
    tree["limits.yaml"] = {
        "records": [{slot: value for slot, value in record.items() if value is not None}]
    }
    return tree


def _request_on_a_corrector(tree: dict[str, Any]) -> dict[str, Any]:
    """``tree`` with the horizontal corrector's setpoint named as the injection request."""
    tree["measurement/SR.yaml"]["instruments"]["injection_request"] = "SCOR:H:SP"
    return tree


def _arm_on_a_corrector(tree: dict[str, Any]) -> dict[str, Any]:
    """``tree`` with the horizontal corrector's setpoint carrying the arm signal."""
    _channel(tree, "SCOR:H:SP")["signal"] = "bpm_arm_command"
    return tree


# --- what builds --------------------------------------------------------------------


def test_a_periodic_model_with_every_signal_and_a_launch_builds(tmp_path: Path) -> None:
    document, _ = built_document(tmp_path, periodic_tree())
    block = measurement(document, "SR")
    assert block["kinds"] == ["trajectory"]
    assert {key: block[key] for key in TRAJECTORY_KEYS} == TRAJECTORY_KEYS


def test_a_single_pass_model_builds_the_kind_beside_orm(tmp_path: Path) -> None:
    tree = line_tree()
    tree["measurement/LINE.yaml"]["kinds"] = ["orm", "trajectory"]
    document, _ = built_document(tmp_path, tree)
    assert measurement(document, "LINE")["kinds"] == ["orm", "trajectory"]


def test_a_busy_flag_the_model_does_not_wire_builds(tmp_path: Path) -> None:
    """The flag only has to be a channel of the facility; the tree's model wires none."""
    tree = periodic_tree()
    assert "SR:INJ:BUSY" not in {entry["address"] for entry in model(tree, "SR")["wiring"]}
    document, _ = built_document(tmp_path, tree)
    assert measurement(document, "SR")["instruments"]["injection_busy"] == "SR:INJ:BUSY"


@pytest.mark.parametrize("slot", ["bunch_clean_enable", "beam_current", "rf_status"])
def test_an_optional_instrument_builds_named_or_left_out(tmp_path: Path, slot: str) -> None:
    left_out, _ = built_document(tmp_path / "left_out", periodic_tree())
    assert slot not in measurement(left_out, "SR")["instruments"]

    tree = periodic_tree()
    tree["records/channels.yaml"].append({"id": "SR:AUX", "on": {"place": "SR"}})
    tree["measurement/SR.yaml"]["instruments"][slot] = "SR:AUX"
    named, _ = built_document(tmp_path / "named", tree)
    assert measurement(named, "SR")["instruments"][slot] == "SR:AUX"


@pytest.mark.parametrize("signal", ["bpm_arm_command", "bpm_armed_status"])
def test_an_arm_signal_on_a_device_outside_the_group_builds(tmp_path: Path, signal: str) -> None:
    """Only the turn-by-turn signals are kept to the group's devices."""
    tree = periodic_tree()
    tree["records/channels.yaml"].append(_signal_channel("QD:AUX", signal, "SR/QD"))
    document, _ = built_document(tmp_path, tree)
    assert measurement(document, "SR")["kinds"] == ["trajectory"]


def test_a_turn_signal_on_another_models_device_builds(tmp_path: Path) -> None:
    """A model's rule reads its own devices, so another model's monitors are not its business."""
    tree = periodic_tree()
    tree["records/channels.yaml"].append(_signal_channel("LBPM:TBT:X", "bpm_tbt_x", "LINE/BPM1"))
    document, _ = built_document(tmp_path, tree)
    assert measurement(document, "SR")["kinds"] == ["trajectory"]


@pytest.mark.parametrize("address", COMMAND_CHANNELS)
def test_a_command_channel_limited_by_both_bounds_and_no_confirm_builds(
    tmp_path: Path, address: str
) -> None:
    document, _ = built_document(tmp_path, _limited(address))
    assert document["limits"]["records"] == [{"address": address, **COMMAND_LIMITS}]


# --- what stops ---------------------------------------------------------------------


@pytest.mark.parametrize("signal", MONITOR_SIGNALS)
def test_a_group_device_without_one_of_the_five_signals_stops(tmp_path: Path, signal: str) -> None:
    assert stop_line(tmp_path, _without_signal(periodic_tree(), signal)) == (
        f"facility: reference-missing: measurement SR — kind trajectory needs a channel of "
        f"signal `{signal}` on every device of `groups.bpm`; device SR/BPM1 of group SR/BPM "
        f"carries none; fix: add a channel of signal `{signal}` on device SR/BPM1, or remove "
        "trajectory from `kinds`"
    )


@pytest.mark.parametrize("signal", TURN_SIGNALS)
def test_a_turn_signal_on_a_device_outside_the_group_stops(tmp_path: Path, signal: str) -> None:
    tree = periodic_tree()
    tree["records/channels.yaml"].append(_signal_channel("QD:TBT", signal, "SR/QD"))
    assert stop_line(tmp_path, tree) == (
        f"facility: value-invalid: measurement SR — channel QD:TBT carries signal `{signal}` on "
        "device SR/QD, which is not in group SR/BPM, the `groups.bpm` of kind trajectory; fix: "
        f"add device SR/QD to group SR/BPM, or remove signal `{signal}` from channel QD:TBT"
    )


def test_a_file_without_corrector_delta_stops(tmp_path: Path) -> None:
    tree = periodic_tree()
    del tree["measurement/SR.yaml"]["corrector_delta"]
    assert stop_line(tmp_path, tree) == (
        "facility: value-invalid: measurement SR — kind trajectory needs `corrector_delta`, "
        "which the file does not state; fix: state `corrector_delta`, or remove trajectory "
        "from `kinds`"
    )


@pytest.mark.parametrize("slot", ["injection_request", "injection_busy"])
def test_a_required_instrument_the_file_does_not_name_stops(tmp_path: Path, slot: str) -> None:
    tree = periodic_tree()
    del tree["measurement/SR.yaml"]["instruments"][slot]
    assert stop_line(tmp_path, tree) == (
        f"facility: reference-missing: measurement SR — kind trajectory needs "
        f"`instruments.{slot}`, which the file does not name; fix: name `instruments.{slot}`, "
        "or remove trajectory from `kinds`"
    )


def test_an_injection_request_the_model_does_not_wire_stops(tmp_path: Path) -> None:
    tree = periodic_tree()
    record = model(tree, "SR")
    record["wiring"] = [entry for entry in record["wiring"] if entry["address"] != "SR:INJ:REQ"]
    assert stop_line(tmp_path, tree) == (
        "facility: reference-missing: measurement SR — kind trajectory needs "
        "`instruments.injection_request`; model SR does not wire SR:INJ:REQ; fix: wire "
        "SR:INJ:REQ in model SR, or name a channel it wires as `instruments.injection_request`"
    )


def test_a_single_pass_model_allows_no_third_kind(tmp_path: Path) -> None:
    tree = line_tree()
    tree["measurement/LINE.yaml"] |= {"kinds": ["orm", "trajectory", "trm"], "quad_delta": 0.001}
    line = stop_line(tmp_path, tree)
    assert line.startswith(
        "facility: reference-missing: measurement LINE — kind trm needs a periodic solve; "
        "model LINE solves single_pass; fix: remove trm from `kinds`"
    )


def test_a_periodic_model_without_a_launch_stops(tmp_path: Path) -> None:
    """The model wires no shot record here; its engine refuses one without a launch."""
    tree = with_trajectory(with_correctors(measured_tree()), "SR", "SR/BPM1", marker="M1")
    del model(tree, "SR")["settings"]["pyat"]["injection"]
    assert stop_line(tmp_path, tree) == (
        "facility: value-invalid: measurement SR — kind trajectory needs "
        "`settings.pyat.injection`, which model SR does not state; fix: state "
        "`settings.pyat.injection` in model SR, or remove trajectory from `kinds`"
    )


def test_a_single_pass_model_stating_a_launch_stops(tmp_path: Path) -> None:
    tree = with_trajectory(measured_tree(), "LINE", "LINE/BPM1")
    model(tree, "LINE")["settings"]["pyat"]["injection"] = {"marker": "M0"}
    assert stop_line(tmp_path, tree) == (
        "facility: engine-invalid: model LINE — settings key pyat.injection is set on a "
        "single_pass model; fix: remove injection or set solve to periodic"
    )


# --- command channels ---------------------------------------------------------------


def test_an_arm_command_that_is_no_setpoint_stops(tmp_path: Path) -> None:
    """The model leaves the channel unwired; a command record is a write."""
    tree = _unwired(periodic_tree(), "SR:BPM:ARM")
    del _channel(tree, "SR:BPM:ARM")["role"]
    assert stop_line(tmp_path, tree) == (
        "facility: value-invalid: measurement SR — kind trajectory writes a command to channel "
        "SR:BPM:ARM, which carries signal `bpm_arm_command`; it is a readback channel; fix: set "
        "`role: setpoint` on channel SR:BPM:ARM"
    )


def test_an_injection_request_that_is_no_setpoint_stops(tmp_path: Path) -> None:
    """The tune reading is a channel the model wires, and no setpoint."""
    tree = periodic_tree()
    tree["measurement/SR.yaml"]["instruments"]["injection_request"] = "SR:TUNE:X"
    assert stop_line(tmp_path, tree) == (
        "facility: value-invalid: measurement SR — kind trajectory writes a command to channel "
        "SR:TUNE:X, which `instruments.injection_request` names; it is a readback channel; fix: "
        "set `role: setpoint` on channel SR:TUNE:X"
    )


@pytest.mark.parametrize(
    ("address", "value_type"),
    [("SR:BPM:ARM", "float"), ("SR:BPM:ARM", "bool"), ("SR:INJ:REQ", "float")],
)
def test_a_command_channel_of_another_value_type_stops(
    tmp_path: Path, address: str, value_type: str
) -> None:
    named = COMMAND_CHANNELS[address]
    tree = _unwired(periodic_tree(), "SR:BPM:ARM") if value_type == "bool" else periodic_tree()
    _channel(tree, address)["value_type"] = value_type
    assert stop_line(tmp_path, tree) == (
        f"facility: value-invalid: measurement SR — kind trajectory writes a command to channel "
        f"{address}, {named}; its value type is {value_type}; fix: set `value_type: int` on "
        f"channel {address}"
    )


def test_a_command_channel_stating_no_value_type_stops(tmp_path: Path) -> None:
    """A channel that states no value type is a float."""
    tree = periodic_tree()
    del _channel(tree, "SR:BPM:ARM")["value_type"]
    assert stop_line(tmp_path, tree) == (
        "facility: value-invalid: measurement SR — kind trajectory writes a command to channel "
        "SR:BPM:ARM, which carries signal `bpm_arm_command`; its value type is float; fix: set "
        "`value_type: int` on channel SR:BPM:ARM"
    )


def test_an_arm_command_outside_the_group_is_held_to_the_same_shape(tmp_path: Path) -> None:
    """Every channel of the arm signal on a device of the model is a command channel."""
    tree = periodic_tree()
    channel = _signal_channel("QD:AUX", "bpm_arm_command", "SR/QD")
    del channel["role"]
    tree["records/channels.yaml"].append(channel)
    assert stop_line(tmp_path, tree) == (
        "facility: value-invalid: measurement SR — kind trajectory writes a command to channel "
        "QD:AUX, which carries signal `bpm_arm_command`; it is a readback channel; fix: set "
        "`role: setpoint` on channel QD:AUX"
    )


def test_an_injection_request_that_is_a_corrector_stops(tmp_path: Path) -> None:
    assert stop_line(tmp_path, _request_on_a_corrector(periodic_tree())) == (
        "facility: value-invalid: measurement SR — kind trajectory writes a command to channel "
        "SCOR:H:SP, which `instruments.injection_request` names; it is a corrector of "
        "`groups.hcor`; fix: name a channel that is no corrector as "
        "`instruments.injection_request`"
    )


def test_an_arm_command_that_is_a_corrector_stops(tmp_path: Path) -> None:
    assert stop_line(tmp_path, _arm_on_a_corrector(periodic_tree())) == (
        "facility: value-invalid: measurement SR — kind trajectory writes a command to channel "
        "SCOR:H:SP, which carries signal `bpm_arm_command`; it is a corrector of `groups.hcor`; "
        "fix: remove signal `bpm_arm_command` from channel SCOR:H:SP"
    )


@pytest.mark.parametrize("address", COMMAND_CHANNELS)
@pytest.mark.parametrize(
    ("changes", "missing"),
    [
        ({"min_value": None}, "`min_value`"),
        ({"max_value": None}, "`max_value`"),
        ({"min_value": None, "max_value": None}, "`min_value`, `max_value`"),
    ],
)
def test_a_command_channel_limited_without_both_bounds_stops(
    tmp_path: Path, address: str, changes: dict[str, Any], missing: str
) -> None:
    assert stop_line(tmp_path, _limited(address, **changes)) == (
        f"facility: limit-invalid: limit {address} — the record of a command channel states no "
        f"{missing}; fix: state {missing}; a command is written inside both bounds"
    )


@pytest.mark.parametrize("address", COMMAND_CHANNELS)
@pytest.mark.parametrize("confirm", [None, True])
def test_a_command_channel_limited_without_confirm_false_stops(
    tmp_path: Path, address: str, confirm: bool | None
) -> None:
    """Leaving ``confirm`` out means true, so the record has to say false."""
    assert stop_line(tmp_path, _limited(address, confirm=confirm)) == (
        f"facility: limit-invalid: limit {address} — the record of a command channel does not "
        "state `confirm: false`; fix: state `confirm: false`; a command is not read back after "
        "the write"
    )


@pytest.mark.parametrize("address", COMMAND_CHANNELS)
def test_a_command_channel_limited_with_a_step_stops(tmp_path: Path, address: str) -> None:
    assert stop_line(tmp_path, _limited(address, max_step=1)) == (
        f"facility: limit-invalid: limit {address} — the record of a command channel states "
        "`max_step`; fix: remove `max_step`; a command is no step from the value before it"
    )


def test_a_locked_command_channel_still_states_both_bounds(tmp_path: Path) -> None:
    """``writable: false`` does not stand in for the bounds."""
    tree = _limited("SR:BPM:ARM", min_value=None, max_value=None, writable=False)
    assert stop_line(tmp_path, tree) == (
        "facility: limit-invalid: limit SR:BPM:ARM — the record of a command channel states no "
        "`min_value`, `max_value`; fix: state `min_value`, `max_value`; a command is written "
        "inside both bounds"
    )


# --- signals and engine wiring ------------------------------------------------------


@pytest.mark.parametrize("signal", SERIES_PLANES)
def test_a_series_signal_the_model_does_not_wire_stops(tmp_path: Path, signal: str) -> None:
    tree = periodic_tree()
    (address,) = [c["id"] for c in tree["records/channels.yaml"] if c.get("signal") == signal]
    words = SERIES_WORDS[signal]
    assert stop_line(tmp_path, _unwired(tree, address)) == (
        f"facility: value-invalid: measurement SR — kind trajectory needs channel {address}, "
        f"which carries signal `{signal}`, wired as {words}; model SR does not wire it; fix: "
        f"wire {address} in model SR as {words}, or remove trajectory from `kinds`"
    )


def test_an_arm_command_the_model_does_not_wire_stops(tmp_path: Path) -> None:
    assert stop_line(tmp_path, _unwired(periodic_tree(), "SR:BPM:ARM")) == (
        "facility: value-invalid: measurement SR — kind trajectory needs channel SR:BPM:ARM, "
        "which carries signal `bpm_arm_command`, wired as a command; model SR does not wire it; "
        "fix: wire SR:BPM:ARM in model SR as a command, or remove trajectory from `kinds`"
    )


@pytest.mark.parametrize(
    ("signal", "axis", "got"),
    [
        ("bpm_tbt_x", "y", "plane y"),
        ("bpm_tbt_y", "sum", "no plane"),
        ("bpm_tbt_sum", "x", "plane x"),
    ],
)
def test_a_series_signal_wired_as_another_plane_stops(
    tmp_path: Path, signal: str, axis: str, got: str
) -> None:
    tree = periodic_tree()
    (address,) = [c["id"] for c in tree["records/channels.yaml"] if c.get("signal") == signal]
    model(_unwired(tree, address), "SR")["wiring"].append(_series(address, "BPM1", axis))
    words = SERIES_WORDS[signal]
    assert stop_line(tmp_path, tree) == (
        f"facility: value-invalid: measurement SR — kind trajectory needs channel {address}, "
        f"which carries signal `{signal}`, wired as {words}; model SR wires it as `turn_series` "
        f"of {got}; fix: wire {address} in model SR as {words}, or remove trajectory from `kinds`"
    )


def test_an_injection_request_wired_as_no_command_stops(tmp_path: Path) -> None:
    """A corrector setpoint is a float the model wires, so the channel's own stops come first."""
    lines = stop_lines(tmp_path, _request_on_a_corrector(periodic_tree()))
    assert (
        "facility: value-invalid: measurement SR — kind trajectory needs channel SCOR:H:SP, "
        "which `instruments.injection_request` names, wired as a command; model SR wires it as "
        "`setpoint` of plane x; fix: wire SCOR:H:SP in model SR as a command, or remove "
        "trajectory from `kinds`"
    ) in lines


@pytest.mark.parametrize(
    ("axis", "words", "signals"),
    [
        ("x", "plane x", "`bpm_tbt_x`"),
        ("y", "plane y", "`bpm_tbt_y`"),
        ("sum", "no plane", "`bpm_tbt_sum` or `bpm_armed_status`"),
    ],
)
def test_a_turn_series_on_a_channel_of_no_matching_signal_stops(
    tmp_path: Path, axis: str, words: str, signals: str
) -> None:
    tree = periodic_tree()
    tree["records/channels.yaml"].append(
        {
            "id": "SR:AUX",
            "value_type": "waveform",
            "shape": [TURN_SAMPLES],
            "on": {"device": "SR/BPM1"},
        }
    )
    model(tree, "SR")["wiring"].append(_series("SR:AUX", "BPM1", axis))
    assert stop_line(tmp_path, tree) == (
        f"facility: value-invalid: measurement SR — model SR wires channel SR:AUX as "
        f"`turn_series` of {words}; the channel carries no signal {signals}; fix: set signal "
        f"{signals} on channel SR:AUX, or remove its wiring record from model SR"
    )


def test_a_turn_series_on_a_channel_of_another_signal_stops(tmp_path: Path) -> None:
    """The armed flag may sit outside the group, and is a series of no plane."""
    tree = periodic_tree()
    channel = _signal_channel("QD:AUX", "bpm_armed_status", "SR/QD")
    tree["records/channels.yaml"].append(
        channel | {"value_type": "waveform", "shape": [TURN_SAMPLES]}
    )
    model(tree, "SR")["wiring"].append(_series("QD:AUX", "BPM1", "x"))
    assert stop_line(tmp_path, tree) == (
        "facility: value-invalid: measurement SR — model SR wires channel QD:AUX as "
        "`turn_series` of plane x; the channel carries no signal `bpm_tbt_x`; fix: set signal "
        "`bpm_tbt_x` on channel QD:AUX, or remove its wiring record from model SR"
    )


@pytest.mark.parametrize("attribute", ["arm", "inject"])
def test_a_command_on_a_channel_of_no_matching_signal_stops(tmp_path: Path, attribute: str) -> None:
    tree = periodic_tree()
    tree["records/channels.yaml"].append(
        {"id": "SR:AUX", "role": "setpoint", "value_type": "int", "on": {"place": "SR"}}
    )
    model(tree, "SR")["wiring"].append({"address": "SR:AUX", "engine": {"attribute": attribute}})
    assert stop_line(tmp_path, tree) == (
        "facility: value-invalid: measurement SR — model SR wires channel SR:AUX as a command; "
        "the channel carries no signal `bpm_arm_command` and `instruments.injection_request` "
        "does not name it; fix: set signal `bpm_arm_command` on channel SR:AUX, name it as "
        "`instruments.injection_request`, or remove its wiring record from model SR"
    )


def test_an_arm_command_wired_on_a_device_outside_the_group_builds(tmp_path: Path) -> None:
    """A command record matches by its channel's signal, wherever the channel sits."""
    tree = periodic_tree()
    tree["records/channels.yaml"].append(_signal_channel("QD:AUX", "bpm_arm_command", "SR/QD"))
    model(tree, "SR")["wiring"].append({"address": "QD:AUX", "engine": {"attribute": "arm"}})
    document, _ = built_document(tmp_path, tree)
    assert measurement(document, "SR")["kinds"] == ["trajectory"]


# --- the trajectory block -----------------------------------------------------------


def _declared(**block: Any) -> dict[str, Any]:
    """The periodic tree with SR declaring a trajectory block: a cleaning command step.

    ``block`` is laid over a block of trigger sample 2, one global setting on
    the place's turns setpoint and the chain arm, inject, clean, readout.
    """
    tree = periodic_tree()
    place = {"place": "SR"}
    tree["records/channels.yaml"] += [
        {"id": "SR:TURNS:SP", "role": "setpoint", "value_type": "int", "on": place},
        {"id": "SR:CLEAN:REQ", "role": "setpoint", "value_type": "int", "on": place},
    ]
    tree["measurement/SR.yaml"]["trajectory"] = {
        "trigger_sample": 2,
        "settings": [
            {
                "id": "turns",
                "label": "Turns",
                "scope": "global",
                "address": "SR:TURNS:SP",
                "timeout_s": 1.0,
            }
        ],
        "steps": [
            {"id": "arm", "label": "Arm", "description": "Arms.", "default": True},
            {"id": "inject", "label": "Inject", "description": "Fires.", "default": True},
            {
                "id": "clean",
                "label": "Clean",
                "description": "Empties.",
                "default": True,
                "command": {"address": "SR:CLEAN:REQ", "value": 1, "timeout_s": 2.0},
            },
            {"id": "readout", "label": "Readout", "description": "Reads.", "default": True},
        ],
        **block,
    }
    return tree


def _block(tree: dict[str, Any]) -> dict[str, Any]:
    block: dict[str, Any] = tree["measurement/SR.yaml"]["trajectory"]
    return block


def _schema_line(path: str, message: str, remedy: str) -> str:
    return (
        f"facility: source-invalid: path models.1.measurement.trajectory.{path} — "
        f"measurement/SR.yaml: {message}; fix: {remedy} measurement/SR.yaml"
    )


def test_a_declared_block_builds_and_is_carried_verbatim(tmp_path: Path) -> None:
    tree = _declared()
    document, _ = built_document(tmp_path, tree)
    assert measurement(document, "SR")["trajectory"] == _block(tree)


def test_a_command_step_without_a_timeout_stops(tmp_path: Path) -> None:
    tree = _declared()
    del _block(tree)["steps"][2]["command"]["timeout_s"]
    assert stop_lines(tmp_path, tree) == [
        _schema_line("steps.2.command.timeout_s", "field required", "add `timeout_s` to")
    ]


def test_a_service_owned_step_stating_a_timeout_stops(tmp_path: Path) -> None:
    tree = _declared()
    _block(tree)["steps"][0]["timeout_s"] = 1.0
    assert stop_lines(tmp_path, tree) == [
        _schema_line(
            "steps.0.timeout_s", "extra inputs are not permitted", "remove `timeout_s` from"
        )
    ]


def test_a_block_without_steps_stops(tmp_path: Path) -> None:
    tree = _declared()
    del _block(tree)["steps"]
    assert stop_lines(tmp_path, tree) == [_schema_line("steps", "field required", "add `steps` to")]


def test_a_negative_trigger_sample_stops(tmp_path: Path) -> None:
    tree = _declared(trigger_sample=-1)
    assert stop_lines(tmp_path, tree) == [
        _schema_line(
            "trigger_sample",
            "input should be greater than or equal to 0",
            "correct `trigger_sample` in",
        )
    ]


def test_a_setting_of_an_unknown_scope_stops(tmp_path: Path) -> None:
    tree = _declared()
    _block(tree)["settings"][0]["scope"] = "beam"
    assert stop_lines(tmp_path, tree) == [
        _schema_line(
            "settings.0.scope", "input should be 'device' or 'global'", "correct `scope` in"
        )
    ]


def test_a_setting_without_a_timeout_stops(tmp_path: Path) -> None:
    tree = _declared()
    del _block(tree)["settings"][0]["timeout_s"]
    assert stop_lines(tmp_path, tree) == [
        _schema_line("settings.0.timeout_s", "field required", "add `timeout_s` to")
    ]


def test_a_done_reading_without_a_value_stops(tmp_path: Path) -> None:
    tree = _declared()
    _block(tree)["steps"][2]["command"]["done"] = {"address": "SR:CLEAN:REQ"}
    assert stop_lines(tmp_path, tree) == [
        _schema_line("steps.2.command.done.value", "field required", "add `value` to")
    ]
