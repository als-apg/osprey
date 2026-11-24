"""A status's seed states ``responds``: after each write to its command it reads a value for a time.

A status channel's seed ``responds: {to, for_s, value}`` names the command
``to`` whose every write makes the simulated status read ``value`` for
``for_s`` seconds, then its ``idle`` again. The build checks the statement
against the command and the status's ``idle`` and stops with
``reference-missing``, ``seed-invalid`` or ``value-invalid``; the simulator
view writes it in the shape the simulator reads, and a subscriber sees the act
in real time.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from osprey_connectors.control_system._va_in_process_connector import VAInProcessConnector
from osprey_connectors.simulation.view import SEEDS_FILE, SimulatorView
from osprey_connectors_tests._write_fakes import arm_switch, writes_enabled_config
from osprey_facility.build import build_facility
from osprey_facility.errors import FacilityBuildError
from tests._builds import FRAMEWORK_CHECKS
from tests._live import async_wait_until
from tests.facility._pyaml_trees import built_document, measured_tree
from tests.facility._synthetic_trees import plain_tree, write_tree
from tests.facility.served_tree import in_process_config, served_tree
from tests.facility.test_command_completion import _further_tree, _marking

#: A command on the plain tree's quadrupole, and the status it pairs with.
COMMAND = "Q1:ON"
STATUS = "Q1:STATE"
SWITCH = {"value_type": "bool", "options": ["OFF", "ON"]}
#: An enum status of three labels.
THREE = {"value_type": "enum", "options": ["IDLE", "RAMPING", "FAULT"]}
#: The seconds the status reads its act in the trees here.
FOR_S = 0.5


def _tree(**changes: dict[str, Any]) -> dict[str, Any]:
    """The plain tree with a bounded command and its responding status; ``changes`` edit by address.

    A slot changed to ``None`` is dropped.
    """
    tree = plain_tree()
    channels = [
        {
            "id": COMMAND,
            "role": "setpoint",
            "signal": "power_on_command",
            "timeout_s": 5.0,
            "on": {"device": "SR/Q1"},
            **SWITCH,
        },
        {
            "id": STATUS,
            "signal": "power_on_status",
            "idle": "OFF",
            "simulation": {"responds": {"to": COMMAND, "for_s": FOR_S}},
            "on": {"device": "SR/Q1"},
            **SWITCH,
        },
    ]
    for channel in channels:
        channel.update(changes.get(channel["id"], {}))
        for key in [key for key, value in channel.items() if value is None]:
            del channel[key]
    tree["records/channels.yaml"] += channels
    return tree


def _responds(**slots: Any) -> dict[str, Any]:
    """The status's seed, ``responds`` to the command with ``slots`` changed; ``None`` drops one."""
    responds = {"to": COMMAND, "for_s": FOR_S, **slots}
    return {"responds": {key: value for key, value in responds.items() if value is not None}}


def _build(tmp_path: Path, tree: dict[str, Any]) -> dict[str, Any]:
    return build_facility(
        write_tree(tmp_path / "facility", tree), project_name="demo", extra_checks=FRAMEWORK_CHECKS
    )


def _seed(document: dict[str, Any], address: str = STATUS) -> dict[str, Any]:
    (channel,) = [c for c in document["channels"] if c["id"] == address]
    return channel["simulation"]


def _stop(tmp_path: Path, tree: dict[str, Any]) -> str:
    with pytest.raises(FacilityBuildError) as stopped:
        _build(tmp_path, tree)
    return stopped.value.format_message()


def _built_stop(tmp_path: Path, tree: dict[str, Any]) -> str:
    with pytest.raises(FacilityBuildError) as stopped:
        built_document(tmp_path, tree)
    return stopped.value.format_message()


# --- clean builds ----------------------------------------------------------------


def test_a_status_states_responds_to_its_command(tmp_path: Path) -> None:
    assert _seed(_build(tmp_path, _tree())) == {"responds": {"to": COMMAND, "for_s": FOR_S}}


def test_a_three_label_status_states_its_value(tmp_path: Path) -> None:
    tree = _tree(
        **{
            COMMAND: THREE,
            STATUS: {**THREE, "idle": "IDLE", "simulation": _responds(value="RAMPING")},
        }
    )

    assert _seed(_build(tmp_path, tree))["responds"]["value"] == "RAMPING"


def test_an_int_status_states_its_value(tmp_path: Path) -> None:
    status = {"value_type": "int", "options": None, "idle": 0, "simulation": _responds(value=1)}

    assert _seed(_build(tmp_path, _tree(**{STATUS: status})))["responds"]["value"] == 1


def test_a_string_status_states_its_value(tmp_path: Path) -> None:
    status = {
        "value_type": "string",
        "options": None,
        "idle": "READY",
        "simulation": _responds(value="BUSY"),
    }

    assert _seed(_build(tmp_path, _tree(**{STATUS: status})))["responds"]["value"] == "BUSY"


def test_a_status_responds_to_a_command_a_model_serves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _marking(monkeypatch, "QD:SP")
    tree = measured_tree()
    (qd,) = [c for c in tree["records/channels.yaml"] if c["id"] == "QD:SP"]
    qd.update(pair="QD:BUSY", timeout_s=2.0)
    responds = {"to": "QD:SP", "for_s": FOR_S, "value": 1}
    tree["records/channels.yaml"].append(
        {
            "id": "QD:BUSY",
            "value_type": "int",
            "idle": 0,
            "simulation": {"responds": responds},
            "on": {"device": "SR/QD"},
        }
    )

    assert _seed(built_document(tmp_path, tree)[0], "QD:BUSY") == {"responds": responds}


def test_a_gated_status_may_respond(tmp_path: Path) -> None:
    gate = {"inputs": {COMMAND: "ON"}, "gated": "OFF"}
    tree = _tree(**{STATUS: {"simulation": {**_responds(), "gate": gate}}})

    assert _seed(_build(tmp_path, tree))["gate"] == gate


def test_follows_none_beside_responds_builds(tmp_path: Path) -> None:
    tree = _tree(**{STATUS: {"simulation": {**_responds(), "follows": "none"}}})

    assert _seed(_build(tmp_path, tree))["follows"] == "none"


# --- stops -----------------------------------------------------------------------


def test_responds_to_a_missing_channel_is_reference_missing(tmp_path: Path) -> None:
    assert _stop(tmp_path, _tree(**{STATUS: {"simulation": _responds(to="Q9:ON")}})) == (
        "facility: reference-missing: channel Q1:STATE — records/channels.yaml "
        "`simulation.responds.to` names channel Q9:ON, which does not exist; fix: add channel "
        "Q9:ON or correct `simulation.responds.to`"
    )


def test_responds_on_a_wired_status_stops_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _marking(monkeypatch, "QF:SP")
    tree = measured_tree()
    for channel in tree["records/channels.yaml"]:
        if channel["id"] == "QF:SP":
            channel["timeout_s"] = 2.0
        elif channel["id"] == "QF:RB":
            channel.update(idle=0.0, simulation={"responds": {"to": "QF:SP", "for_s": FOR_S}})

    assert _built_stop(tmp_path, tree) == (
        "facility: seed-invalid: channel QF:RB — `responds` on a channel model SR wires; fix: "
        "remove `responds`; model `SR` serves QF:RB"
    )


def test_responds_on_a_float_status_stops_the_build(tmp_path: Path) -> None:
    status = {
        "value_type": "float",
        "options": None,
        "idle": 0.0,
        "simulation": _responds(value=1.0),
    }

    assert _stop(tmp_path, _tree(**{STATUS: status})) == (
        "facility: value-invalid: channel Q1:STATE — a float channel carries `responds`; fix: "
        "state `responds` on a bool, enum, int or string status"
    )


def test_responds_on_a_status_without_idle_stops_the_build(tmp_path: Path) -> None:
    tree = _tree(**{COMMAND: {"timeout_s": None}, STATUS: {"idle": None}})

    assert _stop(tmp_path, tree) == (
        "facility: seed-invalid: channel Q1:STATE — `responds` on a status that states no "
        "`idle`; fix: state `idle` on Q1:STATE"
    )


def test_responds_beside_nominal_stops_the_build(tmp_path: Path) -> None:
    tree = _tree(**{STATUS: {"simulation": {**_responds(), "nominal": "OFF"}}})

    assert _stop(tmp_path, tree) == (
        "facility: seed-invalid: channel Q1:STATE — `responds` beside `nominal`; fix: remove "
        "`nominal`; a responding status reads its `idle` when no act runs"
    )


def test_responds_beside_linear_stops_the_build(tmp_path: Path) -> None:
    """A `linear` applies to float channels only, and a float status never responds."""
    tree = _tree(**{STATUS: {"simulation": {**_responds(), "linear": {"Q1:SP": 1.0}}}})

    assert _stop(tmp_path, tree) == (
        "facility: value-invalid: channel Q1:STATE — a bool channel carries `linear`; motion, "
        "`clamp` and `linear` apply to float channels only; fix: remove `linear`"
    )


@pytest.mark.parametrize("for_s", [0, -1, float("nan")])
def test_for_s_must_be_above_zero(tmp_path: Path, for_s: float) -> None:
    assert _stop(tmp_path, _tree(**{STATUS: {"simulation": _responds(for_s=for_s)}})) == (
        f"facility: value-invalid: channel Q1:STATE — `responds.for_s` {for_s!r} is not a "
        "finite number > 0; fix: state `responds.for_s` as the seconds the status reads its act"
    )


def test_responds_to_a_setpoint_that_is_no_command_stops_the_build(tmp_path: Path) -> None:
    assert _stop(tmp_path, _tree(**{STATUS: {"simulation": _responds(to="Q1:SP")}})) == (
        "facility: seed-invalid: channel Q1:STATE — `responds.to` names Q1:SP, which is no "
        "command; fix: name in `responds.to` a command whose `pair` is Q1:STATE"
    )


def test_responds_to_a_command_of_another_status_stops_the_build(tmp_path: Path) -> None:
    tree = _tree()
    tree["records/channels.yaml"] += [
        {
            "id": "Q1:RESET",
            "role": "setpoint",
            "signal": "reset_command",
            "pair": "Q1:FAULT",
            "timeout_s": 1.0,
            "on": {"device": "SR/Q1"},
            **SWITCH,
        },
        {
            "id": "Q1:FAULT",
            "signal": "fault_status",
            "idle": "OFF",
            "on": {"device": "SR/Q1"},
            **SWITCH,
        },
    ]
    tree["records/channels.yaml"][-3]["simulation"] = _responds(to="Q1:RESET")

    assert _stop(tmp_path, tree) == (
        "facility: seed-invalid: channel Q1:STATE — `responds.to` names command Q1:RESET, whose "
        "`pair` is Q1:FAULT; fix: name in `responds.to` a command whose `pair` is Q1:STATE"
    )


def test_responds_to_a_further_address_stops_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = _further_tree(monkeypatch)
    for channel in tree["records/channels.yaml"]:
        if channel["id"] == "QD:BUSY":
            channel["simulation"] = {
                "responds": {"to": "RACK3:UNIT2:SP", "for_s": FOR_S, "value": 1}
            }

    assert _built_stop(tmp_path, tree) == (
        "facility: seed-invalid: channel QD:BUSY — `responds.to` names RACK3:UNIT2:SP, which "
        "wiring record SR/QD:SP states under `also`; fix: name QD:SP; a command is named by its "
        "record's address"
    )


def test_a_value_that_is_no_label_stops_the_build(tmp_path: Path) -> None:
    assert _stop(tmp_path, _tree(**{STATUS: {"simulation": _responds(value="MAYBE")}})) == (
        "facility: value-invalid: channel Q1:STATE — `responds.value` of bool channel Q1:STATE "
        "is 'MAYBE', not a label; fix: type `responds.value` by the channel, as `idle`"
    )


def test_a_three_label_status_without_value_stops_the_build(tmp_path: Path) -> None:
    tree = _tree(**{COMMAND: THREE, STATUS: {**THREE, "idle": "IDLE"}})

    assert _stop(tmp_path, tree) == (
        "facility: seed-invalid: channel Q1:STATE — `responds` states no `value` on a status "
        "without exactly two labels; fix: state `responds.value`"
    )


def test_a_value_equal_to_idle_stops_the_build(tmp_path: Path) -> None:
    assert _stop(tmp_path, _tree(**{STATUS: {"simulation": _responds(value="OFF")}})) == (
        "facility: seed-invalid: channel Q1:STATE — `responds.value` 'OFF' is the status's "
        "`idle`; fix: state as `responds.value` a value other than `idle`"
    )


# --- the view the simulator serves ---------------------------------------------------

#: The served tree's bool command and its status, which reads ``ON`` for ``SERVED_FOR_S``.
SWITCH_COMMAND, SWITCH_STATUS = "C:CMD", "C:STS"
#: The served tree's enum command and its status of three labels, which states its value.
MODE_COMMAND, MODE_STATUS = "M:CMD", "M:STS"
SERVED_FOR_S = 0.3
#: The simulator's tick in the live tests: the idle edge arrives on the tick after ``for_s``.
TICK_S = 0.05


def _served(root: Path) -> Path:
    """A served tree of two commands, each with a responding status, ticking every ``TICK_S``."""
    view = served_tree(
        root,
        {SWITCH_COMMAND: SWITCH_STATUS, MODE_COMMAND: MODE_STATUS},
        channels={
            SWITCH_COMMAND: {"signal": "power_on_command", "timeout_s": 2.0, **SWITCH},
            SWITCH_STATUS: {
                "idle": "OFF",
                "simulation": {"responds": {"to": SWITCH_COMMAND, "for_s": SERVED_FOR_S}},
                **SWITCH,
            },
            MODE_COMMAND: {"signal": "reset_command", "timeout_s": 2.0, **THREE},
            MODE_STATUS: {
                "idle": "IDLE",
                "simulation": {
                    "responds": {"to": MODE_COMMAND, "for_s": SERVED_FOR_S, "value": "RAMPING"}
                },
                **THREE,
            },
        },
    )
    (view.parent.parent / "config.yml").write_text(f"simulation:\n  tick_s: {TICK_S}\n")
    return view


def test_the_released_reader_opens_a_responding_view(tmp_path: Path) -> None:
    seeds = SimulatorView(_served(tmp_path)).document(SEEDS_FILE)["seeds"]

    assert seeds[SWITCH_STATUS] == {
        "nominal": "OFF",
        "responds": {"to": SWITCH_COMMAND, "for_s": SERVED_FOR_S, "value": "ON"},
    }


def _labels(values: list[Any], status: dict[str, Any]) -> list[str]:
    """The labels of a labelled status's values, which the simulator serves by index."""
    return [status["options"][value] for value in values]


@pytest.fixture
def _armed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The login's switch on for the simulator target, with a write tier that allows."""
    arm_switch(monkeypatch, tmp_path / "armed-data")
    monkeypatch.setattr("osprey_connectors.config.get_config_value", writes_enabled_config)


async def _act(
    root: Path, command: str, status: str, value: Any
) -> tuple[VAInProcessConnector, list[tuple[Any, float]], float, list[Any]]:
    """Subscribe to ``status``, write ``value`` to ``command`` and wait for the idle edge.

    Returns:
        The connected connector; each callback's value with its arrival on the
        monotonic clock; the instant before the write; and the values seen
        when the write returned.
    """
    connector = VAInProcessConnector()
    await connector.connect(in_process_config(_served(root), response_delay_ms=0))
    seen: list[tuple[Any, float]] = []
    await connector.subscribe(
        status, lambda reading: seen.append((reading.value, time.monotonic()))
    )
    await async_wait_until(lambda: len(seen) >= 1, what=f"the current value of {status}")

    t0 = time.monotonic()
    await connector.write_channel(command, value)
    at_return = [value for value, _ in seen]
    await async_wait_until(lambda: len(seen) >= 3, what=f"the idle edge of {status}")
    return connector, seen, t0, at_return


@pytest.mark.asyncio
@pytest.mark.usefixtures("_armed")
async def test_a_subscriber_sees_idle_then_non_idle_then_idle(tmp_path: Path) -> None:
    connector, seen, t0, at_return = await _act(tmp_path, SWITCH_COMMAND, SWITCH_STATUS, "ON")
    try:
        assert _labels([value for value, _ in seen], SWITCH) == ["OFF", "ON", "OFF"]
        assert "ON" in _labels(at_return, SWITCH)
        assert seen[2][1] >= t0 + SERVED_FOR_S
    finally:
        await connector.disconnect()


@pytest.mark.asyncio
@pytest.mark.usefixtures("_armed")
async def test_a_read_after_the_idle_edge_reads_idle(tmp_path: Path) -> None:
    connector, _seen, _t0, _at_return = await _act(tmp_path, SWITCH_COMMAND, SWITCH_STATUS, "ON")
    try:
        reading = await connector.read_channel(SWITCH_STATUS)
        assert _labels([reading.value], SWITCH) == ["OFF"]
    finally:
        await connector.disconnect()


@pytest.mark.asyncio
@pytest.mark.usefixtures("_armed")
async def test_a_stated_value_is_the_one_served(tmp_path: Path) -> None:
    connector, seen, _t0, _at_return = await _act(tmp_path, MODE_COMMAND, MODE_STATUS, "RAMPING")
    try:
        assert _labels([value for value, _ in seen], THREE) == ["IDLE", "RAMPING", "IDLE"]
    finally:
        await connector.disconnect()
