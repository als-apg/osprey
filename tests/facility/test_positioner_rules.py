"""A setpoint whose move a command starts names that command as ``started_by``.

The command is a bool, int or float command that states ``timeout_s`` and
whose status states ``idle``; no other setpoint names it. The build stops with
``reference-missing`` on a name that is no channel and ``command-invalid``
where the statement breaks a rule.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from osprey_facility.build import LATER_STAGES, build_facility
from osprey_facility.errors import FacilityBuildError
from osprey_facility.validate import run_stages
from tests._builds import FRAMEWORK_CHECKS
from tests.facility._synthetic_trees import plain_tree, write_tree
from tests.facility.test_command_completion import (
    BUSY,
    FURTHER,
    SETPOINT,
    _further_stop,
    _further_tree,
)

#: The positioner: a setpoint, its readback, its start command and the command's status.
POSITION, READBACK, GO, MOVING = "Q1:SP", "Q1:RB", "Q1:GO", "Q1:MOVING"
#: A command that is its own pair, with no completion.
STOP = "Q1:STOP"

_SIGNALS = [
    {"name": "move_command", "description": "Command that starts a move."},
    {"name": "move_status", "description": "Whether a move runs."},
    {"name": "stop_command", "description": "Command that ends a move."},
]


def _tree(**changes: dict[str, Any]) -> dict[str, Any]:
    """The plain tree with ``POSITION`` started by ``GO``; ``changes`` edit by address."""
    tree = plain_tree()
    tree["signals.yaml"] = list(_SIGNALS)
    tree["records/channels.yaml"] += [
        {
            "id": GO,
            "role": "setpoint",
            "signal": "move_command",
            "value_type": "bool",
            "pair": MOVING,
            "timeout_s": 5.0,
            "on": {"device": "SR/Q1"},
        },
        {
            "id": MOVING,
            "signal": "move_status",
            "value_type": "enum",
            "options": ["MOVING", "DONE"],
            "idle": "DONE",
            "on": {"device": "SR/Q1"},
        },
        {
            "id": STOP,
            "role": "setpoint",
            "signal": "stop_command",
            "value_type": "bool",
            "on": {"device": "SR/Q1"},
        },
    ]
    changes = {POSITION: {"started_by": GO}, **changes}
    for channel in tree["records/channels.yaml"]:
        channel.update(changes.get(channel["id"], {}))
        for key in [key for key, value in channel.items() if value is None]:
            del channel[key]
    return tree


def _build(tmp_path: Path, tree: dict[str, Any]) -> dict[str, Any]:
    return build_facility(
        write_tree(tmp_path / "facility", tree), project_name="demo", extra_checks=FRAMEWORK_CHECKS
    )


def _stop(tmp_path: Path, tree: dict[str, Any]) -> str:
    with pytest.raises(FacilityBuildError) as stopped:
        _build(tmp_path, tree)
    return stopped.value.format_message()


def test_a_positioner_and_its_start_command_build(tmp_path: Path) -> None:
    channels = {channel["id"]: channel for channel in _build(tmp_path, _tree())["channels"]}

    assert channels[POSITION]["started_by"] == GO
    assert channels[GO]["command"] is True
    assert "command" not in channels[POSITION]


def test_a_started_by_naming_no_channel_is_reference_missing(tmp_path: Path) -> None:
    assert _stop(tmp_path, _tree(**{POSITION: {"started_by": "Q1:NOWHERE"}})) == (
        "facility: reference-missing: channel Q1:SP — records/channels.yaml `started_by` names "
        "channel Q1:NOWHERE, which does not exist; fix: add channel Q1:NOWHERE or correct "
        "`started_by`"
    )


#: One case per rule: the edits to the tree, the channel the stop names, its detail and its fix.
_RULES: dict[str, tuple[dict[str, dict[str, Any]], str, str, str]] = {
    "a-not-a-setpoint": (
        {POSITION: {"started_by": None}, READBACK: {"started_by": GO}},
        READBACK,
        "`started_by` on a readback channel",
        "remove `started_by`; only a setpoint's move is started by a command",
    ),
    "b-a-command": (
        {POSITION: {"started_by": None}, STOP: {"started_by": GO}},
        STOP,
        "`started_by` on command Q1:STOP",
        "remove `started_by`; a command is fired, never started by another",
    ),
    "d-itself": (
        {POSITION: {"started_by": POSITION}},
        POSITION,
        "`started_by` names Q1:SP, the setpoint itself",
        "name the command that starts Q1:SP's move",
    ),
    "d-its-pair": (
        {POSITION: {"started_by": READBACK}},
        POSITION,
        "`started_by` names Q1:RB, the setpoint's own pair",
        "name the command that starts Q1:SP's move",
    ),
    "e-no-command": (
        {POSITION: {"started_by": "BPM1:X"}},
        POSITION,
        "`started_by` names BPM1:X, which is no command",
        "name a command; give BPM1:X a command signal",
    ),
    "f-no-completion": (
        {GO: {"timeout_s": None}, MOVING: {"idle": None}},
        POSITION,
        "`started_by` names command Q1:GO, which states no `timeout_s` or whose status states "
        "no `idle`",
        "state `timeout_s` on Q1:GO and `idle` on its status",
    ),
    "f-its-own-pair": (
        {POSITION: {"started_by": STOP}},
        POSITION,
        "`started_by` names command Q1:STOP, which states no `timeout_s` or whose status "
        "states no `idle`",
        "state `timeout_s` on Q1:STOP and `idle` on its status",
    ),
    "h-a-labelled-command": (
        {GO: {"value_type": "enum", "options": ["HOLD", "GO", "PAUSE"]}},
        POSITION,
        "`started_by` names enum channel Q1:GO",
        "name a bool, int or float command; a move writes it 1",
    ),
}


@pytest.mark.parametrize("rule", sorted(_RULES))
def test_started_by_rules(tmp_path: Path, rule: str) -> None:
    changes, channel, detail, fix = _RULES[rule]

    assert _stop(tmp_path, _tree(**changes)) == (
        f"facility: command-invalid: channel {channel} — {detail}; fix: {fix}"
    )


def test_started_by_naming_a_further_address_of_a_command_names_its_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule (g): the command's record states ``timeout_s`` and its status ``idle``."""
    tree = _further_tree(monkeypatch)
    (setpoint,) = [c for c in tree["records/channels.yaml"] if c["id"] == "QF:SP"]
    setpoint["started_by"] = FURTHER
    (busy,) = [c for c in tree["records/channels.yaml"] if c["id"] == BUSY]
    assert busy["idle"] == 0

    assert _further_stop(tmp_path, tree) == (
        f"facility: command-invalid: channel QF:SP — `started_by` names {FURTHER}, which wiring "
        f"record SR/QD:SP states under `also`; fix: name {SETPOINT}"
    )


def test_two_setpoints_naming_one_start_command_stop_the_build(tmp_path: Path) -> None:
    tree = _tree()
    tree["records/channels.yaml"] += [
        {"id": "Q1:TRIM", "role": "setpoint", "on": {"device": "SR/Q1"}, "started_by": GO},
        {"id": "Q1:VERNIER", "role": "setpoint", "on": {"device": "SR/Q1"}, "started_by": GO},
    ]

    report = run_stages(
        write_tree(tmp_path / "facility", tree),
        project_name="demo",
        later=LATER_STAGES,
        extra_checks=FRAMEWORK_CHECKS,
    )

    assert [error.format_message() for error in report.errors] == [
        "facility: command-invalid: channel Q1:TRIM — `started_by` names Q1:GO, which Q1:SP "
        "names too; fix: each start command starts one setpoint; give Q1:TRIM its own command",
        "facility: command-invalid: channel Q1:VERNIER — `started_by` names Q1:GO, which Q1:SP, "
        "Q1:TRIM name too; fix: each start command starts one setpoint; give Q1:VERNIER its own "
        "command",
    ]
