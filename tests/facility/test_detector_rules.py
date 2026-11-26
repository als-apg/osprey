"""A device that states ``acquire`` and ``results`` is a detector.

``acquire`` names a command on the device that states ``timeout_s``, whose
status states ``idle``; ``results`` lists the readbacks on the device it
records. A detector's id is its plan name: it holds no ``/`` or ``.`` and is no
channel's address. The build stops with ``detector-invalid`` where a statement
breaks a rule.
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

#: The detector, its acquire command, the command's status and two results.
DETECTOR = "MCA1"
ACQUIRE, BUSY, COUNTS, SPECTRUM = "MCA1:ACQ", "MCA1:BUSY", "MCA1:COUNTS", "MCA1:SPECTRUM"
#: A readback on the detector it does not record, and a setpoint on it.
TEMPERATURE, GAIN = "MCA1:TEMP", "MCA1:GAIN"

_SIGNALS = [
    {"name": "acquire_command", "description": "Command that starts one acquisition."},
    {"name": "acquire_status", "description": "Whether an acquisition runs."},
    {"name": "move_command", "description": "Command that starts a move."},
]


def _tree(device: dict[str, Any] | None = None, **changes: dict[str, Any]) -> dict[str, Any]:
    """The plain tree with detector ``MCA1``; ``device`` edits its record, ``changes`` channels."""
    tree = plain_tree()
    tree["signals.yaml"] = list(_SIGNALS)
    on = {"device": DETECTOR}
    record = {"id": DETECTOR, "acquire": ACQUIRE, "results": [COUNTS, SPECTRUM]}
    record.update(device or {})
    tree["records/devices.yaml"].append(
        {key: value for key, value in record.items() if value is not None}
    )
    tree["records/channels.yaml"] += [
        {
            "id": ACQUIRE,
            "role": "setpoint",
            "signal": "acquire_command",
            "value_type": "bool",
            "pair": BUSY,
            "timeout_s": 5.0,
            "on": on,
        },
        {"id": BUSY, "signal": "acquire_status", "value_type": "bool", "idle": "FALSE", "on": on},
        {"id": COUNTS, "on": on},
        {"id": SPECTRUM, "value_type": "waveform", "shape": [1024], "on": on},
        {"id": TEMPERATURE, "on": on},
        {"id": GAIN, "role": "setpoint", "on": on},
    ]
    for channel in tree["records/channels.yaml"]:
        channel.update(changes.get(channel["id"], {}))
        for key in [key for key, value in channel.items() if value is None]:
            del channel[key]
    return tree


def _build(tmp_path: Path, tree: dict[str, Any]) -> dict[str, Any]:
    return build_facility(
        write_tree(tmp_path / "facility", tree), project_name="demo", extra_checks=FRAMEWORK_CHECKS
    )


def _errors(tmp_path: Path, tree: dict[str, Any]) -> list[str]:
    report = run_stages(
        write_tree(tmp_path / "facility", tree),
        project_name="demo",
        later=LATER_STAGES,
        extra_checks=FRAMEWORK_CHECKS,
    )
    return [error.format_message() for error in report.errors]


def test_a_device_stating_acquire_and_results_builds(tmp_path: Path) -> None:
    document = _build(tmp_path, _tree())
    (device,) = [d for d in document["devices"] if d["id"] == DETECTOR]

    assert (device["acquire"], device["results"]) == (ACQUIRE, [COUNTS, SPECTRUM])


def _moved_by_acquire(tree: dict[str, Any]) -> dict[str, Any]:
    """``tree`` with ``MCA1:GAIN`` a setpoint whose move ``MCA1:ACQ`` starts."""
    for channel in tree["records/channels.yaml"]:
        if channel["id"] == GAIN:
            channel["started_by"] = ACQUIRE
    return tree


def _renamed(tree: dict[str, Any], old: str, new: str) -> dict[str, Any]:
    """``tree`` with device ``old`` renamed ``new``, every ``on`` naming it moved."""
    for device in tree["records/devices.yaml"]:
        if device["id"] == old:
            device["id"] = new
    for channel in tree["records/channels.yaml"]:
        if channel.get("on") == {"device": old}:
            channel["on"] = {"device": new}
    return tree


#: One case per rule: the tree, the device the stop names, its detail and its fix.
_RULES: dict[str, tuple[dict[str, Any], str, str, str]] = {
    "acquire-without-results": (
        _tree({"results": None}),
        DETECTOR,
        "`acquire` without `results`",
        "state `acquire` and `results` together",
    ),
    "results-without-acquire": (
        _tree({"acquire": None}),
        DETECTOR,
        "`results` without `acquire`",
        "state `acquire` and `results` together",
    ),
    "acquire-no-channel": (
        _tree({"acquire": "MCA1:NOWHERE"}),
        DETECTOR,
        "`acquire` names MCA1:NOWHERE, which is no channel on MCA1",
        "name a command on MCA1",
    ),
    "acquire-on-another-device": (
        _tree({"acquire": "Q1:SP"}),
        DETECTOR,
        "`acquire` names Q1:SP, which is no channel on MCA1",
        "name a command on MCA1",
    ),
    "acquire-not-a-command": (
        _tree({"acquire": GAIN}),
        DETECTOR,
        "`acquire` names MCA1:GAIN, which is no command that states `timeout_s`",
        "state `timeout_s` on MCA1:GAIN and `idle` on its status",
    ),
    "acquire-no-bound": (
        _tree(**{ACQUIRE: {"timeout_s": None}, BUSY: {"idle": None}}),
        DETECTOR,
        "`acquire` names MCA1:ACQ, which is no command that states `timeout_s`",
        "state `timeout_s` on MCA1:ACQ and `idle` on its status",
    ),
    "results-empty": (
        _tree({"results": []}),
        DETECTOR,
        "`results` is empty",
        "list each result once",
    ),
    "results-repeated": (
        _tree({"results": [COUNTS, COUNTS]}),
        DETECTOR,
        "`results` repeats an address",
        "list each result once",
    ),
    "result-no-channel": (
        _tree({"results": [COUNTS, "MCA1:NOWHERE"]}),
        DETECTOR,
        "`results` lists MCA1:NOWHERE, which is no readback on MCA1",
        "list a readback on MCA1",
    ),
    "result-on-another-device": (
        _tree({"results": [COUNTS, "BPM1:X"]}),
        DETECTOR,
        "`results` lists BPM1:X, which is no readback on MCA1",
        "list a readback on MCA1",
    ),
    "result-a-setpoint": (
        _tree({"results": [COUNTS, GAIN]}),
        DETECTOR,
        "`results` lists MCA1:GAIN, which is no readback on MCA1",
        "list a readback on MCA1",
    ),
    "result-of-two-dimensions": (
        _tree(**{SPECTRUM: {"shape": [32, 32]}}),
        DETECTOR,
        "`results` lists MCA1:SPECTRUM, a waveform of 2 dimensions",
        "list a readback of one dimension on MCA1",
    ),
    "acquire-also-a-start": (
        _moved_by_acquire(_tree()),
        DETECTOR,
        "`acquire` names MCA1:ACQ, which a setpoint names under `started_by`",
        "a command starts a positioner or acquires a detector, not both",
    ),
    "id-with-a-slash": (
        _renamed(_tree(), DETECTOR, "ES1/MCA01"),
        "ES1/MCA01",
        "a detector's id is its plan name, and a run cannot name a device whose id holds `/` "
        "or `.`",
        "give the device an id without them",
    ),
}


def _second_detector(tree: dict[str, Any]) -> dict[str, Any]:
    """``tree`` with device ``MCA2`` naming ``MCA1:ACQ`` as its acquire too."""
    tree["records/devices.yaml"].append({"id": "MCA2", "acquire": ACQUIRE, "results": ["MCA2:X"]})
    tree["records/channels.yaml"].append({"id": "MCA2:X", "on": {"device": "MCA2"}})
    return tree


# One command acquires one detector: a channel sits on one device, so a second
# device that names it acquires through a command that is not its own.
_RULES["acquire-named-by-two-devices"] = (
    _second_detector(_tree()),
    "MCA2",
    "`acquire` names MCA1:ACQ, which is no channel on MCA2",
    "name a command on MCA2",
)
_RULES["id-with-a-dot"] = (
    _renamed(_tree(), DETECTOR, "MCA.1"),
    "MCA.1",
    "a detector's id is its plan name, and a run cannot name a device whose id holds `/` or `.`",
    "give the device an id without them",
)
_RULES["id-a-channel-address"] = (
    _renamed(_tree(), DETECTOR, TEMPERATURE),
    TEMPERATURE,
    "detector MCA1:TEMP has the address of a channel as its id",
    "give device MCA1:TEMP an id that is no channel address",
)


@pytest.mark.parametrize("rule", sorted(_RULES))
def test_each_detector_rule_stops_with_its_fix(tmp_path: Path, rule: str) -> None:
    tree, device, detail, fix = _RULES[rule]

    assert _errors(tmp_path, tree) == [
        f"facility: detector-invalid: device {device} — {detail}; fix: {fix}"
    ]


def test_the_build_order_puts_detector_invalid_after_command_invalid(tmp_path: Path) -> None:
    tree = _tree({"results": None})
    tree["records/channels.yaml"][0]["timeout_s"] = 1.0

    assert [line.split(": ", 2)[1] for line in _errors(tmp_path, tree)] == [
        "command-invalid",
        "detector-invalid",
    ]


def test_a_detector_stop_ends_the_build(tmp_path: Path) -> None:
    with pytest.raises(FacilityBuildError) as stopped:
        _build(tmp_path, _tree({"results": None}))

    assert stopped.value.kind == "detector-invalid"
