"""End station ES1 of the example facility: a sample stage and a detector.

The stage moves on a start command and stops on a stop command; the detector
acquires on an acquire command. Each command that has a status names it as its
``pair``, states the longest one act may take, and its status states what it
reads when no act runs and, in the simulation, the act it plays after a write.
"""

from __future__ import annotations

from typing import Any

from tests._builds import BuiltProject

#: Each bounded command of ES1, its status, its ``timeout_s`` and the status's ``idle``.
COMMANDS = {
    "ES1:MOT:STAGE:01:MOVE:SP": ("ES1:MOT:STAGE:01:STATUS:MOVE", 10.0, "DONE"),
    "ES1:DET:MCA:01:ACQUIRE:SP": ("ES1:DET:MCA:01:STATUS:ACQUIRE", 5.0, "FALSE"),
}
STOP = "ES1:MOT:STAGE:01:STOP:SP"
SPECTRUM = "ES1:DET:MCA:01:SPECTRUM:WAVEFORM"


def _channels(built: BuiltProject) -> dict[str, dict[str, Any]]:
    return {channel["id"]: channel for channel in built.facility["channels"]}


def test_the_stage_and_the_detector_state_command_status_idle_and_bound(
    built_demo: BuiltProject,
) -> None:
    channels = _channels(built_demo)
    for command, (status, timeout_s, idle) in COMMANDS.items():
        assert channels[command]["command"] is True, command
        assert channels[command]["pair"] == status
        assert channels[command]["timeout_s"] == timeout_s
        assert channels[status]["idle"] == idle


def test_each_station_status_responds_to_its_command(built_demo: BuiltProject) -> None:
    channels = _channels(built_demo)
    responds = {
        command: channels[status]["simulation"]["responds"]
        for command, (status, _, _) in COMMANDS.items()
    }

    assert responds == {
        "ES1:MOT:STAGE:01:MOVE:SP": {
            "to": "ES1:MOT:STAGE:01:MOVE:SP",
            "for_s": 2.0,
            "value": "MOVING",
        },
        "ES1:DET:MCA:01:ACQUIRE:SP": {"to": "ES1:DET:MCA:01:ACQUIRE:SP", "for_s": 1.0},
    }


def test_the_spectrum_is_a_waveform_of_1024_samples(built_demo: BuiltProject) -> None:
    spectrum = _channels(built_demo)[SPECTRUM]

    assert spectrum["value_type"] == "waveform"
    assert spectrum["shape"] == [1024]
    assert len(spectrum["simulation"]["nominal"]) == 1024


def test_the_stop_command_is_its_own_pair(built_demo: BuiltProject) -> None:
    stop = _channels(built_demo)[STOP]

    assert stop["command"] is True
    assert stop["pair"] == STOP
    assert "timeout_s" not in stop
