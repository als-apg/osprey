"""End station ES1 of the example facility: a sample stage and a detector.

The stage moves on a start command and stops on a stop command; the detector
acquires on an acquire command. Each command that has a status names it as its
``pair``, states the longest one act may take, and its status states what it
reads when no act runs and, in the simulation, the act it plays after a write.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from osprey_connectors.control_system._va_in_process_connector import VAInProcessConnector
from osprey_connectors.simulation.view import VARIABLES_FILE, SimulatorView
from osprey_connectors_tests._write_fakes import arm_switch, writes_enabled_config
from tests._builds import BuiltProject
from tests.facility.served_tree import (
    assert_unmoved,
    assert_write_visible,
    in_process_config,
    unmoved_band,
)

#: Each bounded command of ES1, its status, its ``timeout_s`` and the status's ``idle``.
COMMANDS = {
    "ES1:MOT:STAGE:01:MOVE:SP": ("ES1:MOT:STAGE:01:STATUS:MOVE", 10.0, "DONE"),
    "ES1:DET:MCA:01:ACQUIRE:SP": ("ES1:DET:MCA:01:STATUS:ACQUIRE", 5.0, "FALSE"),
}
STOP = "ES1:MOT:STAGE:01:STOP:SP"
#: The stage's position setpoint, its readback and the command that starts its move.
POSITION, POSITION_RB, MOVE = (
    "ES1:MOT:STAGE:01:POSITION:SP",
    "ES1:MOT:STAGE:01:POSITION:RB",
    "ES1:MOT:STAGE:01:MOVE:SP",
)
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


@pytest.fixture
def _armed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The login's switch on for the simulator target, with a write tier that allows."""
    arm_switch(monkeypatch, tmp_path / "armed-data")
    monkeypatch.setattr("osprey_connectors.config.get_config_value", writes_enabled_config)


@pytest.mark.asyncio
@pytest.mark.usefixtures("_armed")
async def test_the_stage_setpoint_is_started_by_its_move_command(
    built_demo: BuiltProject, tmp_path: Path
) -> None:
    view_dir = tmp_path / "build" / "data" / "simulator"
    shutil.copytree(built_demo.build_dir / "data" / "simulator", view_dir)
    view = SimulatorView.open(view_dir)
    (written,) = [c for c in view.document(VARIABLES_FILE)["channels"] if c["address"] == POSITION]
    assert _channels(built_demo)[POSITION]["started_by"] == MOVE
    assert written["started_by"] == MOVE

    band = unmoved_band(view, POSITION_RB)
    connector = VAInProcessConnector()
    await connector.connect(in_process_config(view_dir, response_delay_ms=0))
    try:
        held = float((await connector.read_channel(POSITION_RB)).value)
        demand = held + 5.0
        assert_write_visible(demand, held, band)

        await connector.write_channel(POSITION, demand)
        assert_unmoved(
            POSITION_RB, held, float((await connector.read_channel(POSITION_RB)).value), band
        )

        await connector.write_channel(MOVE, "TRUE")
        moved = float((await connector.read_channel(POSITION_RB)).value)
        assert_unmoved(POSITION_RB, demand, moved, band)
    finally:
        await connector.disconnect()


def test_the_stage_setpoint_names_its_stop_channel(built_demo: BuiltProject) -> None:
    channels = _channels(built_demo)

    assert channels[POSITION]["started_by"] == MOVE
    assert channels[POSITION]["stopped_by"] == STOP
    assert channels[STOP]["command"] is True
