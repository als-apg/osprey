"""The demo-history session reads channels in the units the facility serves them in.

Every ``channel_read`` result in ``demo-history.yaml`` states a value and a unit. The
unit must be the one ``records/channels.yaml`` gives that channel, or the rehearsed
session would show readings the live facility never produces. The storage-ring BPMs
read in millimetres, so the sector 3 drift (+41 um, +34 um) reads 0.041 mm and
0.034 mm, and the embedded orbit-statistics analysis still reports its recorded
micrometre figures from millimetre inputs.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

TEMPLATE_FACILITY = Path(__file__).resolve().parents[2] / "src/osprey/templates/facilities/example"
SCENARIO_FILE = TEMPLATE_FACILITY / "scenarios" / "demo-history.yaml"
CHANNELS_FILE = TEMPLATE_FACILITY / "records" / "channels.yaml"

BPM13_X = "SR:DIAG:BPM:13:POSITION:X"
BPM14_X = "SR:DIAG:BPM:14:POSITION:X"


def _steps() -> Iterator[dict[str, Any]]:
    scenario = yaml.safe_load(SCENARIO_FILE.read_text(encoding="utf-8"))
    for session in scenario["sessions"]:
        for part in session["parts"]:
            for turn in part["turns"]:
                yield from turn.get("steps", [])


def _channel_reads() -> list[dict[str, Any]]:
    return [
        step["result"] for step in _steps() if step.get("tool", {}).get("name") == "channel_read"
    ]


@pytest.fixture(scope="module")
def channel_units() -> dict[str, str | None]:
    records = yaml.safe_load(CHANNELS_FILE.read_text(encoding="utf-8"))
    return {record["id"]: record.get("unit") for record in records}


def test_the_scenario_has_channel_reads() -> None:
    assert _channel_reads(), "demo-history.yaml carries no channel_read step"


def test_every_reading_states_the_channel_unit(channel_units: dict[str, str | None]) -> None:
    mismatches = []
    for result in _channel_reads():
        for channel, reading in result.items():
            assert channel in channel_units, f"{channel} is not a channel of the example facility"
            if reading.get("unit") != channel_units[channel]:
                mismatches.append((channel, reading.get("unit"), channel_units[channel]))
    assert not mismatches, f"(channel, scenario unit, channels.yaml unit): {mismatches}"


def test_the_sector_3_bpms_read_in_millimetres() -> None:
    seen = 0
    for result in _channel_reads():
        if BPM13_X in result:
            assert result[BPM13_X] == {"value": pytest.approx(0.041), "unit": "mm"}
            assert result[BPM14_X] == {"value": pytest.approx(0.034), "unit": "mm"}
            seen += 1
    assert seen == 2, "both sector 3 reads (before and after the trim) must carry BPM13/BPM14"


def test_the_orbit_statistics_reproduce_their_recorded_result() -> None:
    pytest.importorskip("numpy")
    executes = [
        step
        for step in _steps()
        if step.get("tool", {}).get("name") == "execute"
        and "Orbit statistics" in step["input"].get("description", "")
    ]
    assert len(executes) == 1
    step = executes[0]
    code = step["input"]["code"]
    assert "1e-3" in code and "1e-6" not in code, "the analysis must read the BPMs in mm"
    run = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=60
    )
    assert json.loads(run.stdout) == step["result"]


def test_the_analysis_inputs_are_the_table_readings() -> None:
    reads = [r for r in _channel_reads() if "SR:DIAG:BPM:01:POSITION:X" in r]
    assert reads
    xs = [reads[0][f"SR:DIAG:BPM:{i:02d}:POSITION:X"]["value"] for i in range(1, 7)]
    ys = [reads[0][f"SR:DIAG:BPM:{i:02d}:POSITION:Y"]["value"] for i in range(1, 7)]
    assert xs == pytest.approx([0.012, -0.008, 0.021, 0.004, -0.015, 0.009])
    assert ys == pytest.approx([0.003, -0.002, 0.005, 0.001, -0.004, 0.002])
