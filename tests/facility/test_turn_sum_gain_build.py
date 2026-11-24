"""A monitor's sum gain through the facility build.

The turn-by-turn fixture is built as ``osprey facility validate`` builds it,
with one scenario on top: a scenario may state a ``gain`` on a monitor's
turn-sum record and nothing else there.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey_facility.build import build_facility
from osprey_facility.errors import FacilityBuildError
from tests._builds import FRAMEWORK_CHECKS
from tests.facility._turn_by_turn_fixture import FIXTURE, RING, TURN_SUM, TURN_X

SUM_1, SUM_2 = TURN_SUM["BPM01"], TURN_SUM["BPM02"]


def _tree(tmp_path: Path) -> Path:
    facility_dir = tmp_path / "facility"
    shutil.copytree(FIXTURE, facility_dir)
    return facility_dir


def _with_faults(tmp_path: Path, faults: dict[str, Any]) -> Path:
    facility_dir = _tree(tmp_path)
    (facility_dir / "scenarios").mkdir()
    scenario = {"description": "Monitors that read one charge differently.", "faults": faults}
    (facility_dir / "scenarios" / "uneven-sums.yaml").write_text(yaml.safe_dump(scenario))
    return facility_dir


def _build(facility_dir: Path) -> dict[str, Any]:
    return build_facility(facility_dir, project_name="turn-by-turn", extra_checks=FRAMEWORK_CHECKS)


def test_a_scenario_states_a_gain_on_each_monitors_sum_record(tmp_path: Path) -> None:
    faults = {RING: {SUM_1: {"gain": 0.93}, SUM_2: {"gain": 1.08}}}

    document = _build(_with_faults(tmp_path, faults))

    (scenario,) = [s for s in document["scenarios"] if s["name"] == "uneven-sums"]
    assert scenario["faults"] == faults


def test_a_sum_record_carries_no_other_fault_field(tmp_path: Path) -> None:
    with pytest.raises(FacilityBuildError) as caught:
        _build(_with_faults(tmp_path, {RING: {SUM_1: {"offset": 0.1}}}))

    text = str(caught.value)
    assert f"faults.{RING}.{SUM_1}.offset is not a fault field of {SUM_1}" in text


def test_a_position_array_carries_no_gain(tmp_path: Path) -> None:
    turn_x = TURN_X["BPM01"]
    with pytest.raises(FacilityBuildError) as caught:
        _build(_with_faults(tmp_path, {RING: {turn_x: {"gain": 1.1}}}))

    assert f"faults.{RING}.{turn_x}.gain is not a fault field of {turn_x}" in str(caught.value)


def test_a_sum_gain_off_its_range_stops_the_build(tmp_path: Path) -> None:
    with pytest.raises(FacilityBuildError) as caught:
        _build(_with_faults(tmp_path, {RING: {SUM_1: {"gain": 0.0}}}))

    assert f"faults.{RING}.{SUM_1}.gain" in str(caught.value)
    assert "below its range" in str(caught.value)
