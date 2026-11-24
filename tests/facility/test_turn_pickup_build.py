"""A monitor's pickup and its button gains through the facility build.

The turn-by-turn fixture is built as ``osprey facility validate`` builds it,
with a pickup stated in the ring's settings and one scenario on top: the
monitors' turn-sum records then carry one gain per button, a scenario may
state them, and a malformed pickup statement stops the build.
"""

from __future__ import annotations

import json
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

PLANES: dict[str, Any] = {
    "x": {"plus": [1, 4], "minus": [2, 3]},
    "y": {"plus": [1, 2], "minus": [3, 4]},
}
ROUND: dict[str, Any] = {
    "shape": "round",
    "radius": 0.016,
    "buttons": [45.0, 135.0, 225.0, 315.0],
    "half_width": 14.0,
    **PLANES,
}
TABLE: dict[str, Any] = {"shape": "table", "table": "pickups/flat.json", "buttons": 4, **PLANES}


def _tree(tmp_path: Path, faults: dict[str, Any] | None = None, **block: Any) -> Path:
    facility_dir = tmp_path / "facility"
    shutil.copytree(FIXTURE, facility_dir)
    models = yaml.safe_load((facility_dir / "models.yaml").read_text())
    (ring,) = [model for model in models if model["name"] == RING]
    ring["settings"]["pyat"]["turn_by_turn"].update(block)
    (facility_dir / "models.yaml").write_text(yaml.safe_dump(models, sort_keys=False))
    if faults is not None:
        (facility_dir / "scenarios").mkdir()
        scenario = {"description": "A monitor with uneven buttons.", "faults": faults}
        (facility_dir / "scenarios" / "uneven-buttons.yaml").write_text(yaml.safe_dump(scenario))
    return facility_dir


def _table(facility_dir: Path, *, buttons: int = 4) -> Path:
    grid = [-0.002, -0.001, 0.0, 0.001, 0.002]
    corners = [(1, 1), (-1, 1), (-1, -1), (1, -1)][:buttons]
    values = [[[1.0 + 60.0 * (cx * x + cy * y) for x in grid] for y in grid] for cx, cy in corners]
    path = facility_dir / "decks" / "pickups" / "flat.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"x": grid, "y": grid, "buttons": values}))
    return path


def _build(facility_dir: Path) -> dict[str, Any]:
    return build_facility(facility_dir, project_name="turn-by-turn", extra_checks=FRAMEWORK_CHECKS)


def test_a_scenario_states_the_button_gains_of_a_monitor_with_a_pickup(tmp_path: Path) -> None:
    faults = {RING: {SUM_1: {"button_1": 0.95, "button_3": 0.0, "gain": 1.1}}}

    document = _build(_tree(tmp_path, faults, pickup=ROUND))

    (scenario,) = [s for s in document["scenarios"] if s["name"] == "uneven-buttons"]
    assert scenario["faults"] == faults


def test_a_pickup_of_single_monitors_gives_buttons_to_those_alone(tmp_path: Path) -> None:
    pickups = [{"elements": ["BPM02"], **ROUND}]
    _build(_tree(tmp_path, {RING: {SUM_2: {"button_4": 1.02}}}, pickups=pickups))

    with pytest.raises(FacilityBuildError) as caught:
        _build(_tree(tmp_path / "other", {RING: {SUM_1: {"button_4": 1.02}}}, pickups=pickups))

    text = str(caught.value)
    assert f"faults.{RING}.{SUM_1}.button_4 is not a fault field of {SUM_1}" in text


def test_a_monitor_with_no_pickup_carries_no_button_gain(tmp_path: Path) -> None:
    with pytest.raises(FacilityBuildError) as caught:
        _build(_tree(tmp_path, {RING: {SUM_1: {"button_1": 0.95}}}))

    text = str(caught.value)
    assert f"faults.{RING}.{SUM_1}.button_1 is not a fault field of {SUM_1}" in text


def test_a_button_the_pickup_does_not_have_is_no_fault_field(tmp_path: Path) -> None:
    with pytest.raises(FacilityBuildError) as caught:
        _build(_tree(tmp_path, {RING: {SUM_1: {"button_5": 0.95}}}, pickup=ROUND))

    text = str(caught.value)
    assert f"faults.{RING}.{SUM_1}.button_5 is not a fault field of {SUM_1}" in text


def test_a_position_array_carries_no_button_gain(tmp_path: Path) -> None:
    turn_x = TURN_X["BPM01"]
    with pytest.raises(FacilityBuildError) as caught:
        _build(_tree(tmp_path, {RING: {turn_x: {"button_1": 0.95}}}, pickup=ROUND))

    assert f"faults.{RING}.{turn_x}.button_1 is not a fault field of {turn_x}" in str(caught.value)


def test_a_button_gain_off_its_range_stops_the_build(tmp_path: Path) -> None:
    with pytest.raises(FacilityBuildError) as caught:
        _build(_tree(tmp_path, {RING: {SUM_1: {"button_1": -0.1}}}, pickup=ROUND))

    assert f"faults.{RING}.{SUM_1}.button_1" in str(caught.value)
    assert "below its range" in str(caught.value)


def test_a_table_beside_the_deck_builds(tmp_path: Path) -> None:
    facility_dir = _tree(tmp_path, {RING: {SUM_1: {"button_2": 0.9}}}, pickup=TABLE)
    _table(facility_dir)

    document = _build(facility_dir)

    assert [s["name"] for s in document["scenarios"]].count("uneven-buttons") == 1


def test_a_missing_table_stops_the_build_with_its_resolved_path(tmp_path: Path) -> None:
    facility_dir = _tree(tmp_path, pickup=TABLE)

    with pytest.raises(FacilityBuildError) as caught:
        _build(facility_dir)

    text = str(caught.value)
    assert "pyat.turn_by_turn.pickup.table pickups/flat.json" in text
    assert str((facility_dir / "decks" / "pickups" / "flat.json").resolve()) in text
    assert "cannot be read" in text


def test_a_table_of_the_wrong_shape_stops_the_build(tmp_path: Path) -> None:
    facility_dir = _tree(tmp_path, pickup=TABLE)
    _table(facility_dir, buttons=3)

    with pytest.raises(FacilityBuildError) as caught:
        _build(facility_dir)

    assert "buttons has shape [3, 5, 5]" in str(caught.value)


@pytest.mark.parametrize(
    ("change", "says"),
    [
        ({"radius": 0}, "pickup.radius is 0"),
        ({"x": {"plus": [1, 5], "minus": [2, 3]}}, "pickup.x.plus names button 5"),
        ({"y": {"plus": [1, 2]}}, "pickup.y.minus is None"),
        ({"diameter": 0.006}, "both of half_width and diameter"),
        ({"shape": "oval"}, "pickup.shape is 'oval'"),
    ],
    ids=lambda value: str(value)[:40],
)
def test_a_malformed_pickup_stops_the_build(tmp_path: Path, change: Any, says: str) -> None:
    with pytest.raises(FacilityBuildError) as caught:
        _build(_tree(tmp_path, pickup={**ROUND, **change}))

    assert says in str(caught.value)
