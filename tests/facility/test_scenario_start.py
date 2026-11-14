"""A scenario marked ``start: true`` is in the deployment's start set.

The build derives ``simulation.default_scenarios`` from the marks: the sorted
names of the marked scenarios, nothing when none is marked. A profile that
states a start set beside a mark is refused, so the marks are the one producer;
a dotted ``null`` cancels a value a preset gives. The mark is a fact for the
build alone: ``scenarios.json`` never carries it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from osprey.facility.errors import FacilityBuildError
from osprey.facility.records import FacilityRecords
from osprey.facility.schema._generated.core import Scenario
from osprey.facility.settings import fill_facility_settings
from osprey.facility.sources import slot_names
from osprey.facility.views.simulator import _scenarios_document
from osprey.simulation import DEFAULT_SCENARIOS_KEY

CHANNELS: list[dict[str, Any]] = [{"id": "B:SP", "role": "setpoint"}, {"id": "A:RB"}]


def _document(*scenarios: dict[str, Any]) -> dict[str, Any]:
    return {"channels": CHANNELS, "scenarios": list(scenarios)}


def _fill(tmp_path: Path, profile: dict[str, Any], document: dict[str, Any]) -> dict[str, Any]:
    facility_dir = tmp_path / "facility"
    facility_dir.mkdir(exist_ok=True)
    return fill_facility_settings(
        FacilityRecords(document), facility_dir, tmp_path, {}, profile, None
    )


MARKED = _document(
    {"name": "warm", "start": True},
    {"name": "cold"},
    {"name": "beam", "start": True},
)


# --- the schema -------------------------------------------------------------------


def test_start_is_a_scenario_slot_of_the_schema_and_the_model() -> None:
    assert "start" in slot_names("Scenario")
    assert Scenario(name="warm", start=True).start is True
    assert Scenario(name="warm").start is None


# --- the derived set ----------------------------------------------------------------


def test_the_marked_scenarios_are_the_derived_start_set_sorted(tmp_path: Path) -> None:
    assert _fill(tmp_path, {}, MARKED)[DEFAULT_SCENARIOS_KEY] == ["beam", "warm"]


def test_no_mark_derives_nothing(tmp_path: Path) -> None:
    document = _document({"name": "warm"}, {"name": "cold"})
    assert DEFAULT_SCENARIOS_KEY not in _fill(tmp_path, {}, document)


def test_no_scenarios_derives_nothing(tmp_path: Path) -> None:
    assert DEFAULT_SCENARIOS_KEY not in _fill(tmp_path, {}, {"channels": CHANNELS})


def test_start_false_is_no_mark(tmp_path: Path) -> None:
    document = _document({"name": "warm", "start": False}, {"name": "cold"})
    assert DEFAULT_SCENARIOS_KEY not in _fill(tmp_path, {}, document)
    # A stated value is then checked as it is without a mark.
    assert _fill(tmp_path, {DEFAULT_SCENARIOS_KEY: ["cold"]}, document) == {}
    with pytest.raises(FacilityBuildError) as stop:
        _fill(tmp_path, {DEFAULT_SCENARIOS_KEY: ["hot"]}, document)
    assert "`hot`" in stop.value.detail


def test_only_a_true_mark_counts(tmp_path: Path) -> None:
    document = _document({"name": "warm", "start": True}, {"name": "cold", "start": False})
    assert _fill(tmp_path, {}, document)[DEFAULT_SCENARIOS_KEY] == ["warm"]


# --- a stated value without a mark ------------------------------------------------


def test_without_a_mark_a_stated_value_is_checked_as_today(tmp_path: Path) -> None:
    document = _document({"name": "warm"}, {"name": "cold"})
    assert _fill(tmp_path, {DEFAULT_SCENARIOS_KEY: ["cold"]}, document) == {}
    assert _fill(tmp_path, {DEFAULT_SCENARIOS_KEY: []}, document) == {}
    with pytest.raises(FacilityBuildError) as stop:
        _fill(tmp_path, {DEFAULT_SCENARIOS_KEY: ["warm", "hot"]}, document)
    assert stop.value.kind == "profile-invalid"
    assert stop.value.record_id == DEFAULT_SCENARIOS_KEY
    assert "`hot`" in stop.value.detail


# --- a stated value beside a mark -------------------------------------------------


@pytest.mark.parametrize(
    "profile",
    [
        {DEFAULT_SCENARIOS_KEY: ["warm"]},
        {DEFAULT_SCENARIOS_KEY: ["cold"]},
        {DEFAULT_SCENARIOS_KEY: []},
        {"simulation": {"default_scenarios": ["warm"]}},
    ],
    ids=["the-marked-set", "an-unmarked-name", "empty", "nested"],
)
def test_a_stated_value_beside_a_mark_is_refused(tmp_path: Path, profile: dict[str, Any]) -> None:
    with pytest.raises(FacilityBuildError) as stop:
        _fill(tmp_path, profile, MARKED)

    assert stop.value.kind == "profile-invalid"
    assert stop.value.record_id == DEFAULT_SCENARIOS_KEY
    assert "`beam`, `warm`" in stop.value.detail
    assert "start: true" in stop.value.detail
    remedy = stop.value.remedy
    assert f"remove `{DEFAULT_SCENARIOS_KEY}`" in remedy
    assert f"`{DEFAULT_SCENARIOS_KEY}: null`" in remedy


@pytest.mark.parametrize(
    "profile",
    [
        {DEFAULT_SCENARIOS_KEY: None},
        {"simulation": {"default_scenarios": None}},
    ],
    ids=["dotted", "nested"],
)
def test_a_null_over_a_preset_value_passes_and_the_marks_win(
    tmp_path: Path, profile: dict[str, Any]
) -> None:
    assert _fill(tmp_path, profile, MARKED)[DEFAULT_SCENARIOS_KEY] == ["beam", "warm"]


# --- scenarios.json -----------------------------------------------------------------


def test_scenarios_json_carries_no_start_key() -> None:
    document = _document(
        {"name": "warm", "description": "Warm.", "start": True},
        {"name": "cold", "start": False},
    )
    inputs = SimpleNamespace(records=FacilityRecords(document), served=[])
    scenarios = _scenarios_document(inputs)["scenarios"]  # type: ignore[arg-type]

    assert {entry["name"] for entry in scenarios} >= {"warm", "cold"}
    assert all("start" not in entry for entry in scenarios)
    warm = next(entry for entry in scenarios if entry["name"] == "warm")
    assert warm["description"] == "Warm."
