"""The build derives the profile settings whose value is a fact of the facility.

A facility that ships ``ariel/vocabulary.yml`` turns ARIEL's shorthand
expansion on at that file; one that ships benchmark query sources gets the
benchmark dataset path the benchmarks view writes to. A stated value is the
profile's own, and a stated value the facility cannot back is refused: a served
probe or a freshness channel the file does not hold, a start set naming a
scenario it does not hold.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from osprey.cli.build_profile_archiver import VAArchiverConfig
from osprey.cli.build_profile_load import manifest_rows
from osprey.facility.errors import FacilityBuildError
from osprey.facility.records import FacilityRecords
from osprey.facility.settings import (
    BENCHMARK_DATASET_PATH,
    FACILITY_SETTINGS,
    FRESHNESS_CHANNEL_KEY,
    PROFILE_SOURCE,
    SERVED_PROBE_KEYS,
    VOCABULARY_ENABLED_KEY,
    VOCABULARY_PATH_KEY,
    VOCABULARY_SOURCE,
    fill_facility_settings,
)
from osprey.facility.views.benchmarks import BENCHMARK_DATASET_KEY, QUERY_SOURCES
from osprey.simulation import DEFAULT_SCENARIOS_KEY

#: The architecture page that lists the table.
FACILITY_FILE_PAGE = (
    Path(__file__).resolve().parents[2] / "docs" / "source" / "architecture" / "facility-file.rst"
)

#: A ``list-table`` row's first cell when it is one RST inline literal.
KEY_CELL = re.compile(r"^   \* - ``([^`]+)``$")

#: A facility file with one readback and one setpoint.
DOCUMENT: dict[str, Any] = {
    "channels": [{"id": "B:SP", "role": "setpoint"}, {"id": "A:RB"}],
    "scenarios": [{"name": "warm"}, {"name": "cold"}],
}


def _fill(
    facility_dir: Path,
    profile: dict[str, Any],
    document: dict[str, Any] | None = None,
    va_archiver: VAArchiverConfig | None = None,
) -> dict[str, Any]:
    """Run the table with the facility tree's parent as the project root."""
    return fill_facility_settings(
        FacilityRecords(document or {}),
        facility_dir,
        facility_dir.parent,
        {},
        profile,
        va_archiver,
    )


def _tree(root: Path, *, vocabulary: bool = False, queries: bool = False) -> Path:
    facility_dir = root / "facility"
    facility_dir.mkdir(parents=True)
    if vocabulary:
        path = facility_dir / "ariel" / "vocabulary.yml"
        path.parent.mkdir()
        path.write_text("concepts: []\n")
    if queries:
        sources = facility_dir / QUERY_SOURCES
        sources.mkdir(parents=True)
        (sources / "tree_queries.json").write_text("[]\n")
    return facility_dir


def test_a_tree_with_neither_fact_states_nothing(tmp_path: Path) -> None:
    assert _fill(_tree(tmp_path), {}) == {}


def test_a_missing_tree_states_nothing(tmp_path: Path) -> None:
    assert _fill(tmp_path / "facility", {}) == {}


def test_a_shipped_vocabulary_turns_expansion_on_at_its_file(tmp_path: Path) -> None:
    fill = _fill(_tree(tmp_path, vocabulary=True), {})

    assert fill == {VOCABULARY_ENABLED_KEY: True, VOCABULARY_PATH_KEY: VOCABULARY_SOURCE}
    assert (tmp_path / VOCABULARY_SOURCE.removeprefix("data/")).is_file()


def test_a_stated_false_turns_the_vocabulary_off(tmp_path: Path) -> None:
    fill = _fill(_tree(tmp_path, vocabulary=True), {VOCABULARY_ENABLED_KEY: False})

    assert fill == {}


def test_a_stated_path_is_the_profiles_own(tmp_path: Path) -> None:
    profile = {"ariel": {"vocabulary": {"path": "data/other.yml"}}}
    facility_dir = _tree(tmp_path, vocabulary=True)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "other.yml").write_text("concepts: []\n")

    fill = _fill(facility_dir, profile)

    assert fill == {VOCABULARY_ENABLED_KEY: True}


def test_a_stated_true_without_the_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FacilityBuildError) as stop:
        _fill(_tree(tmp_path), {VOCABULARY_ENABLED_KEY: True})

    assert stop.value.kind == "profile-invalid"
    assert stop.value.record_id == VOCABULARY_ENABLED_KEY
    assert VOCABULARY_SOURCE in stop.value.detail


def test_shipped_query_sources_state_the_dataset_path(tmp_path: Path) -> None:
    fill = _fill(_tree(tmp_path, queries=True), {})

    assert fill == {BENCHMARK_DATASET_KEY: BENCHMARK_DATASET_PATH}


def test_a_stated_dataset_path_with_sources_is_the_location(tmp_path: Path) -> None:
    profile: dict[str, Any] = {BENCHMARK_DATASET_KEY: "data/queries/mine.json"}

    assert _fill(_tree(tmp_path, queries=True), profile) == {}


def test_a_stated_dataset_path_without_sources_is_refused(tmp_path: Path) -> None:
    profile: dict[str, Any] = {BENCHMARK_DATASET_KEY: BENCHMARK_DATASET_PATH}

    with pytest.raises(FacilityBuildError) as stop:
        _fill(_tree(tmp_path), profile)

    assert stop.value.kind == "profile-invalid"
    assert stop.value.record_id == BENCHMARK_DATASET_KEY
    assert QUERY_SOURCES in stop.value.detail


@pytest.mark.parametrize("key", SERVED_PROBE_KEYS)
def test_a_stated_probe_the_file_does_not_hold_is_refused(tmp_path: Path, key: str) -> None:
    with pytest.raises(FacilityBuildError) as stop:
        _fill(_tree(tmp_path), {key: "NOPE:RB"}, DOCUMENT)

    assert stop.value.kind == "profile-invalid"
    assert stop.value.record_id == key
    assert stop.value.sources == (PROFILE_SOURCE,)
    assert "`NOPE:RB` is not a channel of the facility file" in stop.value.detail
    assert "`A:RB`" in stop.value.remedy


@pytest.mark.parametrize("key", SERVED_PROBE_KEYS)
def test_a_stated_probe_the_file_holds_passes(tmp_path: Path, key: str) -> None:
    assert _fill(_tree(tmp_path), {key: "B:SP"}, DOCUMENT) == {}


def test_a_stated_freshness_channel_the_file_does_not_hold_is_refused(tmp_path: Path) -> None:
    archiver = VAArchiverConfig(freshness_channel="NOPE:RB")

    with pytest.raises(FacilityBuildError) as stop:
        _fill(_tree(tmp_path), {}, DOCUMENT, archiver)

    assert stop.value.kind == "profile-invalid"
    assert stop.value.record_id == FRESHNESS_CHANNEL_KEY
    assert "`A:RB`" in stop.value.remedy


def test_a_start_set_naming_a_scenario_the_file_does_not_hold_is_refused(
    tmp_path: Path,
) -> None:
    with pytest.raises(FacilityBuildError) as stop:
        _fill(_tree(tmp_path), {DEFAULT_SCENARIOS_KEY: ["warm", "hot"]}, DOCUMENT)

    assert stop.value.kind == "profile-invalid"
    assert stop.value.record_id == DEFAULT_SCENARIOS_KEY
    assert "`hot`" in stop.value.detail
    assert "`cold`, `warm`" in stop.value.remedy


def test_a_start_set_of_the_files_scenarios_is_never_derived(tmp_path: Path) -> None:
    assert _fill(_tree(tmp_path), {DEFAULT_SCENARIOS_KEY: ["cold"]}, DOCUMENT) == {}
    assert DEFAULT_SCENARIOS_KEY not in _fill(_tree(tmp_path / "x"), {}, DOCUMENT)


def test_a_stated_vocabulary_path_that_is_not_a_file_is_refused(tmp_path: Path) -> None:
    profile = {VOCABULARY_PATH_KEY: "data/missing.yml"}

    with pytest.raises(FacilityBuildError) as stop:
        _fill(_tree(tmp_path, vocabulary=True), profile)

    assert stop.value.kind == "profile-invalid"
    assert stop.value.sources == (PROFILE_SOURCE,)
    assert str(stop.value) == (
        "facility: profile-invalid: path ariel.vocabulary.path — `data/missing.yml` is not a "
        "file; fix: set `ariel.vocabulary.path` to a vocabulary file, or remove it"
    )


def test_a_stated_vocabulary_path_that_is_not_a_string_is_refused(tmp_path: Path) -> None:
    profile = {VOCABULARY_PATH_KEY: 7}

    with pytest.raises(FacilityBuildError) as stop:
        _fill(_tree(tmp_path, vocabulary=True), profile)

    assert stop.value.kind == "profile-invalid"
    assert stop.value.record_id == VOCABULARY_PATH_KEY
    assert stop.value.sources == (PROFILE_SOURCE,)


def test_a_stated_true_at_a_path_that_is_not_a_file_is_refused(tmp_path: Path) -> None:
    profile = {VOCABULARY_ENABLED_KEY: True, VOCABULARY_PATH_KEY: str(tmp_path / "gone.yml")}

    with pytest.raises(FacilityBuildError) as stop:
        _fill(_tree(tmp_path), profile)

    assert stop.value.record_id == VOCABULARY_PATH_KEY


def test_a_stated_path_resolves_against_the_project_root(tmp_path: Path) -> None:
    facility_dir = _tree(tmp_path)
    (tmp_path / "vocab.yml").write_text("concepts: []\n")
    profile = {VOCABULARY_ENABLED_KEY: True, VOCABULARY_PATH_KEY: "vocab.yml"}

    assert _fill(facility_dir, profile) == {}


def test_a_stated_path_with_the_vocabulary_off_is_inert(tmp_path: Path) -> None:
    profile = {VOCABULARY_ENABLED_KEY: False, VOCABULARY_PATH_KEY: "data/missing.yml"}

    assert _fill(_tree(tmp_path, vocabulary=True), profile) == {}


def test_the_table_is_the_manifests_facility_rows() -> None:
    marked = {key for key, row in manifest_rows().items() if (row or {}).get("facility-fact")}

    assert {row.key for row in FACILITY_SETTINGS} == marked


def _section_keys(page: Path, heading: str) -> list[str]:
    """The key cells of the list-table under ``heading``, up to the next ``=`` heading."""
    lines = page.read_text(encoding="utf-8").splitlines()
    start = lines.index(heading) + 2
    keys: list[str] = []
    for n in range(start, len(lines)):
        if n + 1 < len(lines) and lines[n + 1].startswith("===") and lines[n].strip():
            break
        if match := KEY_CELL.match(lines[n]):
            keys.append(match.group(1))
    return keys


def test_the_facility_file_page_lists_the_table() -> None:
    keys = _section_keys(FACILITY_FILE_PAGE, "Settings the facility decides")

    assert keys == [row.key for row in FACILITY_SETTINGS]
