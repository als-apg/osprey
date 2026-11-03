"""The build states the profile keys whose value is a fact of the facility tree.

A facility that ships ``ariel/vocabulary.yml`` turns ARIEL's shorthand
expansion on at that file; one that ships benchmark query sources gets the
benchmark dataset path the benchmarks view writes to. A stated value is the
profile's own, and a stated value the tree cannot back is refused.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from osprey.facility.errors import FacilityBuildError
from osprey.facility.served import (
    BENCHMARK_DATASET_PATH,
    VOCABULARY_ENABLED_KEY,
    VOCABULARY_PATH_KEY,
    VOCABULARY_SOURCE,
    facility_tree_fill,
)
from osprey.facility.views.benchmarks import BENCHMARK_DATASET_KEY, QUERY_SOURCES


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
    assert facility_tree_fill(_tree(tmp_path), {}) == {}


def test_a_missing_tree_states_nothing(tmp_path: Path) -> None:
    assert facility_tree_fill(tmp_path / "facility", {}) == {}


def test_a_shipped_vocabulary_turns_expansion_on_at_its_file(tmp_path: Path) -> None:
    fill = facility_tree_fill(_tree(tmp_path, vocabulary=True), {})

    assert fill == {VOCABULARY_ENABLED_KEY: True, VOCABULARY_PATH_KEY: VOCABULARY_SOURCE}
    assert (tmp_path / VOCABULARY_SOURCE.removeprefix("data/")).is_file()


def test_a_stated_false_turns_the_vocabulary_off(tmp_path: Path) -> None:
    fill = facility_tree_fill(_tree(tmp_path, vocabulary=True), {VOCABULARY_ENABLED_KEY: False})

    assert fill == {}


def test_a_stated_path_is_the_profiles_own(tmp_path: Path) -> None:
    profile = {"ariel": {"vocabulary": {"path": "data/other.yml"}}}

    fill = facility_tree_fill(_tree(tmp_path, vocabulary=True), profile)

    assert fill == {VOCABULARY_ENABLED_KEY: True}


def test_a_stated_true_without_the_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FacilityBuildError) as stop:
        facility_tree_fill(_tree(tmp_path), {VOCABULARY_ENABLED_KEY: True})

    assert stop.value.kind == "profile-invalid"
    assert stop.value.record_id == VOCABULARY_ENABLED_KEY
    assert VOCABULARY_SOURCE in stop.value.detail


def test_shipped_query_sources_state_the_dataset_path(tmp_path: Path) -> None:
    fill = facility_tree_fill(_tree(tmp_path, queries=True), {})

    assert fill == {BENCHMARK_DATASET_KEY: BENCHMARK_DATASET_PATH}


def test_a_stated_dataset_path_with_sources_is_the_location(tmp_path: Path) -> None:
    profile: dict[str, Any] = {BENCHMARK_DATASET_KEY: "data/queries/mine.json"}

    assert facility_tree_fill(_tree(tmp_path, queries=True), profile) == {}


def test_a_stated_dataset_path_without_sources_is_refused(tmp_path: Path) -> None:
    profile: dict[str, Any] = {BENCHMARK_DATASET_KEY: BENCHMARK_DATASET_PATH}

    with pytest.raises(FacilityBuildError) as stop:
        facility_tree_fill(_tree(tmp_path), profile)

    assert stop.value.kind == "profile-invalid"
    assert stop.value.record_id == BENCHMARK_DATASET_KEY
    assert QUERY_SOURCES in stop.value.detail
