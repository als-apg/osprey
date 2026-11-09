"""The benchmarks view: the channel-finder benchmark query file a render carries.

The facility tree ships its query sources under
``data/facility/benchmarks/cross_paradigm/queries/``: ``in_context_queries.json``
for the in_context pipeline and ``tree_queries.json`` for every other mode. A
render whose ``config.yml`` states ``channel_finder.benchmark.dataset_path``
carries the source its ``channel_finder.pipeline_mode`` selects, byte-equal, at
that path.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from osprey.build.modes import VALID_CHANNEL_FINDER_MODES
from osprey.facility import views
from osprey.facility.errors import FacilityBuildError
from osprey.facility.records import FacilityRecords
from osprey.facility.render import render_facility_outputs
from osprey.facility.views.benchmarks import (
    BENCHMARK_DATASET_KEY,
    QUERY_SOURCES,
    benchmarks_wanted,
    write_benchmarks_view,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
QUERIES_DIR = (
    REPO_ROOT / "src/osprey/templates/facilities/example/benchmarks/cross_paradigm/queries"
)
DATASET = "data/benchmarks/queries.json"

EXPECTED_SOURCE = {
    "in_context": "in_context_queries.json",
    "hierarchical": "tree_queries.json",
    "middle_layer": "tree_queries.json",
    "graph": "tree_queries.json",
}


def _config(mode: str | None, dataset_path: str | None = DATASET) -> dict[str, Any]:
    channel_finder: dict[str, Any] = {}
    if mode is not None:
        channel_finder["pipeline_mode"] = mode
    if dataset_path is not None:
        channel_finder["benchmark"] = {"dataset_path": dataset_path}
    return {"channel_finder": channel_finder}


def _inputs(facility_dir: Path, config: dict[str, Any]) -> views.ViewInputs:
    return views.ViewInputs(
        records=FacilityRecords({"channels": []}),
        rendered_config=config,
        facility_dir=facility_dir,
        served=[],
    )


def _facility_with_sources(root: Path) -> Path:
    facility_dir = root / "facility"
    shutil.copytree(QUERIES_DIR, facility_dir / QUERY_SOURCES)
    return facility_dir


def test_shipped_query_files_are_named_by_pipeline() -> None:
    assert sorted(p.name for p in QUERIES_DIR.glob("*.json")) == [
        "in_context_queries.json",
        "tree_queries.json",
    ]


def test_every_mode_has_an_expected_source() -> None:
    assert set(EXPECTED_SOURCE) == set(VALID_CHANNEL_FINDER_MODES)


@pytest.mark.parametrize("mode", sorted(EXPECTED_SOURCE))
def test_the_render_carries_the_modes_query_file(tmp_path: Path, mode: str) -> None:
    facility_dir = _facility_with_sources(tmp_path)
    render_dir = tmp_path / "render"
    inputs = _inputs(facility_dir, _config(mode))

    assert benchmarks_wanted(inputs)[0]
    written = write_benchmarks_view(render_dir / "data", inputs)

    target = render_dir / DATASET
    assert written == [target]
    assert target.read_bytes() == (QUERIES_DIR / EXPECTED_SOURCE[mode]).read_bytes()


def test_an_unstated_dataset_path_is_not_wanted_and_names_the_key(tmp_path: Path) -> None:
    facility_dir = _facility_with_sources(tmp_path)

    wanted, reason = benchmarks_wanted(_inputs(facility_dir, _config("graph", None)))

    assert not wanted
    assert reason == BENCHMARK_DATASET_KEY


def test_an_unstated_dataset_path_answers_without_reading_the_tree() -> None:
    assert benchmarks_wanted(_inputs(Path(), {})) == (False, BENCHMARK_DATASET_KEY)


def test_a_facility_without_query_sources_is_not_wanted(tmp_path: Path) -> None:
    facility_dir = tmp_path / "facility"
    facility_dir.mkdir()

    wanted, reason = benchmarks_wanted(_inputs(facility_dir, _config("graph")))

    assert not wanted
    assert reason == f"data/facility/{QUERY_SOURCES}"


def test_a_missing_mode_file_stops_before_anything_is_written(tmp_path: Path) -> None:
    facility_dir = tmp_path / "facility"
    sources = facility_dir / QUERY_SOURCES
    sources.mkdir(parents=True)
    (sources / "tree_queries.json").write_text("[]\n")
    render_dir = tmp_path / "render"

    with pytest.raises(FacilityBuildError) as stop:
        write_benchmarks_view(render_dir / "data", _inputs(facility_dir, _config("in_context")))

    assert stop.value.kind == "view-unsupported"
    assert stop.value.record_id == views.PIPELINE_MODE_KEY
    assert "in_context_queries.json" in stop.value.detail
    assert not render_dir.exists()


@pytest.mark.parametrize("dataset_path", ["/abs/queries.json", "data/../queries.json", "q.json"])
def test_a_dataset_path_outside_the_renders_data_is_refused(
    tmp_path: Path, dataset_path: str
) -> None:
    facility_dir = _facility_with_sources(tmp_path)
    render_dir = tmp_path / "render"

    with pytest.raises(FacilityBuildError) as stop:
        write_benchmarks_view(
            render_dir / "data", _inputs(facility_dir, _config("graph", dataset_path))
        )

    assert stop.value.kind == "profile-invalid"
    assert stop.value.record_id == BENCHMARK_DATASET_KEY
    assert not render_dir.exists()


def test_two_renders_that_differ_in_mode_each_carry_their_own_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (view,) = [view for view in views.VIEWS if view.name == "benchmarks"]
    monkeypatch.setattr(views, "VIEWS", (view,))
    facility_dir = _facility_with_sources(tmp_path)
    views_written: dict[Any, list[tuple[str, bytes]]] = {}

    for mode in ("in_context", "graph"):
        render_dir = tmp_path / mode
        render_dir.mkdir()
        render_facility_outputs(
            render_dir,
            {"channels": []},
            _config(mode),
            facility_dir,
            views_written=views_written,
        )

    for mode in ("in_context", "graph"):
        carried = (tmp_path / mode / DATASET).read_bytes()
        assert carried == (QUERIES_DIR / EXPECTED_SOURCE[mode]).read_bytes()
