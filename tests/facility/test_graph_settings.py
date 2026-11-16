"""The graph view's two settings: ``facility_graph.claims`` and ``facility_graph.basis``.

Both default to ``all``. The graph ``View`` row reads exactly these two keys, the
writer parses them with :func:`graph_settings` and hands them to
:func:`graph_text`, and an unknown value stops the render before anything is
written, naming the dotted key and the values it allows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from osprey.facility.errors import FacilityBuildError
from osprey.facility.records import FacilityRecords
from osprey.facility.validate import FACILITY_HEADER
from osprey.facility.views import VIEWS, ViewInputs, config_read
from osprey.facility.views import graph as graph_module
from osprey.facility.views.graph import (
    GRAPH_FILE,
    GraphSettings,
    graph_settings,
    graph_text,
    write_graph_view,
)

CLAIMS_KEY = "facility_graph.claims"
BASIS_KEY = "facility_graph.basis"


def _records() -> FacilityRecords:
    return FacilityRecords(
        {
            "schema": FACILITY_HEADER,
            "identity": {"code": "fx"},
            "places": [{"id": "M", "level": "machine"}],
            "devices": [],
            "channels": [],
        }
    )


def _inputs(config: dict[str, Any], root: Path) -> ViewInputs:
    return ViewInputs(records=_records(), rendered_config=config, facility_dir=root, served=[])


def test_the_defaults_are_all_and_all() -> None:
    assert GraphSettings() == GraphSettings(claims="all", basis="all")
    assert graph_settings({}) == GraphSettings("all", "all")
    assert graph_settings({"facility_graph": {}}) == GraphSettings("all", "all")


def test_a_key_left_empty_reads_as_its_default() -> None:
    assert graph_settings({"facility_graph": {"claims": None, "basis": None}}) == GraphSettings()


@pytest.mark.parametrize("claims", ["all", "contested", "none"])
@pytest.mark.parametrize("basis", ["all", "stated"])
def test_every_valid_value_is_taken(claims: str, basis: str) -> None:
    config = {"facility_graph": {"claims": claims, "basis": basis}}

    assert graph_settings(config) == GraphSettings(claims=claims, basis=basis)


def test_one_key_stated_leaves_the_other_at_its_default() -> None:
    assert graph_settings({"facility_graph": {"basis": "stated"}}) == GraphSettings("all", "stated")
    assert graph_settings({"facility_graph": {"claims": "none"}}) == GraphSettings("none", "all")


@pytest.mark.parametrize(
    ("config", "key", "allowed"),
    [
        ({"facility_graph": {"claims": "some"}}, CLAIMS_KEY, ("all", "contested", "none")),
        ({"facility_graph": {"basis": "inferred"}}, BASIS_KEY, ("all", "stated")),
        ({"facility_graph": {"claims": True}}, CLAIMS_KEY, ("all", "contested", "none")),
        ({"facility_graph": {"basis": ["stated"]}}, BASIS_KEY, ("all", "stated")),
    ],
)
def test_an_unknown_value_names_the_key_and_the_allowed_values(
    config: dict[str, Any], key: str, allowed: tuple[str, ...]
) -> None:
    with pytest.raises(FacilityBuildError) as caught:
        graph_settings(config)

    error = caught.value
    assert error.kind == "profile-invalid"
    assert error.record_id == key
    assert key in str(error)
    for value in allowed:
        assert f"`{value}`" in str(error)


def test_a_facility_graph_that_is_not_a_mapping_is_refused() -> None:
    with pytest.raises(FacilityBuildError) as caught:
        graph_settings({"facility_graph": "stated"})

    assert caught.value.kind == "profile-invalid"
    assert "facility_graph" in str(caught.value)


def test_an_unknown_value_stops_the_render_before_anything_is_written(tmp_path: Path) -> None:
    root = tmp_path / "graph"

    with pytest.raises(FacilityBuildError, match=r"facility_graph\.claims"):
        write_graph_view(root, _inputs({"facility_graph": {"claims": "bogus"}}, tmp_path))

    assert not (root / GRAPH_FILE).exists()


def test_the_writer_hands_the_parsed_settings_to_graph_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[GraphSettings] = []
    real = graph_module.graph_text

    def spy(records: FacilityRecords, settings: GraphSettings = GraphSettings()) -> str:
        seen.append(settings)
        return real(records, settings)

    monkeypatch.setattr(graph_module, "graph_text", spy)
    config = {"facility_graph": {"claims": "contested", "basis": "stated"}}

    write_graph_view(tmp_path / "graph", _inputs(config, tmp_path))

    assert seen == [GraphSettings("contested", "stated")]


def test_the_memo_is_keyed_on_the_records_and_the_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    real = graph_module._turtle

    def counting(*args: Any) -> str:
        calls.append(args)
        return real(*args)

    monkeypatch.setattr(graph_module, "_turtle", counting)
    monkeypatch.setattr(graph_module, "_rendered", None)
    records = _records()

    first = graph_text(records)
    assert graph_text(records, GraphSettings()) is first
    assert len(calls) == 1

    graph_text(records, GraphSettings("none", "stated"))
    assert len(calls) == 2

    graph_text(records, GraphSettings("none", "stated"))
    assert len(calls) == 2


def test_the_graph_row_reads_the_two_settings() -> None:
    (view,) = [view for view in VIEWS if view.name == "graph"]

    assert view.reads == (CLAIMS_KEY, BASIS_KEY)


def test_the_row_hands_the_writer_only_its_settings() -> None:
    (view,) = [view for view in VIEWS if view.name == "graph"]
    config = {
        "facility_graph": {"claims": "none", "basis": "stated", "other": 1},
        "channel_finder": {"pipeline_mode": "hierarchical"},
    }

    narrowed = config_read(config, view.reads)

    assert narrowed == {"facility_graph": {"claims": "none", "basis": "stated"}}
    assert graph_settings(narrowed) == GraphSettings("none", "stated")
