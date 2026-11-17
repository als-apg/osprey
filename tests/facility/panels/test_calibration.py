"""Calibration views: a kind over a set, its links, its looks."""

from __future__ import annotations

from osprey.facility.panels import layout, symbols, targets
from osprey.facility.panels.describe import (
    calibration_overview,
    device_card,
    set_overview,
)
from osprey.facility.panels.view import Calibrations, Fact, Grid, Link, Target
from tests.facility.panels._views import nodes, stub_render
from tests.facility.panels.test_symbols import _resolved, contrast


def _cells(view):
    (grid,) = [i for s in view.sections for i in s.items if isinstance(i, Grid)]
    return {c.member.target.id: c.calibration for row in grid.rows for c in row.cells}


def test_a_set_gets_a_calibration_view_per_kind_of_its_class(panel_graph) -> None:
    found = [t for t in targets(panel_graph) if t.view == "calibration"]
    assert [(t.id, t.system) for t in found] == [
        ("XR-QUAD", kind.kind) for kind in panel_graph.calibration_kinds("Quadrupole")
    ]
    assert not [t for t in found if t.id.endswith("-VAC")]  # pumps: no kind


def test_a_cell_says_the_deviation_from_design_or_the_state(panel_graph) -> None:
    cells = _cells(calibration_overview(panel_graph, "XR-QUAD", "integrated_gradient"))
    assert (cells["XR01:QF1"].text, cells["XR01:QF1"].state) == ("+0.21 %", "bench")
    assert (cells["XR01:QD1"].text, cells["XR01:QD1"].state) == ("design", "design")
    assert cells["XR02:QF1"].state == "contested"
    assert cells["XR02:QF1"].text == "+0.01 % / +1.24 %"  # in force, then the other claim
    assert (cells["XR02:QD1"].text, cells["XR02:QD1"].state) == ("no value", "none")
    centre = _cells(calibration_overview(panel_graph, "XR-QUAD", "centre"))
    assert centre["XR01:QF1"].text == "-2.5 · -14.5 um"  # no design value: the value
    live = _cells(calibration_overview(panel_graph, "XR-QUAD", "current_offset"))
    assert live["XR01:QF1"].channels == ("XR01-QF1:Current-SP",)


def test_the_set_overview_and_the_card_link_to_the_calibration_views(panel_graph) -> None:
    view = set_overview(panel_graph, "XR-QUAD")
    (section,) = [s for s in view.sections if s.title == "calibration"]
    assert section.items[0] == Link(
        "calibration: Integrated gradient",
        Target("calibration", "group", "XR-QUAD", system="integrated_gradient"),
    )
    (block,) = [
        s for s in device_card(panel_graph, "XR01:QF1").sections if s.title == "calibration"
    ]
    row = block.items[0].rows[0]
    assert row.link is not None
    assert row.link.target == Target(
        "calibration", "group", "XR-QUAD", system="integrated_gradient"
    )


def test_a_calibration_view_tints_cells_by_state_and_binds_only_held_in_channels(panel_graph):
    doc = stub_render(
        calibration_overview(panel_graph, "XR-QUAD", "integrated_gradient"), panel_graph
    )
    assert not [n for n in nodes(doc) if n["type"].startswith("pv-")]  # static values
    assert not [n for n in nodes(doc) if n.get("rules")]
    fills = {
        n["props"]["fillColor"]["$role"]
        for n in nodes(doc)
        if n["type"] == "rectangle" and isinstance(n["props"].get("fillColor"), dict)
    }
    assert {"Basis_Bench_Tint", "Disagree_Tint", "Field_Background"} <= fills
    texts = [n["props"]["text"] for n in nodes(doc) if n["type"] == "text-label"]
    for word in ("design only", "measured on the bench", "two sources disagree", "no value yet"):
        assert word in texts  # the legend
    assert any("1 BENCH" in t and "1 DISAGREE" in t and "1 NO VALUE" in t for t in texts)
    live = stub_render(calibration_overview(panel_graph, "XR-QUAD", "current_offset"), panel_graph)
    bound = {n["props"]["pv"] for n in nodes(live) if n["type"].startswith("pv-")}
    assert bound == {"XR01-QF1:Current-SP", "XR01-QD1:Current-SP", "XR02-QF1:Current-SP"}


def test_page_text_reads_on_every_tint() -> None:
    for tint in symbols.BASIS_TINTS.values():
        for theme in ("light", "dark"):
            ground = _resolved(tint, theme)
            assert contrast(_resolved(symbols.PAGE_TEXT, theme), ground) >= 4.5, tint
            assert symbols.near_state(ground) is None or tint is symbols.DISAGREE_TINT


def test_the_set_overview_lists_its_calibration_views_as_chips(panel_graph) -> None:
    doc = stub_render(set_overview(panel_graph, "XR-QUAD"), panel_graph)
    chips = [
        n for n in nodes(doc)
        if n["type"] == "rectangle" and str(n["props"].get("tooltip", "")).startswith("calibration: ")
    ]  # fmt: skip
    assert len(chips) == len(panel_graph.calibration_kinds("Quadrupole"))
    assert all(c["geometry"]["height"] == layout.CHIP_HEIGHT for c in chips)


def test_a_place_says_per_kind_how_many_are_at_their_wanted_basis(panel_graph) -> None:
    from osprey.facility.panels.describe import health_rollup, place_view

    view = place_view(panel_graph, "XR/XR01")
    titles = [s.title for s in view.sections]
    assert titles.index("calibration") == titles.index("key numbers") + 1
    (section,) = [s for s in view.sections if s.title == "calibration"]
    first = section.items[0]
    assert isinstance(first, Link)
    assert first.label == "Integrated gradient: 1 of 2 at wanted basis"  # QF1 bench, QD1 design
    assert first.target == Target("calibration", "group", "XR-QUAD", system="integrated_gradient")
    rollup = health_rollup(panel_graph, "XR")
    (summary,) = [s for s in rollup.sections if s.title == "calibration"]
    assert summary.items[0].label == "Integrated gradient: 1 of 4 at wanted basis"
    doc = stub_render(view, panel_graph)
    chips = [n for n in nodes(doc) if "at wanted basis" in str(n["props"].get("tooltip"))]
    assert len(chips) == len(section.items)
    assert len(section.items) == len(panel_graph.calibration_kinds("Quadrupole"))  # pumps: none


_TRIM = "XR01-QF1:Trim-SP"


def _second_holder(document) -> None:
    """A second channel on XR01:QF1 that states the signal ``current_offset`` is held in."""
    document["channels"].append(
        {"id": _TRIM, "signal": "current_setpoint", "on": {"device": "XR01:QF1"}}
    )


def test_a_value_held_in_several_channels_shows_them_all_in_channel_order(tmp_path) -> None:
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import PANELS_DIR, write_fixture_facility

    graph = load_graph(write_fixture_facility(tmp_path / "f.json", _second_holder), PANELS_DIR)
    names = tuple(
        c.name for c in graph.channels("XR01:QF1", "device") if c.signal == "current_setpoint"
    )
    assert names == ("XR01-QF1:Current-SP", _TRIM)
    (block,) = [s for s in device_card(graph, "XR01:QF1").sections if s.title == "calibration"]
    (row,) = [r for r in block.items[0].rows if r.title == "Current offset"]
    assert row.channels == names and row.chip == "EPICS"
    doc = stub_render(device_card(graph, "XR01:QF1"), graph)
    fields = [n for n in nodes(doc) if n["type"].startswith("pv-") and n["props"]["pv"] in names]
    side_by_side = [
        (a, b) for a in fields for b in fields
        if (a["props"]["pv"], b["props"]["pv"]) == names
        and a["geometry"]["y"] == b["geometry"]["y"]
        and a["geometry"]["x"] + a["geometry"]["width"] <= b["geometry"]["x"]
    ]  # fmt: skip
    assert len(side_by_side) == 1  # the calibration row; the card's own rows stack
    cells = _cells(calibration_overview(graph, "XR-QUAD", "current_offset"))
    assert cells["XR01:QF1"].channels == names
    bound = stub_render(calibration_overview(graph, "XR-QUAD", "current_offset"), graph)
    assert _TRIM in {n["props"]["pv"] for n in nodes(bound) if n["type"].startswith("pv-")}


def test_a_kind_shows_only_on_the_devices_under_its_place(panel_graph) -> None:
    def block(device):
        return [s for s in device_card(panel_graph, device).sections if s.title == "calibration"]

    (here,) = block("XR02:BEND1")
    (calibrations,) = here.items
    assert isinstance(calibrations, Calibrations)
    assert [r.title for r in calibrations.rows] == ["Field centre"]
    assert calibrations.rows[0].link is None  # no set of dipoles
    assert block("XR01:BEND1") == []
    from osprey.facility.panels.describe import place_view

    def summary(place):
        sections = [s for s in place_view(panel_graph, place).sections if s.title == "calibration"]
        return [i.label if isinstance(i, Link) else i.text for s in sections for i in s.items]

    assert "Field centre: 0 of 1 at wanted basis" in summary("XR/XR02")
    assert not [t for t in summary("XR/XR01") if t.startswith("Field centre")]
    facts = [
        i for s in place_view(panel_graph, "XR/XR02").sections
        if s.title == "calibration" for i in s.items if isinstance(i, Fact)
    ]  # fmt: skip
    assert len(facts) == 1


def test_a_cut_cell_shows_its_whole_text_on_hover(panel_graph, monkeypatch) -> None:
    monkeypatch.setattr(layout, "SET_CELL_MAX", 60)  # too narrow for "a / b"
    doc = stub_render(
        calibration_overview(panel_graph, "XR-QUAD", "integrated_gradient"), panel_graph
    )
    tips = [str(n["props"].get("tooltip")) for n in nodes(doc) if n["type"] == "rectangle"]
    assert any(t.endswith("+0.01 % / +1.24 %") and "XR02" in t for t in tips)
