"""Calibration views: a kind over a set, its links, its looks."""

from __future__ import annotations

from osprey.facility.panels import layout, symbols, targets
from osprey.facility.panels.describe import (
    calibration_overview,
    device_card,
    set_overview,
)
from osprey.facility.panels.view import Calibrations, CalibrationTallies, Grid, Link, Target
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
        "Integrated gradient",  # the kind's short name, else its title; no prefix
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
        if n["type"] == "rectangle" and n["props"].get("actions")
        and "/calibration-" in n["props"]["actions"][0]["file"]
    ]  # fmt: skip
    kinds = panel_graph.calibration_kinds("Quadrupole")
    assert len(chips) == len(kinds)
    assert [c["props"]["tooltip"] for c in chips] == [k.tally_title for k in kinds]
    assert len({c["geometry"]["y"] for c in chips}) == 1  # one line
    assert all(c["geometry"]["height"] == layout.CHIP_HEIGHT for c in chips)


def _tallies(view):
    (section,) = [s for s in view.sections if s.title == "calibration"]
    (item,) = section.items
    assert isinstance(item, CalibrationTallies)
    return item


def test_a_place_tallies_each_kind_once_and_folds_those_not_started(panel_graph) -> None:
    from osprey.facility.panels.describe import health_rollup, place_view

    view = place_view(panel_graph, "XR/XR01")
    titles = [s.title for s in view.sections]
    assert titles.index("calibration") == titles.index("key numbers") + 1
    found = _tallies(view)
    shown = [(t.title, t.good, t.total, t.state) for t in found.tallies]
    assert shown == [
        ("Integrated gradient", 1, 2, "below_wanted"),  # QF1 bench, QD1 design
        ("Excitation curve", 1, 2, "none"),  # QD1 has none: the worst state present
        ("Magnetic centre", 0, 2, "none"),
    ]
    assert found.not_started == 2  # strength scale, current offset: no value anywhere
    assert found.tallies[0].link.target == Target(
        "calibration", "group", "XR-QUAD", system="integrated_gradient"
    )
    assert found.more.target == Target("rollup", "place", "XR/XR01", by="place")
    rollup = _tallies(health_rollup(panel_graph, "XR"))
    assert (rollup.tallies[0].good, rollup.tallies[0].total) == (1, 4)
    assert rollup.tallies[0].state == "contested"  # XR02:QF1
    assert rollup.more is None  # the roll-up is where "more" leads


def test_kinds_of_one_short_title_are_summed_over_their_classes(tmp_path) -> None:
    from osprey.facility.panels import load_graph
    from osprey.facility.panels.describe import place_view
    from tests.facility.panels._fixture import copy_tables, write_fixture_facility

    tables = copy_tables(tmp_path / "panels")
    rows = (tables / "calibration_kinds.csv").read_text().splitlines()
    rows[0] += ",short"
    rows[1:] = [r + (",Centre" if r.startswith("centre,") else ",") for r in rows[1:]]
    (tables / "calibration_kinds.csv").write_text("\n".join(rows) + "\n")
    graph = load_graph(write_fixture_facility(tmp_path / "f.json"), tables)
    assert graph.calibration_kind("centre", "Dipole").short == "Centre"
    found = _tallies(place_view(graph, "XR/XR02"))
    titles = [t.title for t in found.tallies]
    assert "Centre" not in titles  # no value anywhere under XR/XR02: folded
    assert found.not_started == 4  # integrated gradient has a value; four kinds have none
    assert len({t.title for t in found.tallies}) == len(found.tallies)


def _many_kinds(tmp_path):
    """Twelve kinds of six classes, each with one bench value, under XR."""
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import copy_tables, write_fixture_facility

    devices = ["XR01:QF1", "XR01:QD1", "XR02:QF1", "XR02:QD1", "XR01:BEND1", "XR02:BEND1"]

    def classes(document) -> None:
        for record in document["devices"]:
            if record["id"] in devices:
                record["class"] = f"C{devices.index(record['id'])}"

    tables = copy_tables(tmp_path / "panels")
    kinds = ["kind,title,class,shape,unit,wanted_basis,tolerance,tolerance_source,held_in,order"]
    claims = [(tables / "calibrations.csv").read_text().splitlines()[0]]
    for k in range(12):
        for c, device in enumerate(devices):
            kinds.append(f"k{k},Quantity number {k} of the magnet,C{c},scalar,T,bench,,,,{k}")
            claims.append(f"v{k}-{c},k{k},{device},1.0,T,,bench,,2024-01-01,D,A,Released,,accepted")
    (tables / "calibration_kinds.csv").write_text("\n".join(kinds) + "\n")
    (tables / "calibrations.csv").write_text("\n".join(claims) + "\n")
    (tables / "calibration_curves.csv").write_text("id,x,y\n")
    return load_graph(write_fixture_facility(tmp_path / "f.json", classes), tables)


def test_the_chips_take_one_line_and_fold_the_rest_into_more(tmp_path) -> None:
    from osprey.facility.panels.describe import place_view

    graph = _many_kinds(tmp_path)
    view = place_view(graph, "XR")
    found = _tallies(view)
    assert len(found.tallies) == 12 and found.tallies[0].total == 6
    doc = stub_render(view, graph)
    chips = [
        n for n in nodes(doc)
        if n["type"] == "rectangle" and str(n["props"].get("tooltip", "")).startswith("Quantity")
    ]  # fmt: skip
    more = [n for n in nodes(doc) if str(n["props"].get("tooltip", "")).endswith(" more")]
    assert chips and more
    assert len({n["geometry"]["y"] for n in [*chips, *more]}) == 1  # one line
    right = max(n["geometry"]["x"] + n["geometry"]["width"] for n in [*chips, *more])
    assert right <= layout.MARGIN + layout.STRIP_WIDTH
    assert "Quantity number 0 of the magnet 6 / 6" in {n["props"]["tooltip"] for n in chips}
    texts = {n["props"]["text"] for n in nodes(doc) if n["type"] == "text-label"}
    shown = {n["props"]["tooltip"] for n in chips}
    assert f"+{12 - len(shown)} more" in texts


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
    assert side_by_side  # the calibration row
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

    there = _tallies(place_view(panel_graph, "XR/XR02"))
    assert "Field centre" not in [t.title for t in there.tallies]  # no value: folded
    assert there.not_started == 5  # four quadrupole kinds and the dipole's, under XR02 only


def test_a_cut_cell_shows_its_whole_text_on_hover(panel_graph, monkeypatch) -> None:
    monkeypatch.setattr(layout, "SET_CELL_MAX", 60)  # too narrow for "a / b"
    doc = stub_render(
        calibration_overview(panel_graph, "XR-QUAD", "integrated_gradient"), panel_graph
    )
    tips = [str(n["props"].get("tooltip")) for n in nodes(doc) if n["type"] == "rectangle"]
    assert any(t.endswith("+0.01 % / +1.24 %") and "XR02" in t for t in tips)


def test_a_column_is_headed_by_the_short_name_its_members_share(panel_graph) -> None:
    doc = stub_render(
        calibration_overview(panel_graph, "XR-QUAD", "integrated_gradient"), panel_graph
    )
    texts = [n["props"]["text"] for n in nodes(doc) if n["type"] == "text-label"]
    assert "F1" in texts  # XR01:QF1 and XR02:QF1 both state F1
    assert "#2" in texts and "#1" not in texts  # XR02:QD1 states none: the number stays


def test_the_count_line_names_the_states_in_the_legends_order(panel_graph) -> None:
    doc = stub_render(
        calibration_overview(panel_graph, "XR-QUAD", "integrated_gradient"), panel_graph
    )
    texts = [n["props"]["text"] for n in nodes(doc) if n["type"] == "text-label"]
    (line,) = [t for t in texts if t.startswith("4 MEMBERS")]
    words = ["DESIGN", "BENCH", "DISAGREE", "NO VALUE"]
    legend = ["design only", "measured on the bench", "two sources disagree", "no value yet"]
    assert [line.index(w) for w in words] == sorted(line.index(w) for w in words)
    assert [texts.index(w) for w in legend] == sorted(texts.index(w) for w in legend)


def _claims(tmp_path, rows: str):
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import copy_tables, write_fixture_facility

    tables = copy_tables(tmp_path / "panels")
    with (tables / "calibrations.csv").open("a") as handle:
        handle.write(rows)
    return load_graph(write_fixture_facility(tmp_path / "f.json"), tables)


def test_a_contested_claim_alone_is_shown_contested_and_is_no_base_value(tmp_path) -> None:
    from osprey.facility.panels.describe import calibration_cell, calibration_row

    graph = _claims(
        tmp_path,
        "ig/XR02QD1/a,integrated_gradient,XR02:QD1,4.91,T,,bench,stretched wire,"
        "2024-01-01,XR-MAG-TEST,A.3,Released,,contested\n",
    )
    found = graph.calibration("XR02:QD1", "integrated_gradient")
    assert found.in_force is None and found.state == "contested"
    kind = found.kind
    row = calibration_row(graph, "XR02:QD1", kind)
    assert row.shown is None and row.state == "contested"  # no value in force
    assert row.note == "contested, not in force: 4.91 T (XR-MAG-TEST A.3)"
    assert calibration_cell(graph, "XR02:QD1", kind).text == "contested"


def test_beside_an_accepted_claim_a_contested_one_is_named_not_used(tmp_path) -> None:
    from osprey.facility.panels.describe import calibration_row

    graph = _claims(
        tmp_path,
        "ig/QF1/late,integrated_gradient,XR01:QF1,-5.5,T,at 73.33 A,bench,rotating coil,"
        "2025-01-01,XR-MAG-ROT,B,Released,,contested\n",
    )
    found = graph.calibration("XR01:QF1", "integrated_gradient")
    assert found.in_force.id == "ig/QF1/bench"  # the newest accepted, not the newer contested
    row = calibration_row(graph, "XR01:QF1", found.kind)
    assert row.state == "contested" and row.note.startswith("contested, not in force: -5.5 T")


def test_a_cell_of_several_live_numbers_is_as_tall_as_its_lines(panel_graph) -> None:
    from dataclasses import replace

    view = calibration_overview(panel_graph, "XR-QUAD", "current_offset")
    channels = ("XR01-QF1:Current-SP",) * 4  # a tuple of four numbers

    def widen(item):
        if not isinstance(item, Grid):
            return item
        rows = tuple(
            replace(
                row,
                cells=tuple(
                    replace(c, calibration=replace(c.calibration, channels=channels))
                    for c in row.cells
                ),
            )
            for row in item.rows
        )
        return replace(item, rows=rows)

    sections = tuple(replace(s, items=tuple(widen(i) for i in s.items)) for s in view.sections)
    doc = stub_render(replace(view, sections=sections), panel_graph)
    fields = [n["geometry"] for n in nodes(doc) if n["type"].startswith("pv-")]
    boxes = [
        n["geometry"]
        for n in nodes(doc)
        if n["type"] == "rectangle" and n["props"].get("lineStyle") in ("solid", "dashed")
    ]
    for f in fields:  # every number inside its cell's box
        assert any(
            b["x"] <= f["x"] and f["y"] >= b["y"] and f["y"] + f["height"] <= b["y"] + b["height"]
            for b in boxes
        ), f
    legend = next(n for n in nodes(doc) if n["props"].get("text") == "design only")
    assert legend["geometry"]["y"] >= max(f["y"] + f["height"] for f in fields)


def test_a_long_basis_and_date_wrap_whole_instead_of_being_cut(panel_graph) -> None:
    from dataclasses import replace

    view = device_card(panel_graph, "XR01:QF1")
    chip = "survey · laser tracker over the whole girder"
    date = "Friday, August 14, 2026"

    def long(item):
        if not isinstance(item, Calibrations):
            return item
        rows = (replace(item.rows[0], chip=chip, date=date), *item.rows[1:])
        return replace(item, rows=rows)

    sections = tuple(replace(s, items=tuple(long(i) for i in s.items)) for s in view.sections)
    doc = stub_render(replace(view, sections=sections), panel_graph)
    texts = [n["props"]["text"] for n in nodes(doc) if n["type"] == "text-label"]
    assert not [t for t in texts if t.endswith("…") and (t[:6] in chip or t[:6] in date)]
    joined = " ".join(texts)
    for word in ("laser", "girder", "August", "2026"):
        assert word in joined
