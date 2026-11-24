"""The set overview's looks: an equal grid with column and row heads."""

from __future__ import annotations

from osprey_panels import layout
from osprey_panels.describe import set_overview
from osprey_panels.view import (
    Fact,
    Grid,
    GridCell,
    GridRow,
    Link,
    Section,
    Target,
    View,
)
from osprey_panels_tests._fixture import PANELS_DIR
from osprey_panels_tests._views import nodes, printed, stub_render


def _texts(doc: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for n in nodes(doc):
        if n["type"] == "text-label":
            out.setdefault(n["props"]["text"], []).append(n)
    return out


def _cells(doc: dict) -> dict[str, dict]:
    """Each cell's box by the member it opens."""
    return {
        n["props"]["tooltip"]: n["geometry"]
        for n in nodes(doc)
        if n["type"] == "rectangle"
        and n["props"].get("actions")
        and n["geometry"]["width"] >= layout.SET_CELL_WIDTH
        and "/calibration-" not in n["props"]["actions"][0]["file"]  # a chip
    }


def test_the_set_is_an_equal_grid_under_column_heads_beside_row_heads(panel_graph) -> None:
    doc = stub_render(set_overview(panel_graph, "XR-QUAD"), panel_graph)
    texts = _texts(doc)
    (h2,) = texts["#2"]
    h1 = min(texts["F1"], key=lambda n: n["geometry"]["y"])  # column 1: every member states F1
    assert h1["geometry"]["y"] == h2["geometry"]["y"]
    cells = _cells(doc)
    widths = {g["width"] for g in cells.values()}
    assert len(widths) == 1 and min(widths) >= 96
    assert all(g["height"] >= 24 for g in cells.values())
    for member, head in (("XR01:QF1", h1), ("XR01:QD1", h2), ("XR02:QF1", h1), ("XR02:QD1", h2)):
        g = cells[member]
        centre = head["geometry"]["x"] + head["geometry"]["width"] / 2
        assert abs(g["x"] + g["width"] / 2 - centre) < 1, member
    (row,) = texts["XR01"]  # the row head: the place, short
    assert row["geometry"]["x"] == layout.MARGIN
    assert row["geometry"]["y"] == cells["XR01:QF1"]["y"]


def _unpaired_qd1(tmp_path):
    """The fixture with no pair on XR01:QD1's setpoint (the build pairs it by signal)."""
    from osprey_panels.model import load_graph
    from osprey_panels_tests._fixture import PANELS_DIR, write_fixture_facility

    def change(document) -> None:
        for channel in document["channels"]:
            if channel["id"] == "XR01-QD1:Current-SP":
                channel.pop("pair", None)

    return load_graph(write_fixture_facility(tmp_path / "f.json", change), PANELS_DIR)


def test_a_cell_holds_the_readback_or_the_short_name_with_its_light(tmp_path) -> None:
    panel_graph = _unpaired_qd1(tmp_path)
    doc = stub_render(set_overview(panel_graph, "XR-QUAD"), panel_graph)
    cells = _cells(doc)
    inside = lambda g, c: c["x"] <= g["x"] and g["x"] + g["width"] <= c["x"] + c["width"]  # noqa: E731
    values = {  # each readback, a number held to its cell
        printed(doc, n): n["geometry"]
        for n in nodes(doc)
        if n["type"] == "text-label" and n.get("rules")
    }
    assert inside(values["XR01-QF1:Current-RB"], cells["XR01:QF1"])
    assert not [n for n in nodes(doc) if n["type"] in ("pv-text-input", "pv-button")]
    # no pair: its card's first key tile stands in the cell (its signals are its roles)
    (grid,) = [
        i
        for s in set_overview(panel_graph, "XR-QUAD").sections
        if s.title == "members"
        for i in s.items
    ]
    (qd1,) = [c for row in grid.rows for c in row.cells if c.member.label == "XR01:QD1"]
    assert qd1.pair is None and qd1.tile is not None
    bound = [
        n["geometry"]
        for n in nodes(doc)
        if "XR01-QD1:" in str(n["props"].get("pv", "")) + str(n.get("rules", ""))
    ]
    assert any(inside(g, cells["XR01:QD1"]) for g in bound)
    lights = [
        n["geometry"]
        for n in nodes(doc)
        if n["type"] == "polygon" and abs(n["geometry"]["y"] - cells["XR01:QD1"]["y"]) < layout.ROW
    ]
    assert any(inside(g, cells["XR01:QD1"]) for g in lights)


def test_more_columns_than_fit_wrap_to_a_block_with_its_row_heads_again(panel_graph) -> None:
    many = tuple(
        GridCell(Link(f"X:M{i}", Target("card", "device", f"X:M{i}")), ()) for i in range(14)
    )
    rows = (
        GridRow(Link("XR/XR01", Target("card", "place", "XR/XR01")), many),
        GridRow(Link("XR/XR02", Target("card", "place", "XR/XR02")), many[:3]),
        GridRow(Fact("row", "not placed"), many[:2]),
    )
    view = View(Target("set", "group", "X"), "x", (), (Section("members", (Grid(None, rows),)),))
    doc = stub_render(view, panel_graph)
    texts = _texts(doc)
    assert len(texts["XR01"]) == 2  # once per block it has members in
    assert len(texts["NOT PLACED"]) == 1  # its two members are all in the first block
    assert len(texts["#1"]) == 1 and len(texts["#14"]) == 1
    assert texts["#14"][0]["geometry"]["y"] > texts["#1"][0]["geometry"]["y"]
    for g in _cells(doc).values():
        assert g["x"] + g["width"] <= layout.MARGIN + layout.STRIP_WIDTH


def test_few_columns_widen_to_the_most_and_the_row_heads_are_narrow(panel_graph) -> None:
    doc = stub_render(set_overview(panel_graph, "XR-QUAD"), panel_graph)
    cells = _cells(doc)
    assert {g["width"] for g in cells.values()} == {layout.SET_CELL_MAX}
    assert min(g["x"] for g in cells.values()) == layout.MARGIN + layout.SET_ROW_HEAD


def test_a_cell_has_one_light_the_deviation_light_when_its_pair_has_one(panel_graph) -> None:
    doc = stub_render(set_overview(panel_graph, "XR-QUAD"), panel_graph)
    for member, g in _cells(doc).items():
        lights = [
            n for n in nodes(doc)
            if n["type"] == "polygon"
            and g["x"] <= n["geometry"]["x"] < g["x"] + g["width"]
            and g["y"] <= n["geometry"]["y"] < g["y"] + g["height"]
        ]  # fmt: skip
        assert len(lights) == 1, member
    boxes = [  # no box of the engine under a field: the viewer frames it
        n for n in nodes(doc)
        if n["type"] == "rectangle" and not n["props"].get("actions")
        and any(
            v["type"] == "pv-label" and v["geometry"]["x"] == n["geometry"]["x"]
            and v["geometry"]["y"] == n["geometry"]["y"]
            for v in nodes(doc)
        )
    ]  # fmt: skip
    assert not boxes


def test_rows_are_grouped_under_their_top_level_place_with_a_line_saying_what(
    tmp_path,
) -> None:
    panel_graph = _set_with_units(
        tmp_path, {f"XR0{n}-Q{k}1:Current-RB": "A" for n in (1, 2) for k in "FD"}
    )
    doc = stub_render(set_overview(panel_graph, "XR-QUAD"), panel_graph)
    texts = _texts(doc)
    (group,) = texts["XR"]
    assert group["geometry"]["y"] < texts["XR01"][0]["geometry"]["y"]
    (what,) = [t for t in texts if str(t).startswith("each cell: readback of")]
    assert what.endswith("(A)")


def test_rows_of_two_machines_get_two_groups_with_a_gap(panel_graph) -> None:
    one = (GridCell(Fact("member", "M"), ()),)
    rows = (
        GridRow(Link("XR/XR01", Target("card", "place", "XR/XR01")), one),
        GridRow(Link("XR/XR02", Target("card", "place", "XR/XR02")), one),
        GridRow(Fact("row", "not placed"), one),
    )
    view = View(Target("set", "group", "X"), "x", (), (Section("members", (Grid(None, rows),)),))
    doc = stub_render(view, panel_graph)
    texts = _texts(doc)
    assert len(texts["XR"]) == 1 and len(texts["NOT PLACED"]) == 1
    assert "each cell: the member's short name and its light" in texts
    heads = sorted(t["geometry"]["y"] for k in ("XR", "NOT PLACED") for t in texts[k])
    rows_y = sorted(t["geometry"]["y"] for t in texts["XR02"])
    assert heads[1] - rows_y[0] > layout.ROW  # a gap between the groups


def _wide_set(long: int, short_rows: int) -> View:
    def cells(prefix: str, n: int) -> tuple[GridCell, ...]:
        return tuple(
            GridCell(Link(f"{prefix}:M{i}", Target("card", "device", f"{prefix}:M{i}")), ())
            for i in range(n)
        )

    rows = (
        GridRow(Link("XR/XR01", Target("card", "place", "XR/XR01")), cells("A", long)),
        *(
            GridRow(Link("XR/XR02", Target("card", "place", "XR/XR02")), cells(f"B{k}", 2))
            for k in range(short_rows)
        ),
    )
    return View(Target("set", "group", "X"), "x", (), (Section("members", (Grid(None, rows),)),))


def test_a_wrapped_block_is_as_tall_as_the_rows_it_draws(panel_graph) -> None:
    doc = stub_render(_wide_set(60, 20), panel_graph)
    cells = _cells(doc)
    top = min(g["y"] for g in cells.values())
    texts = [  # headings and column heads; a row head counts only with its cells
        n for n in nodes(doc)
        if n["type"] == "text-label" and n["geometry"]["y"] >= top
        and n["props"]["text"] not in ("XR01", "XR02")
    ]  # fmt: skip
    lines = len({n["geometry"]["y"] for n in texts} | {g["y"] for g in cells.values()})
    blocks = len(_texts(doc)["#1"]) + len([t for t in _texts(doc) if t in ("#9", "#17")])
    budget = (lines * (layout.ROW + layout.ROW_GAP) + blocks * layout.SET_BLOCK_GAP) * 1.1
    bottom = max(g["y"] + g["height"] for g in cells.values())
    assert bottom - top <= budget, (bottom - top, budget)
    assert any(name.startswith("A:M") and g["y"] > top for name, g in cells.items())  # wraps
    assert len(_texts(doc)["XR02"]) == 20  # the short rows only in the first block


def test_a_set_of_more_members_than_the_most_counts_the_rest(panel_graph) -> None:
    doc = stub_render(_wide_set(300, 0), panel_graph)
    drawn = len(_cells(doc))
    assert layout.SET_MEMBERS <= drawn < layout.SET_MEMBERS + 20
    (line,) = [t for t in _texts(doc) if str(t).endswith("more members: open by place")]
    assert line.startswith(f"{300 - drawn} ")


def test_a_group_of_one_row_too_long_for_a_block_is_a_flow_at_full_width(panel_graph) -> None:
    def cells(prefix: str, n: int) -> tuple[GridCell, ...]:
        return tuple(
            GridCell(Link(f"{prefix}{i}", Target("card", "device", f"{prefix}{i}")), ())
            for i in range(n)
        )

    rows = (
        GridRow(Link("XR/XR01", Target("card", "place", "XR/XR01")), cells("A", 3)),
        GridRow(Link("XR/XR02", Target("card", "place", "XR/XR02")), cells("B", 3)),
        GridRow(Fact("row", "not placed"), cells("C", 40)),
    )
    view = View(Target("set", "group", "X"), "x", (), (Section("members", (Grid(None, rows),)),))
    doc = stub_render(view, panel_graph)
    texts = _texts(doc)
    assert len(texts["NOT PLACED"]) == 1  # one heading, no row head per line
    assert "#4" not in texts and len(texts["#1"]) == 1  # column heads: the grid group only
    flow = {k: g for k, g in _cells(doc).items() if k.startswith("C")}
    assert len(flow) == 40
    assert min(g["x"] for g in flow.values()) == layout.MARGIN  # full width, no row head
    right = max(g["x"] + g["width"] for g in flow.values())
    assert right <= layout.MARGIN + layout.STRIP_WIDTH
    assert right > layout.MARGIN + layout.STRIP_WIDTH - layout.SET_CELL_MAX
    lines = sorted({g["y"] for g in flow.values()})
    assert len(lines) > 1 and min(lines) > max(
        g["y"] for k, g in _cells(doc).items() if k[0] in "AB"
    )


def test_each_group_is_headed_by_the_short_names_its_own_members_share(panel_graph) -> None:
    def cell(name: str, short: str | None) -> GridCell:
        return GridCell(Link(name, Target("card", "device", name)), (), short_name=short)

    rows = (
        GridRow(Link("XR/XR01", Target("card", "place", "XR/XR01")), (cell("A1", "F1"), cell("A2", "D1"))),
        GridRow(Fact("row", "not placed"), (cell("C1", None), cell("C2", None))),
    )  # fmt: skip
    view = View(Target("set", "group", "X"), "x", (), (Section("members", (Grid(None, rows),)),))
    texts = _texts(stub_render(view, panel_graph))
    (f1,), (d1,), (n1,), (n2,) = texts["F1"], texts["D1"], texts["#1"], texts["#2"]
    (heading,) = texts["NOT PLACED"]
    assert f1["geometry"]["y"] == d1["geometry"]["y"] < heading["geometry"]["y"]
    assert n1["geometry"]["y"] == n2["geometry"]["y"] > heading["geometry"]["y"]


def _set_with_units(tmp_path, units: dict[str, str]):
    """The fixture's quadrupole set, its readbacks given *units* (channel to unit)."""
    from osprey_panels.model import load_graph
    from osprey_panels_tests._fixture import copy_tables, write_fixture_facility

    def change(document) -> None:
        for channel in document["channels"]:
            if channel["id"] in units:
                channel["unit"] = units[channel["id"]]

    tables = copy_tables(tmp_path / "panels")
    return load_graph(write_fixture_facility(tmp_path / "f.json", change), tables)


def _cell_widgets(doc: dict, cell: dict) -> list[dict]:
    """The widgets a set cell draws: those whose middle lies in its box, the box aside."""
    out = []
    for n in nodes(doc):
        g = n["geometry"]
        mx, my = g["x"] + g["width"] / 2, g["y"] + g["height"] / 2
        if n["props"].get("actions") or n["type"] == "rectangle":
            continue
        inside = cell["x"] <= mx <= cell["x"] + cell["width"]
        if inside and cell["y"] <= my <= cell["y"] + cell["height"] and not _layer(g, out):
            out.append(g)
    return out


def _layer(g: dict, found: list[dict]) -> bool:
    """A light's layers are one widget: they share its box, its mark grown about
    the box's middle to one line of text."""
    return any(
        g["x"] == o["x"]
        and g["width"] == o["width"]
        and abs((g["y"] + g["height"] / 2) - (o["y"] + o["height"] / 2)) < 0.01
        for o in found
    )


def _apart(a: dict, b: dict) -> bool:
    return (
        a["x"] + a["width"] <= b["x"] + 0.01
        or b["x"] + b["width"] <= a["x"] + 0.01
        or a["y"] + a["height"] <= b["y"] + 0.01
        or b["y"] + b["height"] <= a["y"] + 0.01
    )


def _no_overlap_in_cells(doc: dict) -> None:
    cells = _cells(doc)
    assert cells
    for member, cell in cells.items():
        widgets = _cell_widgets(doc, cell)
        assert len(widgets) >= 2, member  # a value or a name, and a light
        for i, a in enumerate(widgets):
            assert cell["x"] - 0.01 <= a["x"], member
            assert a["x"] + a["width"] <= cell["x"] + cell["width"] + 0.01, member
            for b in widgets[i + 1 :]:
                assert _apart(a, b), (member, a, b)


def test_a_set_cell_with_a_long_unit_keeps_its_number_and_light_apart(tmp_path) -> None:
    long = "1/Meter^2"
    graph = _set_with_units(
        tmp_path,
        {f"XR0{n}-Q{k}1:Current-RB": long for n in (1, 2) for k in "FD"},
    )
    doc = stub_render(set_overview(graph, "XR-QUAD"), graph)
    _no_overlap_in_cells(doc)
    texts = _texts(doc)
    assert any(t.endswith(f"({long})") for t in texts)  # stated once, over the grid
    assert long not in texts  # not in a cell
    assert not [n for n in nodes(doc) if n["type"] == "pv-label"]  # the viewer would add it


def test_a_set_whose_units_differ_keeps_each_unit_clear_of_the_light(tmp_path) -> None:
    from dataclasses import replace

    graph = _set_with_units(
        tmp_path, {"XR01-QF1:Current-RB": "1/Meter^2", "XR01-QF1:Current-SP": "A"}
    )
    view = set_overview(graph, "XR-QUAD")
    section = next(s for s in view.sections if any(isinstance(i, Grid) for i in s.items))
    grid = next(i for i in section.items if isinstance(i, Grid))
    cell = next(c for r in grid.rows for c in r.cells if c.pair is not None)
    assert cell.pair is not None
    other = replace(cell, pair=replace(cell.pair, readback="XR01-QF1:Current-SP"))
    rows = (GridRow(grid.rows[0].label, (cell, other)),)
    doc = stub_render(View(view.id, view.title, (), (Section("members", (replace(grid, rows=rows),)),)), graph)  # fmt: skip
    _no_overlap_in_cells(doc)
    texts = _texts(doc)
    assert "1/Meter^2" in texts and "A" in texts  # each cell states its own
    assert not any(t.endswith(("(1/Meter^2)", "(A)")) for t in texts)


def test_a_group_whose_members_share_a_unit_states_it_after_its_heading(tmp_path) -> None:
    from dataclasses import replace

    graph = _set_with_units(
        tmp_path, {"XR01-QF1:Current-RB": "1/Meter^2", "XR01-QF1:Current-SP": "A"}
    )
    view = set_overview(graph, "XR-QUAD")
    section = next(s for s in view.sections if any(isinstance(i, Grid) for i in s.items))
    grid = next(i for i in section.items if isinstance(i, Grid))
    cell = next(c for r in grid.rows for c in r.cells if c.pair is not None)
    assert cell.pair is not None
    other = replace(cell, pair=replace(cell.pair, readback="XR01-QF1:Current-SP"))
    rows = (GridRow(Fact("row", "ring"), (cell,)), GridRow(Fact("row", "booster"), (other,)))
    doc = stub_render(View(view.id, view.title, (), (Section("members", (replace(grid, rows=rows),)),)), graph)  # fmt: skip
    _no_overlap_in_cells(doc)
    texts = _texts(doc)
    for heading, unit in (("RING", "1/Meter^2"), ("BOOSTER", "A")):
        (head,) = texts[heading]
        (stated,) = texts[unit]  # once, after the heading, in no cell
        assert stated["geometry"]["y"] == head["geometry"]["y"]
        assert stated["geometry"]["x"] > head["geometry"]["x"]
    assert not [n for n in nodes(doc) if n["type"] == "pv-label"]  # the viewer would add it


def test_a_member_with_no_light_shows_the_value_its_card_shows_first(tmp_path) -> None:
    from osprey_panels import load_graph
    from osprey_panels_tests._fixture import copy_tables, write_fixture_facility
    from osprey_panels_tests.test_card import ROLES_HEADER

    def unpaired(document) -> None:
        for record in document["channels"]:
            record.pop("pair", None)  # no primary pair: the cell has no readback field

    tables = copy_tables(tmp_path / "panels")
    (tables / "channel_roles.csv").write_text(
        ROLES_HEADER + "Quadrupole,current,*:Current-RB,device,value,,,,,stated,S,1,l\n"
    )
    health = (tables / "health.csv").read_text().splitlines(keepends=True)
    kept = [line for line in health if "-Q" not in line.split(",", 1)[0]]  # no magnet rule
    (tables / "health.csv").write_text("".join(kept))
    graph = load_graph(write_fixture_facility(tmp_path / "f.json", unpaired), tables)
    view = set_overview(graph, "XR-QUAD")
    cells = [c for s in view.sections for g in s.items if hasattr(g, "rows") for r in g.rows for c in r.cells]  # fmt: skip
    assert cells and all(c.channels == () and c.pair is None for c in cells)
    assert {c.tile.channel for c in cells} == {
        "XR01-QF1:Current-RB", "XR01-QD1:Current-RB", "XR02-QF1:Current-RB", "XR02-QD1:Current-RB"
    }  # fmt: skip
    doc = stub_render(view, graph)
    shown = {n["props"]["pv"] for n in nodes(doc) if n["type"] == "pv-label"}
    assert "XR01-QF1:Current-RB" in shown  # drawn in the cell, not an empty box


def _grouped(tmp_path, *groups):
    from osprey_panels import load_graph
    from osprey_panels_tests._fixture import write_fixture_facility

    def change(document) -> None:
        document["devices"].append({"id": "XR01:ODD1", "place": "XR/XR01"})  # no class
        document["groups"].extend(groups)

    return load_graph(write_fixture_facility(tmp_path / "f.json", change), PANELS_DIR)


def test_a_section_is_never_a_set_and_a_family_needs_one_class(tmp_path) -> None:
    from osprey_panels.describe import targets

    graph = _grouped(
        tmp_path,
        {"id": "XR-ZONE", "kind": "section", "members": ["XR01:QF1"]},
        {"id": "XR-FAM", "kind": "family", "members": ["XR01:QF1", "XR02:QF1"]},
    )
    sets = {t.id for t in targets(graph) if t.view == "set"}
    assert "XR-FAM" in sets and "XR-ZONE" not in sets


def test_check_names_a_family_that_is_no_set_for_want_of_a_members_class(tmp_path) -> None:
    from osprey_panels import emit

    graph = _grouped(
        tmp_path,
        {"id": "XR-HALF", "kind": "family", "members": ["XR01:QF1", "XR02:QF1", "XR01:ODD1"]},
    )
    emit.write(graph, tmp_path / "out")
    (note,) = [p for p in emit.check(graph, tmp_path / "out") if p.kind == "unset_group"]
    assert note.note and note.detail.startswith("XR-HALF is no set: XR01:ODD1 is no device")


def test_a_set_over_one_page_is_paged_by_whole_rows_and_each_page_links_the_others(
    tmp_path, monkeypatch, panel_facility
) -> None:
    import importlib

    from osprey_panels import emit, load_graph
    from osprey_panels.describe import targets
    from osprey_panels.view import Grid, Target

    described = importlib.import_module("osprey_panels.describe")
    monkeypatch.setattr(described, "SET_PAGE", 3)  # two sector rows of two: one a page
    graph = load_graph(panel_facility, PANELS_DIR)  # a fresh graph: its index is built now
    pages = [t for t in targets(graph) if t.view == "set" and t.id == "XR-QUAD"]
    assert [t.system for t in pages] == [None, "2"]
    rows = []
    for n, target in enumerate(pages, 1):
        view = set_overview(graph, "XR-QUAD", n)
        assert view.id == target
        links = [i.target for s in view.sections if s.title == "pages" for i in s.items]
        assert links == pages
        (grid,) = [i for s in view.sections for i in s.items if isinstance(i, Grid)]
        rows += [(r.label.label, len(r.cells)) for r in grid.rows]
    assert rows == [("XR/XR01", 2), ("XR/XR02", 2)]  # each row whole, each member once
    assert pages[0] == Target("set", "group", "XR-QUAD")  # links elsewhere open page 1
    emit.write(graph, tmp_path / "out")
    assert not [
        p for p in emit.check(graph, tmp_path / "out") if p.kind in ("missing", "over_budget")
    ]
