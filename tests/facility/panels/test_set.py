"""The set overview's looks: an equal grid with column and row heads."""

from __future__ import annotations

from osprey.facility.panels import layout
from osprey.facility.panels.describe import set_overview
from osprey.facility.panels.view import (
    Fact,
    Grid,
    GridCell,
    GridRow,
    Link,
    Section,
    Target,
    View,
)
from tests.facility.panels._views import nodes, stub_render


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
    }


def test_the_set_is_an_equal_grid_under_column_heads_beside_row_heads(panel_graph) -> None:
    doc = stub_render(set_overview(panel_graph, "XR-QUAD"), panel_graph)
    texts = _texts(doc)
    (h1,), (h2,) = texts["#1"], texts["#2"]
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


def test_a_cell_holds_the_readback_or_the_short_name_with_its_light(panel_graph) -> None:
    doc = stub_render(set_overview(panel_graph, "XR-QUAD"), panel_graph)
    cells = _cells(doc)
    inside = lambda g, c: c["x"] <= g["x"] and g["x"] + g["width"] <= c["x"] + c["width"]  # noqa: E731
    values = {n["props"]["pv"]: n["geometry"] for n in nodes(doc) if n["type"] == "pv-label"}
    assert inside(values["XR01-QF1:Current-RB"], cells["XR01:QF1"])
    assert not [n for n in nodes(doc) if n["type"] in ("pv-text-input", "pv-button")]
    (name,) = _texts(doc)["D1"]  # no pair: the short name names.csv states
    assert inside(name["geometry"], cells["XR01:QD1"])
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
    panel_graph,
) -> None:
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
