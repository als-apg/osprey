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
        GridRow(Fact("row", "not placed"), many[:2]),
    )
    view = View(Target("set", "group", "X"), "x", (), (Section("members", (Grid(None, rows),)),))
    doc = stub_render(view, panel_graph)
    texts = _texts(doc)
    assert len(texts["XR01"]) == 2 and len(texts["not placed"]) == 2  # once per block
    assert len(texts["#1"]) == 1 and len(texts["#14"]) == 1
    assert texts["#14"][0]["geometry"]["y"] > texts["#1"][0]["geometry"]["y"]
    for g in _cells(doc).values():
        assert g["x"] + g["width"] <= layout.MARGIN + layout.STRIP_WIDTH
