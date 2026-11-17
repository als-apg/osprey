"""The roll-up's looks: the children as large tiles, the by-system rows beside them."""

from __future__ import annotations

from osprey.facility.panels import layout, symbols
from osprey.facility.panels.describe import health_rollup
from osprey.facility.panels.view import Group
from tests.facility.panels._views import nodes, rollup_view, stub_render


def _section(view, title):
    (section,) = [s for s in view.sections if s.title == title]
    return section


def _tiles(doc: dict) -> list[dict]:
    """The children's tiles: the boxes that open a screen, in the tiles' area."""
    return [
        n
        for n in nodes(doc)
        if n["type"] == "rectangle"
        and n["props"].get("actions")
        and n["geometry"]["width"] >= layout.ROLLUP_TILE_WIDTH
        and n["geometry"]["x"] < layout.MARGIN + layout.ROLLUP_LEFT
    ]


def test_a_place_roll_up_carries_its_rows_by_system(panel_graph) -> None:
    view = health_rollup(panel_graph, "XR/XR01")
    (group,) = _section(view, "by system").items
    assert isinstance(group, Group) and group.form == "rows"
    *_band, systems = health_rollup(panel_graph, "XR/XR01", "system").sections
    assert group.nodes == systems.items[0].children
    assert view.sections[-1].title == "roll-up"  # the tree stays last
    by_system = health_rollup(panel_graph, "XR/XR01", "system")
    assert not [s for s in by_system.sections if s.title == "by system"]


def test_the_children_are_large_tiles_each_opening_its_screen(panel_graph) -> None:
    view = health_rollup(panel_graph, "XR/XR01")
    doc = stub_render(view, panel_graph)
    (root,) = view.sections[-1].items
    linked = [c for c in root.children if c.link is not None]
    tiles = _tiles(doc)
    assert len(tiles) == len(linked)
    for tile in tiles:
        g = tile["geometry"]
        assert g["width"] >= 96 and g["height"] >= 56
    doc = stub_render(health_rollup(panel_graph, "XR"), panel_graph)
    texts = [n["props"]["text"] for n in nodes(doc) if n["type"] == "text-label"]
    assert {"XR01", "XR02"} <= set(texts)  # the child's name, short: the parent's prefix cut


def test_the_by_system_rows_stand_right_of_the_tiles(panel_graph) -> None:
    doc = stub_render(health_rollup(panel_graph, "XR/XR01"), panel_graph)
    heading = [
        n for n in nodes(doc) if n["type"] == "text-label" and n["props"]["text"] == "BY SYSTEM"
    ]
    assert heading and heading[0]["geometry"]["x"] >= layout.MARGIN + layout.ROLLUP_LEFT
    tiles = _tiles(doc)
    assert all(
        t["geometry"]["x"] + t["geometry"]["width"] <= heading[0]["geometry"]["x"] for t in tiles
    )


def test_a_childs_part_lights_stand_inside_its_tile(panel_graph, monkeypatch) -> None:
    from tests.facility.panels.test_layout import _limit_between_one_channel_and_three

    limit = _limit_between_one_channel_and_three(panel_graph)
    monkeypatch.setattr(layout, "RULE_EXPRESSION_LIMIT", limit)  # "three" is in parts
    doc = stub_render(rollup_view(), panel_graph)
    parts = [n for n in nodes(doc) if str(n["props"].get("tooltip", "")).startswith("part ")]
    assert parts
    (tile,) = [
        n["geometry"]
        for n in nodes(doc)
        if n["type"] == "rectangle"
        and n["props"].get("fillColor") == symbols.FIELD_BACKGROUND
        and any(
            t["props"].get("text") == "three"
            and n["geometry"]["x"]
            <= t["geometry"]["x"]
            < n["geometry"]["x"] + n["geometry"]["width"]
            and n["geometry"]["y"]
            <= t["geometry"]["y"]
            < n["geometry"]["y"] + n["geometry"]["height"]
            for t in nodes(doc)
            if t["type"] == "text-label"
        )
    ]
    for part in parts:
        g = part["geometry"]
        assert tile["x"] <= g["x"] and g["x"] + g["width"] <= tile["x"] + tile["width"]
        assert tile["y"] <= g["y"] and g["y"] + g["height"] <= tile["y"] + tile["height"]
