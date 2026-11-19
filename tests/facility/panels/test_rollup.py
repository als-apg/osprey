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
        and n["geometry"]["height"] > layout.CHIP_HEIGHT  # not a calibration chip
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


def test_a_child_over_the_limit_is_an_open_tile_and_all_tiles_have_one_height(
    panel_graph, monkeypatch
) -> None:
    from tests.facility.panels.test_layout import _limit_between_one_channel_and_three

    limit = _limit_between_one_channel_and_three(panel_graph)
    monkeypatch.setattr(layout, "RULE_EXPRESSION_LIMIT", limit)  # "three" is over it
    doc = stub_render(rollup_view(), panel_graph)
    assert not [n for n in nodes(doc) if str(n["props"].get("tooltip", "")).startswith("part ")]
    assert not [n for n in nodes(doc) if n["geometry"]["width"] == layout.MOSAIC_LIGHT]
    tiles = [
        n["geometry"]
        for n in nodes(doc)
        if n["type"] == "rectangle" and n["props"].get("fillColor") == symbols.FIELD_BACKGROUND
        and layout.ROLLUP_TILE_WIDTH <= n["geometry"]["width"] < layout.ROLLUP_LEFT
    ]  # fmt: skip
    assert tiles and {t["height"] for t in tiles} == {layout.ROLLUP_TILE_HEIGHT}
    texts = [n["props"].get("text") for n in nodes(doc) if n["type"] == "text-label"]
    assert "3 channels · open" in texts


def test_the_title_stands_in_the_header_bar_after_the_id(panel_graph) -> None:
    view = health_rollup(panel_graph, "XR/XR01")
    doc = stub_render(view, panel_graph)
    texts = [n for n in nodes(doc) if n["type"] == "text-label"]
    (rest,) = [n for n in texts if n["props"].get("text") == "health by place"]
    assert rest["geometry"]["y"] < layout.HEADER_HEIGHT
    assert rest["props"].get("foregroundColor") == symbols.MUTED_TEXT
    assert not [n for n in texts if n["props"].get("text") == view.title]
    assert [n for n in texts if str(n["props"].get("text")).startswith("model ")]


def test_a_metrics_row_applies_only_where_the_place_level_matches(panel_graph, monkeypatch) -> None:
    from dataclasses import replace

    from osprey.facility.panels.tables import Metric
    from osprey.facility.panels.view import Band

    ring = Metric(
        view="rollup", level="ring", order=1.0, title="Ring number", measure="count",
        signal="status", cls=None, unit=None, caption=None, source=None,
    )  # fmt: skip
    tables = replace(panel_graph._tables, metrics=(ring,))
    monkeypatch.setattr(panel_graph, "_tables", tables)
    assert panel_graph.thing("XR/XR01", "place").level != "ring"
    assert panel_graph.metrics("rollup", "ring") == [ring]
    view = health_rollup(panel_graph, "XR/XR01")  # a place of another level: no band
    assert not [i for s in view.sections for i in s.items if isinstance(i, Band)]


# -- the ring --------------------------------------------------------------------


class _Ringed:
    """A stand-in graph: place R with children A, B, C; C spans the ring's start."""

    spans = {"R/A": [10.0, 20.0], "R/B": [40.0, 55.0], "R/C": [0.5, 80.0, 99.0]}

    def within(self, _place_id):
        from osprey.facility.panels.model import Thing

        return [Thing(c, "place", None, "R") for c in ("R/C", "R/A", "R/B")]

    def contents(self, place_id):
        from osprey.facility.panels.model import Thing

        return [
            Thing(f"{place_id}#{i}", "device", None, place_id)
            for i in range(len(self.spans[place_id]))
        ]

    def beam_position(self, thing):
        place, i = thing.id.split("#")
        return "R-RING", self.spans[place][int(i)]

    def frame_length(self, frame):
        return 100.0 if frame == "R-RING" else None


def test_the_ring_orders_children_by_where_their_span_starts() -> None:
    from osprey.facility.panels.describe import ring_arcs

    assert ring_arcs(_Ringed(), "R") == ["R/A", "R/B", "R/C"]  # C starts at 80 m, not 0.5 m


def _ring_view(ring_graph):
    return health_rollup(ring_graph, "XR")


def test_a_ring_roll_up_has_its_children_on_the_ring_in_beam_order(ring_graph, panel_graph) -> None:
    view = _ring_view(ring_graph)
    (group,) = _section(view, "ring").items
    assert isinstance(group, Group) and group.form == "ring"
    assert [n.label for n in group.nodes] == ["XR/XR01", "XR/XR02"]
    # no periodic model states the ring: no arcs
    assert not [s for s in health_rollup(panel_graph, "XR").sections if s.title == "ring"]
    assert not [s for s in health_rollup(ring_graph, "XR", "system").sections if s.title == "ring"]


def test_the_ring_draws_an_arc_per_child_with_its_name_and_link(ring_graph) -> None:
    doc = stub_render(_ring_view(ring_graph), ring_graph)
    big = [
        n for n in nodes(doc)
        if n["type"] == "polygon" and n["geometry"]["height"] > 2 * layout.RING_INNER
    ]  # fmt: skip
    assert len(big) == 2  # two half rings, each as tall as the ring
    for arc in big:
        assert arc.get("rules") or arc["props"]["lineStyle"] == "dashed" or any(
            t.startswith("needs:") for t in arc.get("tags", ())
        )  # fmt: skip
    texts = {n["props"].get("text"): n for n in nodes(doc) if n["type"] == "text-label"}
    for name in ("XR01", "XR02", "XR"):
        assert texts[name]["props"].get("fontSize", layout.TEXT_SIZE) >= layout.SMALL_SIZE
    links = [
        n for n in nodes(doc)
        if n["props"].get("actions") and n["props"].get("tooltip") in ("XR/XR01", "XR/XR02")
    ]  # fmt: skip
    assert len(links) == 2
    a, b = (n["geometry"] for n in links)
    assert a["x"] + a["width"] <= b["x"] or b["x"] + b["width"] <= a["x"] or (
        a["y"] + a["height"] <= b["y"] or b["y"] + b["height"] <= a["y"]
    )  # fmt: skip
    assert not [t for t in _tiles(doc) if t["props"]["tooltip"] in ("XR/XR01", "XR/XR02")]
