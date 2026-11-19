"""How the screens look: measurable floors, checked on every rendered fixture view.

These are floors, not the acceptance: the acceptance is the rendered screen
beside the approved mockup.
"""

from __future__ import annotations

from itertools import combinations

import pytest

from osprey.facility.panels import layout, symbols
from osprey.facility.panels.describe import place_view
from tests.facility.panels._views import (
    ABOVE,
    SYMBOLS,
    element,
    fixture_views,
    invented_views,
    strip_view,
    stub_render,
    with_symbols,
)

#: The colour roles each element kind may draw in.
#: A calibration chip's outline and text: its basis, or disagree when contested.
_BASIS = {role["$role"] for role in symbols.BASIS_ROLES.values()}

#: A drawn card's own device (Focus, Focus_Tint) and a control's tag (Control_Action).
_CARD = {"Focus", "Control_Action"}
ALLOWED = {
    ("text-label", "foregroundColor"): {"Page_Text", "Text_Muted", *_BASIS, *_CARD},
    ("text-label", "backgroundColor"): {"Transparent", "Field_Background"},
    ("line", "lineColor"): {"Page_Text", "Text_Muted", "MINOR", "MAJOR", "DISCONNECTED"},
    # Page_Text: a drawn outline, as the strip's symbols (the rack's frame)
    ("rectangle", "lineColor"): {
        "Box_Border",
        "Transparent",
        "Page_Text",
        "Text_Muted",
        *_BASIS,
        *_CARD,
    },
    ("rectangle", "fillColor"): {
        "Field_Background",
        "Tile_Background",
        "Transparent",  # a control's tag, an outline round a choice or an entry
        "Focus_Tint",
        *(t["$role"] for t in symbols.BASIS_TINTS.values()),
    },  # fmt: skip
}


def _nodes(doc: dict) -> list[dict]:
    return doc["root"]["children"]


def _box(n: dict) -> tuple[float, float, float, float]:
    g = n["geometry"]
    return g["x"], g["y"], g["x"] + g["width"], g["y"] + g["height"]


def _docs(graph) -> list[tuple[str, dict]]:
    views = [*fixture_views(graph), *invented_views()]
    return [(f"{v.id.view}:{v.id.id}", stub_render(v, graph)) for v in views]


@pytest.fixture(scope="module")
def docs(request):
    graph = request.getfixturevalue("panel_graph")
    return _docs(graph)


def test_the_type_floors_are_pinned() -> None:
    assert (layout.TEXT_SIZE, layout.SMALL_SIZE, layout.TITLE_SIZE) == (13, 11, 20)
    assert layout.STRIP_LABEL_SIZE == layout.HEADING_SIZE == layout.SMALL_SIZE


def test_no_text_is_below_the_floor(panel_graph) -> None:
    for name, doc in _docs(panel_graph):
        for n in _nodes(doc):
            if "fontSize" in n["props"]:
                assert n["props"]["fontSize"] >= layout.SMALL_SIZE, (name, n["props"])


def test_no_two_text_boxes_on_a_screen_overlap(panel_graph) -> None:
    for name, doc in _docs(panel_graph):
        texts = [_box(n) for n in _nodes(doc) if n["type"] == "text-label" and n["props"]["text"]]
        for a, b in combinations(texts, 2):
            apart = a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]
            assert apart, (name, a, b)


def test_no_widget_is_outside_the_page(panel_graph) -> None:
    """Except an entry's Apply button while nothing is entered: it waits off the
    canvas, and its rule brings it onto the page (see test_emit)."""
    for name, doc in _docs(panel_graph):
        size = doc["metadata"]["size"]
        for n in _nodes(doc):
            left, top, right, bottom = _box(n)
            if left == layout.APPLY_AWAY:
                (moves,) = [r for r in n["rules"] if r["targetProperty"] == "x"]
                left = moves["conditions"][0]["value"]
                right = left + n["geometry"]["width"]
            assert left >= 0 and top >= 0, (name, n["type"], n["geometry"])
            assert right <= size["width"] + 0.5 and bottom <= size["height"] + 0.5, (name, n)


def test_every_colour_is_a_role_allowed_for_its_element(panel_graph) -> None:
    for name, doc in _docs(panel_graph):
        for n in _nodes(doc):
            for (kind, prop), roles in ALLOWED.items():
                if n["type"] == kind and isinstance(n["props"].get(prop), dict):
                    assert n["props"][prop]["$role"] in roles, (name, kind, prop)


def test_a_bend_on_a_strip_is_at_least_48_px_tall_and_never_below_36(panel_graph) -> None:
    elements = [element(f"B{i}", float(i), "bend") for i in range(8)]
    doc = stub_render(strip_view(elements), panel_graph)
    bends = [n for n in _nodes(doc) if str(n["props"].get("tooltip", "")).startswith("B")]
    bends = [n for n in bends if n["type"] == "polygon"]
    assert bends and all(n["geometry"]["height"] >= 48 for n in bends)
    assert layout.MIN_STRIP_SCALE * bends[0]["geometry"]["height"] >= 36  # compressed


def test_a_strip_of_eight_or_more_spans_nine_tenths_of_the_width(panel_graph) -> None:
    for count in (8, 20, 300):
        elements = [element(f"E{i}", i * 0.7, "bpm") for i in range(count)]
        doc = stub_render(strip_view(elements), panel_graph)
        shapes = [
            n
            for n in _nodes(doc)
            if n["type"] == "polygon" and str(n["props"].get("tooltip", "")).startswith("E")
        ]
        left = min(_box(n)[0] for n in shapes)
        right = max(_box(n)[2] for n in shapes)
        assert right - left >= 0.9 * layout.STRIP_WIDTH, count


def test_a_badge_is_a_circle_of_at_least_14_px_on_a_stem(panel_graph) -> None:
    view = place_view(with_symbols(panel_graph, SYMBOLS, ABOVE), "XR/XR01")
    doc = stub_render(view, panel_graph)
    (circle,) = [
        n
        for n in _nodes(doc)
        if n["type"] == "polygon"
        and n["props"].get("tooltip") == "XR01:IP1"
        and len(n["props"]["points"]) > 8  # not its ring
    ]
    assert circle["geometry"]["width"] >= 14 and circle["geometry"]["height"] >= 14
    assert len(circle["props"]["points"]) >= 12  # a circle, not a box


def test_leaders_run_below_the_names_and_never_cross(panel_graph) -> None:
    elements = [element(f"E{i}", i * 0.37 + (i % 3) * 0.1, "bpm") for i in range(30)]
    doc = stub_render(strip_view(elements), panel_graph)
    names = [
        n for n in _nodes(doc) if n["type"] == "text-label" and n["props"]["text"].startswith("E")
    ]
    leaders = [
        n
        for n in _nodes(doc)
        if n["type"] == "line"
        and "tooltip" not in n["props"]
        and n["props"]["lineColor"] == symbols.MUTED_TEXT
        and n["geometry"]["height"] >= layout.LEADER_DROP
    ]
    assert len(leaders) == len(elements)
    bottom = max(_box(n)[3] for n in names)
    ends = []
    for n in leaders:
        (a, b) = n["props"]["points"]
        top, low = (a, b) if a["y"] < b["y"] else (b, a)
        assert n["geometry"]["y"] >= bottom  # a clear band under the names
        ends.append((n["geometry"]["x"] + top["x"], n["geometry"]["x"] + low["x"]))
    ends.sort()
    assert [low for _top, low in ends] == sorted(low for _top, low in ends)  # no crossing


def test_a_place_view_names_its_place_once(panel_graph) -> None:
    view = place_view(with_symbols(panel_graph, SYMBOLS, ABOVE), "XR/XR01")
    doc = stub_render(view, panel_graph)
    texts = [n["props"]["text"] for n in _nodes(doc) if n["type"] == "text-label"]
    assert sum("XR/XR01" in t for t in texts) == 1  # in the header
