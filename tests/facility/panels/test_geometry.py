"""The geometry every written screen holds to, read from the files the engine writes.

Nothing is rendered: text is measured in the viewer's font (:mod:`metrics`), as
the layout measures it. Every screen of the fixture facility is checked:

- a text label's text fits its box, in width and in line height, so the viewer
  never shrinks or clips it;
- no widget reaches outside the canvas, but the Apply buttons parked away by
  design (:data:`layout.APPLY_AWAY`);
- no two text labels draw over each other;
- the page never asks for a face the viewer's font set lacks: a plot carries no
  title of its own.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from osprey.facility.panels import layout, load_graph, metrics, symbols
from osprey.facility.panels.emit import load_draw, write
from tests._by_path import load_module
from tests.facility.panels._fixture import PANELS_DIR, write_fixture_facility

#: The script that writes :mod:`metrics` from the viewer's font file.
FONT_METRICS = Path(__file__).parents[3] / "scripts" / "panel_font_metrics.py"

#: Below this many px of shared width or height, two texts only touch.
TOUCH = 2


def _nodes(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        if "type" in node and "geometry" in node:
            yield node
        for value in node.values():
            yield from _nodes(value)
    elif isinstance(node, list):
        for value in node:
            yield from _nodes(value)


def _ink(text: str, size: float) -> float:
    """The width of *text*'s letters at *size*, and the viewer's inset either side."""
    em = sum(metrics.WIDTHS.get(c, metrics.UNLISTED) for c in text)
    return em * size / metrics.PER_EM + 2 * layout.LABEL_INSET


@pytest.fixture(scope="module")
def screens(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Any]]:
    tmp = tmp_path_factory.mktemp("geometry")
    graph = load_graph(write_fixture_facility(tmp / "facility.json"), PANELS_DIR)
    write(graph, tmp / "out", draw=load_draw(PANELS_DIR / "draw.json"))
    return {
        str(p.relative_to(tmp / "out")): json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((tmp / "out").rglob("*.screen.json"))
    }


def _texts(doc: dict[str, Any]) -> list[tuple[str, str, float, float, float, float]]:
    """Each text label's id, text and the box its letters take: x, y, width, height."""
    found = []
    for node in _nodes(doc["root"]):
        props, g = node["props"], node["geometry"]
        if node["type"] != "text-label" or not str(props.get("text", "")).strip():
            continue
        size = props.get("fontSize", layout.TEXT_SIZE)
        w = min(_ink(props["text"], size), g["width"])
        align = props.get("align", "left")
        x = {"left": g["x"], "right": g["x"] + g["width"] - w}.get(
            align, g["x"] + (g["width"] - w) / 2
        )
        y = g["y"] + (g["height"] - size) / 2  # one line, centred in its box
        found.append((node["id"], props["text"], x, y, w, size))
    return found


def test_every_text_fits_its_box(screens):
    cut = [
        (name, node["id"], node["props"]["text"])
        for name, doc in screens.items()
        for node in _nodes(doc["root"])
        if node["type"] == "text-label"
        and node["props"].get("text")
        and _ink(node["props"]["text"], node["props"].get("fontSize", layout.TEXT_SIZE))
        > node["geometry"]["width"] + 0.01
    ]
    assert cut == []


def test_every_text_label_is_one_line_high(screens):
    low = [
        (name, node["props"]["text"], node["geometry"]["height"], node["props"]["fontSize"])
        for name, screen in screens.items()
        for node in _nodes(screen["root"])
        if node["type"] == "text-label"
        and node["geometry"]["height"] < layout.line_height(node["props"]["fontSize"])
    ]
    assert low == []


def test_a_lights_mark_gets_a_line_of_its_size():
    _shape, mark = symbols.light(10, 30, layout.LIGHT_SIZE, "unknown")
    drawn = layout._Screen("card-x", "x").add(mark)
    size = mark["props"]["fontSize"]
    assert drawn["geometry"]["height"] >= layout.line_height(size) > layout.LIGHT_SIZE
    # grown about the light's middle, so the mark stays centred in it
    middle = drawn["geometry"]["y"] + drawn["geometry"]["height"] / 2
    assert middle == pytest.approx(30 + layout.LIGHT_SIZE / 2, abs=0.5)
    # and every mark a light shows is measured and fits its width
    for glyph in (symbols.UNKNOWN_MARK, symbols.DISAGREE_MARK, symbols.AGGREGATE_MARK):
        assert glyph in metrics.WIDTHS
        ink = metrics.WIDTHS[glyph] * size / metrics.PER_EM + 2 * layout.LABEL_INSET
        assert ink <= drawn["geometry"]["width"]


def test_every_widget_stands_on_whole_pixels(screens):
    """Edges on fractions of a pixel draw soft: every box, and every point of a
    line or shape, is on a whole pixel."""
    off = []
    for name, doc in screens.items():
        for node in _nodes(doc["root"]):
            g = node["geometry"]
            points = node["props"].get("points") or ()
            values = [g["x"], g["y"], g["width"], g["height"]]
            values += [v for p in points for v in (p["x"], p["y"])]
            if any(not float(v).is_integer() for v in values):
                off.append((name, node["id"]))
    assert off == []


def test_the_canvas_ends_one_margin_under_the_content(screens):
    """Every screen, of every kind, ends :data:`layout.PAGE_END` under its lowest
    widget; nothing more, so the viewer draws it as large as it can."""
    wrong = {}
    for name, doc in screens.items():
        bottom = max(
            n["geometry"]["y"] + n["geometry"]["height"]
            for n in _nodes(doc["root"])
            if n["type"] != "absolute-canvas" and n["geometry"]["x"] != layout.APPLY_AWAY
        )
        height = doc["metadata"]["size"]["height"]
        if height != doc["root"]["geometry"]["height"] or height - bottom != layout.PAGE_END:
            wrong[name] = height - bottom
    assert wrong == {}


def test_no_widget_reaches_outside_the_canvas(screens):
    outside = []
    for name, doc in screens.items():
        size = doc["metadata"]["size"]
        for node in _nodes(doc["root"]):
            g = node["geometry"]
            if node["type"] == "absolute-canvas" or g["x"] == layout.APPLY_AWAY:
                continue
            if (
                g["width"] < 0
                or g["height"] < 0
                or g["x"] < 0
                or g["y"] < 0
                or g["x"] + g["width"] > size["width"] + 0.01
                or g["y"] + g["height"] > size["height"] + 0.01
            ):
                outside.append((name, node["id"], node["type"]))
    assert outside == []


#: The widgets that print a channel's value in a box of their own.
VALUE_WIDGETS = frozenset({"pv-label", "pv-text-input", "pv-progress-bar"})


def test_no_text_lies_under_a_value(screens):
    under = []
    for name, doc in screens.items():
        values = [n for n in _nodes(doc["root"]) if n["type"] in VALUE_WIDGETS]
        for t_id, text, x, y, w, h in _texts(doc):
            for value in values:
                g = value["geometry"]
                if (
                    x < g["x"] + g["width"] - TOUCH
                    and g["x"] < x + w - TOUCH
                    and y < g["y"] + g["height"] - TOUCH
                    and g["y"] < y + h - TOUCH
                ):
                    under.append((name, t_id, text, value["id"]))
    assert under == []


def test_no_text_lies_over_another(screens):
    over = []
    for name, doc in screens.items():
        texts = _texts(doc)
        for i, (a_id, a_text, ax, ay, aw, ah) in enumerate(texts):
            for b_id, b_text, bx, by, bw, bh in texts[i + 1 :]:
                if (
                    ax < bx + bw - TOUCH
                    and bx < ax + aw - TOUCH
                    and ay < by + bh - TOUCH
                    and by < ay + ah - TOUCH
                ):
                    over.append((name, a_id, a_text, b_id, b_text))
    assert over == []


def test_a_plot_carries_no_title_of_its_own(screens):
    titled = [
        (name, node["id"])
        for name, doc in screens.items()
        for node in _nodes(doc["root"])
        if node["type"] == "waveform-plot" and "title" in node["props"]
    ]
    assert titled == []


def test_the_metrics_measure_the_font_the_layout_names():
    assert metrics.FONT
    assert metrics.UNLISTED == max(metrics.WIDTHS.values())
    # every printable character a panel writes is measured, not guessed
    assert all(chr(c) in metrics.WIDTHS for c in range(0x20, 0x7F))
    assert layout.text_width("ab", 10) < layout.text_width("AB", 10)
    assert layout.text_width("ab", 10, caps=True) == layout.text_width("AB", 10)


def test_the_metrics_table_is_the_font_file_it_names():
    script = load_module("panel_font_metrics", FONT_METRICS)
    family, version, widths, line = script.measure()
    # regenerate the module from the font file: it is the module in the tree
    assert metrics.FONT == family
    assert metrics.WIDTHS == widths
    assert metrics.LINE == line
    assert f"from {family}, {version};" in (metrics.__doc__ or "")
    assert (script.FONT_FILE.parent / "OFL.txt").is_file()  # the font's licence beside it


def test_a_cut_text_fits_and_is_counted():
    cuts: list[str] = []
    token = layout._CUTS.set(cuts)
    try:
        text = "a long name that a narrow box cannot hold"
        cut = layout._cut(text, 80, layout.SMALL_SIZE)
    finally:
        layout._CUTS.reset(token)
    assert cut.endswith(layout._ELLIPSIS)
    assert layout.text_width(cut, layout.SMALL_SIZE) <= 80
    assert cuts == [text]


def test_a_checkout_s_name_columns_take_their_longest_names_and_stay_on_the_page():
    from osprey.facility.panels.view import CheckRow

    def row(channel: str, thing: str | None) -> CheckRow:
        return CheckRow(channel, thing, True, True, True, True)

    short = [row("XR01-PS1:On-Sts", "XR01:PS1")]
    assert layout.check_columns(short) == (layout.NAME_WIDTH, layout.NAME_WIDTH)
    long_name = "XR01-PS1:" + "VeryLongReadbackName" * 2 + "-RB"
    channel_w, thing_w = layout.check_columns([*short, row(long_name, "XR01:PS1")])
    assert channel_w >= layout.text_width(long_name, layout.TEXT_SIZE)  # whole, no cut
    assert thing_w == layout.NAME_WIDTH
    room = layout.STRIP_WIDTH - layout.CHECK_FLAGS * layout.CELL_WIDTH
    assert channel_w + thing_w <= room  # the flag columns still fit the page
    # both longer than the room: the thing column keeps its least, the channel the rest
    huge = "X" * 120
    channel_w, thing_w = layout.check_columns([row(huge, huge)])
    assert thing_w == layout.NAME_LEAST and channel_w + thing_w == room


def test_a_line_of_prose_takes_the_lines_it_needs_and_loses_no_text():
    text = (
        "source XR-0001-0002 B.5 (drawing) released in XR-0003-0004 A.12 feature 7, "
        "path XR-0005-0006/XR-0003-0004/XR-0001-0002; the rest of the note follows"
    )
    width = 240
    lines = layout._prose(text, width, layout.SMALL_SIZE)
    assert len(lines) > layout.TILE_LINES  # more than the old limit of two
    assert all(
        layout.text_width(line, layout.SMALL_SIZE) <= width + layout.TEXT_PAD for line in lines
    )
    assert not any(line.endswith("…") for line in lines)  # nothing cut
    # breaking at a space, a slash, a comma or a semicolon keeps every character
    assert "".join(lines).replace(" ", "") == text.replace(" ", "")


def test_every_cut_label_carries_its_whole_text_as_a_tooltip(screens):
    def covers(box: dict[str, Any], g: dict[str, Any]) -> bool:
        b = box["geometry"]
        return (
            b["x"] <= g["x"] + 0.01
            and b["y"] <= g["y"] + 0.01
            and b["x"] + b["width"] >= g["x"] + g["width"] - 0.01
            and b["y"] + b["height"] >= g["y"] + g["height"] - 0.01
        )

    missing, shown = [], 0
    for name, doc in screens.items():
        found = list(_nodes(doc["root"]))
        hovers = [n for n in found if n["type"] == "rectangle" and n["props"].get("tooltip")]
        for node in found:
            text = node["props"].get("text", "")
            if node["type"] != "text-label" or node.get("rules") or not text.endswith("…"):
                continue
            kept = text[:-1].rstrip()
            boxes = [h for h in hovers if covers(h, node["geometry"])]
            if any(kept in h["props"]["tooltip"] or h["props"].get("actions") for h in boxes):
                shown += 1
            else:
                missing.append((name, text))
    assert missing == []
    assert shown  # the fixture cuts some labels, so the rule is exercised


def test_a_fact_s_long_note_wraps_beside_it_and_the_row_grows(panel_graph):
    from osprey.facility.panels.view import Fact, Section, Target, View
    from tests.facility.panels._views import stub_render

    caveat = "read from the drawing's own frame, " * 8 + "the rest of what it assumes"
    here = Target("card", "device", "XR01:PS1")
    view = View(
        here,
        "XR01:PS1",
        (),
        (Section("where", (Fact("pose", "s 1.0 m", caveat=caveat), Fact("after", "next"))),),
    )
    doc = stub_render(view, panel_graph)
    texts = [n for n in _nodes(doc["root"]) if n["type"] == "text-label"]
    note = [n for n in texts if n["props"]["text"] and n["props"]["text"] in caveat]
    assert len(note) > 1 and not any(n["props"]["text"].endswith("…") for n in note)
    after = next(n for n in texts if n["props"]["text"] == "after")
    assert after["geometry"]["y"] >= max(n["geometry"]["y"] + n["geometry"]["height"] for n in note)


def test_conditions_several_calibration_rows_share_are_stated_once_under_them(panel_graph):
    from osprey.facility.panels.view import CalibrationRow, Calibrations, Section, Target, View
    from tests.facility.panels._views import stub_render

    frame = "in the support frame; x across, y up, z along the beam, angles seen from upstream"

    def row(title: str, conditions: str | None) -> CalibrationRow:
        return CalibrationRow(
            title, conditions, "0.1 mrad", None, "stated", "stated", False, None, None, None
        )

    table = Calibrations((row("Roll", frame), row("Yaw", frame), row("Gain", "at 20 C")))
    view = View(
        Target("card", "device", "XR01:PS1"), "XR01:PS1", (), (Section("calibration", (table,)),)
    )
    doc = stub_render(view, panel_graph)
    texts = [n for n in _nodes(doc["root"]) if n["type"] == "text-label"]
    words = " ".join(n["props"]["text"] for n in texts)
    assert words.count("support frame") == 1  # once, not per row
    assert "Roll, Yaw: in the support frame" in words
    assert "at 20 C" in words  # a row's own conditions stay with it
    gain = next(n for n in texts if n["props"]["text"] == "Gain")
    shared = next(n for n in texts if n["props"]["text"].startswith("Roll, Yaw:"))
    assert shared["geometry"]["y"] > gain["geometry"]["y"]  # under the rows
    assert not any(n["props"]["text"].endswith("…") for n in texts if "frame" in n["props"]["text"])


def test_a_name_with_no_separator_breaks_where_a_small_letter_meets_a_capital():
    width = layout.text_width("Electronics", layout.SMALL_SIZE) + 4
    assert layout._wrap_label("ElectronicsChassis", width, layout.SMALL_SIZE, 2) == [
        "Electronics",
        "Chassis",
    ]
    # a name that fits, or one with separators, breaks as before
    assert layout._wrap_label("PowerSupply", 400, layout.SMALL_SIZE, 2) == ["PowerSupply"]
    assert layout._wrap_label("XR01-PS1:On-Sts", 400, layout.SMALL_SIZE, 2) == ["XR01-PS1:On-Sts"]


def _frames(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """The section frames of *doc*: the boxes in the margin that sections stand in."""
    left = layout.MARGIN - layout.PANEL_PAD
    return [
        n["geometry"]
        for n in _nodes(doc["root"])
        if n["type"] == "rectangle"
        and n["props"].get("fillColor") == layout.PANEL_BACKGROUND
        and n["props"].get("lineColor") == layout.BOX_BORDER
        and n["geometry"]["x"] in (left, left + layout.HALF_FRAME + layout.PANEL_GAP)
        and n["geometry"]["width"] in (layout.HALF_FRAME, layout.STRIP_WIDTH + 2 * layout.PANEL_PAD)
    ]


def test_narrow_frames_stand_side_by_side_in_two_columns(screens):
    """Width before height: frames whose content needs at most half the width
    stand in two columns, each beside a frame of the other column, and hold
    everything drawn there; no two frames overlap."""
    pairs = 0
    for name, doc in screens.items():
        frames = _frames(doc)
        halves = [g for g in frames if g["width"] == layout.HALF_FRAME]
        lefts = [g for g in halves if g["x"] == layout.MARGIN - layout.PANEL_PAD]
        rights = [g for g in halves if g not in lefts]
        runs: list[list[dict[str, Any]]] = []  # frames set together, top to foot
        for a in sorted(halves, key=lambda g: g["y"]):
            if runs and a["y"] <= max(g["y"] + g["height"] for g in runs[-1]) + layout.PANEL_GAP:
                runs[-1].append(a)
            else:
                runs.append([a])
        for run in runs:  # a half frame always has a column beside it
            assert {g["x"] for g in run} == {lefts[0]["x"], rights[0]["x"]}, (name, run)
        for a in halves:
            pairs += a in lefts
            inside = [
                n["geometry"]
                for n in _nodes(doc["root"])
                if n["type"] != "absolute-canvas"
                and n["geometry"] not in halves
                and a["y"] < n["geometry"]["y"] < a["y"] + a["height"]
                and a["x"] <= n["geometry"]["x"] < a["x"] + a["width"]
            ]
            for g in inside:
                assert g["x"] + g["width"] <= a["x"] + a["width"], (name, g)
        for a, b in itertools.combinations(frames, 2):  # frames never overlap
            assert (
                a["x"] + a["width"] <= b["x"] or b["x"] + b["width"] <= a["x"]
                or a["y"] + a["height"] <= b["y"] or b["y"] + b["height"] <= a["y"]
            ), name  # fmt: skip
    assert pairs >= 5  # the fixture has short sections to pair


def _grouped_frames(panel_graph, sizes: tuple[int, ...]) -> list[dict[str, Any]]:
    """The frames of a channel list of groups of one stated grouping, *sizes*
    rows each, in order."""
    import dataclasses

    from osprey.facility.panels.describe import channel_list
    from osprey.facility.panels.view import Section, ValueRow
    from tests.facility.panels._views import stub_render

    def group(k: int, n: int) -> Section:
        rows = tuple(
            ValueRow(f"row {i}", f"XR01-PS1:G{k}R{i}-RB", None, None, False) for i in range(n)
        )
        return Section(f"group {k}", rows, "record type")

    base = channel_list(panel_graph, "XR01:PS1")
    view = dataclasses.replace(base, sections=tuple(group(k, n) for k, n in enumerate(sizes)))
    return _frames(stub_render(view, panel_graph))


def test_a_short_frame_beside_a_long_one_keeps_its_own_height(panel_graph):
    """A frame is stretched to its neighbour's height by at most a third of its
    own: a short group beside a long one keeps its height, and the page under it
    stays plain, no empty frame."""
    (alone,) = _grouped_frames(panel_graph, (6,))
    short, long = _grouped_frames(panel_graph, (6, 30))
    assert (short["y"], short["width"]) == (long["y"], layout.HALF_FRAME)
    assert short["x"] < long["x"]
    assert short["height"] == alone["height"] < long["height"]
    near, near_too = _grouped_frames(panel_graph, (6, 7))  # near in height: one height
    assert near["height"] == near_too["height"]
    assert all(type(v) is int for g in (near, near_too) for v in g.values())  # whole pixels
    assert near_too["height"] - _grouped_frames(panel_graph, (6,))[0]["height"] <= (
        layout.STRETCH_SHARE * alone["height"]
    )


def test_short_frames_stack_beside_a_long_one(panel_graph):
    """Three short groups beside one long one stack in one column, each at its
    own height, the long one alone in the other, read downward."""
    own = [_grouped_frames(panel_graph, (n,))[0]["height"] for n in (4, 5, 6)]
    for sizes in ((4, 5, 6, 30), (30, 4, 5, 6)):
        frames = _grouped_frames(panel_graph, sizes)
        long = max(frames, key=lambda g: g["height"])
        stack = [g for g in frames if g is not long]
        assert len({g["x"] for g in stack}) == 1 and stack[0]["x"] != long["x"]
        assert sorted(g["height"] for g in stack) == sorted(own)  # none stretched
        assert stack[0]["y"] == long["y"]
        for upper, lower in itertools.pairwise(sorted(stack, key=lambda g: g["y"])):
            assert lower["y"] == upper["y"] + upper["height"] + layout.PANEL_GAP
        assert stack[-1]["y"] + stack[-1]["height"] < long["y"] + long["height"]


def _paired_titles(doc: dict[str, Any]) -> list[tuple[str, str]]:
    """The headings of each two frames standing side by side, left then right."""
    half = layout.HALF_FRAME
    frames = [
        n["geometry"] for n in _nodes(doc["root"])
        if n["type"] == "rectangle" and n["geometry"]["width"] == half
        and n["props"].get("fillColor") == symbols.PANEL_BACKGROUND
    ]  # fmt: skip

    def title(g: dict[str, Any]) -> str:
        heads = sorted(
            (n for n in _nodes(doc["root"]) if n["type"] == "text-label"
             and abs(n["geometry"]["y"] - (g["y"] + layout.PANEL_PAD)) < 1
             and g["x"] <= n["geometry"]["x"] < g["x"] + g["width"]),
            key=lambda n: n["geometry"]["x"],
        )  # fmt: skip
        return str(heads[0]["props"]["text"]) if heads else ""

    rows: dict[float, list[dict[str, Any]]] = {}
    for g in frames:
        rows.setdefault(g["y"], []).append(g)
    return [
        (title(a), title(b))
        for a, b in (sorted(r, key=lambda g: g["x"]) for r in rows.values() if len(r) == 2)
    ]


def test_frames_pair_only_with_their_own_role(screens, panel_graph):
    """An action block never stands beside a reference list; on a channel list two
    groups pair only as stated groups of one grouping, and a group of no stated
    kind (the not stated, the leftovers) stands alone."""
    from osprey.facility.panels.describe import channel_list
    from osprey.facility.panels.view import Link, Section, Target, ValueRow

    pairs = [p for doc in screens.values() for p in _paired_titles(doc)]
    assert pairs  # some do pair
    actions = [p for p in pairs if any(t.endswith("OPERATE") for t in p)]
    assert not [p for p in actions if any("ALL CHANNELS" in t for t in p)]
    reference = Section("links", (Link("x", Target("card", "device", "X")),))
    live = Section("values", (ValueRow("a", "XR01-X:A-RB", None, None, False),))
    assert layout.frame_role(reference, grouped=False) == "reference"
    assert layout.frame_role(live, grouped=False) == "live"
    assert layout.frame_role(live, grouped=True) is None  # a group of no stated kind
    stated = Section("record type ai", live.items, "record type")
    assert layout.frame_role(stated, grouped=True) == "group: record type"
    view = channel_list(panel_graph, "XR01:PS1")
    unstated = [s for s in view.sections if s.grouping is None]
    assert all(view.sections.index(s) >= len(view.sections) - len(unstated) for s in unstated)
