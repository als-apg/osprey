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


def _cut_label(screen, x: float, y: float, whole: str) -> None:
    screen.add(
        {
            "type": "text-label",
            "geometry": {"x": x, "y": y, "width": 60, "height": 16},
            "props": {"text": layout._Cut(whole[:8] + "…", whole), "fontSize": 11},
        }
    )


def test_a_link_over_two_cut_labels_joins_their_whole_texts_line_by_line():
    screen = layout._Screen("card-x", "x")
    _cut_label(screen, 10, 10, "first label of a stated length")
    _cut_label(screen, 10, 30, "second label of a stated length")
    _cut_label(screen, 200, 10, "a label outside the link")
    tip = screen.link_tip("first label", 0, 0, 100, 60)
    # the link's own label is in the first text, so it is not added again
    assert tip == "first label of a stated length\nsecond label of a stated length"
    assert screen.link_tip("Open the device", 0, 0, 100, 60).splitlines()[0] == "Open the device"


def test_a_link_s_tooltip_adds_no_text_another_line_already_holds():
    screen = layout._Screen("card-x", "x")
    _cut_label(screen, 10, 10, "XR02S Straight")  # a place's last part, cut
    _cut_label(screen, 10, 30, "+1.02 % / +3.35 %")  # the figures the label holds
    tip = screen.link_tip("XR/XR02/XR02S Straight: +1.02 % / +3.35 %", 0, 0, 100, 60)
    assert tip == "XR/XR02/XR02S Straight: +1.02 % / +3.35 %"
    # and a whole path drawn after its last part keeps the path alone
    screen = layout._Screen("card-x", "x")
    _cut_label(screen, 10, 10, "XR02S Straight")
    _cut_label(screen, 10, 30, "XR/XR02/XR02S Straight")
    assert screen.link_tip("XR02S", 0, 0, 100, 60) == "XR/XR02/XR02S Straight"


def test_a_hover_label_is_one_line(screens):
    """The viewer draws ``hoverLabel`` as one line at the widget, breaks and all
    run together; the lines a link adds go to its tooltip only."""
    broken = [
        (name, node["id"])
        for name, doc in screens.items()
        for node in _nodes(doc["root"])
        if "\n" in str(node["props"].get("hoverLabel", ""))
    ]
    assert broken == []


def test_no_hover_label_is_longer_than_its_bound(screens):
    """A hover label is drawn as one line over what stands beside its box: on
    every screen it holds at most :data:`HOVER_LABEL_MOST` characters, and on a
    link it is the link's own label, which opens its tooltip."""
    long, other = [], []
    for name, doc in screens.items():
        for node in _nodes(doc["root"]):
            said = str(node["props"].get("hoverLabel", ""))
            if len(said) > layout.HOVER_LABEL_MOST:
                long.append((name, said))
            tip = str(node["props"].get("tooltip", ""))
            if said and node["props"].get("actions") and not tip.startswith(said.rstrip("…")):
                other.append((name, said, tip))
    assert long == []
    assert other == []


def test_a_hover_label_past_its_bound_is_cut_and_keeps_one_line():
    bound = layout.HOVER_LABEL_MOST
    assert layout._hover_text("open XR01:QF1") == "open XR01:QF1"
    cut = layout._hover_text("x" * (bound + 30))
    assert len(cut) == bound and cut.endswith("…")
    assert layout._hover_text("first line\nsecond line") == "first line"


def test_a_link_over_more_cut_labels_than_a_bubble_shows_counts_the_rest():
    screen = layout._Screen("card-x", "x")
    many = layout.TOOLTIP_LINES + 5
    for i in range(many):
        _cut_label(screen, 10, 20 * i, f"label number {i:02d} of a stated length")
    lines = screen.link_tip("label number 00", 0, 0, 100, 20 * many).splitlines()
    assert len(lines) == layout.TOOLTIP_LINES
    assert lines[-1] == f"and {many - layout.TOOLTIP_LINES + 1} more"


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

    missing, shown, linked = [], 0, 0
    for name, doc in screens.items():
        found = list(_nodes(doc["root"]))
        hovers = [n for n in found if n["type"] == "rectangle" and n["props"].get("tooltip")]
        for node in found:
            text = node["props"].get("text", "")
            if node["type"] != "text-label" or node.get("rules") or not text.endswith("…"):
                continue
            kept = text[:-1].rstrip()
            boxes = [h for h in hovers if covers(h, node["geometry"])]
            # the pointer reaches the box drawn last: a link over the label too
            if boxes and kept in boxes[-1]["props"]["tooltip"]:
                shown += 1
                linked += bool(boxes[-1]["props"].get("actions"))
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
    assert "Roll, Yaw" not in words  # the marks point at the rows: titles not said again
    assert "at 20 C" in words  # a row's own conditions stay with it
    gain = next(n for n in texts if n["props"]["text"] == "Gain")
    shared = next(n for n in texts if n["props"]["text"].startswith("in the support frame"))
    assert shared["geometry"]["y"] > gain["geometry"]["y"]  # under the rows
    assert not any(n["props"]["text"].endswith("…") for n in texts if "frame" in n["props"]["text"])


def test_shared_conditions_stand_apart_under_a_rule_tied_to_their_rows_by_a_mark(panel_graph):
    """The shared line must not read as the last row's note: half a row of space and
    a thin rule set it off, and a reference mark, readable at the card's small size,
    follows the title of every row it holds for and starts the line; a second shared
    text gets the next mark."""
    from osprey.facility.panels.view import CalibrationRow, Calibrations, Section, Target, View
    from tests.facility.panels._views import stub_render

    def row(title: str, conditions: str | None) -> CalibrationRow:
        return CalibrationRow(
            title, conditions, "0.1 mrad", None, "stated", "stated", False, None, None, None
        )

    frame, warm = "in the support frame", "at 20 C"
    rows = (row("Roll", frame), row("Gain", warm), row("Yaw", frame), row("Offset", warm))
    rows += (row("Tilt", "alone"),)
    view = View(
        Target("card", "device", "XR01:PS1"), "XR01:PS1", (),
        (Section("calibration", (Calibrations(rows),)),),
    )  # fmt: skip
    doc = stub_render(view, panel_graph)
    texts = [n for n in _nodes(doc["root"]) if n["type"] == "text-label"]

    def at(text: str) -> dict[str, Any]:
        return next(n for n in texts if n["props"]["text"] == text)

    marks = {m: [n for n in texts if n["props"]["text"] == m] for m in ("1", "2")}
    for mark, titles in (("1", ("Roll", "Yaw")), ("2", ("Gain", "Offset"))):
        assert len(marks[mark]) == len(titles) + 1  # after each title, and at its line
        for title in titles:
            g = at(title)["geometry"]
            assert any(
                m["geometry"]["y"] == g["y"] and m["geometry"]["x"] > g["x"] for m in marks[mark]
            )
        assert all(m["props"]["fontSize"] >= layout.SMALL_SIZE for m in marks[mark])
    first = at(frame)  # the text alone: its mark points at its rows
    second = at(warm)
    assert first["geometry"]["y"] < second["geometry"]["y"]  # in the order the rows state
    line_mark = max(marks["1"], key=lambda m: m["geometry"]["y"])
    assert line_mark["geometry"]["y"] == first["geometry"]["y"]
    assert line_mark["geometry"]["x"] < first["geometry"]["x"]
    assert "alone" in " ".join(n["props"]["text"] for n in texts)  # its own row keeps it
    last = max(
        n["geometry"]["y"] + n["geometry"]["height"]
        for n in texts
        if n["geometry"]["y"] < first["geometry"]["y"] - layout.ROW_GAP
    )
    rules = [
        n for n in _nodes(doc["root"])
        if n["type"] == "line" and last <= n["geometry"]["y"] <= first["geometry"]["y"]
    ]  # fmt: skip
    (rule,) = rules  # one thin rule between the rows and the shared lines
    assert rule["geometry"]["y"] - last >= (layout.ROW + layout.ROW_GAP) / 2


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


def _texts_at(doc: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    found: dict[str, list[dict[str, Any]]] = {}
    for node in _nodes(doc["root"]):
        text = node["props"].get("text") or node["props"].get("label")
        if isinstance(text, str):
            found.setdefault(text, []).append(node)
    return found


def test_one_short_row_of_chips_shares_its_caption_s_line(screens):
    """A section of chips that fit beside their caption is one line high."""
    seen = 0
    for doc in screens.values():
        texts = _texts_at(doc)
        for caption in ("SETS", "CALIBRATION"):
            for head in texts.get(caption, []):
                y = head["geometry"]["y"]
                beside = [
                    n
                    for n in _nodes(doc["root"])
                    if n["type"] == "rectangle"
                    and n["props"].get("actions")
                    and n["geometry"]["x"] > head["geometry"]["x"] + head["geometry"]["width"]
                    and abs(n["geometry"]["y"] - y) < layout.ROW
                ]
                seen += bool(beside)
    assert seen >= 3  # the fixture's sets and calibration chips


def test_a_count_shares_its_heading_s_line(screens):
    for name in ("set/set-xr-quad", "calibration/calibration-xr-quad-centre"):
        doc = screens[f"{name}.screen.json"]
        texts = _texts_at(doc)
        (head,) = texts["MEMBERS"]
        count = next(
            n for t, ns in texts.items() if t.endswith("ROWS") or " ROWS · " in t for n in ns
        )
        assert count["geometry"]["y"] == head["geometry"]["y"], name
        assert count["props"]["foregroundColor"] == head["props"]["foregroundColor"]  # muted


def test_a_one_line_legend_sits_on_the_line_it_explains(screens):
    doc = screens["calibration/calibration-xr-quad-centre.screen.json"]
    texts = _texts_at(doc)
    (head,) = texts["MEMBERS"]
    for word in ("design only", "surveyed"):
        assert texts[word][0]["geometry"]["y"] == head["geometry"]["y"]


def test_a_strip_with_nothing_placed_draws_no_legend_of_looks(panel_graph):
    from osprey.facility.panels.view import TrayEntry
    from tests.facility.panels._views import strip_view, stub_render

    doc = stub_render(
        strip_view([], (TrayEntry("XR01:QF1", None, "no position", None),)), panel_graph
    )
    assert not any(layout._LOOKS_LEGEND == t for t in _texts_at(doc))


def test_the_header_names_the_page_once(screens):
    twice = {}
    for name, doc in screens.items():
        words = [
            n["props"].get("text") or n["props"].get("label")
            for n in _nodes(doc["root"])
            if n["geometry"]["y"] < layout.HEADER_HEIGHT
            and n["type"] in ("text-label", "pv-button")
        ]
        words = [w for w in words if w and w != "›" and not w.startswith("model ")]
        repeated = {w for w in words if words.count(w) > 1}
        if repeated:
            twice[name] = repeated
    assert twice == {}


def _items(doc: dict[str, Any]) -> dict[str, dict[int, tuple[float, float, float, float]]]:
    """Per item kind, per stated order: the box its widgets take (x0, y0, x1, y1)."""
    found: dict[str, dict[int, tuple[float, float, float, float]]] = {}
    for node in _nodes(doc["root"]):
        comment = (node.get("meta") or {}).get("comment", "")
        tags = dict(t.split(": ", 1) for t in comment.split("; ") if ": " in t)
        if "item" not in tags or "order" not in tags:
            continue
        g = node["geometry"]
        box = (g["x"], g["y"], g["x"] + g["width"], g["y"] + g["height"])
        held = found.setdefault(tags["item"], {}).get(int(tags["order"]))
        if held is not None:
            box = (min(held[0], box[0]), min(held[1], box[1]),
                   max(held[2], box[2]), max(held[3], box[3]))  # fmt: skip
        found[tags["item"]][int(tags["order"])] = box
    return found


def test_items_read_in_the_order_their_view_states(screens):
    """Every widget of a view item names it (kind, and its order where the view
    states one); the items read on the screen in that order along the way their
    kind reads (:data:`layout.ITEM_TAGS`): across, left to right and on in rows
    below; down, top to bottom and on in columns to the right."""
    wrong, seen = [], set()
    for name, doc in screens.items():
        for kind, boxes in _items(doc).items():
            assert kind in layout.ITEM_TAGS, kind
            seen.add(kind)
            ordered = [boxes[k] for k in sorted(boxes)]
            for i in range(1, len(ordered)):
                before, done = ordered[i - 1], ordered[:i]
                x0, y0 = ordered[i][0], ordered[i][1]
                if layout.ITEM_TAGS[kind] == "across":
                    on = x0 > before[0] or y0 >= max(b[3] for b in done) - 0.01
                else:
                    on = (
                        y0 > before[1]
                        or (y0 == before[1] and x0 > before[0])
                        or x0 >= max(b[2] for b in done) - 0.01
                    )
                if not on:
                    wrong.append((name, kind))
    assert wrong == []
    assert {"key-tile", "chain-block", "grid-row", "rack-unit", "strip-element"} <= seen


def test_item_tags_are_no_part_of_a_drawing():
    from osprey.facility.panels.emit import _without_meta

    plain = {"root": {"children": [{"id": "w-0001", "props": {}}]}, "metadata": {}}
    tagged = {"root": {"children": [{"id": "w-0001", "props": {}, "meta": {"comment": "item: x"}}]},
              "metadata": {}}  # fmt: skip
    assert _without_meta(plain, "0") == _without_meta(tagged, "0")
    with pytest.raises(ValueError, match="no item kind"):
        layout.item_tags("no-such-kind")
