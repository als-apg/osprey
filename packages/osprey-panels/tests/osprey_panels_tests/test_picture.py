"""A device class's drawing on its card, from ``pictures.csv``: the body, the parts, the legend.

Every class, number and name here is invented.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import pytest

from osprey_panels import layout, symbols
from osprey_panels.describe import device_card
from osprey_panels.model import load_graph
from osprey_panels.tables import COLUMNS, PanelTableError
from osprey_panels_tests._fixture import (
    add_part,
    copy_tables,
    read_fixture_tables,
    write_fixture_facility,
)
from osprey_panels_tests._views import bound, nodes, stub_render, with_symbols

PICTURES = "pictures.csv"
HEADER = ",".join(c for c in COLUMNS[PICTURES] if c not in ("under", "devices"))  # reads
CITE = "XR-DRAWING,B,released,sheet 2"
COLOUR = "#2f9e52"  # the fixture's quadrupole colour


def _row(part, kind, x, y, *, w="", h="", angle="", signal="", cls="Quadrupole") -> str:
    return f"{cls},end,{part},{kind},{x},{y},0,{CITE},{w},{h},{angle},{signal}"


def _six_poles() -> list[str]:
    """An invented six-pole body: a yoke, poles at 30 + 60 k degrees, a coil on each,
    two corrector windings, a bore."""
    rows = [_row("yoke", "yoke", 0, 0, w=300)]
    for k in range(6):
        angle = 30 + 60 * k
        t = math.radians(angle)
        rows.append(_row(f"pole {k + 1}", "pole", 60 * math.cos(t), 60 * math.sin(t), w=30, h=80, angle=angle))  # fmt: skip
        rows.append(_row(f"coil {k + 1}", "coil", 75 * math.cos(t), 75 * math.sin(t), w=12, h=40, angle=angle))  # fmt: skip
    rows.append(_row("corrector winding 1", "corrector_winding", 0, 120, w=4, h=30, angle=0))
    rows.append(_row("corrector winding 2", "corrector_winding", 0, -120, w=4, h=30, angle=0))
    rows.append(_row("bore", "bore", 0, 0, w=24))
    return rows


def _tables(tmp_path: Path, rows: list[str]) -> Path:
    tables = copy_tables(tmp_path / "panels")
    (tables / PICTURES).write_text("\n".join([HEADER, *rows]) + "\n")
    return tables


def _graph(tmp_path: Path, rows: list[str], rule: bool = False, change=None):
    tables = _tables(tmp_path, rows)
    if rule:
        with (tables / "health.csv").open("a") as handle:
            handle.write("XR01-QF1:Current-SP,1,bad,screen,stated,xr-screens,4d5e6f7,m.screen:1\n")
    return load_graph(write_fixture_facility(tmp_path / "f.json", change), tables)


def _switches(document) -> None:
    """Two thermal-switch devices, each ``part_of`` the magnet, each with a status channel."""
    for n in (1, 2):
        add_part(document, f"XR01:QF1-TS{n}", "XR01:QF1", f"XR01-QF1-TS{n}:Open-Sts")


def _card(graph) -> dict:
    view = device_card(with_symbols(graph, {"Quadrupole": "quadrupole"}), "XR01:QF1")
    return stub_render(view, graph, colours={"quadrupole": COLOUR})


def _by_type(doc: dict, kind: str) -> list[dict]:
    return [n for n in nodes(doc) if n["type"] == kind]


def _tipped(doc: dict, tip: str) -> list[dict]:
    return [n for n in nodes(doc) if n["props"].get("tooltip") == tip]


def _texts(doc: dict) -> list[str]:
    return [n["props"]["text"] for n in _by_type(doc, "text-label")]


def _centre(node: dict) -> tuple[float, float]:
    g = node["geometry"]
    return g["x"] + g["width"] / 2, g["y"] + g["height"] / 2


# -- the table ---------------------------------------------------------------------


def test_the_old_eleven_column_file_still_reads(tmp_path) -> None:
    tables = copy_tables(tmp_path / "panels")
    header = "device_class,view,part,kind,x_mm,y_mm,z_mm,document,version,state,locator"
    (tables / PICTURES).write_text(f"{header}\nQuadrupole,end,pole 1,pole,1,2,3,d,v,s,l\n")
    (row,) = read_fixture_tables(tables).pictures
    assert (row.kind, row.w_mm, row.h_mm, row.angle_deg, row.signal) == ("pole", *[None] * 4)
    assert not row.shape  # a pole with no size: a part at a point, as before


def test_the_drawing_columns_read(tmp_path) -> None:
    rows = read_fixture_tables(_tables(tmp_path, _six_poles())).pictures
    pole = next(r for r in rows if r.part == "pole 2")
    assert (pole.w_mm, pole.h_mm, pole.angle_deg, pole.shape) == (30, 80, 90, True)
    assert next(r for r in rows if r.kind == "bore").h_mm is None


@pytest.mark.parametrize(
    ("row", "word"),
    [
        (_row("pole 1", "pole", 0, 50, w=0), "w_mm is 0"),
        (_row("coil 1", "coil", 0, 50, w=5, h=-1), "h_mm is -1"),
        (_row("pole 1", "pole", 0, 50, w="wide"), "not a number"),
    ],
)
def test_a_shape_row_states_a_positive_size(tmp_path, row: str, word: str) -> None:
    with pytest.raises(PanelTableError, match=word) as caught:
        read_fixture_tables(_tables(tmp_path, [row]))
    assert caught.value.file == PICTURES and caught.value.line == 2


# -- the signal --------------------------------------------------------------------


def test_a_signal_resolves_by_name_or_by_pattern_on_the_device(tmp_path) -> None:
    rows = [
        _row("switch 1", "thermal_switch", 50, 50, signal="current_readback"),
        _row("switch 2", "thermal_switch", -50, 50, signal="*:Current-SP"),
        _row("switch 3", "thermal_switch", -50, -50, signal="*:Nothing-*"),
        _row("switch 4", "thermal_switch", 50, -50),
    ]
    graph = _graph(tmp_path, rows)
    assert graph.picture_channel("XR01:QF1", "*:Current-SP") == "XR01-QF1:Current-SP"
    (section,) = [s for s in device_card(graph, "XR01:QF1").sections if s.title == "picture"]
    (picture,) = section.items
    assert [p.channel for p in picture.parts] == [
        "XR01-QF1:Current-RB",
        "XR01-QF1:Current-SP",
        None,
        None,
    ]


# -- the drawing -------------------------------------------------------------------


def test_six_poles_are_drawn_at_their_angles_each_with_its_coils_two_sides(tmp_path) -> None:
    doc = _card(_graph(tmp_path, _six_poles()))
    yoke, _window = _by_type_tip(doc, "ellipse", "yoke")
    cx, cy = _centre(yoke)
    for k in range(6):
        (pole,) = _tipped(doc, f"pole {k + 1}")
        assert pole["type"] == "polygon"
        g, pts = pole["geometry"], pole["props"]["points"]
        px = g["x"] + sum(q["x"] for q in pts) / len(pts)
        py = g["y"] + sum(q["y"] for q in pts) / len(pts)
        angle = math.degrees(math.atan2(cy - py, px - cx)) % 360
        assert angle == pytest.approx(30 + 60 * k, abs=1)
        sides = _tipped(doc, f"coil {k + 1}")
        assert len(sides) == 2 and all(s["type"] == "polygon" for s in sides)
        (ax, ay), (bx, by) = map(_centre, sides)
        # the two sides lie either side of the pole's axis, at the same reach
        assert math.dist((ax, ay), (cx, cy)) == pytest.approx(math.dist((bx, by), (cx, cy)), 0.01)
    assert len(_tipped(doc, "corrector winding 1")) == 2  # a band outside each coil block
    (bore,) = _tipped(doc, "bore")
    (beam,) = _tipped(doc, "beam")
    assert _centre(beam) == pytest.approx(_centre(bore), abs=0.5)
    assert _centre(bore) == pytest.approx((cx, cy), abs=0.5)


def _by_type_tip(doc: dict, kind: str, tip: str) -> list[dict]:
    return [n for n in _tipped(doc, tip) if n["type"] == kind]


def test_the_body_is_painted_by_material_and_the_class_colour_is_not_used(tmp_path) -> None:
    doc = _card(_graph(tmp_path, _six_poles()))
    yoke, window = (n["props"] for n in _by_type_tip(doc, "ellipse", "yoke"))
    assert yoke["fillColor"] == symbols.STEEL_DARK
    assert window["fillColor"] == symbols.PANEL_BACKGROUND  # a ring: its inside is clear
    pole, coil = (_tipped(doc, t)[0]["props"] for t in ("pole 1", "coil 1"))
    assert pole["fillColor"] == symbols.STEEL and coil["fillColor"] == symbols.COPPER
    assert coil["lineColor"] == symbols.COPPER_EDGE and not _tipped(doc, "coil 1")[0].get("rules")
    assert _tipped(doc, "corrector winding 1")[0]["props"]["fillColor"] == symbols.WINDING
    bore = next(n for n in _tipped(doc, "bore") if n["type"] == "ellipse")
    assert bore["props"]["fillColor"] == symbols.BORE
    assert _tipped(doc, "beam")  # the point; the word only where the bore has room
    assert COLOUR not in json.dumps(doc)  # material, not the class's strip colour


def test_a_coil_whose_row_reads_a_ruled_channel_has_a_state_outline(tmp_path) -> None:
    rows = [r.replace("coil 2,coil", "coil 2,coil") for r in _six_poles()]
    rows = [r + "*:Current-SP" if r.startswith("Quadrupole,end,coil 2,") else r for r in rows]
    doc = _card(_graph(tmp_path, rows, rule=True))
    sides = _tipped(doc, "coil 2")
    assert len(sides) == 2
    for side in sides:
        targets = {r["targetProperty"] for r in side.get("rules", ())}
        assert targets == {"lineColor", "lineWidth"}  # its fill stays copper
        assert bound(doc, side) == ("XR01-QF1:Current-SP",)
    assert not _tipped(doc, "coil 1")[0].get("rules")


def test_a_part_shows_its_state_where_it_sits_and_one_without_a_signal_is_plain(
    tmp_path,
) -> None:
    rows = [
        *_six_poles(),
        _row("switch 1", "thermal_switch", 90, 90, signal="*:Current-SP"),  # ruled: a light
        _row("gauge 1", "thermocouple", -90, 90, signal="current_readback"),  # a value
        _row("switch 2", "thermal_switch", -90, -90),  # no signal: plain
    ]
    doc = _card(_graph(tmp_path, rows, rule=True))
    top = next(n for n in _by_type(doc, "text-label") if n["props"]["text"] == "END VIEW")
    below = top["geometry"]["y"]  # on the drawing, not in the signals table

    def on_drawing(kind: str, tip_or_pv: str, key: str) -> list[dict]:
        return [
            n
            for n in _by_type(doc, kind)
            if n["props"].get(key) == tip_or_pv and n["geometry"]["y"] > below
        ]

    (light,) = on_drawing("polygon", "switch 1", "tooltip")
    assert light.get("rules") and bound(doc, light) == ("XR01-QF1:Current-SP",)
    (gauge,) = _tipped(doc, "gauge 1")  # a plain dot, its number beside it, outward
    gx, gy = _centre(gauge)
    number = min(  # the one beside it; the grid below lists it again
        on_drawing("pv-label", "XR01-QF1:Current-RB", "pv"), key=lambda n: abs(_centre(n)[1] - gy)
    )
    assert _centre(number)[1] == pytest.approx(gy, abs=1)
    assert number["geometry"]["x"] + number["geometry"]["width"] <= gx
    (plain,) = _tipped(doc, "switch 2")
    assert plain["type"] == "ellipse" and not plain.get("rules")
    assert plain["props"]["fillColor"] == symbols.MUTED_TEXT  # plain: no state colour
    for name in ("switch 1", "gauge 1", "switch 2"):
        assert name not in _texts(doc)  # no names on the drawing: the legend has them
    leaders = [
        n
        for n in _by_type(doc, "line")
        if n["geometry"]["y"] > below
        and n["geometry"]["x"] < layout.MARGIN + layout.CARD_LEFT
        and "tooltip" not in n["props"]
    ]
    assert not leaders


def test_a_ruled_readback_drawn_by_its_role_default_still_shows_its_number(tmp_path) -> None:
    rows = [*_six_poles(), _row("gauge 1", "thermocouple", -90, 90, signal="current_readback")]
    tables = _tables(tmp_path, rows)
    types = tables / "record_types.csv"
    types.write_text(types.read_text().replace("Current-RB,ai,value,", "Current-RB,ai,,"))
    with (tables / "health.csv").open("a") as handle:
        handle.write("XR01-QF1:Current-RB,1,bad,screen,stated,xr-screens,4d5e6f7,m.screen:2\n")
    graph = load_graph(write_fixture_facility(tmp_path / "f.json"), tables)
    assert graph.channel("XR01-QF1:Current-RB").drawn_as == "value"  # the readback default
    doc = _card(graph)
    (gauge,) = _tipped(doc, "gauge 1")
    gy = _centre(gauge)[1]
    beside = [
        n
        for n in _by_type(doc, "pv-label")
        if n["props"].get("pv") == "XR01-QF1:Current-RB" and abs(_centre(n)[1] - gy) < 1
    ]
    assert beside  # an empty drawn_as draws as the stated "value" does


def test_the_picture_is_a_drawing_of_its_own_after_the_relations(tmp_path) -> None:
    doc = _card(_graph(tmp_path, _six_poles()))
    heads = {n["props"]["text"]: n["geometry"] for n in _by_type(doc, "text-label")}
    view, relations = heads["END VIEW"], heads["1 · RELATIONS"]
    assert view["x"] == relations["x"] == layout.MARGIN
    assert view["y"] > relations["y"]
    drawn = [n for t in ("yoke", "pole 1", "coil 1", "bore") for n in _tipped(doc, t)]
    assert all(
        n["geometry"]["x"] + n["geometry"]["width"] <= layout.MARGIN + layout.STRIP_WIDTH
        for n in drawn
    )
    assert "PARTS" not in _texts(doc) and "PICTURE" not in _texts(doc)


def test_parts_with_no_body_are_drawn_in_a_frame_that_says_so_with_the_scale(
    tmp_path,
) -> None:
    rows = [
        _row("switch 1", "thermal_switch", 120, 40),
        _row("switch 2", "thermal_switch", -80, 40),
        _row("manifold 1", "manifold", 200, -60),
    ]
    doc = _card(_graph(tmp_path, rows))
    texts = _texts(doc)
    assert "outline not stated" in texts
    assert any(t.endswith(" mm") for t in texts)  # the scale bar's length
    assert [n for n in _by_type(doc, "polygon") if n["props"].get("lineStyle") == "dashed"]
    assert _tipped(doc, "beam axis")  # the frame's one reference
    for name in ("switch 1", "switch 2", "manifold 1"):
        assert _tipped(doc, name)  # at its position, named by its tooltip


def test_water_and_terminals_stand_in_a_column_beside_the_yoke_under_one_word(tmp_path) -> None:
    rows = [
        *_six_poles(),  # a yoke of radius 150
        _row("manifold 1", "manifold", 180, 20),
        _row("manifold 2", "manifold", 180, -20),
        _row("terminal 1", "terminal", "", ""),  # no position: dashed, left of the yoke
        _row("switch 1", "thermal_switch", "", "", angle=90),  # its coil: a dot on it
    ]
    doc = _card(_graph(tmp_path, rows))
    yoke, _window = _by_type_tip(doc, "ellipse", "yoke")
    right = yoke["geometry"]["x"] + yoke["geometry"]["width"]
    blocks = {n["props"].get("tooltip"): n for n in _by_type(doc, "rectangle")}
    one, two = blocks["manifold 1"], blocks["manifold 2"]
    for water in (one, two):
        assert water["props"]["fillColor"] == symbols.WATER and water["props"]["cornerRadius"] > 0
        assert water["geometry"]["x"] > right  # a column right of the yoke
    assert one["geometry"]["x"] == two["geometry"]["x"]
    assert one["geometry"]["y"] < two["geometry"]["y"]
    terminal = blocks["terminal 1"]  # no position: in the row of parts not placed
    assert terminal["props"]["lineStyle"] == "dashed"
    assert _centre(terminal)[1] < yoke["geometry"]["y"] + yoke["geometry"]["height"] / 4
    assert _centre(terminal)[0] < yoke["geometry"]["x"]
    texts = _texts(doc)
    assert texts.count("water") == 1 and "TB" in texts
    (dot,) = _tipped(doc, "switch 1")
    blocks_of_coil = [_centre(n) for n in _tipped(doc, "coil 2")]  # the coil at 90 degrees
    assert dot["type"] == "ellipse"
    assert min(math.dist(_centre(dot), c) for c in blocks_of_coil) < 20  # on its block


def test_a_sensor_with_no_place_is_a_hollow_ring_beside_the_drawing(tmp_path) -> None:
    rows = [*_six_poles(), _row("thermal switch 1", "thermal_switch", "", "")]
    doc = _card(_graph(tmp_path, rows))
    (ring,) = _tipped(doc, "thermal switch 1")
    assert ring["type"] == "ellipse" and ring["props"]["fill"] is False
    assert "thermal switch" in _texts(doc)


def test_each_coil_number_stands_on_the_yoke_ring_at_its_angle(tmp_path) -> None:
    doc = _card(_graph(tmp_path, _six_poles()))
    yoke, window = _by_type_tip(doc, "ellipse", "yoke")
    cx, cy = _centre(yoke)
    outer, inner = yoke["geometry"]["width"] / 2, window["geometry"]["width"] / 2
    for k in range(6):
        (label,) = [
            n
            for n in _by_type(doc, "text-label")
            if n["props"]["text"] == str(k + 1)
            and n["props"]["fontSize"] == layout.COIL_NUMBER_SIZE
        ]
        x, y = _centre(label)
        assert inner < math.dist((x, y), (cx, cy)) < outer  # on the ring
        angle = math.degrees(math.atan2(cy - y, x - cx)) % 360
        assert angle == pytest.approx(30 + 60 * k, abs=2)


def test_the_legend_lists_each_part_once_by_kind_with_its_source(tmp_path) -> None:
    rows = [
        *_six_poles(),
        _row("thermal switch 1", "thermal_switch", 90, 90),
        _row("thermal switch 2", "thermal_switch", -90, 90),
        _row("odd one", "thermal_switch", 0, -140),
    ]
    doc = _card(_graph(tmp_path, rows))
    texts = _texts(doc)
    assert "1, 2, 3 (odd one)" in texts  # a numbered member by its number, the rest named
    assert "1, 2, 3, 4, 5, 6" in texts  # the coils
    assert "1, 2" in texts  # the corrector windings
    for kind in ("thermal switch", "coil", "corrector winding"):
        assert texts.count(kind) == 1
    assert "pole" not in texts and "yoke" not in texts  # the iron is the drawing, not a part
    assert texts.count("source XR-DRAWING B released") == 1
    chips = [n for n in _by_type(doc, "polygon") if n["props"].get("tooltip") == "no rule"]
    assert not chips  # no second block of chips


def test_a_class_with_no_rows_has_no_picture(tmp_path) -> None:
    doc = _card(_graph(tmp_path, [_row("yoke", "yoke", 0, 0, w=10, cls="Dipole")]))
    assert "END VIEW" not in _texts(doc)


def test_a_signal_resolves_on_a_part_device_of_the_magnet(tmp_path) -> None:
    rows = [_row("switch 1", "thermal_switch", 90, 90, signal="*TS1:Open-Sts")]
    graph = _graph(tmp_path, rows, change=_switches)
    assert graph.picture_matches("XR01:QF1", "*TS1:Open-Sts") == [
        ("XR01:QF1-TS1", "XR01-QF1-TS1:Open-Sts")
    ]
    assert graph.picture_channel("XR01:QF1", "*:Open-Sts") == "XR01-QF1-TS1:Open-Sts"
    assert len(graph.picture_matches("XR01:QF1", "*:Open-Sts")) == 2  # two parts: no pick
    assert graph.picture_channel("XR01:QF1", "*:Current-SP") == "XR01-QF1:Current-SP"  # own first


# -- shapes with no size, other pole counts, the yoke as a polygon -----------------


def _unsized(poles: int, first: float, *, bore: str = "bore", gap: float = 30) -> list[str]:
    """An invented class of *poles* poles from *first* degrees, no pole or coil sized;
    its bore (or its gap) sized."""
    rows = [_row(bore, bore, 0, 0, w=gap)]
    for k in range(poles):
        angle = first + 360 / poles * k
        rows.append(_row(f"pole {k + 1}", "pole", 0, 0, angle=angle))
        rows.append(_row(f"coil {k + 1}", "coil", 0, 0, angle=angle))
    return rows


@pytest.mark.parametrize(("poles", "first"), [(4, 45), (6, 30)])
def test_a_stated_yoke_sets_the_proportions_however_small_the_bore(
    tmp_path, poles: int, first: float
) -> None:
    """A yoke twelve times the bore: the coils keep the mockup's proportion to the
    yoke, the bore is drawn as stated, and the pole tips reach for it apart."""
    from osprey_panels.picture import COIL_FROM, COIL_TO, YOKE, body

    rows = [_row("yoke", "yoke", "", "", w=420, h=420), *_unsized(poles, first, gap=35)]
    pictures = read_fixture_tables(_tables(tmp_path, rows)).pictures
    drawn = body([r for r in pictures if r.shape])
    assert drawn.unit == pytest.approx(210 / YOKE)  # mm per unit from the yoke
    (bore,) = [s for s in drawn.shapes if s.role == "bore"]
    (x1, _y1), (x2, _y2) = bore.points
    assert x2 - x1 == pytest.approx(35)  # as stated
    coils = [s for s in drawn.shapes if s.role == "coil"]
    reach = [math.hypot(*pt) for s in coils for pt in s.points]
    unit = drawn.unit
    assert min(reach) >= COIL_FROM * unit * 0.95 and max(reach) <= COIL_TO * unit * 1.2
    tips = [s for s in drawn.shapes if s.role == "pole"]
    nose = min(math.hypot(*pt) for pt in tips[0].points)
    assert 17.5 < nose < 17.5 * 2  # close to the bore, never into it
    for a, b in itertools.pairwise(tips):  # neighbouring poles never touch
        assert min(math.dist(p, q) for p in a.points for q in b.points) > 1


@pytest.mark.parametrize(("poles", "first"), [(2, 90), (4, 45), (6, 30)])
def test_poles_with_an_angle_and_no_size_draw_from_the_bore_and_say_so(
    tmp_path, poles: int, first: float
) -> None:
    from osprey_panels.picture import body

    rows = _unsized(poles, first, bore="gap" if poles == 2 else "bore")
    pictures = read_fixture_tables(_tables(tmp_path, rows)).pictures
    drawn = body([r for r in pictures if r.shape])
    assert not drawn.sized and drawn.beam == (0, 0)
    for k in range(poles):
        (pole,) = [s for s in drawn.shapes if s.name == f"pole {k + 1}"]
        tip = min(math.hypot(a, b) for a, b in pole.points)
        assert tip >= 15 * 0.99  # no pole reaches into the bore (radius 15)
        sides = [s for s in drawn.shapes if s.name == f"coil {k + 1}"]
        assert len(sides) == 2
        for side in sides:  # beside the pole, not across it
            assert not _overlap(side.points, pole.points)
    assert [c.number for c in drawn.coils] == [str(k + 1) for k in range(poles)]
    doc = _card(_graph(tmp_path / "g", rows))
    texts = _texts(doc)
    frame = "gap" if poles == 2 else "bore"
    assert f"scaled from the {frame}; pole, coil at default proportions" in texts
    assert not any(t.startswith("size not stated") for t in texts)
    for k in range(poles):
        assert str(k + 1) in texts  # each coil's number at the coil


@pytest.mark.parametrize(
    ("frame", "caption"),
    [
        (
            [_row("yoke", "yoke", "", "", w=420, h=420)],
            "scaled from the yoke; pole, coil at default proportions",
        ),  # fmt: skip
        ([], "size not stated: default proportions · schematic, not to scale"),
    ],
)
def test_the_scale_caption_names_what_the_drawing_is_scaled_from(
    tmp_path, frame: list[str], caption: str
) -> None:
    rows = [*frame, *(r for r in _unsized(4, 45) if ",bore," not in r)]
    texts = _texts(_card(_graph(tmp_path, rows)))
    assert caption in texts
    assert sum(t.startswith(("scaled from", "size not stated")) for t in texts) == 1


def test_a_drawing_whose_every_shape_is_sized_has_no_caption(tmp_path) -> None:
    texts = _texts(_card(_graph(tmp_path, _six_poles())))
    assert not [t for t in texts if t.startswith(("scaled from", "size not stated"))]


def _overlap(p: tuple, q: tuple) -> bool:
    """Whether two convex polygons overlap by more than a touch (separating axes)."""
    for poly in (p, q):
        for i in range(len(poly)):
            (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
            nx, ny = y1 - y2, x2 - x1
            a = [nx * x + ny * y for x, y in p]
            b = [nx * x + ny * y for x, y in q]
            if max(a) <= min(b) + 1e-6 or max(b) <= min(a) + 1e-6:
                return False
    return True


def test_a_yoke_stated_as_a_polygon_is_drawn_filled_and_bounds_the_poles(tmp_path) -> None:
    from osprey_panels.picture import body

    hexagon = [
        _row(
            f"o{k}",
            "outline",
            100 * math.cos(math.radians(60 * k)),
            100 * math.sin(math.radians(60 * k)),
        )
        for k in range(6)
    ]
    rows = [*_unsized(4, 45), *hexagon]
    pictures = read_fixture_tables(_tables(tmp_path, rows)).pictures
    drawn = body([r for r in pictures if r.shape], [r for r in pictures if r.kind == "outline"])
    yoke, window = [s for s in drawn.shapes if s.row is not None and s.row.kind == "outline"]
    assert len(yoke.points) == 6 and yoke.role == "yoke" and window.role == "window"
    inner = 100 * math.cos(math.radians(30))  # the hexagon's nearest edge
    for shape in drawn.shapes:
        if shape.row.kind == "pole":
            assert max(math.hypot(a, b) for a, b in shape.points) < inner
    doc = _card(_graph(tmp_path / "g", rows))
    (polygon,) = _by_type_tip(doc, "polygon", "yoke")
    (cavity,) = _by_type_tip(doc, "ellipse", "yoke")
    assert polygon["props"]["fillColor"] == symbols.STEEL_DARK
    assert polygon["props"].get("lineStyle") != "dashed"  # stated
    assert cavity["props"]["fillColor"] == symbols.PANEL_BACKGROUND  # convex: a round cavity


def test_two_poles_put_each_winding_band_between_its_coil_and_the_gap(tmp_path) -> None:
    from osprey_panels.picture import body

    rows = [*_unsized(2, 90, bore="gap")]
    rows += [
        _row("trim 1", "corrector_winding", "", "", angle=90),
        _row("trim 2", "corrector_winding", "", "", angle=270),
    ]
    pictures = read_fixture_tables(_tables(tmp_path, rows)).pictures
    drawn = body([r for r in pictures if r.shape])
    coils = [s for s in drawn.shapes if s.role == "coil"]
    bands = [s for s in drawn.shapes if s.role == "winding"]
    assert len(coils) == len(bands) == 4
    for band in bands:
        assert not any(_overlap(band.points, coil.points) for coil in coils)
        (above,) = [c for c in coils if _same_side(c, band) and _beside(c, band)]
        # between its coil and the gap: nearer the beam than its coil
        assert max(abs(b) for _a, b in band.points) < min(abs(b) for _a, b in above.points)
    doc = _card(_graph(tmp_path / "g", rows))
    for name in ("trim 1", "trim 2"):
        for node in _tipped(doc, name):  # a box the viewer draws where its points say
            assert node["geometry"]["height"] >= layout.POLYGON_BOX


def _same_side(p, q) -> bool:
    return (sum(b for _a, b in p.points) > 0) == (sum(b for _a, b in q.points) > 0)


def _beside(p, q) -> bool:
    return (sum(a for a, _b in p.points) > 0) == (sum(a for a, _b in q.points) > 0)


# -- parts with no position --------------------------------------------------------


def test_a_row_may_leave_its_position_empty_and_old_positions_read_unchanged(tmp_path) -> None:
    rows = [
        "Quadrupole,end,yoke,yoke,,,,X,A,s,l,300,,,",
        "Quadrupole,end,switch 1,thermal_switch,,,,X,A,s,l,,,90,",
        "Quadrupole,end,switch 2,thermal_switch,,,,X,A,s,l,,,,",
        _row("switch 3", "thermal_switch", 50, -40),
    ]
    yoke, one, two, three = read_fixture_tables(_tables(tmp_path, rows)).pictures
    assert (yoke.x_mm, yoke.y_mm, yoke.z_mm) == (None, None, None)
    assert one.placed and not one.positioned  # its coil's angle
    assert not two.placed
    assert three.positioned and (three.x_mm, three.y_mm) == (50, -40)


def test_a_yoke_and_a_bore_with_no_position_are_centred(tmp_path) -> None:
    from osprey_panels.picture import body

    rows = [
        "Quadrupole,end,yoke,yoke,,,,X,A,s,l,300,,,",
        "Quadrupole,end,bore,bore,,,,X,A,s,l,24,,,",
        "Quadrupole,end,pole 1,pole,,,,X,A,s,l,,,45,",
    ]
    pictures = read_fixture_tables(_tables(tmp_path, rows)).pictures
    drawn = body([r for r in pictures if r.shape])
    for shape in drawn.shapes:
        if shape.role in ("yoke", "window", "bore"):
            (a1, b1), (a2, b2) = shape.points
            assert ((a1 + a2) / 2, (b1 + b2) / 2) == (0, 0)
    assert drawn.beam == (0, 0)


def test_parts_at_a_coils_angle_sit_on_it_side_by_side(tmp_path) -> None:
    rows = [
        *_unsized(4, 45),
        "Quadrupole,end,thermal switch 1,thermal_switch,,,,X,A,s,l,,,135,",
        "Quadrupole,end,thermocouple 1,thermocouple,,,,X,A,s,l,,,135,",
    ]
    doc = _card(_graph(tmp_path, rows))
    blocks = [_centre(n) for n in _tipped(doc, "coil 2")]  # the coil at 135 degrees
    (switch,) = _tipped(doc, "thermal switch 1")
    (couple,) = _tipped(doc, "thermocouple 1")
    for dot in (switch, couple):
        assert dot["props"]["fill"] is True  # placed: a filled dot
        assert min(math.dist(_centre(dot), c) for c in blocks) < 26  # on the coil, to a pixel
    gap = math.dist(_centre(switch), _centre(couple))
    assert gap >= switch["geometry"]["width"]  # side by side, not on top of each other
    assert "not placed" not in " ".join(_texts(doc))


def test_a_part_with_no_position_and_no_angle_is_not_placed(tmp_path) -> None:
    rows = [
        *_unsized(4, 45),
        "Quadrupole,end,thermal switch 1,thermal_switch,,,,X,A,s,l,,,,",
        "Quadrupole,end,thermal switch 2,thermal_switch,,,,X,A,s,l,,,,",
        "Quadrupole,end,thermocouple 1,thermocouple,,,,X,A,s,l,,,45,",
    ]
    doc = _card(_graph(tmp_path, rows))
    for name in ("thermal switch 1", "thermal switch 2"):
        (ring,) = _tipped(doc, name)
        assert ring["type"] == "ellipse" and ring["props"]["fill"] is False
    texts = _texts(doc)
    assert "1, 2 (not placed)" in texts  # the legend says so
    assert "1" in texts and "thermocouple" in texts
    assert not any(t.startswith("1 (") and "not placed" in t for t in texts if t != "1, 2 (not placed)")  # fmt: skip


def test_parts_not_placed_with_no_body_stand_in_the_frames_corner(tmp_path) -> None:
    rows = [
        _row("thermal switch 1", "thermal_switch", 120, 40),
        "Quadrupole,end,thermal switch 2,thermal_switch,,,,X,A,s,l,,,,",
    ]
    doc = _card(_graph(tmp_path, rows))
    (placed,) = _tipped(doc, "thermal switch 1")
    (ring,) = _tipped(doc, "thermal switch 2")
    assert placed["props"]["fill"] is True and ring["props"]["fill"] is False
    assert "1, 2 (2 not placed)" in _texts(doc)


def test_a_picture_with_no_part_placed_is_a_row_of_blocks_not_an_empty_frame(tmp_path) -> None:
    from osprey_panels import layout

    rows = [
        "Quadrupole,end,thermal switch 1,thermal_switch,,,,X,A,s,l,,,,",
        "Quadrupole,end,thermal switch 2,thermal_switch,,,,X,A,s,l,,,,",
    ]
    doc = _card(_graph(tmp_path, rows))
    texts = _texts(doc)
    assert "arrangement not stated" in texts
    assert "thermal switch 1" in texts and "thermal switch 2" in texts  # named on their blocks
    assert not any("not placed" in t for t in texts)  # no legend list of them
    dashed = [n for n in _by_type(doc, "rectangle") if n["props"].get("lineStyle") == "dashed"]
    assert not [n for n in dashed if n["geometry"]["width"] >= layout.PICTURE_SIZE / 2]
    blocks = [
        n for n in _by_type(doc, "rectangle") if n["geometry"]["width"] == layout.ARRANGED_BLOCK
    ]
    assert len(blocks) == 2 and blocks[0]["geometry"]["y"] == blocks[1]["geometry"]["y"]


def test_check_counts_the_parts_not_placed_per_class_as_information(tmp_path) -> None:
    from osprey_panels.emit import check, unplaced_parts, write

    rows = [
        *_unsized(4, 45),
        "Quadrupole,end,thermal switch 1,thermal_switch,,,,X,A,s,l,,,,",
        "Quadrupole,end,thermal switch 2,thermal_switch,,,,X,A,s,l,,,135,",
        "Quadrupole,end,manifold 1,manifold,,,,X,A,s,l,,,,",
    ]
    graph = _graph(tmp_path, rows)
    assert unplaced_parts(graph) == {"Quadrupole": 2}
    out = tmp_path / "out"
    write(graph, out, only=lambda t: t.view == "card" and t.id == "XR01:QF1")
    assert not [p for p in check(graph, out) if "placed" in p.kind or "placed" in p.detail]


def test_parts_not_placed_stand_above_the_yoke_on_no_side(tmp_path) -> None:
    from osprey_panels.picture import body

    rows = [
        *_unsized(2, 90, bore="gap"),
        _row("thermal switch 1", "thermal_switch", "", ""),
        _row("thermocouple 1", "thermocouple", "", ""),
        _row("manifold 1", "manifold", "", ""),
        _row("terminal 1", "terminal", "", ""),
    ]
    pictures = read_fixture_tables(_tables(tmp_path, rows)).pictures
    drawn = body([r for r in pictures if r.shape], [], [r for r in pictures if not r.shape])
    top = max(b for s in drawn.shapes if s.role == "yoke" for _a, b in s.points)
    assert not any(b.placed for b in drawn.blocks)  # no water column, no side claimed
    for at, half in [(b.at, b.size[1] / 2) for b in drawn.blocks] + [
        (d.at, d.radius) for d in drawn.dots
    ]:
        assert at[1] - half > top  # in the rows above the yoke, none over it


def test_a_terminal_not_placed_is_a_terminal_block_clear_of_the_rows_beside(tmp_path) -> None:
    from osprey_panels.picture import TERMINAL_BLOCK, WATER_BLOCK, body

    rows = [
        *_six_poles(),
        _row("manifold 1", "manifold", "", ""),
        _row("terminal TB1", "terminal", "", ""),
        _row("terminal TB2", "terminal", "", ""),
        _row("thermal switch 1", "thermal_switch", "", ""),
    ]
    pictures = read_fixture_tables(_tables(tmp_path, rows)).pictures
    drawn = body([r for r in pictures if r.shape], [], [r for r in pictures if not r.shape])
    (water,) = [b for b in drawn.blocks if b.role == "water"]
    terminals = [b for b in drawn.blocks if b.role == "terminal"]
    unit = water.size[0] / WATER_BLOCK[0]
    assert [b.size for b in terminals] == [(TERMINAL_BLOCK[0] * unit, TERMINAL_BLOCK[1] * unit)] * 2
    assert all(b.label for b in terminals)
    spans: dict[float, tuple[float, float]] = {}  # each row's centre: its lowest and highest
    for y, half in [(b.at[1], b.size[1] / 2) for b in drawn.blocks] + [
        (d.at[1], d.radius) for d in drawn.dots if not d.placed
    ]:
        low, high = spans.get(round(y, 6), (y - half, y + half))
        spans[round(y, 6)] = (min(low, y - half), max(high, y + half))
    ordered = sorted(spans.values())
    assert len(ordered) == 3  # one row per kind
    for (_low, high), (low_next, _high) in itertools.pairwise(ordered):
        assert high < low_next  # no row reaches into the next


def test_a_water_or_terminal_part_with_only_an_angle_stands_on_that_side(tmp_path) -> None:
    from osprey_panels.picture import body

    rows = [
        *_six_poles(),
        _row("manifold 1", "manifold", "", "", angle=0),
        _row("manifold 2", "manifold", "", "", angle=0),
        _row("terminal TB1", "terminal", "", "", angle=180),
        _row("terminal TB2", "terminal", "", ""),
    ]
    pictures = read_fixture_tables(_tables(tmp_path, rows)).pictures
    drawn = body([r for r in pictures if r.shape], [], [r for r in pictures if not r.shape])
    yoke = max(abs(a) for s in drawn.shapes if s.role == "yoke" for a, _b in s.points)
    by_part = {b.row.part: b for b in drawn.blocks if b.row is not None}
    for name in ("manifold 1", "manifold 2"):
        assert by_part[name].placed and by_part[name].at[0] > yoke  # the +x column
    assert by_part["terminal TB1"].placed and by_part["terminal TB1"].at[0] < -yoke
    assert not by_part["terminal TB2"].placed  # no angle, no position: not placed
    assert "water" in [w.text for w in drawn.words]  # the column's word


# -- rows under a place --------------------------------------------------------


UNDER_HEADER = HEADER + ",under"


def _under_rows(place: str) -> list[str]:
    """An invented four-pole body stated for the class's devices under *place*."""
    return [r + f",{place}" for r in _unsized(4, 45)]


def test_a_row_under_a_place_draws_only_the_devices_there(tmp_path) -> None:
    tables = copy_tables(tmp_path / "panels")
    (tables / PICTURES).write_text("\n".join([UNDER_HEADER, *_under_rows("XR/XR01")]) + "\n")
    graph = load_graph(write_fixture_facility(tmp_path / "f.json"), tables)
    assert graph.pictures("Quadrupole", "XR01:QF1")
    assert not graph.pictures("Quadrupole", "XR02:QF1")  # same class, another place
    assert len(graph.pictures("Quadrupole")) == 9  # the class's rows, whatever the place
    titles = lambda device: [s.title for s in device_card(graph, device).sections]  # noqa: E731
    assert "picture" in titles("XR01:QF1")
    assert "picture" not in titles("XR02:QF1")


def test_a_device_outside_the_rows_place_gets_no_note_and_the_class_still_counts(
    tmp_path,
) -> None:
    from osprey_panels.emit import check, picture_classes, write

    tables = copy_tables(tmp_path / "panels")
    rows = [*_under_rows("XR/XR01"), _row("switch 1", "thermal_switch", "", "", signal="ghost") + ",XR/XR01"]  # fmt: skip
    (tables / PICTURES).write_text("\n".join([UNDER_HEADER, *rows]) + "\n")
    graph = load_graph(write_fixture_facility(tmp_path / "f.json"), tables)
    out = tmp_path / "out"
    write(graph, out, only=lambda t: t.view == "card" and t.id in ("XR01:QF1", "XR02:QF1"))
    notes = [p for p in check(graph, out) if p.kind == "unbound_picture"]
    assert len(notes) == 1 and "XR01:QF1" in notes[0].detail  # none for XR02:QF1
    classes = picture_classes(graph)
    assert "Quadrupole" in classes["drawing"] and "Quadrupole" not in classes["none"]


def test_a_file_without_the_under_column_applies_to_every_device(tmp_path) -> None:
    graph = _graph(tmp_path, _unsized(4, 45))  # the header has no under column
    assert all(row.under is None for row in graph.pictures("Quadrupole"))
    assert graph.pictures("Quadrupole", "XR01:QF1") == graph.pictures("Quadrupole", "XR02:QF1")


# -- check -------------------------------------------------------------------------


def test_check_notes_a_signal_that_resolves_to_no_channel_and_sorts_the_classes(
    tmp_path,
) -> None:
    from osprey_panels.emit import check, picture_classes, write

    rows = [
        *_six_poles(),
        _row("switch 1", "thermal_switch", 90, 90, signal="ghost_status"),
        _row("switch 1", "thermal_switch", 9, 9, cls="Dipole"),
    ]
    graph = _graph(tmp_path, rows)
    out = tmp_path / "out"
    write(graph, out, only=lambda t: t.view == "card" and t.id == "XR01:QF1")
    found = [p for p in check(graph, out) if p.kind == "unbound_picture"]
    assert len(found) == 1 and not found[0].failing
    assert "ghost_status" in found[0].detail and "XR01:QF1" in found[0].detail
    classes = picture_classes(graph)
    assert classes["drawing"] == ["Quadrupole"]
    assert classes["parts only"] == ["Dipole"]
    assert "Pump" in classes["none"]


def test_check_notes_a_signal_that_matches_two_part_devices(tmp_path) -> None:
    from osprey_panels.emit import check, write

    rows = [_row("switch 1", "thermal_switch", 90, 90, signal="*:Open-Sts")]
    graph = _graph(tmp_path, rows, change=_switches)
    out = tmp_path / "out"
    write(graph, out, only=lambda t: t.view == "card" and t.id == "XR01:QF1")
    (found,) = [p for p in check(graph, out) if p.kind == "ambiguous_picture"]
    assert not found.failing
    assert "XR01:QF1-TS1" in found.detail and "XR01:QF1-TS2" in found.detail


def test_a_picture_whose_rows_state_schematic_says_not_to_scale(tmp_path) -> None:
    rows = [
        "Quadrupole,end,thermal switch 1,thermal_switch,,,,X,A,schematic,l,,,,",
        "Quadrupole,end,thermal switch 2,thermal_switch,,,,X,A,schematic,l,,,,",
    ]
    texts = _texts(_card(_graph(tmp_path, rows)))
    assert "arrangement not stated · schematic, not to scale" in texts


def test_a_picture_wholly_at_default_proportions_says_not_to_scale(tmp_path) -> None:
    rows = [r for r in _unsized(4, 45) if ",bore," not in r]
    texts = _texts(_card(_graph(tmp_path, rows)))
    assert "size not stated: default proportions · schematic, not to scale" in texts


def test_a_scaled_picture_of_stated_rows_says_nothing_of_schematic(tmp_path) -> None:
    texts = _texts(_card(_graph(tmp_path, _six_poles())))
    assert not [t for t in texts if "not to scale" in t]


def test_a_row_s_devices_pattern_takes_or_leaves_out_devices_of_its_class(tmp_path) -> None:
    def graph_with(devices: str):
        tables = copy_tables(tmp_path / devices.replace("*", "s").replace("!", "not-"))
        rows = [f"{row},{devices}" for row in _under_rows("")]
        (tables / PICTURES).write_text("\n".join([UNDER_HEADER + ",devices", *rows]) + "\n")
        return load_graph(write_fixture_facility(tmp_path / f"{len(devices)}.json"), tables)

    only = graph_with("XR01:*")
    assert only.pictures("Quadrupole", "XR01:QF1")
    assert not only.pictures("Quadrupole", "XR02:QF1")
    but = graph_with("!XR01:QF1")  # every device of the class but that one
    assert not but.pictures("Quadrupole", "XR01:QF1")
    assert but.pictures("Quadrupole", "XR02:QF1")


def test_a_signal_resolves_on_a_device_any_relation_names(panel_graph) -> None:
    # the magnet's own channels and parts hold no supply state; its powered_by does
    assert panel_graph.picture_matches("XR01:QF1", "*:On-Sts") == [("XR01:PS1", "XR01-PS1:On-Sts")]
