"""Symbols: the element shapes, the six looks of a light or symbol, and the theme-role table."""

from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path

import pytest

from osprey_panels import symbols
from osprey_panels.symbols import (
    ALARM_ROLES,
    APPEARANCES,
    KINDS,
    STATES,
    WIDTH_UNITS,
    draw,
    light,
    look,
)
from osprey_panels_tests._harness import REPO_ROOT

PANELS = Path(symbols.__file__).parent
HEX = re.compile(r"#[0-9a-fA-F]{6}")
BRIEF_KINDS = (
    "bend",
    "quadrupole_f",
    "quadrupole_d",
    "sextupole",
    "skew_quadrupole",
    "corrector",
    "bpm",
    "neg_pump",
    "ion_pump",
    "gauge",
    "gate_valve",
    "bellows",
    "kicker",
    "scraper",
    "generic",
)
DRAWING = {"rectangle", "ellipse", "polygon", "line"}
AXIS = 100.0
UNIT = 10.0


def _colours(value):
    if isinstance(value, dict) and "$role" in value:
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _colours(item)
    elif isinstance(value, list):
        for item in value:
            yield from _colours(item)


def _extent(nodes):
    top = min(n["geometry"]["y"] for n in nodes)
    bottom = max(n["geometry"]["y"] + n["geometry"]["height"] for n in nodes)
    left = min(n["geometry"]["x"] for n in nodes)
    right = max(n["geometry"]["x"] + n["geometry"]["width"] for n in nodes)
    return left, top, right, bottom


def test_every_kind_of_the_brief_is_drawn_and_one_without_polarity_besides() -> None:
    assert set(BRIEF_KINDS) <= set(KINDS)
    assert set(KINDS) - set(BRIEF_KINDS) == {"quadrupole", "pump"}


@pytest.mark.parametrize("kind", KINDS)
def test_a_symbol_is_drawing_widgets_centred_on_x_and_as_wide_as_its_kind(kind) -> None:
    nodes = draw(kind, 50.0, AXIS, unit=UNIT)
    assert nodes
    assert {n["type"] for n in nodes} <= DRAWING
    assert all("id" not in n for n in nodes)
    left, _, right, _ = _extent(nodes)
    assert (left + right) / 2 == pytest.approx(50.0)
    assert right - left == pytest.approx(WIDTH_UNITS[kind] * UNIT)


@pytest.mark.parametrize("kind", KINDS)
def test_every_colour_of_a_symbol_is_a_theme_role(kind) -> None:
    for node in draw(kind, 0.0, AXIS, unit=UNIT):
        for key, value in node["props"].items():
            if key.endswith("Color"):
                assert isinstance(value, dict) and "$role" in value, (kind, key)


def test_every_magnet_is_centred_on_the_axis_and_its_polarity_is_in_its_shape() -> None:
    for kind in ("quadrupole_f", "quadrupole_d", "quadrupole", "bend", "sextupole"):
        _, top, _, bottom = _extent(draw(kind, 0.0, AXIS, unit=UNIT))
        assert (top + bottom) / 2 == pytest.approx(AXIS), kind
    (f,) = draw("quadrupole_f", 0.0, AXIS, unit=UNIT)
    (d,) = draw("quadrupole_d", 0.0, AXIS, unit=UNIT)
    w = f["geometry"]["width"]
    # at the axis a focusing one is at its widest, a defocusing one at its narrowest
    assert max(p["x"] for p in f["props"]["points"]) == pytest.approx(w)
    assert min(p["x"] for p in d["props"]["points"] if p["x"] > 0) < w


def test_a_pump_of_no_stated_kind_is_a_circle_on_the_chamber_with_the_letter_p() -> None:
    (circle,) = draw("pump", 0.0, AXIS, unit=UNIT)
    assert len(circle["props"]["points"]) >= 12
    g = circle["geometry"]
    assert g["y"] + g["height"] == pytest.approx(AXIS)  # it sits on the chamber
    _, letter = symbols.badge(0.0, 0.0, 20.0, "pump", 11)
    assert letter["props"]["text"] == "P"


def test_an_ion_pump_is_a_box_with_upright_fins_inside() -> None:
    body, *fins = draw("ion_pump", 0.0, AXIS, unit=16.0)  # as a card's strip draws it
    g = body["geometry"]
    assert len(body["props"]["points"]) == 4  # a box, not a triangle
    assert g["y"] + g["height"] / 2 == pytest.approx(AXIS)
    assert len(fins) == 4 and all(f["type"] == "line" for f in fins)
    for fin in fins:
        f = fin["geometry"]
        assert g["x"] < f["x"] < g["x"] + g["width"] and g["y"] < f["y"]
        assert f["y"] + f["height"] < g["y"] + g["height"]
    assert len(draw("ion_pump", 0.0, AXIS, unit=1.0)) == 1  # too small for fins: plain


def test_an_unknown_kind_is_refused() -> None:
    with pytest.raises(ValueError, match="no symbol"):
        draw("wiggler", 0.0, AXIS)


#: Every role the theme table may hold: neutral page roles, Disagree and the alarm roles.
THEME_ROLES = {
    "Basis_Design",
    "Basis_Bench",
    "Basis_Survey",
    "Basis_Beam",
    "Basis_Bench_Tint",
    "Basis_Survey_Tint",
    "Basis_Beam_Tint",
    "Disagree_Tint",
    "Curve_2",
    "Curve_3",
    "Page_Background",
    "Page_Text",
    "Field_Background",
    "Tile_Background",
    "Box_Border",
    "Text_Muted",
    "Normal_Fill",
    "Transparent",
    "Disagree",
    "MINOR",
    "MAJOR",
    "DISCONNECTED",
    # what a magnet's drawing is made of: material, never a type or a state
    "Material_Steel",
    "Material_Steel_Dark",
    "Material_Steel_Edge",
    "Material_Copper",
    "Material_Copper_Edge",
    "Material_Winding",
    "Material_Water",
    "Material_Winding_2",
    "Material_Bore",
    # a drawn card: its own device in focus, and the look of a control
    "Focus",
    "Focus_Tint",
    "Control_Action",
    "Control_Action_Text",
    "Control_Choice_Selected",
}
NEUTRAL_ROLES = {
    "Page_Background",
    "Page_Text",
    "Field_Background",
    "Tile_Background",
    "Box_Border",
    "Text_Muted",
    "Normal_Fill",
}


def test_the_theme_table_holds_no_type_colour() -> None:
    table, _ = _between(
        (PANELS / "symbols.py").read_text(), "# --- theme roles", "# --- end theme roles"
    )
    named = set(re.findall(r'role\("([^"]+)"', table))
    assert named == THEME_ROLES
    defined = {v["$role"] for v in vars(symbols).values() if isinstance(v, dict) and "$role" in v}
    assert defined == THEME_ROLES


@pytest.mark.parametrize("kind", KINDS)
def test_a_symbol_with_no_state_is_neutral_and_carries_no_mark(kind) -> None:
    nodes = draw(kind, 0.0, AXIS, unit=UNIT)
    assert {c["$role"] for c in _colours(nodes)} <= NEUTRAL_ROLES
    assert all(n["type"] != "text-label" for n in nodes)


def _seen(nodes) -> tuple:
    return tuple(
        (
            n["type"],
            tuple(
                (k, v["$role"] if isinstance(v, dict) and "$role" in v else json.dumps(v))
                for k, v in sorted(n["props"].items())
            ),
        )
        for n in nodes
    )


@pytest.mark.parametrize("kind", KINDS)
def test_a_symbol_in_each_look_differs_from_its_other_five_looks(kind) -> None:
    drawn = {state: _seen(draw(kind, 0.0, AXIS, state, unit=UNIT)) for state in APPEARANCES}
    for a, b in combinations(APPEARANCES, 2):
        assert drawn[a] != drawn[b], (kind, a, b)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("state", APPEARANCES)
def test_a_symbol_in_a_state_takes_the_lights_look(kind, state) -> None:
    nodes = draw(kind, 0.0, AXIS, state, unit=UNIT)
    shown = look(state)
    if state == symbols.NONE:  # no rule: the look, and no mark
        assert all(n["type"] != "text-label" for n in nodes)
        nodes = [*nodes, {"type": "text-label", "props": {"text": ""}}]
    *shapes, mark = nodes
    assert shapes
    for node in shapes:
        props = symbols.styled(node["type"])
        assert set(props) <= set(symbols.LIGHT_PROPS) and "lineColor" in props
        for prop in props:
            assert node["props"][prop] == shown[prop], (kind, state, prop)
    assert mark["type"] == "text-label"
    assert symbols.styled("text-label") == (symbols.MARK_PROP,)
    assert mark["props"]["text"] == shown["text"]


def _outline(kind: str) -> tuple:
    """A kind's outline, whatever its fill: types, proportions, offset from the axis, points."""
    nodes = draw(kind, 0.0, AXIS, unit=UNIT)
    left, top, right, bottom = _extent(nodes)
    w, h = right - left, bottom - top
    return tuple(
        (
            n["type"],
            round(h / w, 2),
            round(((top + bottom) / 2 - AXIS) / h, 2),
            n["props"].get("rotation", 0),
            tuple(
                (round(p["x"] / w, 2), round(p["y"] / h, 2)) for p in n["props"].get("points", ())
            ),
        )
        for n in nodes
    )


def test_the_shape_alone_tells_the_kind() -> None:
    for a, b in combinations(KINDS, 2):
        assert _outline(a) != _outline(b), (a, b)


def _appearance(state: str) -> tuple:
    shown = look(state)
    return tuple(
        shown[p]["$role"] if isinstance(shown[p], dict) else shown[p]
        for p in (*symbols.LIGHT_PROPS, symbols.MARK_PROP)
    )


def test_the_five_states_the_aggregate_and_none_look_pairwise_different() -> None:
    assert len(STATES) == 5
    assert symbols.APPEARANCES == (*STATES, "aggregate", "none")
    no_rule = look("none")
    assert (no_rule["fill"], no_rule["lineColor"]["$role"], no_rule["text"]) == (
        True,
        "Page_Text",  # a clear outline in the text role, never the border role
        "",
    )
    for a, b in combinations(symbols.APPEARANCES, 2):
        assert _appearance(a) != _appearance(b), (a, b)
    aggregate = look("aggregate")
    assert aggregate["text"] == symbols.AGGREGATE_MARK
    roles = {c["$role"] for c in _colours(dict(aggregate))}
    assert not roles & ALARM_ROLES  # an aggregate is not a state


def test_normal_is_neutral_with_no_state_colour() -> None:
    shown = look("normal")
    roles = {c["$role"] for c in _colours(dict(shown))}
    assert not roles & ALARM_ROLES
    assert "OK" not in roles  # never green
    assert shown["fill"] is True and shown["text"] == ""


def test_unknown_is_hollow_and_dashed_and_disagree_carries_a_mark() -> None:
    unknown, disagree = look("unknown"), look("disagree")
    assert unknown["fill"] is False and unknown["lineStyle"] == "dashed"
    assert unknown["text"] == symbols.UNKNOWN_MARK
    assert disagree["text"] == symbols.DISAGREE_MARK != symbols.UNKNOWN_MARK
    assert disagree["fillColor"]["$role"] not in ALARM_ROLES


def test_a_light_is_a_polygon_and_a_mark_in_the_look_of_its_state() -> None:
    shape, mark = light(10.0, 20.0, 14.0, "unknown")
    assert shape["type"] == "polygon" and len(shape["props"]["points"]) == 8
    assert shape["props"]["lineStyle"] == "dashed" and shape["props"]["fill"] is False
    assert mark["type"] == "text-label" and mark["props"]["text"] == symbols.UNKNOWN_MARK
    assert mark["geometry"] == {"x": 10.0, "y": 20.0, "width": 14.0, "height": 14.0}


def _between(text: str, start: str, end: str) -> tuple[str, str]:
    head, rest = text.split(start, 1)
    inside, tail = rest.split(end, 1)
    return inside, head + tail


def test_no_hex_colour_outside_the_theme_role_table() -> None:
    table, rest = _between(
        (PANELS / "symbols.py").read_text(), "# --- theme roles", "# --- end theme roles"
    )
    assert HEX.search(table)
    assert not HEX.search(rest)
    assert not HEX.search((PANELS / "layout.py").read_text())


def test_each_look_is_pinned_to_its_role_and_pattern_by_name() -> None:
    def shown(state: str) -> tuple:
        x = look(state)
        return (
            x["fill"],
            x["fillColor"]["$role"],
            x["lineColor"]["$role"],
            x["lineStyle"],
            x["text"],
        )

    assert shown("normal") == (True, "Normal_Fill", "Page_Text", "solid", "")
    assert shown("warn") == (True, "MINOR", "MINOR", "solid", "")
    assert shown("bad") == (True, "MAJOR", "MAJOR", "solid", "")
    assert shown("unknown") == (False, "Page_Background", "DISCONNECTED", "dashed", "?")
    assert shown("disagree") == (True, "Disagree", "Page_Text", "solid", symbols.DISAGREE_MARK)
    assert shown("aggregate") == (
        True,
        "Field_Background",
        "Page_Text",
        "solid",
        symbols.AGGREGATE_MARK,
    )


@pytest.mark.parametrize("kind", KINDS)
def test_every_closed_symbol_is_a_polygon_so_its_outline_can_be_dashed(kind) -> None:
    for node in draw(kind, 0.0, AXIS, "unknown", unit=UNIT):
        assert node["type"] in ("polygon", "line", "text-label"), (kind, node["type"])


# --------------------------------------------------------------------------- contrast

THEME_FILE = REPO_ROOT / "src/osprey/templates/apps/demo/data/canopy/theme/osprey.json"


def _resolved(colour: dict, theme: str) -> str:
    """A role's colour in *theme* (``light`` or ``dark``): the shipped theme's when it
    names the role, else the role's own fallback."""
    roles = json.loads(THEME_FILE.read_text())["roles"]
    if colour["$role"] in roles:
        return roles[colour["$role"]][theme]
    return colour["fallback" if theme == "light" else "fallbackDark"]


def _luminance(hex_colour: str) -> float:
    def channel(v: int) -> float:
        c = v / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (int(hex_colour[i : i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def contrast(a: str, b: str) -> float:
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


#: The grounds a panel screen draws on: the page, a section's frame, a field or chip.
GROUNDS = (symbols.PAGE_BACKGROUND, symbols.PANEL_BACKGROUND, symbols.FIELD_BACKGROUND)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_text_and_outlines_reach_the_contrast_floor_on_every_ground(theme) -> None:
    for ground in GROUNDS:
        bg = _resolved(ground, theme)
        for text in (symbols.PAGE_TEXT, symbols.MUTED_TEXT):
            assert contrast(_resolved(text, theme), bg) >= 4.5, (text["$role"], ground["$role"])
        for state in ("normal", "none", "disagree", "aggregate"):
            stroke = look(state)["lineColor"]
            assert contrast(_resolved(stroke, theme), bg) >= 3, (state, ground["$role"])
    # a light's mark is page text on the normal fill
    fill = _resolved(symbols.NORMAL_FILL, theme)
    assert contrast(_resolved(symbols.PAGE_TEXT, theme), fill) >= 4.5


def test_normal_and_none_differ_at_a_glance() -> None:
    normal, none = look("normal"), look("none")
    assert normal["lineColor"] == none["lineColor"] == symbols.PAGE_TEXT
    # no rule: filled with the ground it stands on, so it reads as an outline
    assert none["fillColor"] == symbols.PANEL_BACKGROUND
    assert normal["fillColor"] == symbols.NORMAL_FILL


@pytest.mark.parametrize("fill", [None, "#2f9e52"])
def test_the_seven_appearances_of_a_strip_symbol_are_told_apart(fill) -> None:
    def seen(state: str) -> tuple:
        shapes = symbols.coloured(
            "quadrupole_f", 0.0, AXIS, unit=UNIT, fill=fill, dashed=state == symbols.NONE
        )
        ring = symbols.ring(0, 0, 10, 10, 6, state)
        word = symbols.tag(5, 0, 64, 16, 11, state)
        return (
            tuple(n["props"]["lineStyle"] for n in shapes),
            tuple(json.dumps(n["props"].get("fillColor")) for n in shapes),
            ring["props"]["lineColor"]["$role"],
            ring["props"]["lineStyle"],
            word["props"]["text"],
        )

    for a, b in combinations(APPEARANCES, 2):
        assert seen(a) != seen(b), (fill, a, b)
    # the fill never says the state
    assert len({seen(state)[1] for state in APPEARANCES}) == 1


def test_a_state_ring_is_heavier_than_an_outline_and_colours_only_by_state() -> None:
    node = symbols.ring(0, 0, 10, 10, 6, "bad")
    assert node["props"]["lineWidth"] == 2.5 and node["props"]["fill"] is False
    assert node["props"]["lineColor"] == symbols.ALARM_MAJOR
    assert symbols.ring_look("warn")["lineColor"] == symbols.ALARM_MINOR
    assert symbols.ring_look("disagree")["text"] == symbols.DISAGREE_MARK
    assert symbols.ring_look("unknown")["lineStyle"] == "dashed"
    assert symbols.ring_look("normal")["lineColor"] == symbols.TRANSPARENT


def test_the_colour_distance_is_cie76_and_the_state_threshold_refuses_reds() -> None:
    assert symbols.colour_distance("#000000", "#ffffff") == pytest.approx(100, abs=0.1)
    assert symbols.near_state("#d8453b") == "MAJOR"  # a toned-down red is still a fault
    assert symbols.near_state("#4c78d8") is None
