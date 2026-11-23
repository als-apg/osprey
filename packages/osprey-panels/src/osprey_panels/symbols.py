"""Symbols of the panels' screens: the theme roles, the lights' looks and the element shapes.

Every colour any panel screen draws is a theme role from the table below: a
role name with light and dark fallbacks, so the viewer's theme decides the
colour. Colour means state only: OSPREY's own roles are neutral, ``Disagree``
marks sources that disagree, and the scheme's alarm roles (``MINOR``,
``MAJOR``, ``DISCONNECTED``) are used only where a colour means a state.

A light has seven looks: one per health state, an aggregate's, and ``none``
for a thing with no rule (not a state: it has no rules). ``normal`` is
neutral, never green: an operator's eye goes to what is not normal.
``unknown`` is hollow and dashed, ``disagree`` and ``aggregate`` carry a mark,
``none`` is an outline in the text role filled with the ground it sits on, so none of them reads as a
colour alone and a thing with no rule never looks normal.

An element symbol's shape tells its type. It takes the same look a light
takes, so a symbol in warn is a warn light of that shape, and one with no rule
is in the ``none`` look. Every closed shape is a polygon, a circle a
many-sided one, because the pinned viewer dashes the outline of lines and
polygons only.

:func:`draw` gives the widget nodes of one element symbol, without ids;
:mod:`~osprey_panels.layout` places them and numbers them. Shapes are
written in multiples of a unit the caller passes, so this module holds ratios
and :mod:`~osprey_panels.layout` holds every size.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from osprey_panels.health import State

__all__ = [
    "AGGREGATE_MARK",
    "ALARM_ROLES",
    "APPEARANCES",
    "BADGE_LETTERS",
    "CONFLICT_MARK",
    "RING_PROPS",
    "STATE_DISTANCE",
    "TAG_PROP",
    "colour_distance",
    "coloured",
    "near_state",
    "ring",
    "ring_look",
    "tag",
    "DISAGREE_MARK",
    "INFERRED_MARK",
    "KINDS",
    "LIGHT_PROPS",
    "MARK_PROP",
    "NONE",
    "STATES",
    "UNKNOWN_MARK",
    "WIDTH_UNITS",
    "badge",
    "draw",
    "light",
    "look",
    "role",
    "styled",
]

Node = dict[str, Any]


def role(name: str, light: str, dark: str) -> dict[str, str]:
    """A theme colour role with its light and dark fallbacks."""
    return {"$role": name, "fallback": light, "fallbackDark": dark}


# --- theme roles: the only colour literals of the panels' screens -------------
PAGE_BACKGROUND = role("Page_Background", "#f4f5f5", "#111217")
PAGE_TEXT = role("Page_Text", "#24292e", "#ccccdc")
FIELD_BACKGROUND = role("Field_Background", "#fafbfc", "#181b1f")
PANEL_BACKGROUND = role("Tile_Background", "#ffffff", "#22252b")
BOX_BORDER = role("Box_Border", "#d8d9da", "#2c3235")
MUTED_TEXT = role("Text_Muted", "#6e6e77", "#8e8e9a")
NORMAL_FILL = role("Normal_Fill", "#b0b5bb", "#4a555c")
DISAGREE = role("Disagree", "#8e44ad", "#c39bd3")
ALARM_MINOR = role("MINOR", "#e0b400", "#f2cc0c")
ALARM_MAJOR = role("MAJOR", "#c4162a", "#e05454")
ALARM_DISCONNECTED = role("DISCONNECTED", "#86878f", "#6e6e77")
TRANSPARENT = role("Transparent", "#00000000", "#00000000")
# how a calibration value was found: an outline chip each, never a state colour
BASIS_DESIGN = role("Basis_Design", "#6e6e77", "#9aa3ad")
BASIS_BENCH = role("Basis_Bench", "#1f6fb8", "#5aa5e6")
BASIS_SURVEY = role("Basis_Survey", "#0f766e", "#3cc0b8")
BASIS_BEAM = role("Basis_Beam", "#1e7a46", "#46bd80")
# the ground of a calibration view's cell, a tint of its basis
BENCH_TINT = role("Basis_Bench_Tint", "#e3eefa", "#1f3347")
SURVEY_TINT = role("Basis_Survey_Tint", "#dff3f1", "#173a38")
BEAM_TINT = role("Basis_Beam_Tint", "#e1f2e8", "#1a3627")
DISAGREE_TINT = role("Disagree_Tint", "#efe3f5", "#33263d")
# what a magnet's drawing is made of: material, never state (a picture's own colours)
STEEL = role("Material_Steel", "#9aa3b2", "#4b5364")  # poles, a terminal block
STEEL_DARK = role("Material_Steel_Dark", "#c3c9d3", "#3a4150")  # the yoke
STEEL_EDGE = role("Material_Steel_Edge", "#d3d7de", "#353b47")
COPPER = role("Material_Copper", "#c06f25", "#c9803c")  # a main coil
COPPER_EDGE = role("Material_Copper_Edge", "#8a4d18", "#8a5526")
WINDING = role("Material_Winding", "#2f6fb0", "#6fa8dc")  # a corrector winding
WATER = role("Material_Water", "#1f7fa6", "#4aa3c7")  # a manifold
WINDING_2 = role("Material_Winding_2", "#7a4fb3", "#b58be0")  # a second winding, stacked
BORE = role("Material_Bore", "#eef0f3", "#15181d")  # the bore, the gap
#: The curves of one plot, in turn: page text first, then two hues that are no state.
CURVE_2 = role("Curve_2", "#1f6fb8", "#5aa5e6")
CURVE_3 = role("Curve_3", "#b45309", "#f0a050")
CURVES: tuple[dict[str, str], ...] = (PAGE_TEXT, CURVE_2, CURVE_3)
#: A drawn card's own device, highlighted (no state); a control's look.
FOCUS = role("Focus", "#1f6fb8", "#5aa5e6")
FOCUS_TINT = role("Focus_Tint", "#e6f0fa", "#1a2a3a")
CONTROL_ACTION = role("Control_Action", "#1f6fb8", "#3d8bd6")
CONTROL_ACTION_TEXT = role("Control_Action_Text", "#ffffff", "#ffffff")
CONTROL_CHOICE_SELECTED = role("Control_Choice_Selected", "#dbe9f7", "#1d3a57")
# --- end theme roles -----------------------------------------------------------

#: The colour of each calibration basis, and of a contested value (the disagree role).
BASIS_ROLES: Mapping[str, dict[str, str]] = MappingProxyType(
    {
        "design": BASIS_DESIGN,
        "bench": BASIS_BENCH,
        "survey": BASIS_SURVEY,
        "beam": BASIS_BEAM,
        "contested": DISAGREE,
    }
)
#: The ground of a calibration view's cell by state; a design value stands on the field.
BASIS_TINTS: Mapping[str, dict[str, str]] = MappingProxyType(
    {
        "design": FIELD_BACKGROUND,
        "bench": BENCH_TINT,
        "survey": SURVEY_TINT,
        "beam": BEAM_TINT,
        "contested": DISAGREE_TINT,
    }
)

#: The scheme roles that carry alarm meaning; no other colour does.
ALARM_ROLES = frozenset({"MINOR", "MAJOR", "DISCONNECTED"})

#: The glyph beside an item whose fact or rule is inferred rather than stated.
INFERRED_MARK = "≈"
#: The mark inside a light whose sources disagree.
DISAGREE_MARK = "≠"
#: The mark inside a light whose state is unknown.
UNKNOWN_MARK = "?"
#: The mark inside an aggregate: a roll-up too large for one rule.
AGGREGATE_MARK = "\u03a3"
#: The mark beside a rack unit whose device is claimed into more than one unit.
CONFLICT_MARK = "!"

#: The health states, worst first after ``normal``.
STATES: tuple[State, ...] = ("normal", "warn", "bad", "unknown", "disagree")

#: The look of a thing with no rule: not a state, it carries no rules and no mark.
NONE = "none"

#: Every look a light can take: the five states, an aggregate's, and :data:`NONE`.
APPEARANCES: tuple[str, ...] = (*STATES, "aggregate", NONE)

#: The props of a light's shape that its rules set; the mark's text is :data:`MARK_PROP`.
LIGHT_PROPS = ("fill", "fillColor", "lineColor", "lineStyle")
MARK_PROP = "text"

_LOOKS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "normal": {
            "fill": True,
            "fillColor": NORMAL_FILL,
            "lineColor": PAGE_TEXT,
            "lineStyle": "solid",
            "text": "",
        },
        "warn": {
            "fill": True,
            "fillColor": ALARM_MINOR,
            "lineColor": ALARM_MINOR,
            "lineStyle": "solid",
            "text": "",
        },
        "bad": {
            "fill": True,
            "fillColor": ALARM_MAJOR,
            "lineColor": ALARM_MAJOR,
            "lineStyle": "solid",
            "text": "",
        },
        "unknown": {
            "fill": False,
            "fillColor": PAGE_BACKGROUND,
            "lineColor": ALARM_DISCONNECTED,
            "lineStyle": "dashed",
            "text": UNKNOWN_MARK,
        },
        "disagree": {
            "fill": True,
            "fillColor": DISAGREE,
            "lineColor": PAGE_TEXT,
            "lineStyle": "solid",
            "text": DISAGREE_MARK,
        },
        "aggregate": {
            "fill": True,
            "fillColor": FIELD_BACKGROUND,
            "lineColor": PAGE_TEXT,
            "lineStyle": "solid",
            "text": AGGREGATE_MARK,
        },
        NONE: {
            "fill": True,
            "fillColor": PANEL_BACKGROUND,
            "lineColor": PAGE_TEXT,
            "lineStyle": "solid",
            "text": "",
        },
    }
)


def look(state: str) -> Mapping[str, Any]:
    """The props a light takes in *state* (one of :data:`APPEARANCES`): its shape's :data:`LIGHT_PROPS` and the mark's text."""
    return _LOOKS[state]


# --- shape ratios, in units of the caller's ``unit`` ---------------------------
# Width of each kind along the beam; the strip reserves this much per symbol.
WIDTH_UNITS: Mapping[str, float] = MappingProxyType(
    {
        "bend": 2.6,
        "quadrupole_f": 1.4,
        "quadrupole_d": 1.4,
        "quadrupole": 1.4,
        "sextupole": 1.4,
        "skew_quadrupole": 1.6,
        "corrector": 0.7,
        "bpm": 0.9,
        "neg_pump": 1.1,
        "ion_pump": 1.1,
        "pump": 1.1,
        "gauge": 1.1,
        "gate_valve": 1.0,
        "bellows": 0.7,
        "kicker": 2.1,
        "scraper": 1.4,
        "generic": 1.0,
    }
)
#: Every symbol kind :func:`draw` knows.
KINDS: tuple[str, ...] = tuple(WIDTH_UNITS)

_BEND_HEIGHT = 3.75  # across the axis
_QUAD_HEIGHT = 2.4  # across the axis; the polarity is in the shape
_LENS = 0.25  # how far a quadrupole's sides bow in or out, in parts of its width
_SEXTUPOLE_HEIGHT = 2.75
_CORRECTOR_HEIGHT = 2.0
_NEG_HEIGHT = 0.4  # a getter strip lies flat along the chamber
_VALVE_HEIGHT = 1.15
_BELLOWS_HEIGHT = 0.9
_BELLOWS_FOLDS = 4
_KICKER_HEIGHT = 1.75
_SCRAPER_HEIGHT = 1.75
_SKEW_TURN = 45  # degrees
_ION_PUMP_HEIGHT = 0.86  # an ion pump's box, in parts of its width
_FINS = 4  # its cooling fins, upright inside the box
_FIN_INSET = 0.14  # between the box and its fins, in parts of its width
_FINS_LEAST = 16  # px: a narrower box is drawn plain
_LINE = 1.5  # line width of a drawn outline, in px, at the least
_LINE_PER_UNIT = 0.1  # a symbol's outline grows with its unit
_THICK = 2  # line width of the bellows and the kicker, at the least
_MARK_FONT = 0.9  # a light's mark, in parts of the light's size
_OCTAGON = 8
_CIRCLE_SIDES = 16  # a circle is drawn as a polygon of this many sides
# --- end shape ratios ----------------------------------------------------------

# A symbol with no state is drawn in these: an outline and a neutral fill.
_OUTLINE = PAGE_TEXT
_FILL = NORMAL_FILL

# The props a look sets on each widget type of a symbol; a closed shape takes
# all of a light's, a line has no fill, a mark takes its text.
_STYLED: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {"line": ("lineColor", "lineStyle"), "text-label": (MARK_PROP,)}
)


def styled(node_type: str) -> tuple[str, ...]:
    """The props of a symbol's widget of *node_type* that its look, and so its rules, set."""
    return _STYLED.get(node_type, LIGHT_PROPS)


def _geometry(x: float, y: float, w: float, h: float) -> dict[str, float]:
    return {"x": x, "y": y, "width": w, "height": h}


def _shape(kind: str, x: float, y: float, w: float, h: float, *, fill: bool, **more: Any) -> Node:
    props: dict[str, Any] = {"lineColor": _OUTLINE, "lineWidth": _LINE, "fill": fill}
    if fill:
        props["fillColor"] = _FILL
    props.update(more)
    return {"type": kind, "props": props, "geometry": _geometry(x, y, w, h)}


def _polygon(points: list[tuple[float, float]], *, fill: bool, **more: Any) -> Node:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, y0 = min(xs), min(ys)
    return _shape(
        "polygon",
        x0,
        y0,
        max(xs) - x0,
        max(ys) - y0,
        fill=fill,
        points=[{"x": px - x0, "y": py - y0} for px, py in points],
        **more,
    )


def draw(
    kind: str, x: float, y: float, state: str | None = None, *, unit: float = 1.0
) -> list[Node]:
    """The widget nodes of one element symbol, without ids.

    Args:
        kind: One of :data:`KINDS`.
        x: The centre of the symbol along the beam.
        y: The axis the symbol sits on. A focusing quadrupole is drawn above
            it, a defocusing one below, a quadrupole with no polarity centred
            on it; every other kind is centred on it.
        state: One of :data:`APPEARANCES`: the shapes take its look (the
            props :func:`styled` names) and, but for :data:`NONE`, a mark is
            drawn over them, last. ``None`` draws the symbol neutral, with no
            mark.
        unit: The size of one shape unit, in px.

    Raises:
        ValueError: *kind* is not a symbol kind.
    """
    if kind not in WIDTH_UNITS:
        raise ValueError(f"no symbol {kind!r}; the kinds are {', '.join(KINDS)}")
    w = WIDTH_UNITS[kind] * unit
    left = x - w / 2
    nodes = _draw(kind, left, y, w, unit)
    for node in nodes:  # an outline of comparable weight to a filled symbol
        node["props"]["lineWidth"] = max(node["props"]["lineWidth"], unit * _LINE_PER_UNIT)
    if state is None:
        return nodes
    shown = look(state)
    for node in nodes:
        for prop in styled(node["type"]):
            node["props"][prop] = shown[prop]
    if state != NONE:
        nodes.append(_symbol_mark(nodes, shown[MARK_PROP], unit))
    return nodes


def _symbol_mark(nodes: list[Node], text: str, unit: float) -> Node:
    """A symbol's mark: a square over the symbol's centre, at least one unit wide."""
    left = min(n["geometry"]["x"] for n in nodes)
    top = min(n["geometry"]["y"] for n in nodes)
    right = max(n["geometry"]["x"] + n["geometry"]["width"] for n in nodes)
    bottom = max(n["geometry"]["y"] + n["geometry"]["height"] for n in nodes)
    side = max(right - left, bottom - top, unit)
    cx, cy = (left + right) / 2, (top + bottom) / 2
    return {
        "type": "text-label",
        "props": {
            "text": text,
            "fontSize": round(unit * _MARK_FONT),
            "align": "center",
            "foregroundColor": PAGE_TEXT,
            "backgroundColor": TRANSPARENT,
        },
        "geometry": _geometry(cx - side / 2, cy - side / 2, side, side),
    }


def _box(left: float, top: float, w: float, h: float, *, fill: bool, **more: Any) -> Node:
    """A rectangle as a polygon, so its outline can be dashed."""
    corners = [(left, top), (left + w, top), (left + w, top + h), (left, top + h)]
    return _polygon(corners, fill=fill, **more)


def _ring(cx: float, cy: float, r: float, sides: int) -> list[tuple[float, float]]:
    """The corners of a regular polygon of *sides* on a circle of radius *r*, one at angle 0.

    With a number of sides divisible by four the polygon is as wide and as tall as the circle.
    """
    step = 2 * math.pi / sides
    return [(cx + r * math.cos(step * i), cy + r * math.sin(step * i)) for i in range(sides)]


def _draw(kind: str, left: float, y: float, w: float, unit: float) -> list[Node]:
    if kind in ("quadrupole_f", "quadrupole_d"):
        # centred on the axis: a focusing one bows out like a convex lens, a
        # defocusing one is waisted like a concave one
        h = _QUAD_HEIGHT * unit
        top, bottom, right = y - h / 2, y + h / 2, left + w
        bow = _LENS * w
        if kind == "quadrupole_f":
            points = [
                (left + bow, top),
                (right - bow, top),
                (right, y - h / 4),
                (right, y + h / 4),
                (right - bow, bottom),
                (left + bow, bottom),
                (left, y + h / 4),
                (left, y - h / 4),
            ]
        else:
            points = [
                (left, top),
                (right, top),
                (right - bow, y),
                (right, bottom),
                (left, bottom),
                (left + bow, y),
            ]
        return [_polygon(points, fill=True)]
    if kind in ("quadrupole", "bend"):
        h = (_QUAD_HEIGHT if kind == "quadrupole" else _BEND_HEIGHT) * unit
        return [_box(left, y - h / 2, w, h, fill=True)]
    if kind == "sextupole":
        h = _SEXTUPOLE_HEIGHT * unit
        q = w / 4
        points = [
            (left + q, y - h / 2),
            (left + w - q, y - h / 2),
            (left + w, y),
            (left + w - q, y + h / 2),
            (left + q, y + h / 2),
            (left, y),
        ]
        return [_polygon(points, fill=True)]
    if kind == "skew_quadrupole":  # a square turned by _SKEW_TURN: a diamond
        half = w / 2
        cx = left + half
        assert _SKEW_TURN == 45
        points = [(cx, y - half), (cx + half, y), (cx, y + half), (cx - half, y)]
        return [_polygon(points, fill=False)]
    if kind == "corrector":
        h = _CORRECTOR_HEIGHT * unit
        return [_box(left, y - h / 2, w, h, fill=True)]
    if kind == "bpm":
        return [_polygon(_ring(left + w / 2, y, w / 2, _CIRCLE_SIDES), fill=False)]
    if kind == "neg_pump":
        h = _NEG_HEIGHT * unit
        return [_box(left, y - h / 2, w, h, fill=False)]
    if kind == "pump":  # a circle on top of the chamber: a pump of no stated kind
        r = w / 2
        return [_polygon(_ring(left + r, y - r, r, _CIRCLE_SIDES), fill=True)]
    if kind == "ion_pump":  # a box with cooling fins: how an operator knows the pump
        h = _ION_PUMP_HEIGHT * w
        top = y - h / 2
        body = _box(left, top, w, h, fill=True)
        if w < _FINS_LEAST:
            return [body]
        inset = _FIN_INSET * w
        step = (w - 2 * inset) / (_FINS - 1)
        fins: list[Node] = []
        for i in range(_FINS):
            x = left + inset + i * step
            props = {
                "lineColor": STEEL,
                "lineWidth": 1,
                "points": [{"x": 0, "y": 0}, {"x": 0, "y": h - 2 * inset}],
                "arrowStart": False,
                "arrowEnd": False,
            }
            fins.append(
                {
                    "type": "line",
                    "props": props,
                    "geometry": _geometry(x, top + inset, 1, h - 2 * inset),
                }
            )
        return [body, *fins]
    if kind == "gauge":  # a triangle pointing up
        top, bottom = y - w / 2, y + w / 2
        return [_polygon([(left + w / 2, top), (left + w, bottom), (left, bottom)], fill=False)]
    if kind == "gate_valve":
        h = _VALVE_HEIGHT * unit
        top, bottom = y - h / 2, y + h / 2
        points = [(left, top), (left + w, bottom), (left + w, top), (left, bottom)]
        return [_polygon(points, fill=True)]
    if kind == "bellows":
        h = _BELLOWS_HEIGHT * unit
        step = w / _BELLOWS_FOLDS
        points = [
            (left + i * step, y - h / 2 if i % 2 == 0 else y + h / 2)
            for i in range(_BELLOWS_FOLDS + 1)
        ]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x0, y0 = min(xs), min(ys)
        props = {
            "lineColor": _OUTLINE,
            "lineWidth": _THICK,
            "points": [{"x": px - x0, "y": py - y0} for px, py in points],
            "arrowStart": False,
            "arrowEnd": False,
        }
        return [{"type": "line", "props": props, "geometry": _geometry(x0, y0, w, h)}]
    if kind == "kicker":
        h = _KICKER_HEIGHT * unit
        return [_box(left, y - h / 2, w, h, fill=False, lineWidth=_THICK)]
    if kind == "scraper":
        h = _SCRAPER_HEIGHT * unit
        return [_box(left, y - h, w, h, fill=True)]
    # generic
    return [_box(left, y - w / 2, w, w, fill=False)]


#: The letter in a badge of a kind; any other kind shows its first letter.
BADGE_LETTERS: Mapping[str, str] = MappingProxyType(
    {"neg_pump": "N", "ion_pump": "I", "pump": "P", "gauge": "G"}
)


def badge(cx: float, cy: float, size: float, kind: str, font: float) -> list[Node]:
    """A device off the beam axis: a circle of *size* centred on *cx*, *cy*, its kind's letter in it.

    The circle (a many-sided polygon) is neutral, as :func:`draw` leaves a
    symbol with no state; the caller gives it a look. The letter, at *font*,
    is the second node and never changes.
    """
    circle = _polygon(_ring(cx, cy, size / 2, _CIRCLE_SIDES), fill=True)
    letter = BADGE_LETTERS.get(kind, kind[:1].upper())
    mark = {
        "type": "text-label",
        "props": {
            "text": letter,
            "fontSize": font,
            "align": "center",
            "foregroundColor": PAGE_TEXT,
            "backgroundColor": TRANSPARENT,
        },
        "geometry": _geometry(cx - size / 2, cy - size / 2, size, size),
    }
    return [circle, mark]


# --- the strip: type colour fills, state rings ---------------------------------
#: What a state adds around a strip symbol: its ring (``lineColor``,
#: ``lineStyle``) and the word above the ring (``text``). Normal adds nothing;
#: no rule adds no ring and dashes the symbol's own outline instead.
RING_PROPS = ("lineColor", "lineStyle")
TAG_PROP = "text"
_RINGS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "normal": {"lineColor": TRANSPARENT, "lineStyle": "solid", "text": ""},
        "warn": {"lineColor": ALARM_MINOR, "lineStyle": "solid", "text": "WARNING"},
        "bad": {"lineColor": ALARM_MAJOR, "lineStyle": "solid", "text": "FAULT"},
        "unknown": {"lineColor": ALARM_DISCONNECTED, "lineStyle": "dashed", "text": UNKNOWN_MARK},
        "disagree": {"lineColor": DISAGREE, "lineStyle": "solid", "text": DISAGREE_MARK},
        "aggregate": {"lineColor": PAGE_TEXT, "lineStyle": "solid", "text": AGGREGATE_MARK},
        NONE: {"lineColor": TRANSPARENT, "lineStyle": "solid", "text": ""},
    }
)
_RING_WIDTH = 2.5
_RING_CORNER = 0.3  # the ring's cut corners, in parts of its pad


def ring_look(state: str) -> Mapping[str, Any]:
    """What *state* (one of :data:`APPEARANCES`) sets on a strip symbol's ring and tag."""
    return _RINGS[state]


def coloured(
    kind: str, x: float, y: float, *, unit: float, fill: str | None = None, dashed: bool = False
) -> list[Node]:
    """A strip symbol in its type colour: *fill* (a colour the drawing settings
    give its kind), else the panel's ground; outlined in the text role, the
    outline dashed when the thing has no rule. Nothing of it changes with state."""
    nodes = draw(kind, x, y, unit=unit)
    for node in nodes:
        props = node["props"]
        props["lineColor"] = PAGE_TEXT
        props["lineStyle"] = "dashed" if dashed else "solid"
        if node["type"] == "polygon":
            props["fill"] = True
            props["fillColor"] = fill if fill is not None else PANEL_BACKGROUND
    return nodes


def ring(left: float, top: float, right: float, bottom: float, pad: float, state: str) -> Node:
    """The ring around a box, *pad* outside it, with cut corners, in *state*'s look."""
    c = pad * _RING_CORNER
    x0, y0, x1, y1 = left - pad, top - pad, right + pad, bottom + pad
    points = [
        (x0 + c, y0),
        (x1 - c, y0),
        (x1, y0 + c),
        (x1, y1 - c),
        (x1 - c, y1),
        (x0 + c, y1),
        (x0, y1 - c),
        (x0, y0 + c),
    ]
    node = _polygon(points, fill=False, lineWidth=_RING_WIDTH)
    for prop in RING_PROPS:
        node["props"][prop] = _RINGS[state][prop]
    return node


def tag(cx: float, bottom: float, width: float, height: float, font: float, state: str) -> Node:
    """The word above a ring, centred on *cx*, ending at *bottom*, in *state*'s look."""
    return {
        "type": "text-label",
        "props": {
            "text": _RINGS[state][TAG_PROP],
            "fontSize": font,
            "align": "center",
            "foregroundColor": PAGE_TEXT,
            "backgroundColor": TRANSPARENT,
        },
        "geometry": _geometry(cx - width / 2, bottom - height, width, height),
    }


# --- colour distance --------------------------------------------------------------


def _lab(hex_colour: str) -> tuple[float, float, float]:
    """CIE L*a*b* (D65) of an ``#rrggbb`` colour."""

    def linear(v: int) -> float:
        c = v / 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (linear(int(hex_colour[i : i + 2], 16)) for i in (1, 3, 5))
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > (6 / 29) ** 3 else t / (3 * (6 / 29) ** 2) + 4 / 29

    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def colour_distance(a: str, b: str) -> float:
    """How far apart two ``#rrggbb`` colours look: CIE76 delta E (about 2.3 is the
    smallest difference an eye sees side by side; 20 and more reads as another colour)."""
    return math.dist(_lab(a), _lab(b))


#: A type colour nearer than this (CIE76 delta E) to a state colour could be read as
#: that state: about the distance at which two flat fills stop reading as different
#: colours at a glance, rather than shades of one.
STATE_DISTANCE = 20.0

#: The state colours a type colour must stay clear of: bad, warn, disagree.
STATE_ROLES = (ALARM_MAJOR, ALARM_MINOR, DISAGREE)


def near_state(
    colour: str, extra: Mapping[str, Mapping[str, str]] = MappingProxyType({})
) -> str | None:
    """The state role whose colour *colour* is within :data:`STATE_DISTANCE` of, or ``None``.

    Each role is taken in both themes: its fallbacks, and the colours *extra*
    (a theme's ``roles``) gives it.
    """
    for state in STATE_ROLES:
        name = state["$role"]
        shades = [state["fallback"], state["fallbackDark"], *extra.get(name, {}).values()]
        if any(colour_distance(colour, shade) < STATE_DISTANCE for shade in shades):
            return name
    return None


_ARC_STEP = 3  # degrees between two corners of an arc's curved edges


def arc(
    cx: float, cy: float, inner: float, outer: float, start: float, end: float, state: str
) -> Node:
    """An arc of a ring in the look of *state*, as one polygon: the part of the ring
    between radii *inner* and *outer* from angle *start* to *end*, in degrees
    clockwise from twelve o'clock (where the ring's first arc begins)."""
    steps = max(1, math.ceil((end - start) / _ARC_STEP))
    angles = [math.radians(start + (end - start) * i / steps) for i in range(steps + 1)]
    edge = [(cx + outer * math.sin(a), cy - outer * math.cos(a)) for a in angles]
    edge += [(cx + inner * math.sin(a), cy - inner * math.cos(a)) for a in reversed(angles)]
    shown = look(state)
    shape = _polygon(edge, fill=shown["fill"])
    for prop in LIGHT_PROPS:
        shape["props"][prop] = shown[prop]
    return shape


def light(x: float, y: float, size: float, state: str) -> list[Node]:
    """A light's two nodes, without ids: its shape (an octagon) and its mark.

    The shape is a polygon because the pinned viewer draws a dashed outline on
    lines and polygons only. *x*, *y* is the top-left corner of its box.
    """
    shown = look(state)
    r = size / 2
    cx, cy = x + r, y + r
    k = r * (2**0.5 - 1)  # half an octagon's side, for a circumradius of about r
    points = [
        (cx - k, cy - r),
        (cx + k, cy - r),
        (cx + r, cy - k),
        (cx + r, cy + k),
        (cx + k, cy + r),
        (cx - k, cy + r),
        (cx - r, cy + k),
        (cx - r, cy - k),
    ]
    assert len(points) == _OCTAGON
    shape = _polygon(points, fill=shown["fill"])
    for prop in LIGHT_PROPS:
        shape["props"][prop] = shown[prop]
    mark = {
        "type": "text-label",
        "props": {
            "text": shown["text"],
            "fontSize": round(size * _MARK_FONT),
            "align": "center",
            "foregroundColor": PAGE_TEXT,
            "backgroundColor": TRANSPARENT,
        },
        "geometry": _geometry(x, y, size, size),
    }
    return [shape, mark]
