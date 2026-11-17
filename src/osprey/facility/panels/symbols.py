"""Symbols of the panels' screens: the theme roles, the lights' looks and the element shapes.

Every colour any panel screen draws is a theme role from the table below: a
role name with light and dark fallbacks, so the viewer's theme decides the
colour. Colour means state only: OSPREY's own roles are neutral, ``Disagree``
marks sources that disagree, and the scheme's alarm roles (``MINOR``,
``MAJOR``, ``DISCONNECTED``) are used only where a colour means a state.

A light has six looks: one per health state, and an aggregate's. ``normal`` is
neutral, never green: an operator's eye goes to what is not normal.
``unknown`` is hollow and dashed, ``disagree`` and ``aggregate`` carry a mark,
so none of them reads as a colour alone.

An element symbol is neutral, its shape tells its type. In a state it takes
the same look a light takes, so a symbol in warn is a warn light of that shape.

:func:`draw` gives the widget nodes of one element symbol, without ids;
:mod:`~osprey.facility.panels.layout` places them and numbers them. Shapes are
written in multiples of a unit the caller passes, so this module holds ratios
and :mod:`~osprey.facility.panels.layout` holds every size.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from osprey.facility.panels.health import State

__all__ = [
    "AGGREGATE_MARK",
    "ALARM_ROLES",
    "APPEARANCES",
    "CONFLICT_MARK",
    "DISAGREE_MARK",
    "INFERRED_MARK",
    "KINDS",
    "LIGHT_PROPS",
    "MARK_PROP",
    "STATES",
    "UNKNOWN_MARK",
    "WIDTH_UNITS",
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
BOX_BORDER = role("Box_Border", "#d8d9da", "#2c3235")
LED_OFF = role("Led_Off", "#ecedef", "#22252b")
DISAGREE = role("Disagree", "#8e44ad", "#c39bd3")
ALARM_MINOR = role("MINOR", "#e0b400", "#f2cc0c")
ALARM_MAJOR = role("MAJOR", "#c4162a", "#e05454")
ALARM_DISCONNECTED = role("DISCONNECTED", "#86878f", "#6e6e77")
TRANSPARENT = role("Transparent", "#00000000", "#00000000")
# --- end theme roles -----------------------------------------------------------

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

#: Every look a light can take: the five states, and an aggregate's.
APPEARANCES: tuple[str, ...] = (*STATES, "aggregate")

#: The props of a light's shape that its rules set; the mark's text is :data:`MARK_PROP`.
LIGHT_PROPS = ("fill", "fillColor", "lineColor", "lineStyle")
MARK_PROP = "text"

_LOOKS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "normal": {
            "fill": True,
            "fillColor": LED_OFF,
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
    }
)


def look(state: str) -> Mapping[str, Any]:
    """The props a light takes in *state* (one of :data:`APPEARANCES`): its shape's :data:`LIGHT_PROPS` and the mark's text."""
    return _LOOKS[state]


# --- shape ratios, in units of the caller's ``unit`` ---------------------------
# Width of each kind along the beam; the strip reserves this much per symbol.
WIDTH_UNITS: Mapping[str, float] = MappingProxyType(
    {
        "bend": 3.0,
        "quadrupole_f": 1.5,
        "quadrupole_d": 1.5,
        "quadrupole": 1.5,
        "sextupole": 1.2,
        "skew_quadrupole": 1.2,
        "corrector": 0.6,
        "bpm": 0.8,
        "neg_pump": 1.0,
        "ion_pump": 1.0,
        "gauge": 0.6,
        "gate_valve": 1.0,
        "bellows": 1.0,
        "kicker": 2.0,
        "scraper": 0.4,
        "generic": 1.0,
    }
)
#: Every symbol kind :func:`draw` knows.
KINDS: tuple[str, ...] = tuple(WIDTH_UNITS)

_MAGNET_HEIGHT = 1.2  # a quadrupole's height off the axis, a bend's across it
_SEXTUPOLE_HEIGHT = 1.0
_CORRECTOR_HEIGHT = 1.6
_NEG_HEIGHT = 0.3  # a getter strip lies flat along the chamber
_VALVE_HEIGHT = 1.2
_BELLOWS_HEIGHT = 1.0
_BELLOWS_FOLDS = 4
_KICKER_HEIGHT = 1.0
_SCRAPER_HEIGHT = 1.4
_SKEW_TURN = 45  # degrees
_LINE = 1  # line width of a drawn outline, in px, whatever the unit
_THICK = 2  # line width of the bellows and the kicker
_MARK_FONT = 0.9  # a light's mark, in parts of the light's size
_OCTAGON = 8
# --- end shape ratios ----------------------------------------------------------

# A symbol with no state is drawn in these: an outline and a neutral fill.
_OUTLINE = PAGE_TEXT
_FILL = LED_OFF

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
            props :func:`styled` names) and a mark is drawn over them, last.
            ``None`` draws the symbol neutral, with no mark.
        unit: The size of one shape unit, in px.

    Raises:
        ValueError: *kind* is not a symbol kind.
    """
    if kind not in WIDTH_UNITS:
        raise ValueError(f"no symbol {kind!r}; the kinds are {', '.join(KINDS)}")
    w = WIDTH_UNITS[kind] * unit
    left = x - w / 2
    nodes = _draw(kind, left, y, w, unit)
    if state is None:
        return nodes
    shown = look(state)
    for node in nodes:
        for prop in styled(node["type"]):
            node["props"][prop] = shown[prop]
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


def _draw(kind: str, left: float, y: float, w: float, unit: float) -> list[Node]:
    if kind == "quadrupole_f":
        h = _MAGNET_HEIGHT * unit
        return [_shape("rectangle", left, y - h, w, h, fill=True)]
    if kind == "quadrupole_d":
        h = _MAGNET_HEIGHT * unit
        return [_shape("rectangle", left, y, w, h, fill=True)]
    if kind in ("quadrupole", "bend"):
        h = _MAGNET_HEIGHT * unit
        return [_shape("rectangle", left, y - h / 2, w, h, fill=True)]
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
    if kind == "skew_quadrupole":
        return [_shape("rectangle", left, y - w / 2, w, w, fill=False, rotation=_SKEW_TURN)]
    if kind == "corrector":
        h = _CORRECTOR_HEIGHT * unit
        return [_shape("rectangle", left, y - h / 2, w, h, fill=True)]
    if kind == "bpm":
        return [_shape("ellipse", left, y - w / 2, w, w, fill=False)]
    if kind == "neg_pump":
        h = _NEG_HEIGHT * unit
        return [_shape("rectangle", left, y - h / 2, w, h, fill=False)]
    if kind == "ion_pump":  # a triangle pointing down, the pump hanging off the chamber
        top, bottom = y - w / 2, y + w / 2
        return [_polygon([(left, top), (left + w, top), (left + w / 2, bottom)], fill=True)]
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
        return [_shape("rectangle", left, y - h / 2, w, h, fill=False, lineWidth=_THICK)]
    if kind == "scraper":
        h = _SCRAPER_HEIGHT * unit
        return [_shape("rectangle", left, y - h, w, h, fill=True)]
    # generic
    return [_shape("rectangle", left, y - w / 2, w, w, fill=False)]


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
