"""Layout: a view description drawn as a Canopy Screen-IR 1.1 document.

This is the only module of the panels with geometry. :func:`render` walks a
:class:`~osprey.facility.panels.view.View` top to bottom, one section after
another, and draws every item with the pinned viewer's widgets. Colours are
the theme roles of :mod:`~osprey.facility.panels.symbols`.

States are the viewer's to compute. A light binds its channels and carries
the format's rules, written from the channels' health rows (see
:func:`ladder`): the viewer shows the state the rows give the live value, and
``unknown`` before a value arrives. Nothing here reads a value.

Where a view needs a widget the format lacks, the nearest one is drawn and the
node says so in its ``meta.comment`` (``format-request: <name>``); the needs
are :data:`FORMAT_REQUESTS`, and :func:`needs` lists those a document hits. A
light whose rule the format cannot write is drawn ``unknown`` and counted
(:func:`unexpressed`).

Every size, spacing and font size is in the constants block below.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from osprey.facility.panels import health, strip, symbols
from osprey.facility.panels.describe import light as describe_light
from osprey.facility.panels.health import State
from osprey.facility.panels.model import Citation, PanelGraph
from osprey.facility.panels.symbols import (
    BOX_BORDER,
    FIELD_BACKGROUND,
    PAGE_BACKGROUND,
    PAGE_TEXT,
    TRANSPARENT,
)
from osprey.facility.panels.view import (
    CheckRow,
    Elevation,
    Fact,
    Grid,
    Light,
    Link,
    Node,
    Picture,
    Placement,
    RackUnit,
    Section,
    Strip,
    StripElement,
    Target,
    Totals,
    ValueRow,
    View,
)

__all__ = [
    "FORMAT_REQUESTS",
    "MIN_STRIP_SCALE",
    "Ladder",
    "ladder",
    "aggregates",
    "needs",
    "render",
    "rule_length",
    "screen_id",
    "text_width",
    "unexpressed",
]

# --- constants: every size, spacing and font size of a panel screen -----------
PAGE_WIDTH = 1200
MARGIN = 16
TITLE_SIZE = 20
HEADING_SIZE = 16
TEXT_SIZE = 13
SMALL_SIZE = 11
CHAR_WIDTH = 0.6  # estimated width of one character, in parts of the font size
TEXT_PAD = 6  # added to an estimated text width
ROW = 24  # height of one row of text, a field or a button
ROW_GAP = 4
SECTION_GAP = 16
RULE_GAP = 6  # between a heading and the line under it
GAP = 8  # between two widgets on one row
INDENT = 20  # one level of a roll-up tree
LABEL_WIDTH = 200
VALUE_WIDTH = 130
BUTTON_PAD = 16
MAX_BUTTON_WIDTH = 420
LIGHT_SIZE = 14
MARK_WIDTH = 14
LINE_WIDTH = 1
AXIS_WIDTH = 2
CELL_WIDTH = 110  # a grid's or a checkout's column
NAME_WIDTH = 260  # a grid's or a checkout's first column
STRIP_WIDTH = PAGE_WIDTH - 2 * MARGIN
MIN_STRIP_SCALE = 0.5  # below this a strip breaks into rows
SYMBOL_UNIT = 14  # one shape unit of a symbol, in px, at scale 1
STRIP_ABOVE = 40  # height of the lane above the beam, from the axis
STRIP_LEVEL = 16  # second level of the above lane, further up
STRIP_BELOW = 28  # room under the axis
LOCATOR_GAP = 10
LOCATOR_BAR = 6
TICK = 5
TICK_LABEL_WIDTH = 60
MAX_TICKS = 12
TICK_STEPS = (1, 2, 5)
TICK_BASE = 10
TRAY_PAD = 8
PICTURE_SIZE = 260
PICTURE_PAD = 24
PART_MARKER = 8
PART_LABEL_WIDTH = 140
RACK_U = 22  # one rack unit
RACK_WIDTH = 420
RACK_LABEL_WIDTH = 56
UNPOSITIONED_GAP = 40
# The longest expression a light's rule may compile to. The pinned viewer
# compiles a rule's ladder into one Fox expression (``v0 if (c0) else (...)``);
# CANOPY's formula engine refuses one longer than ``max_expression_len``
# (canopy/formula_engine/config.py at 166989aa: 2048 characters).
RULE_EXPRESSION_LIMIT = 2048
TOOLTIP_CHANNELS = 8  # a light's tooltip names this many channels, else counts them
MOSAIC_LIGHT = 10  # one child's light in an aggregate's mosaic
MOSAIC_GAP = 3
MAX_TABLE_ROWS = 400  # rows of one table drawn; the rest are counted under it
SCREEN_ID_PREFIX = "panels"
# --- end constants -------------------------------------------------------------

#: Widgets the format lacks, drawn with the nearest one: (name, why).
FORMAT_REQUESTS: tuple[tuple[str, str], ...] = (
    (
        "state light",
        "a light with five states (normal, warn, bad, unknown, disagree); pv-led has on and "
        "off, so a light is a polygon whose rules set fill, colours and line style",
    ),
    (
        "hatched fill",
        "disagree should read as hatched; no drawing widget fills with a hatch, so the light "
        "carries a mark",
    ),
    (
        "table",
        "a grid of members by channel and the checkout are tables; there is no table "
        "container that scrolls or pages, so every cell is placed on its own and a long "
        "table is cut, its remaining rows counted",
    ),
    (
        "clickable text",
        "a link in a list or a tree reads as text that opens a screen; the nearest widget is "
        "a related-display button",
    ),
    (
        "label condition",
        "a health row that names a text label cannot be written as a rule condition: the "
        "viewer gives an enum channel's value as its index, and the label's index is not known",
    ),
    (
        "aggregate state",
        "a roll-up whose rule would be longer than the format's expression limit: the worst "
        "state over many channels, computed by the viewer or a server, drawn as one light; "
        "today it is an aggregate mark over a mosaic of its children's lights",
    ),
    (
        "long rule",
        "one channel whose health rows give a rule longer than the format's expression "
        "limit; its light is drawn unknown",
    ),
)
_REQUEST_NAMES = frozenset(name for name, _ in FORMAT_REQUESTS)

_REQUEST_TAG = "format-request: "
_UNKNOWN_TAG = "unknown: "
_AGGREGATE_ROW = "aggregate: row"
_AGGREGATE_TILE = "aggregate: tile"
_AGGREGATE_SYMBOL = "aggregate: symbol"
_AGGREGATE_TEXT = "{} channels, open for detail"
_IF_ELSE = " if () else ()"  # what the viewer adds around each arm when it compiles a ladder
_NAVIGATION_TARGET = "tab"
_HOVER_LABEL = "hoverLabel"
_TOOLTIP = "tooltip"
_YES, _NO = "yes", "no"
_NO_RULE = "no rule"
_NOT_PLACED = "not placed"
_NO_UNIT = "no unit"
_ROWS_NOT_DRAWN = "rows not drawn: the screen holds a table of at most this many rows"
_CONTESTED = "contested"
_STATE_ORDER: tuple[State, ...] = ("bad", "warn", "disagree")
_SEGMENT = re.compile(r"[^A-Za-z0-9_-]+")


def text_width(text: str, size: float) -> float:
    """A box wide enough for *text* at *size* px: the one text-width estimate."""
    return math.ceil(len(text) * size * CHAR_WIDTH) + TEXT_PAD


def screen_id(target: Target) -> str:
    """The document id of a target's screen when the caller names none."""
    name = _SEGMENT.sub("_", target.id).strip("_") or "_"
    view = target.view if target.by is None else f"{target.view}-{target.by}"
    return f"{SCREEN_ID_PREFIX}/{view}/{target.kind}-{name}"


# --------------------------------------------------------------------------- rules


@dataclass(frozen=True)
class Ladder:
    """A light's rule: the channels it binds and its arms, first match wins.

    Attributes:
        channels: The bound channels; ``{i}`` in an arm is ``channels[i]``.
        arms: ``(expression, state)`` in order; no arm matching is ``unknown``.
        unwritten: Channels whose rows no condition can say; they add no arm,
            and the light is never ``normal``.
    """

    channels: tuple[str, ...]
    arms: tuple[tuple[str, State], ...]
    unwritten: tuple[str, ...] = ()


def _number(value: float) -> str:
    return repr(value)


def _condition(row_form: health.Form, token: str) -> str | None:
    """A rule condition over *token* for one health row form; ``None`` when none can say it."""
    if row_form.kind == "compare":
        return f"{token} {row_form.op} {_number(row_form.number)}"
    if row_form.kind == "bit":
        weight = 2**row_form.bit
        return f"{token} == {token} // 1 and ({token} // {weight}) % 2 == {row_form.wanted}"
    if row_form.number is None:
        return None
    return f"{token} == {_number(row_form.number)}"


def _any(conditions: Sequence[str]) -> str | None:
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return " or ".join(f"({c})" for c in conditions)


def _exclusive(a: health.Form, b: health.Form) -> bool:
    """Whether no value can match both forms: two literals of different numbers."""
    return (
        a.kind == b.kind == "literal"
        and a.number is not None
        and b.number is not None
        and a.number != b.number
    )


def _channel_arms(graph: PanelGraph, item: Light, token: str) -> dict[str, str | None] | str:
    """The condition under which one channel is in each state, or why none can be written.

    Rows with one condition are taken together, so agreeing sources write it
    once; two conditions no value can meet both of never disagree.
    """
    found: dict[str, tuple[health.Form, set[str]]] = {}
    for row in graph.rule(item.channel):
        row_form = health.form(row.value)
        if row_form is None:
            continue  # the engine never matches it either
        condition = _condition(row_form, token)
        if condition is None:
            return "label condition"
        found.setdefault(condition, (row_form, set()))[1].add(row.state)
    conds = list(found.items())

    def meets(i: int, j: int) -> bool:
        return not _exclusive(conds[i][1][0], conds[j][1][0])

    pairs = [
        f"({a}) and ({b})"
        for i, (a, (_, states_a)) in enumerate(conds)
        for j, (b, (_, states_b)) in enumerate(conds)
        if i < j and len(states_a) == len(states_b) == 1 and states_a != states_b and meets(i, j)
    ]
    mixed = [c for c, (_, states) in conds if len(states) > 1]
    arms: dict[str, str | None] = {"disagree": _any([*mixed, *pairs])}
    for state in ("bad", "warn", "normal"):
        terms = []
        for i, (c, (_, states)) in enumerate(conds):
            if states != {state}:
                continue
            against = _any(
                [d for j, (d, (_, other)) in enumerate(conds) if other != {state} and meets(i, j)]
            )
            terms.append(c if against is None else f"({c}) and not ({against})")
        arms[state] = _any(terms)
    # A light the rows flag is never drawn normal: where its rule says normal it
    # shows what the flag says (disagree), or unknown (rows the engine skips).
    if arms["normal"] is not None and item.disagree:
        arms["disagree"] = _any([c for c in (arms["disagree"], arms["normal"]) if c])
        arms["normal"] = None
    elif item.unsupported:
        arms["normal"] = None
    return arms


def ladder(graph: PanelGraph, lights: Sequence[Light], *, node: bool = False) -> Ladder | str:
    """The rule of one light over *lights*' channels: the worst state among them.

    A channel is ``disagree`` where rows of different states match its value,
    and in the state of the matching rows otherwise; rows the engine cannot
    evaluate are left out, as the engine leaves them out. The light is the
    worst channel (``bad > warn > disagree > unknown > normal``), so it is
    ``normal`` only when every channel is.

    A channel whose rows no condition can say adds no arm, so the others'
    ``bad``, ``warn`` and ``disagree`` still show and the light is never
    ``normal``; when that is every channel, the light cannot be written.

    Args:
        graph: The panel graph.
        lights: The channels the light reads.
        node: The light is a roll-up node's: over the limit it is an
            aggregate even with one channel. Otherwise only a light over
            several channels is.

    Returns:
        The ladder, or the :data:`FORMAT_REQUESTS` name of why the format
        cannot write it.
    """
    too_long = "aggregate state" if node or len(lights) > 1 else "long rule"
    per_channel: list[dict[str, str | None]] = []
    unwritten: list[str] = []
    reason = ""
    written = 0
    for i, item in enumerate(lights):
        arms = _channel_arms(graph, item, f"{{{i}}}")
        if isinstance(arms, str):
            reason = arms
            unwritten.append(item.channel)
            arms = dict.fromkeys(("disagree", "bad", "warn", "normal"))
        per_channel.append(arms)
        written += sum(len(c) for c in arms.values() if c)
        if written > RULE_EXPRESSION_LIMIT:
            return too_long  # every condition appears at least once in the ladder
    if unwritten and len(unwritten) == len(lights):
        return reason
    out: list[tuple[str, State]] = []
    for state in _STATE_ORDER:
        found = _any([c for arms in per_channel if (c := arms[state])])
        if found is not None:
            out.append((found, state))
    normals = [arms["normal"] for arms in per_channel]
    if normals and all(normals):
        joined = normals[0] if len(normals) == 1 else " and ".join(f"({c})" for c in normals)
        out.append((joined, "normal"))  # type: ignore[arg-type]
    rule = Ladder(tuple(item.channel for item in lights), tuple(out), tuple(unwritten))
    return too_long if rule_length(rule) > RULE_EXPRESSION_LIMIT else rule


def rule_length(rule: Ladder) -> int:
    """The length of the expression the viewer compiles a light's rule to, at most.

    Each arm is its condition, the value written as a literal (the longest
    look a light's rules write) and the ``if () else ()`` around it; the
    default ends the ladder.
    """
    longest = max(
        len(json.dumps(symbols.look(state)[prop]))
        for state in symbols.STATES
        for prop in (*symbols.LIGHT_PROPS, symbols.MARK_PROP)
    )
    return sum(len(c) + longest + len(_IF_ELSE) for c, _ in rule.arms) + longest


def _tolerance_ladder(row: ValueRow) -> Ladder | None:
    """Within the tolerance normal, outside it warn: setpoint ``{0}``, readback ``{1}``."""
    if row.tolerance is None or row.readback is None:
        return None
    term, amount = row.tolerance
    if term == "relative":
        within = f"({{1}} - {{0}}) ** 2 <= ({_number(amount)} * {{0}}) ** 2"
    else:
        within = f"({{1}} - {{0}}) ** 2 <= {_number(amount * amount)}"
    return Ladder((row.channel, row.readback), ((within, "normal"), (f"not ({within})", "warn")))


# --------------------------------------------------------------------------- the document


def _n(value: float) -> float | int:
    """A coordinate rounded to a tenth, integral values as int."""
    rounded = round(value, 1)
    return int(rounded) if rounded == int(rounded) else rounded


def _geometry(x: float, y: float, w: float, h: float) -> dict[str, float | int]:
    return {"x": _n(x), "y": _n(y), "width": _n(w), "height": _n(h)}


@dataclass
class _Screen:
    """One Screen-IR document under construction, ids in build order."""

    doc_id: str
    title: str
    nodes: list[dict[str, Any]] = field(default_factory=list)
    unknown: int = 0
    _widgets: int = 1
    _rules: int = 0

    def add(
        self,
        node: dict[str, Any],
        *,
        rules: list[dict[str, Any]] | None = None,
        tags: Sequence[str] = (),
    ) -> dict[str, Any]:
        self._widgets += 1
        g = node["geometry"]
        out: dict[str, Any] = {
            "id": f"w-{self._widgets:04d}",
            "type": node["type"],
            "props": node["props"],
            "geometry": _geometry(g["x"], g["y"], g["width"], g["height"]),
        }
        if "points" in out["props"]:
            out["props"]["points"] = [
                {"x": _n(p["x"]), "y": _n(p["y"])} for p in out["props"]["points"]
            ]
        if rules:
            out["rules"] = rules
        if tags:
            out["meta"] = {"comment": "; ".join(tags)}
        self.nodes.append(out)
        return out

    def widget(self, kind: str, x, y, w, h, props: dict[str, Any], **kw: Any) -> dict[str, Any]:
        return self.add({"type": kind, "props": props, "geometry": _geometry(x, y, w, h)}, **kw)

    def rule(self, name: str, target: str, ladder: Ladder, values: Mapping[str, Any]) -> dict:
        self._rules += 1
        return {
            "id": f"r-{self._rules:04d}",
            "name": name,
            "targetProperty": target,
            "default": values["unknown"],
            "pvs": [{"name": channel, "trigger": True} for channel in ladder.channels],
            "conditions": [
                {"expression": expression, "value": values[state]}
                for expression, state in ladder.arms
            ],
        }

    def document(self, height: float) -> dict[str, Any]:
        size = {"width": PAGE_WIDTH, "height": _n(height)}
        description = f"Generated from the facility's panel view {self.doc_id}."
        if self.unknown:
            description += (
                f" {self.unknown} light(s) drawn unknown: the format cannot write their rule."
            )
        root = {
            "id": "w-0001",
            "type": "absolute-canvas",
            "props": {**size, "backgroundColor": PAGE_BACKGROUND},
            "geometry": {"x": 0, "y": 0, **size},
            "children": self.nodes,
        }
        return {
            "schemaVersion": "1.1",
            "kind": "screen",
            "id": self.doc_id,
            "metadata": {
                "title": self.title,
                "source": {"type": "osprey-panels"},
                "size": size,
                "description": description,
            },
            "macros": [],
            "formulas": [],
            "root": root,
        }


def _need(name: str) -> str:
    assert name in _REQUEST_NAMES, name
    return _REQUEST_TAG + name


def needs(doc: Mapping[str, Any]) -> list[str]:
    """The :data:`FORMAT_REQUESTS` names a rendered document hits, each once, in order."""
    found: list[str] = []
    for node in _walk(doc["root"]):
        for tag in node.get("meta", {}).get("comment", "").split("; "):
            if tag.startswith(_REQUEST_TAG) and tag[len(_REQUEST_TAG) :] not in found:
                found.append(tag[len(_REQUEST_TAG) :])
    return found


def aggregates(doc: Mapping[str, Any]) -> int:
    """How many roll-up rows of a rendered document are drawn as an aggregate."""
    return sum(
        _AGGREGATE_ROW in n.get("meta", {}).get("comment", "").split("; ")
        for n in _walk(doc["root"])
    )


def unexpressed(doc: Mapping[str, Any]) -> int:
    """How many lights of a rendered document are drawn unknown because no rule could be written."""
    return sum(
        any(t.startswith(_UNKNOWN_TAG) for t in n.get("meta", {}).get("comment", "").split("; "))
        for n in _walk(doc["root"])
    )


def _walk(node: Mapping[str, Any]):
    yield node
    for child in node.get("children", ()):
        yield from _walk(child)


# --------------------------------------------------------------------------- the page


def _citation_text(citation: Citation | None) -> str:
    if citation is None:
        return ""
    parts = [citation.document, citation.version, citation.state, citation.locator]
    return " ".join(p for p in parts if p)


class _Page:
    """The drawing of one view: a cursor that moves down the page."""

    def __init__(
        self, view: View, graph: PanelGraph, screens: Callable[[Target], str], doc_id: str
    ) -> None:
        self.view = view
        self.graph = graph
        self.screens = screens
        self.s = _Screen(doc_id, view.title)
        self.y: float = MARGIN

    # -- primitives -----------------------------------------------------------

    def text(self, x, y, text: str, *, size=TEXT_SIZE, w=None, fg=PAGE_TEXT, tags=()) -> float:
        """A text label; returns its right edge."""
        w = text_width(text, size) if w is None else w
        props = {
            "text": text,
            "fontSize": size,
            "align": "left",
            "foregroundColor": fg,
            "backgroundColor": TRANSPARENT,
        }
        self.s.widget("text-label", x, y, w, ROW, props, tags=tags)
        return x + w

    def mark(self, x, y, glyph: str) -> float:
        return self.text(x, y, glyph, w=MARK_WIDTH)

    def line(self, points: Sequence[tuple[float, float]], colour, width=LINE_WIDTH) -> None:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x0, y0 = min(xs), min(ys)
        props = {
            "lineColor": colour,
            "lineWidth": width,
            "points": [{"x": px - x0, "y": py - y0} for px, py in points],
            "arrowStart": False,
            "arrowEnd": False,
        }
        self.s.widget("line", x0, y0, max(xs) - x0, max(ys) - y0, props)

    def box(self, x, y, w, h, *, line=BOX_BORDER, fill=FIELD_BACKGROUND, dashed=False) -> None:
        if dashed:
            points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            node = {
                "type": "polygon",
                "props": {
                    "lineColor": line,
                    "lineWidth": LINE_WIDTH,
                    "lineStyle": "dashed",
                    "fill": False,
                    "points": [{"x": px - x, "y": py - y} for px, py in points],
                },
                "geometry": {"x": x, "y": y, "width": w, "height": h},
            }
            self.s.add(node)
            return
        props = {"lineColor": line, "lineWidth": LINE_WIDTH, "fill": True, "fillColor": fill}
        self.s.widget("rectangle", x, y, w, h, props)

    def button(self, x, y, link: Link, *, label: str | None = None) -> float:
        """A link to another screen; returns the right edge of what it drew."""
        text = link.label if label is None else label
        w = min(text_width(text, TEXT_SIZE) + BUTTON_PAD, MAX_BUTTON_WIDTH)
        props = {"label": text, "file": self.screens(link.target), "fontSize": TEXT_SIZE}
        self.s.widget(
            "related-display-button", x, y, w, ROW, props, tags=(_need("clickable text"),)
        )
        right = x + w
        if link.basis == "inferred":
            right = self.mark(right + GAP, y, symbols.INFERRED_MARK)
        if link.standing == _CONTESTED:
            right = self.text(right + GAP, y, _CONTESTED, size=SMALL_SIZE)
        return right

    def ref(self, x, y, item: Link | Fact) -> float:
        if isinstance(item, Link):
            return self.button(x, y, item)
        return self.text(x, y, item.text or item.label)

    def light(
        self,
        x,
        y,
        lights: Sequence[Light],
        *,
        rule: Ladder | str | None = None,
        size: float = LIGHT_SIZE,
        tile: bool = False,
    ) -> float:
        """One light over *lights*; returns its right edge (after an inferred mark).

        A light whose rule is too long for the format is the aggregate look; a
        *tile* is one in an aggregate's mosaic and carries no inferred mark.
        """
        if rule is None:
            rule = ladder(self.graph, lights)
        aggregate = rule == "aggregate state"
        look = "aggregate" if aggregate else "unknown"
        shape, mark = symbols.light(x, y + (ROW - size) / 2, size, look)
        names = ", ".join(item.channel for item in lights[:TOOLTIP_CHANNELS])
        if len(lights) > TOOLTIP_CHANNELS:
            names = f"{len(lights)} channels"
        shape["props"][_TOOLTIP] = names
        tags = [_need("state light")]
        shape_rules: list[dict] = []
        mark_rules: list[dict] = []
        if aggregate:
            tags += [_need("aggregate state"), _AGGREGATE_TILE if tile else _AGGREGATE_ROW]
        elif isinstance(rule, str):
            tags += [_need(rule), _UNKNOWN_TAG + rule]
            self.s.unknown += 1
        else:
            if rule.unwritten:
                # the writable channels' arms are drawn; the unwritten ones add none
                tags.append(_need("label condition"))
            if rule.arms:
                self.rules(shape_rules, mark_rules, names, rule)
        self.s.add(shape, rules=shape_rules, tags=tags)
        mark_tags = [_need("hatched fill")] if mark_rules else []
        self.s.add(mark, rules=mark_rules, tags=mark_tags)
        right = x + size
        if not tile and any(item.inferred for item in lights):
            right = self.mark(right + GAP, y, symbols.INFERRED_MARK)
        return right

    def rules(self, shape_rules: list, mark_rules: list, names: str, rule: Ladder) -> None:
        """A light's rules: one per shape prop and one for its mark, all from *rule*."""
        looks = {state: symbols.look(state) for state in symbols.STATES}
        for prop in symbols.LIGHT_PROPS:
            values = {state: looks[state][prop] for state in symbols.STATES}
            shape_rules.append(self.s.rule(f"{names} {prop}", prop, rule, values))
        values = {state: looks[state][symbols.MARK_PROP] for state in symbols.STATES}
        mark_rules.append(self.s.rule(f"{names} mark", symbols.MARK_PROP, rule, values))

    def field(self, x, y, channel: str, *, writable: bool, w=VALUE_WIDTH) -> float:
        common = {
            "pv": channel,
            "fontSize": TEXT_SIZE,
            "foregroundColor": PAGE_TEXT,
            "backgroundColor": FIELD_BACKGROUND,
        }
        if writable:
            self.s.widget("pv-text-input", x, y, w, ROW, common)
        else:
            props = {**common, "showUnits": True, "alarmSensitive": True, "alarmBorder": True}
            props["align"] = "right"
            self.s.widget("pv-label", x, y, w, ROW, props)
        return x + w

    def next_row(self, height: float = ROW) -> None:
        self.y += height + ROW_GAP

    # -- the view -------------------------------------------------------------

    def draw(self) -> dict[str, Any]:
        x = MARGIN
        if self.view.crumbs:
            for i, crumb in enumerate(self.view.crumbs):
                if i:
                    x = self.text(x, self.y, "›", w=MARK_WIDTH) + GAP
                x = self.button(x, self.y, crumb) + GAP
            self.next_row()
        self.text(MARGIN, self.y, self.view.title, size=TITLE_SIZE, w=STRIP_WIDTH)
        self.next_row()
        for section in self.view.sections:
            self.section(section)
        return self.s.document(self.y + MARGIN)

    def section(self, section: Section) -> None:
        self.y += SECTION_GAP
        self.text(MARGIN, self.y, section.title, size=HEADING_SIZE, w=STRIP_WIDTH)
        self.y += ROW + RULE_GAP
        self.line([(MARGIN, self.y), (MARGIN + STRIP_WIDTH, self.y)], BOX_BORDER)
        self.y += RULE_GAP
        previous: object = None
        rows = left_out = 0
        for item in section.items:
            if isinstance(item, CheckRow):
                if not isinstance(previous, CheckRow):
                    self.check_header()
                    rows = 0
                rows += 1
                if rows > MAX_TABLE_ROWS:
                    left_out += 1
                    previous = item
                    continue
            self.item(item)
            previous = item
        if left_out:
            self.left_out(left_out)

    def left_out(self, count: int) -> None:
        """The note under a table cut at :data:`MAX_TABLE_ROWS`."""
        text = f"{count} more {_ROWS_NOT_DRAWN}"
        self.text(MARGIN, self.y, text, w=STRIP_WIDTH, tags=(_need("table"),))
        self.next_row()

    def item(self, item: object) -> None:
        draw = {
            ValueRow: self.value_row,
            Light: self.light_row,
            Link: self.link_row,
            Fact: self.fact_row,
            Placement: self.placement_row,
            Grid: self.grid,
            Strip: self.strip,
            Picture: self.picture,
            Elevation: self.elevation,
            Node: self.node,
            CheckRow: self.check_row,
            Totals: self.totals,
        }[type(item)]
        draw(item)

    # -- rows -----------------------------------------------------------------

    def value_row(self, row: ValueRow) -> None:
        x = self.text(MARGIN, self.y, row.label, w=LABEL_WIDTH)
        x = self.field(x + GAP, self.y, row.channel, writable=row.writable)
        if row.readback is not None:
            x = self.field(x + GAP, self.y, row.readback, writable=False)
        deviation = _tolerance_ladder(row)
        if deviation is not None:
            x = self.light(x + GAP, self.y, [Light(row.channel)], rule=deviation)
        if row.inferred:
            self.mark(x + GAP, self.y, symbols.INFERRED_MARK)
        self.next_row()

    def light_row(self, item: Light) -> None:
        x = self.light(MARGIN, self.y, [item])
        self.text(x + GAP, self.y, item.channel)
        self.next_row()

    def link_row(self, link: Link) -> None:
        self.button(MARGIN, self.y, link)
        self.next_row()

    def fact_row(self, fact: Fact) -> None:
        x = self.text(MARGIN, self.y, fact.label, w=LABEL_WIDTH)
        x = self.text(x + GAP, self.y, fact.text)
        if fact.basis == "inferred":
            x = self.mark(x + GAP, self.y, symbols.INFERRED_MARK)
        if fact.standing == _CONTESTED:
            x = self.text(x + GAP, self.y, _CONTESTED, size=SMALL_SIZE)
        note = " · ".join(t for t in (fact.caveat, _citation_text(fact.citation)) if t)
        if note:
            self.text(x + GAP, self.y, note, size=SMALL_SIZE)
        self.next_row()

    def placement_row(self, item: Placement) -> None:
        x = self.button(MARGIN, self.y, item.link)
        parts = []
        if item.unit is not None:
            parts.append(f"unit {item.unit}")
        if item.size_u is not None:
            parts.append(f"{item.size_u} U")
        if item.citation is not None:
            parts.append(_citation_text(item.citation))
        if parts:
            self.text(x + GAP, self.y, " · ".join(parts), size=SMALL_SIZE)
        self.next_row()

    def grid(self, grid: Grid) -> None:
        x0 = MARGIN + NAME_WIDTH
        tags = (_need("table"),)
        for j, column in enumerate(grid.columns):
            self.text(x0 + j * CELL_WIDTH, self.y, column, size=SMALL_SIZE, w=CELL_WIDTH, tags=tags)
        self.next_row()
        shown = zip(grid.rows[:MAX_TABLE_ROWS], grid.cells[:MAX_TABLE_ROWS], strict=True)
        for ref, cells in shown:
            self.ref(MARGIN, self.y, ref)
            for j, cell in enumerate(cells):
                x = x0 + j * CELL_WIDTH
                if isinstance(cell, Light):
                    self.light(x, self.y, [cell])
                elif isinstance(cell, ValueRow):
                    self.field(x, self.y, cell.channel, writable=cell.writable, w=CELL_WIDTH - GAP)
            self.next_row()
        if len(grid.rows) > MAX_TABLE_ROWS:
            self.left_out(len(grid.rows) - MAX_TABLE_ROWS)

    def check_header(self) -> None:
        tags = (_need("table"),)
        x = self.text(MARGIN, self.y, "channel", size=SMALL_SIZE, w=NAME_WIDTH, tags=tags)
        x = self.text(x, self.y, "thing", size=SMALL_SIZE, w=NAME_WIDTH, tags=tags)
        for title in ("on device", "typed", "rule", "source"):
            x = self.text(x, self.y, title, size=SMALL_SIZE, w=CELL_WIDTH, tags=tags)
        self.next_row()

    def check_row(self, row: CheckRow) -> None:
        x = self.text(MARGIN, self.y, row.channel, w=NAME_WIDTH)
        x = self.text(x, self.y, row.thing, w=NAME_WIDTH)
        for flag in (row.on_device, row.typed, row.has_rule, row.has_source):
            x = self.text(x, self.y, _YES if flag else _NO, w=CELL_WIDTH)
        self.next_row()

    def totals(self, totals: Totals) -> None:
        text = (
            f"{totals.channels} channels · {totals.on_device} on a device · "
            f"{totals.typed} typed · {totals.has_rule} with a rule · "
            f"{totals.has_source} with a source"
        )
        self.text(MARGIN, self.y, text, w=STRIP_WIDTH)
        self.next_row()

    def node(self, node: Node, depth: int = 0) -> None:
        """A roll-up node's row; the root's direct children follow it, one level only.

        A node whose rule is too long for the format is drawn as an aggregate:
        its mark and channel count, and on the root a mosaic of its children's
        lights (each child that is itself too long is an aggregate tile).
        """
        x = MARGIN + depth * INDENT
        rule: Ladder | str | None = None
        if node.no_rule:
            x = self.text(x, self.y, _NO_RULE, size=SMALL_SIZE)
        else:
            lights = [describe_light(self.graph, c) for c in node.channels]
            rule = ladder(self.graph, lights, node=True)
            x = self.light(x, self.y, lights, rule=rule)
        if node.link is not None:
            x = self.button(x + GAP, self.y, node.link, label=node.label)
        else:
            x = self.text(x + GAP, self.y, node.label)
        if rule == "aggregate state":
            self.text(x + GAP, self.y, _AGGREGATE_TEXT.format(len(node.channels)), size=SMALL_SIZE)
        self.next_row()
        if depth:
            return
        if rule == "aggregate state":
            self.mosaic(node.children, MARGIN + INDENT)
        for child in node.children:
            self.node(child, 1)

    def mosaic(self, children: Sequence[Node], x0: float) -> None:
        """One small light per child that reads a rule, in rows across the page."""
        x = x0
        for child in children:
            if child.no_rule:
                continue
            if x + MOSAIC_LIGHT > MARGIN + STRIP_WIDTH:
                self.y += MOSAIC_LIGHT + MOSAIC_GAP
                x = x0
            lights = [describe_light(self.graph, c) for c in child.channels]
            rule = ladder(self.graph, lights, node=True)
            top = self.y - (ROW - MOSAIC_LIGHT) / 2
            self.light(x, top, lights, rule=rule, size=MOSAIC_LIGHT, tile=True)
            x += MOSAIC_LIGHT + MOSAIC_GAP
        self.y += MOSAIC_LIGHT + ROW_GAP

    # -- the strip ------------------------------------------------------------

    def strip(self, item: Strip) -> None:
        if item.elements:
            widths = {kind: units * SYMBOL_UNIT for kind, units in symbols.WIDTH_UNITS.items()}
            for row in _strip_rows(item.elements, widths):
                self.strip_row(row, widths, item.extent_m)
        if item.tray:
            self.tray(item)

    def strip_row(
        self,
        row: Sequence[StripElement],
        widths: Mapping[str, float],
        extent: tuple[float, float] | None,
    ) -> None:
        drawn = [
            e if e.symbol in symbols.WIDTH_UNITS else replace(e, symbol="generic") for e in row
        ]
        placed, scale = strip.arrange_scaled(drawn, widths, STRIP_WIDTH)
        axis = self.y + STRIP_ABOVE + STRIP_LEVEL
        self.line([(MARGIN, axis), (MARGIN + STRIP_WIDTH, axis)], PAGE_TEXT, AXIS_WIDTH)
        del scale  # p.width is already scaled; the unit follows from it
        for p in placed:
            x = MARGIN + p.x
            y = axis
            if p.lane == "above":
                y = axis - STRIP_ABOVE - p.level * STRIP_LEVEL
                self.line([(x, y), (x, axis)], BOX_BORDER)
            self.symbol(p, x, y, p.width / symbols.WIDTH_UNITS[p.symbol])
            if p.link is not None:
                half = p.width / 2
                reach = STRIP_LEVEL if p.lane == "above" else STRIP_BELOW
                props = {
                    "lineColor": TRANSPARENT,
                    "lineWidth": 0,
                    "fill": False,
                    _HOVER_LABEL: p.thing,
                    "actions": [
                        {
                            "type": "open_display",
                            "file": self.screens(p.link.target),
                            "target": _NAVIGATION_TARGET,
                        }
                    ],
                }
                self.s.widget("rectangle", x - half, y - reach, p.width, 2 * reach, props)
        self.y = axis + STRIP_BELOW + LOCATOR_GAP
        if extent is not None:
            self.locator(row, extent)

    def symbol(self, element: strip.Placed, x: float, y: float, unit: float) -> None:
        """A device's symbol, in the live state of its rule channels.

        The symbol reads the ladder a roll-up node of the device would read: the
        worst of its channels. With no rule channel it is neutral and carries no
        mark; over the format's limit it is drawn in the aggregate look.
        """
        if not element.channels:
            for node in symbols.draw(element.symbol, x, y, unit=unit):
                node["props"][_TOOLTIP] = element.thing
                self.s.add(node)
            return
        lights = [describe_light(self.graph, c) for c in element.channels]
        rule = ladder(self.graph, lights, node=True)
        aggregate = rule == "aggregate state"
        drawn = symbols.draw(
            element.symbol, x, y, "aggregate" if aggregate else "unknown", unit=unit
        )
        tags = [_need("state light")]
        if aggregate:
            tags += [_need("aggregate state"), _AGGREGATE_SYMBOL]
        elif isinstance(rule, str):
            tags += [_need(rule), _UNKNOWN_TAG + rule]
            self.s.unknown += 1
        elif rule.unwritten:
            tags.append(_need("label condition"))
        live = rule if isinstance(rule, Ladder) and rule.arms else None
        *shapes, mark = drawn
        for node in shapes:
            node["props"][_TOOLTIP] = element.thing
            rules = [] if live is None else self.look_rules(element.thing, node["type"], live)
            self.s.add(node, rules=rules, tags=tags)
            tags = []  # the needs are recorded once, on the first shape
        mark["props"][_TOOLTIP] = element.thing
        rules = [] if live is None else self.look_rules(element.thing, mark["type"], live)
        self.s.add(mark, rules=rules, tags=[_need("hatched fill")] if rules else [])

    def look_rules(self, name: str, node_type: str, rule: Ladder) -> list[dict]:
        """The rules that give a symbol's widget of *node_type* the look of *rule*'s state."""
        looks = {state: symbols.look(state) for state in symbols.STATES}
        return [
            self.s.rule(f"{name} {prop}", prop, rule, {s: looks[s][prop] for s in symbols.STATES})
            for prop in symbols.styled(node_type)
        ]

    def locator(self, row: Sequence[StripElement], extent: tuple[float, float]) -> None:
        lo, hi = extent
        span = hi - lo
        bar = self.y
        self.box(MARGIN, bar, STRIP_WIDTH, LOCATOR_BAR)
        shifted = [replace(e, s_m=e.s_m - lo) for e in row]
        for _thing, x in strip.locator(shifted, span, STRIP_WIDTH):
            self.line(
                [(MARGIN + x, bar - TICK), (MARGIN + x, bar + LOCATOR_BAR)], PAGE_TEXT, AXIS_WIDTH
            )
        for metres, x in _ticks(lo, hi):
            top = bar + LOCATOR_BAR
            self.line([(MARGIN + x, top), (MARGIN + x, top + TICK)], PAGE_TEXT)
            label = f"{metres:g} m"
            self.text(
                MARGIN + x - TICK_LABEL_WIDTH / 2,
                top + TICK,
                label,
                size=SMALL_SIZE,
                w=TICK_LABEL_WIDTH,
            )
        self.y = bar + LOCATOR_BAR + TICK + ROW + ROW_GAP

    def tray(self, item: Strip) -> None:
        top = self.y
        self.y += TRAY_PAD
        self.text(MARGIN + TRAY_PAD, self.y, _NOT_PLACED, size=SMALL_SIZE)
        self.next_row()
        for entry in item.tray:
            x = MARGIN + TRAY_PAD
            if entry.link is not None:
                x = self.button(x, self.y, entry.link, label=entry.thing)
            else:
                x = self.text(x, self.y, entry.thing)
            self.text(x + GAP, self.y, entry.reason, size=SMALL_SIZE)
            self.next_row()
        self.box(MARGIN, top, STRIP_WIDTH, self.y - top, dashed=True)
        self.y += TRAY_PAD

    # -- picture and elevation ------------------------------------------------

    def picture(self, picture: Picture) -> None:
        top = self.y
        self.box(MARGIN, top, PICTURE_SIZE, PICTURE_SIZE, fill=PAGE_BACKGROUND)
        self.text(MARGIN + PICTURE_SIZE + GAP, top, f"{picture.view} view", size=SMALL_SIZE)
        coords = [_picture_xy(picture.view, part) for part in picture.parts]
        reach = max([max(abs(a), abs(b)) for a, b in coords] or [0]) or 1
        scale = (PICTURE_SIZE / 2 - PICTURE_PAD) / reach
        cx, cy = MARGIN + PICTURE_SIZE / 2, top + PICTURE_SIZE / 2
        for part, (a, b) in zip(picture.parts, coords, strict=True):
            px, py = cx + a * scale, cy - b * scale
            channel = part.channel  # stated by the pictures row; no table states one yet
            if channel is not None and self.graph.rule(channel):
                self.light(px - LIGHT_SIZE / 2, py - ROW / 2, [describe_light(self.graph, channel)])
                right = px + LIGHT_SIZE / 2
            else:
                half = PART_MARKER / 2
                props = {"lineColor": PAGE_TEXT, "lineWidth": LINE_WIDTH, "fill": False}
                props[_TOOLTIP] = part.part
                self.s.widget("ellipse", px - half, py - half, PART_MARKER, PART_MARKER, props)
                right = px + half
            self.text(right + GAP / 2, py - ROW / 2, part.part, size=SMALL_SIZE, w=PART_LABEL_WIDTH)
        self.y = top + PICTURE_SIZE + ROW_GAP

    def elevation(self, elevation: Elevation) -> None:
        top = self.y
        y = top
        for unit in elevation.units:
            y = self.rack_unit(MARGIN, y, unit)
        side = MARGIN + RACK_WIDTH + UNPOSITIONED_GAP
        self.text(side, top, _NO_UNIT, size=SMALL_SIZE)
        y_side = top + ROW + ROW_GAP
        for unit in elevation.unpositioned:
            x = self.ref(side, y_side, unit.link)
            for item in unit.lights:
                x = self.light(x + GAP, y_side, [item])
            y_side += ROW + ROW_GAP
        self.y = max(y, y_side) + ROW_GAP

    def rack_unit(self, x0: float, y: float, unit: RackUnit) -> float:
        h = (unit.size_u or 1) * RACK_U
        self.box(x0, y, RACK_WIDTH, h, dashed=unit.conflicting)
        label = "" if unit.position is None else f"U{unit.position}"
        x = self.text(x0 + GAP, y, label, size=SMALL_SIZE, w=RACK_LABEL_WIDTH)
        if unit.conflicting:
            x = self.mark(x, y, symbols.CONFLICT_MARK)
        x = self.ref(x + GAP, y, unit.link)
        for item in unit.lights:
            x = self.light(x + GAP, y, [item])
        if unit.standing == _CONTESTED:
            self.text(x + GAP, y, _CONTESTED, size=SMALL_SIZE)
        return y + max(h, ROW)


def _picture_xy(view: str, part) -> tuple[float, float]:
    """A part's drawn coordinates: x, y on an end view; z, y on a side view."""
    return (part.z_mm, part.y_mm) if view == "side" else (part.x_mm, part.y_mm)


def _ticks(lo: float, hi: float) -> list[tuple[float, float]]:
    """The metre ticks of a locator of ``lo`` to ``hi`` metres: (metres, x)."""
    span = hi - lo
    if span <= 0:
        return [(lo, STRIP_WIDTH / 2)]
    exponent = math.floor(math.log10(span / MAX_TICKS)) if span > MAX_TICKS else 0
    step = next(
        base * TICK_BASE**e
        for e in range(exponent, exponent + 2)
        for base in TICK_STEPS
        if span / (base * TICK_BASE**e) <= MAX_TICKS
    )
    first = math.ceil(lo / step)
    out = []
    k = first
    while k * step <= hi:
        metres = k * step
        out.append((metres, (metres - lo) / span * STRIP_WIDTH))
        k += 1
    return out


def _strip_rows(
    elements: Sequence[StripElement], widths: Mapping[str, float]
) -> list[list[StripElement]]:
    """The strip in rows, each as long as fits at no less than :data:`MIN_STRIP_SCALE`."""

    def fits(chunk: Sequence[StripElement]) -> bool:
        return strip.arrange_scaled(chunk, widths, STRIP_WIDTH)[1] >= MIN_STRIP_SCALE

    rows: list[list[StripElement]] = []
    start = 0
    while start < len(elements):
        lo, hi = start + 1, len(elements)  # one element always makes a row
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if fits(elements[start:mid]):
                lo = mid
            else:
                hi = mid - 1
        rows.append(list(elements[start:lo]))
        start = lo
    return rows


def render(
    view: View,
    *,
    graph: PanelGraph,
    screens: Callable[[Target], str],
    doc_id: str | None = None,
) -> dict[str, Any]:
    """The view as a Screen-IR 1.1 document.

    Args:
        view: What the screen shows.
        graph: The panel graph, for the health rows the lights' rules are written from.
        screens: The screen file a link to a target opens, relative to this
            screen's directory.
        doc_id: The document's id; :func:`screen_id` of the view's target by default.

    Returns:
        The document, as JSON-ready data. Equal inputs give equal documents.
    """
    return _Page(view, graph, screens, doc_id or screen_id(view.id)).draw()
