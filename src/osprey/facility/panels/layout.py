"""Layout: a view description drawn as a Canopy Screen-IR 1.1 document.

This is the only module of the panels with geometry. :func:`render` walks a
:class:`~osprey.facility.panels.view.View` top to bottom, one section after
another, and draws every item with the pinned viewer's widgets. Colours are
the theme roles of :mod:`~osprey.facility.panels.symbols`.

States are the viewer's to compute. A light's rule over its channels, written from
their health rows (see :func:`ladder`), is a list of arms, each a condition that
holds when the light is in that arm's state; each prop of the light has a rule
(:func:`light_rule`) over the light's channels whose conditions are those arms,
first that holds wins, and set the prop to the state's look. The viewer
shows the state the rows give the live value, and ``unknown`` before a value
arrives. Nothing here reads a value.

Where a view needs a widget the format lacks, the nearest one is drawn and the
node says so in its ``meta.comment`` (``format-request: <name>``); the needs
are :data:`FORMAT_REQUESTS`, and :func:`needs` lists those a document hits. A
light whose rule the format cannot write is drawn ``unknown`` and counted
(:func:`unexpressed`).

Every size, spacing and font size is in the constants block below.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from osprey.facility.panels import health, strip, symbols
from osprey.facility.panels.describe import NO_POSITION, RELATION_TITLES
from osprey.facility.panels.describe import light as describe_light
from osprey.facility.panels.health import State
from osprey.facility.panels.model import Citation, PanelGraph
from osprey.facility.panels.symbols import (
    BOX_BORDER,
    FIELD_BACKGROUND,
    MUTED_TEXT,
    PAGE_BACKGROUND,
    PAGE_TEXT,
    PANEL_BACKGROUND,
    TRANSPARENT,
)
from osprey.facility.panels.view import (
    Band,
    CalibrationRow,
    Calibrations,
    CheckRow,
    Elevation,
    Fact,
    Grid,
    GridCell,
    GridRow,
    Group,
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
    Tile,
    Totals,
    ValueRow,
    View,
)

__all__ = [
    "FORMAT_REQUESTS",
    "MIN_STRIP_SCALE",
    "OUTLINE_KIND",
    "compiled_rules",
    "light_rule",
    "metric_formula",
    "WIDGETS",
    "Ladder",
    "ladder",
    "aggregates",
    "needs",
    "render",
    "picture_labels",
    "engine_expression",
    "engine_length",
    "stem_levels",
    "strip_labels",
    "text_width",
    "unexpressed",
]

# --- constants: every size, spacing and font size of a panel screen -----------
PAGE_WIDTH = 1200
MARGIN = 32  # the left edge of every section's content
# type: body 13, small 11 and nothing smaller anywhere (a text-label below
# SMALL_SIZE is raised to it), headings small and muted, the page title 20
TITLE_SIZE = 20
HEADING_SIZE = 11
TEXT_SIZE = 13
SMALL_SIZE = 11
CHAR_WIDTH = 0.6  # estimated width of one character, in parts of the font size
CAPS_WIDTH = 0.72  # the same for capitals: a title in capitals is wider
TEXT_PAD = 10  # added to an estimated text width: the label's own padding and a margin
HEADER_HEIGHT = 40  # the bar at the top of every screen
HEADER_PAD = 8  # the bar's widgets from its top
PANEL_PAD = 16  # a section's frame around its content: the frame stands in the margin
PANEL_GAP = 12  # between two section frames
ROW = 24  # height of one row of text, a field or a button
ROW_GAP = 4
SECTION_GAP = 16
STAMP_DIGITS = 8  # of the model's content hash, in the header
GAP = 8  # between two widgets on one row
INDENT = 20  # one level of a roll-up tree
LABEL_WIDTH = 200
VALUE_WIDTH = 130
UNIT_WIDTH = 44  # a value's unit, beside it
UNIT_GAP = 2
BUTTON_PAD = 16
MAX_BUTTON_WIDTH = 420
LIGHT_SIZE = 14
MARK_WIDTH = 16
LINE_WIDTH = 1
AXIS_WIDTH = 2
CELL_WIDTH = 110  # a checkout's column
NAME_WIDTH = 260  # a checkout's first column
# a card's calibration table: each column's left edge from the content's, and the
# basis chip's width
CAL_COLUMNS = (0, 250, 470, 650, 745, 895)
CAL_CHIP_WIDTH = 170
CHIP_INSET = 2  # a basis chip stands this far inside its row, above and below
SET_CELL_WIDTH = 112  # one member of a set overview at the least: its value or name, a light
SET_CELL_MAX = 220  # the widest a cell grows to when the columns are few
SET_ROW_HEAD = 64  # a set overview's row heads: the place, short
SET_BLOCK_GAP = 24  # between two blocks of a set overview's columns
SET_MEMBERS = 200  # a set overview draws blocks until this many cells are drawn
STRIP_WIDTH = PAGE_WIDTH - 2 * MARGIN
MIN_STRIP_SCALE = 0.6  # below this a strip breaks into rows: a bend stays >= 45 px tall
SYMBOL_UNIT = 20  # one shape unit of a symbol, in px, at scale 1: a bend is 75 px tall
STRIP_AXIS = 38  # the axis's room each side: half the tallest symbol, a bend
BADGE = 20  # a device off the axis: its circle
STEM_LOW = 52  # from the axis to the centre of a badge, first level
STEM_HIGH = 84  # second level, for a badge that would touch its neighbour
BADGE_GAP = 2  # between two badges of one level
STRIP_LABEL_TOP = STRIP_AXIS + 18  # a symbol's name, below the axis, with air between
STRIP_LABEL_STEP = 18  # the second line of names, further down
STRIP_LABEL_SIZE = SMALL_SIZE
STRIP_LABEL_WIDTH = 96  # the most a symbol's name may take along the strip
STRIP_LABEL_GAP = 6  # between two names on one line
STRIP_BELOW = STRIP_LABEL_TOP + 2 * STRIP_LABEL_STEP  # room under the axis
LEADER_GAP = 8  # between the names and the leaders
LEADER_DROP = 40  # the leaders' run down to the locator
LOCATOR_BAR = 1  # the to-scale bar: a line
LOCATOR_MARK = 8  # how far a device's mark reaches above the bar
LOCATOR_BIG = 12  # the mark of a magnet
TICK = 5
TICK_LABEL_WIDTH = 48
MAX_TICKS = 16
TICK_STEPS = (1, 2, 5)
TICK_BASE = 10
LEGEND_SYMBOL = 12  # one shape unit of a symbol in the legend: 60 % of the strip's
LEGEND_GAP = 20  # between two entries of the legend
TRAY_PAD = 10
CHIP_HEIGHT = 26
CHIP_GAP = 8
CHIP_LIGHT = 12
LEGEND_ROW = 52
BIG_SIZE = 22  # a key number
TILE_PAD = 10
MIN_TILE_WIDTH = 150  # the band takes the fewest rows whose tiles are this wide
LINE_HEIGHT = 1.5  # a line of text, in parts of its size: the viewer shrinks what overflows
NUMBER_HEIGHT = 36  # a key number's box: one line at the big size, and its border
NUMBER_PAD = 22  # a value display's own padding and border, across
SMALL_LINE = 16  # one line of small text
TILE_LINES = 2  # the most lines a tile's title or caption takes
VALUE_DIGITS = 7  # room for a value's number: "2.6e-10"
COUNT_DIGITS = 3  # the least room for a count's number: what a dash takes
RING_PAD = 6  # a state ring's distance from its symbol
TAG_HEIGHT = 16  # the word above a ring
VALUE_STEP = 16  # one live value under a symbol's name
STRIP_VALUE_WIDTH = 44  # every value under a symbol: "-412.0" at the small size
BADGE_VALUE_WIDTH = 64  # a value beside a badge
TAG_WIDTH = 64  # room for the longest word a ring carries, at the small size
CARD_LEFT = 600  # a device card's left column, its signals, with the gap to the right one
COLUMN_GAP = 24
CARD_LABEL_WIDTH = 240  # a signal's name with its unit, in the card's table
CARD_ROWS = 40  # signal rows a card draws; the rest are counted by how they are drawn
CHAIN_BOX = 150  # one thing of a card's fed-by chain
CHAIN_HEIGHT = 44
CHAIN_ARROW = 40  # the run of an arrow between two boxes of the chain
ROLLUP_LEFT = 700  # a roll-up's tiles, with the gap to its by-system rows
ROLLUP_TILE_WIDTH = 150  # the least width of a child's tile
ROLLUP_TILE_HEIGHT = 64
RING_OUTER = 185  # a ring roll-up's arcs: outer and inner radius
RING_INNER = 120
RING_GAP = 2  # degrees left free at each end of an arc
TURN = 360  # degrees round the ring
RING_PAD = 16  # above and below the ring
RING_LABEL_WIDTH = 60  # an arc's name and mark, centred on the arc
RING_MARK = 12  # the font size of an arc's mark, under its name
SUMMARY_ROW = 36  # one row of a by-system summary
SUMMARY_COLUMNS = 2
PICTURE_SIZE = 260
PICTURE_PAD = 24
PART_MARKER = 8
PART_LABEL_WIDTH = 140
RACK_U = 16  # one rack unit: a 48 U rack is 768 px tall
RACK_WIDTH = 420  # a chassis bar, the rack's inner width
RACK_LABEL_WIDTH = 40  # the unit numbers, left of the rack
RACK_PAD = 6  # the rack's frame round its chassis
RACK_SOURCES = 8  # documents a rack names; the rest are counted
RACK_COLUMN = RACK_LABEL_WIDTH + RACK_WIDTH + 2 * RACK_PAD + 2 * COLUMN_GAP  # the facts' x
# The longest expression the formula engine takes. The viewer compiles each rule
# of a light into one expression, with the light's screen formula inlined once
# per condition (see engine_expression); the engine refuses one longer than
# ``max_expression_len`` (canopy/formula_engine/config.py at 166989aa: 2048).
RULE_EXPRESSION_LIMIT = 2048
TOOLTIP_CHANNELS = 8  # a light's tooltip names this many channels, else counts them
MOSAIC_LIGHT = 10  # one child's light in an aggregate's mosaic
MOSAIC_GAP = 3
MAX_TABLE_ROWS = 400  # rows of one table drawn; the rest are counted under it
WAVEFORM_WIDTH = 520
WAVEFORM_HEIGHT = 140
COMMAND_LABEL = "send"  # what a command's button reads
COMMAND_VALUE = 1  # what pressing a command's button writes
# What one screen may bind. The formula engine allows 1,000 subscriptions per
# client (``max_subscriptions_per_client``, canopy/formula_engine/config.py at
# 166989aa), and the viewer subscribes once per rule with a condition; a
# roll-up binding some 28,000 channels on one screen made the Channel Access
# backend report its virtual circuit unresponsive (seen live, 2026-11-17).
SCREEN_RULE_BUDGET = 800
SCREEN_CHANNEL_BUDGET = 4000
INTEGER_RUN = 3  # the fewest successive integer values a rule writes as one range test
BIT_TESTS = 6  # the most bits a channel's rule writes as a table of their values
# --- end constants -------------------------------------------------------------

#: The widget a value is drawn as, by its ``drawn_as`` in ``record_types.csv``:
#: (on a device card's own signals and settings, anywhere else). Writable widgets
#: are in the first column only, and only for a writable channel. A channel with
#: no row (or ``untyped``) is drawn as a ``setpoint`` when writable.
WIDGETS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "value": ("pv-label", "pv-label"),  # an analogue value, with its unit
        "setpoint": ("pv-text-input", "pv-label"),
        "state": ("pv-label", "pv-label"),  # an enum: the label's text is its state
        "command": ("pv-button", "pv-label"),
        "waveform": ("waveform-plot", "pv-label"),
        "text": ("pv-label", "pv-label"),
        "untyped": ("pv-label", "pv-label"),
    }
)
_WRITABLE_WIDGETS = frozenset({"pv-text-input", "pv-button"})

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
        "a grid of a set's members and the checkout are tables; there is no table "
        "container that scrolls or pages, so every cell is placed on its own and a long "
        "table is cut, its remaining rows counted",
    ),
    (
        "clickable text",
        "neutral navigation: a link in a list or a tree reads as text that opens a screen; "
        "the format's navigation buttons take no colour props (related-display-button none, "
        "regular-button only flat), so a link is a text label under an outlined box whose "
        "action opens the screen: two widgets where one would do",
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
        "today it is an aggregate mark, drawn in parts where the screen's budget allows",
    ),
    (
        "server-side aggregate",
        "server-side aggregate over more channels than a screen can bind: a roll-up level "
        "whose summary needs more rules or channels than one screen's budget "
        "(SCREEN_RULE_BUDGET, SCREEN_CHANNEL_BUDGET); today the level says so in a line",
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
_SUMMARY_TEXT = "{} channels: a live summary of this level needs a server-side aggregate"
_PART = "part {} of {}"
_ID_PARTS = re.compile(r"[:/]")
_PARTS = "parts"
#: The kind of a pictures row that is a point of the class's outline, not a part.
OUTLINE_KIND = "outline"
_ELLIPSIS = "\u2026"
_FORMULA_PREFIX = "formula://"
_TOKEN = re.compile(r"\{(\d+)\}")
_NAVIGATION_TARGET = "tab"
_HOVER_LABEL = "hoverLabel"
_TOOLTIP = "tooltip"
_YES, _NO = "yes", "no"
_NO_RULE = "no rule"
_NOT_PLACED = "not placed"
_NO_UNIT = "no unit"
_ROWS_NOT_DRAWN = "rows not drawn: the screen holds a table of at most this many rows"
_CONTESTED = "contested"
_STRIP_TITLE = "IN BEAM ORDER → · schematic, not to scale"
_LOOKS_LEGEND = (
    "fill = component type · red ring = fault · amber ring = warning · dashed = no health rule"
)
_LOCATOR_CAPTION = "to scale · position along the beam (m)"
_NO_RULE_YET = "no health rule yet"
_TRAY_TITLE = "in our lists, not placed"
_NO_DATA = "no data yet"
_NEEDS_SERVER = "needs a server-side aggregate"
_MANY_VALUES = "{} channels state it: the row names no one"
_DASH = "—"
_COUNTS = frozenset({"count_true", "count_false", "count_in_tolerance"})
_VALUES = "values: "
_DROPPED = "{} values not shown: the screen's budget is spent"
_MORE_CHANNELS = "{} more {} channels not drawn"
_MORE_ROWS = "{} more not drawn"
_NOT_AT_A_UNIT = "not at a unit"
_BY_CLASS = "by class"
_SOURCES = "sources"
_RACK_COUNTS = "{} at a stated unit · {} not at a unit"
_UNITS_FROM = "{} · {} devices"
_SET_READBACK = "each cell: readback of {}"
_CAL_HEADS = ("parameter", "value in force", "basis", "date", "source", "history")
_NO_VALUE_YET = "no value yet"
_SET_MORE = "{} more members: open by place"
_SET_NAMES = "each cell: the member's short name and its light"
_NO_CLASS = "no class stated"
_BY_SYSTEM_TITLE = "by system"
_TILE_OPEN = "{} channels · open"
_SIGNAL_HEADS = ("signal", "setpoint", "readback")
_CHAIN_CAPTION = "power flows right to left"
_THIS_DEVICE = "this device"
_FED_BY = "fed by"
_ALSO = "also"
_CARD_LEFT = frozenset({"signals", "settings", "health", "sources"})
_CARD_RIGHT = frozenset({"where"})
_READS = "{} rule channel{}"
_MAGNETS = frozenset(
    {"bend", "quadrupole", "quadrupole_f", "quadrupole_d", "sextupole", "skew_quadrupole"}
)
#: The word for each symbol in a strip's legend.
SYMBOL_WORDS: Mapping[str, str] = MappingProxyType(
    {
        "bend": "bend",
        "quadrupole_f": "focusing quadrupole",
        "quadrupole_d": "defocusing quadrupole",
        "quadrupole": "quadrupole",
        "sextupole": "sextupole",
        "skew_quadrupole": "skew quadrupole",
        "corrector": "corrector",
        "bpm": "BPM",
        "neg_pump": "NEG pump",
        "ion_pump": "ion pump",
        "pump": "pump",
        "gauge": "gauge",
        "gate_valve": "gate valve",
        "bellows": "bellows",
        "kicker": "kicker",
        "scraper": "scraper",
        "generic": "other device",
    }
)
_STATE_ORDER: tuple[State, ...] = ("bad", "warn", "disagree")


def text_width(text: str, size: float) -> float:
    """A box wide enough for *text* at *size* px: the one text-width estimate."""
    return math.ceil(len(text) * size * CHAR_WIDTH) + TEXT_PAD


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


#: For each arm of a ladder, the states an earlier arm has already ruled out for every
#: channel: where a channel is in one of them, the arm may hold or not.
_RULED_OUT: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "bad": frozenset(),
        "warn": frozenset({"bad"}),
        "disagree": frozenset({"bad", "warn"}),
        "normal": frozenset({"bad", "warn", "disagree"}),
    }
)


def _shown(state: State, item: Light) -> State:
    """The state a light draws: a flagged light is never normal (disagree, else unknown)."""
    if state != "normal":
        return state
    if item.disagree:
        return "disagree"
    return "unknown" if item.unsupported else state


def _interval(points: Sequence[float], first: int, last: int, token: str) -> str:
    """The test that a value is in segments *first* to *last* (see :func:`_channel_arms`).

    Segment ``2j + 1`` is the point ``points[j]``; segment ``2j`` is the open
    interval below it, and segment ``2 * len(points)`` the one above the last.
    """
    if first == last and first % 2:
        return f"{token} == {_number(points[first // 2])}"
    low: tuple[float, str] | None = None
    if first % 2:
        low = (points[first // 2], "<=")
    elif first:
        low = (points[first // 2 - 1], "<")
    high: tuple[float, str] | None = None
    if last % 2:
        high = (points[last // 2], "<=")
    elif last < 2 * len(points):
        high = (points[last // 2], "<")
    if low and high:
        return f"{_number(low[0])} {low[1]} {token} {high[1]} {_number(high[0])}"
    if low:
        return f"{token} {'>=' if low[1] == '<=' else '>'} {_number(low[0])}"
    if high:
        return f"{token} {high[1]} {_number(high[0])}"
    return f"{token} == {token}"  # every number; NaN is no number


def _cover(points: Sequence[float], marks: Sequence[str], token: str) -> str | None:
    """The shortest test true on every ``T`` segment and false on every ``F`` one.

    A ``D`` segment may go either way. Each maximal run of segments with no
    ``F`` that holds a ``T`` is one interval; a run of at least
    :data:`INTEGER_RUN` single successive integers is one integer range test.
    """
    runs: list[tuple[int, int]] = []
    start = None
    for k, mark in enumerate([*marks, "F"]):
        if mark != "F" and start is None:
            start = k
        elif mark == "F" and start is not None:
            if "T" in marks[start:k]:
                runs.append((start, k - 1))
            start = None
    terms: list[str] = []
    i = 0
    while i < len(runs):
        j = i
        while j + 1 < len(runs) and _successive(points, runs[j], runs[j + 1]):
            j += 1
        if j - i + 1 >= INTEGER_RUN:
            low, high = points[runs[i][0] // 2], points[runs[j][0] // 2]
            terms.append(
                f"{token} == {token} // 1 and {_number(low)} <= {token} <= {_number(high)}"
            )
        else:
            terms += [_interval(points, a, b, token) for a, b in runs[i : j + 1]]
        i = j + 1
    return " or ".join(terms) if terms else None


def _successive(points: Sequence[float], a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Whether runs *a* and *b* are single points at successive integers."""
    if a[0] != a[1] or b[0] != b[1] or not a[0] % 2 or not b[0] % 2:
        return False
    p, q = points[a[0] // 2], points[b[0] // 2]
    return p == int(p) and q == p + 1


def _bit_test(token: str, bit: int, wanted: int) -> str:
    shifted = token if bit == 0 else f"{token} // {2**bit}"
    return f"{shifted} % 2 == {wanted}"


def _bits_of(value: int, bits: Sequence[int]) -> tuple[int, ...]:
    return tuple((value >> bit) & 1 for bit in bits)


def _segment_values(
    points: Sequence[float], k: int, bits: Sequence[int]
) -> tuple[float | None, dict[tuple[int, ...], int]]:
    """A value of segment *k* that is no integer (``None`` when it has none) and, for
    each combination of *bits* that an integer in it has, one such integer."""
    period = 2 ** (max(bits) + 1)
    if k % 2:
        point = points[k // 2]
        if point == int(point):
            return None, {_bits_of(int(point), bits): int(point)}
        return point, {}
    low = points[k // 2 - 1] if k else None
    high = points[k // 2] if k < 2 * len(points) else None
    if low is None and high is None:
        other, ints = 1 / 2, range(period)
    elif low is None:
        other = math.floor(high) - 1 / 2  # type: ignore[arg-type]
        last = math.ceil(high) - 1  # type: ignore[arg-type]
        ints = range(last - period + 1, last + 1)
    elif high is None:
        other = math.floor(low) + 1 + 1 / 2
        first = math.floor(low) + 1
        ints = range(first, first + period)
    else:
        other = (low + high) / 2
        while other == int(other):  # halve toward the low end until it is no integer
            other = (low + other) / 2
        first, last = math.floor(low) + 1, math.ceil(high) - 1
        ints = range(first, min(last, first + period - 1) + 1)
    found: dict[tuple[int, ...], int] = {}
    for value in ints:
        found.setdefault(_bits_of(value, bits), value)
    return other, found


def _cubes(
    bits: Sequence[int], true: set[tuple[int, ...]], false: set[tuple[int, ...]], token: str
) -> list[str]:
    """The fewest bit tests, ORed, that hold for every *true* combination and no *false*
    one: a greedy cover by cubes; ``""`` is the cube of every combination."""
    cubes = sorted(itertools.product((None, 0, 1), repeat=len(bits)), key=_fixed)
    allowed = [q for q in cubes if not any(_in_cube(c, q) for c in false)]
    chosen: list[tuple[int | None, ...]] = []
    left = set(true)
    while left:  # the cube that covers most of what is left, fewest bits on a tie
        best = max(allowed, key=lambda q: (sum(_in_cube(c, q) for c in left), -_fixed(q)))
        chosen.append(best)
        left = {c for c in left if not _in_cube(c, best)}
    return [
        " and ".join(
            _bit_test(token, bit, w) for bit, w in zip(bits, q, strict=True) if w is not None
        )
        for q in chosen
    ]


def _both(first: str, second: str) -> str:
    """*first* and *second*; an ``or`` in either is put in parentheses."""
    return " and ".join(f"({t})" if " or " in t else t for t in (first, second) if t)


def _bit_arms(
    graph: PanelGraph, item: Light, token: str, bits: Sequence[int], points: Sequence[float]
) -> dict[str, str | None]:
    """The conditions of a channel with bit tests among its rows.

    A bit test matches integers only. On a segment of the number line (see
    :func:`_channel_arms`) every value that is no integer has one state, and
    an integer's state follows from the segment and its *bits*: each is read
    from :func:`health.channel_state` at a value that has it. A condition is
    the test that the value is no integer and in the segments where that
    state holds, or an integer, in a run of segments, with the fewest bit
    tests that say it; a combination no integer of a segment has may go
    either way.
    """
    combos = list(itertools.product((0, 1), repeat=len(bits)))
    other: list[State | None] = []
    ints: list[dict[tuple[int, ...], State]] = []
    for k in range(2 * len(points) + 1):
        value, found = _segment_values(points, k, bits)
        state = health.channel_state(graph, item.channel, value) if value is not None else None
        other.append(_shown(state, item) if state is not None else None)
        ints.append(
            {
                c: _shown(health.channel_state(graph, item.channel, v), item)
                for c, v in found.items()
            }
        )
    guard = f"{token} == {token} // 1"
    arms: dict[str, str | None] = {}
    for state, ruled_out in _RULED_OUT.items():

        def mark(found: State | None, state=state, ruled_out=ruled_out) -> str:
            if found is None or found in ruled_out:
                return "D"
            return "T" if found == state else "F"

        parts: list[str] = []
        apart = _cover(points, [mark(s) for s in other], token)
        if apart is not None:
            parts.append(_both(f"{token} != {token} // 1", apart))
        # runs of segments whose integers want no conflicting marks, merged left to right
        runs: list[list] = []
        for k, states in enumerate(ints):
            if not states:
                continue
            pattern = {c: mark(states.get(c)) for c in combos}
            last = runs[-1] if runs else None
            if last is not None and not any({last[2][c], pattern[c]} == {"T", "F"} for c in combos):
                last[1] = k
                for c in combos:
                    if pattern[c] != "D":
                        last[2][c] = pattern[c]
            else:
                runs.append([k, k, pattern])
        terms: list[str] = []
        for i, (_first, _last, pattern) in enumerate(runs):
            true = {c for c in combos if pattern[c] == "T"}
            if not true:
                continue
            low = 0 if i == 0 else runs[i - 1][1] + 1
            high = 2 * len(points) if i == len(runs) - 1 else runs[i + 1][0] - 1
            where = _interval(points, low, high, token)
            where = "" if where == f"{token} == {token}" else where
            false = {c for c in combos if pattern[c] == "F"}
            tests = _cubes(bits, true, false, token)
            which = "" if "" in tests else " or ".join(tests)
            terms.append(_both(where, which))
        if terms:
            parts.append(guard if "" in terms else _both(guard, " or ".join(terms)))
        arms[state] = " or ".join(parts) if parts else None
    return arms


def _fixed(cube: tuple[int | None, ...]) -> int:
    return sum(w is not None for w in cube)


def _in_cube(combo: tuple[int, ...], cube: tuple[int | None, ...]) -> bool:
    return all(w is None or w == c for c, w in zip(combo, cube, strict=True))


def _channel_arms(graph: PanelGraph, item: Light, token: str) -> dict[str, str | None] | str:
    """The shortest condition under which one channel is in each state, or why none can be.

    The numbers the channel's rows name cut the number line into segments
    (open intervals and the points between them); the channel's state, as
    the light draws it, is constant on each, and is read from
    :func:`health.channel_state` at one value of each, so the conditions say
    exactly what the engine says. The condition of an arm may hold or not
    where an earlier arm has already ruled the channel out
    (:data:`_RULED_OUT`). A channel with bit tests is :func:`_bit_arms`'s;
    one with more than :data:`BIT_TESTS` bits is written row by row.
    """
    forms = [f for row in graph.rule(item.channel) if (f := health.form(row.value)) is not None]
    if any(f.kind == "literal" and f.number is None for f in forms):
        return "label condition"
    bits = sorted({f.bit for f in forms if f.kind == "bit"})  # type: ignore[type-var]
    if len(bits) > BIT_TESTS:
        return _row_arms(graph, item, token)
    points = sorted({f.number for f in forms if f.number is not None})
    if bits:
        return _bit_arms(graph, item, token, bits, points)  # type: ignore[arg-type]
    samples: list[float] = []
    for j, point in enumerate(points):
        samples.append(point - 1 if j == 0 else (points[j - 1] + point) / 2)
        samples.append(point)
    samples.append(points[-1] + 1 if points else 0)
    states = [_shown(health.channel_state(graph, item.channel, v), item) for v in samples]
    arms: dict[str, str | None] = {}
    for state, ruled_out in _RULED_OUT.items():
        marks = ["T" if s == state else "D" if s in ruled_out else "F" for s in states]
        arms[state] = _cover(points, marks, token)
    return arms


def _row_arms(graph: PanelGraph, item: Light, token: str) -> dict[str, str | None] | str:
    """The condition under which one channel is in each state, written row by row.

    For a channel with a bit test, which no interval of values can say.

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

    mixed = [c for c, (_, states) in conds if health.disagree(states)]
    pairs = [
        f"({a}) and ({b})"
        for i, (a, (_, states_a)) in enumerate(conds)
        for j, (b, (_, states_b)) in enumerate(conds)
        if i < j
        and a not in mixed
        and b not in mixed
        and health.disagree(states_a | states_b)
        and meets(i, j)
    ]
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
            return too_long  # every condition is written at least once: a cheap early exit
    if unwritten and len(unwritten) == len(lights):
        return reason
    out: list[tuple[str, State]] = []
    for state in _STATE_ORDER:
        found = [c for arms in per_channel if (c := arms[state])]
        if found:
            out.append((" or ".join(found), state))
    normals = [arms["normal"] for arms in per_channel]
    if normals and all(normals):
        joined = " and ".join(f"({c})" if " or " in c else c for c in normals)  # type: ignore[operator]
        out.append((joined, "normal"))  # type: ignore[arg-type]
    rule = Ladder(tuple(item.channel for item in lights), tuple(out), tuple(unwritten))
    return too_long if engine_length(rule) > RULE_EXPRESSION_LIMIT else rule


def light_rule(rule: Ladder, values: Mapping[str, Any]) -> dict[str, Any]:
    """One prop's rule of a light: its channels, and its arms as the conditions.

    The first arm that holds gives its state's value, none holding gives
    ``unknown``'s, the default. Successive arms of one value share a
    condition, and the arms after the last one whose value is not the
    default's are left out. The arms are written into the rule itself, not
    into a screen formula: the viewer inlines a formula with longer tokens
    (``{<formula>.<name>}``) for each ``{N}`` that names it, so the
    compiled rule is shorter this way (see :func:`compiled_rules`).
    """
    default = values["unknown"]
    groups: list[tuple[Any, list[str]]] = []
    for condition, state in rule.arms:
        value = values[state]
        if groups and groups[-1][0] == value:
            groups[-1][1].append(condition)
        else:
            groups.append((value, [condition]))
    while groups and groups[-1][0] == default:
        groups.pop()
    conditions = [
        {"expression": " or ".join(f"({c})" if len(arms) > 1 else c for c in arms), "value": value}
        for value, arms in groups
    ]
    pvs = [{"name": channel, "trigger": True} for channel in rule.channels]
    return {"default": default, "pvs": pvs, "conditions": conditions}


def _literal(value: Any) -> str:
    """A rule value as the viewer writes it into the expression (its ``pyLiteral``)."""
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int | float):
        return str(int(value)) if float(value).is_integer() else repr(value)
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


_WIRE_TOKEN = re.compile(r"\{([^{}]+)\}")


def engine_expression(rule: Mapping[str, Any], formulas: Mapping[str, Mapping[str, Any]]) -> str:
    """The one expression the viewer gives the formula engine for a conditional rule.

    This is the viewer's rule compiler, step for step: a ``formula://`` pv's
    screen formula has each binding name made a ``{<formula>.<name>}`` token
    and is inlined, in parentheses, for that pv's ``{N}`` in every condition;
    the conditions become ``value if (condition) else (...)`` ending in the
    default. The engine refuses it when it is longer than
    :data:`RULE_EXPRESSION_LIMIT`.
    """
    inlined: dict[str, str] = {}
    for index, pv in enumerate(rule.get("pvs", ())):
        name = pv["name"]
        if not name.startswith(_FORMULA_PREFIX):
            continue
        decl = formulas[name[len(_FORMULA_PREFIX) :]]
        wire = decl["expression"]
        keys = sorted(decl.get("bindings", {}), key=len, reverse=True)
        if not _WIRE_TOKEN.search(wire):
            for key in keys:
                wire = re.sub(
                    rf"(?<![A-Za-z0-9_{{]){re.escape(key)}(?![A-Za-z0-9_}}])", f"{{{key}}}", wire
                )
        for key in keys:
            wire = wire.replace(f"{{{key}}}", f"{{{decl['name']}.{key}}}")
        inlined[str(index)] = f"({wire})"

    def condition(text: str) -> str:
        return _WIRE_TOKEN.sub(lambda m: inlined.get(m.group(1), m.group(0)), text)

    out = _literal(rule["default"])
    for item in reversed(rule.get("conditions", ())):
        out = f"{_literal(item['value'])} if ({condition(item['expression'])}) else ({out})"
    return out


def compiled_rules(rule: Ladder) -> list[str]:
    """Every rule a light drawn with *rule* writes (one per prop of any look), as the
    viewer compiles it for the formula engine (:func:`engine_expression`)."""
    looks = {state: symbols.look(state) for state in symbols.STATES}
    return [
        engine_expression(light_rule(rule, {s: looks[s][prop] for s in symbols.STATES}), {})
        for prop in (*symbols.LIGHT_PROPS, symbols.MARK_PROP)
    ]


def engine_length(rule: Ladder) -> int:
    """The longest expression the engine receives for a light drawn with *rule*."""
    return max(len(text) for text in compiled_rules(rule))


def ring_cost(rule: Ladder | str | None) -> tuple[int, frozenset[str]]:
    """What a strip symbol's ring and tag drawn with *rule* spend of a screen's budget
    (see :func:`light_cost`)."""
    if not isinstance(rule, Ladder) or not rule.arms:
        return 0, frozenset()
    looks = {state: symbols.ring_look(state) for state in symbols.STATES}
    rules = sum(
        bool(light_rule(rule, {s: looks[s][prop] for s in symbols.STATES})["conditions"])
        for prop in (*symbols.RING_PROPS, symbols.TAG_PROP)
    )
    return rules, frozenset(rule.channels)


def light_cost(rule: Ladder | str | None) -> tuple[int, frozenset[str]]:
    """What a light drawn with *rule* spends of a screen's budget: its rules that the
    viewer subscribes (those with a condition), and the channels they bind."""
    if not isinstance(rule, Ladder) or not rule.arms:
        return 0, frozenset()
    looks = {state: symbols.look(state) for state in symbols.STATES}
    rules = sum(
        bool(light_rule(rule, {s: looks[s][prop] for s in symbols.STATES})["conditions"])
        for prop in (*symbols.LIGHT_PROPS, symbols.MARK_PROP)
    )
    return rules, frozenset(rule.channels)


@dataclass
class Budget:
    """What is left of one screen's budget (:data:`SCREEN_RULE_BUDGET`,
    :data:`SCREEN_CHANNEL_BUDGET`)."""

    rules: int = field(default_factory=lambda: SCREEN_RULE_BUDGET)
    channels: set[str] = field(default_factory=set)

    def take(self, costs: Sequence[tuple[int, frozenset[str]]]) -> bool:
        """Spend *costs* when all of them fit; whether they did."""
        rules = sum(cost[0] for cost in costs)
        channels = self.channels.union(*(cost[1] for cost in costs))
        if rules > self.rules or len(channels) > SCREEN_CHANNEL_BUDGET:
            return False
        self.rules -= rules
        self.channels = channels
        return True


def tile_positions(
    count: int, first_x: float, line_x: float, line_top: float
) -> tuple[list[tuple[float, float]], float]:
    """Where *count* small lights go: ``(x, dy)`` each, and the height they take.

    The first line starts at *first_x*, each further line at *line_x*, the
    second *line_top* below the first and each next one a tile and a gap
    lower; a line ends at the page's right margin.
    """
    step = MOSAIC_LIGHT + MOSAIC_GAP
    right = MARGIN + STRIP_WIDTH
    out: list[tuple[float, float]] = []
    x, dy = first_x, 0.0
    for _ in range(count):
        if x + MOSAIC_LIGHT > right and x != line_x:
            x, dy = line_x, (line_top if dy == 0 else dy + step)
        out.append((x, dy))
        x += step
    return out, (out[-1][1] + MOSAIC_LIGHT if out else 0.0)


def chunks(graph: PanelGraph, lights: Sequence[Light]) -> list[list[Light]]:
    """*lights* in consecutive parts, each a rule within :data:`RULE_EXPRESSION_LIMIT`.

    The bound is the one :func:`ladder` applies, never a channel count: a part
    grows while the lengths of its channels' own rules fit the limit, and a
    part whose joint rule is still too long is halved. A channel too long on
    its own is a part alone (its light is then ``long rule``).
    """
    found: list[list[Light]] = []
    part: list[Light] = []
    cost = 0
    for item in lights:
        one = ladder(graph, [item])
        length = engine_length(one) if isinstance(one, Ladder) else 0
        if part and cost + length > RULE_EXPRESSION_LIMIT:
            found.append(part)
            part, cost = [], 0
        part.append(item)
        cost += length
    if part:
        found.append(part)

    def fit(part: list[Light]) -> list[list[Light]]:
        if len(part) == 1 or ladder(graph, part, node=True) != "aggregate state":
            return [part]
        half = len(part) // 2
        return fit(part[:half]) + fit(part[half:])

    return [piece for part in found for piece in fit(part)]


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
    formulas: list[dict[str, Any]] = field(default_factory=list)
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
        if node["type"] == "text-label" and node["props"].get("fontSize", SMALL_SIZE) < SMALL_SIZE:
            node["props"]["fontSize"] = SMALL_SIZE  # the floor: nothing smaller anywhere
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
        """One prop's rule: the value of the first arm that holds, else unknown's."""
        self._rules += 1
        return {
            "id": f"r-{self._rules:04d}",
            "name": name,
            "targetProperty": target,
            **light_rule(ladder, values),
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
            **({"formulas": self.formulas} if self.formulas else {}),
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
        self,
        view: View,
        graph: PanelGraph,
        screens: Callable[[Target], str],
        doc_id: str,
        colours: Mapping[str, str] = MappingProxyType({}),
    ) -> None:
        self.view = view
        self.graph = graph
        self.screens = screens
        self.s = _Screen(doc_id, view.title)
        self.y: float = MARGIN
        # writable widgets only on a device card's own signals and settings
        self.on_card = view.id.view == "card" and view.id.kind == "device"
        self.budget = Budget()  # spent by the strip's symbols, chips and summary rows
        self.colours: Mapping[str, str] = colours  # a symbol's type colour, by symbol name
        self.shown: dict[
            str, dict[str | None, str | None]
        ] = {}  # strip values: symbol, signal, unit
        self.dropped = 0  # strip values the budget left out
        self.taken: list[tuple[float, float, float, float]] = []  # a strip row's boxes
        self.x0: float = MARGIN  # the left edge of the column drawn in
        self.right: float = PAGE_WIDTH - MARGIN + PANEL_PAD  # where text is cut
        self.lit: set[str] = set()  # channels whose light a card's table drew
        self.facts_bottom: float = 0  # a rack's right column, where it ends
        self.cell_w: float = SET_CELL_WIDTH  # a set overview's cell width
        self.drawn = 0  # a set overview's cells drawn so far
        self.parts_listed: set[str] = set()  # a card's parts listed without a picture

    # -- primitives -----------------------------------------------------------

    def text(
        self,
        x,
        y,
        text: str,
        *,
        size=TEXT_SIZE,
        w=None,
        h=ROW,
        fg=PAGE_TEXT,
        bg=TRANSPARENT,
        align="left",
        tags=(),
    ) -> float:
        """A text label; returns its right edge. Text that would run past the right
        margin is cut there, an ellipsis marking the cut."""
        w = text_width(text, size) if w is None else w
        right = self.right  # the inner edge of a section's frame, or the column's
        if x + w > right:
            w = max(right - x, 0)
            text = _cut(text, w, size)
        props = {
            "text": text,
            "fontSize": size,
            "align": align,
            "foregroundColor": fg,
            "backgroundColor": bg,
        }
        self.s.widget("text-label", x, y, w, h, props, tags=tags)
        return x + w

    def mark(self, x, y, glyph: str) -> float:
        return self.text(x, y, glyph, w=MARK_WIDTH)

    def line(
        self,
        points: Sequence[tuple[float, float]],
        colour,
        width=LINE_WIDTH,
        *,
        tooltip=None,
        arrow: bool = False,
    ) -> None:
        """A line through *points*; its box is never narrower or lower than its stroke.

        A vertical or horizontal line has a box of no width or no height, which
        the viewer may draw as nothing; the box is widened by the stroke and the
        points sit in its middle.
        """
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x0, y0 = min(xs), min(ys)
        w, h = max(xs) - x0, max(ys) - y0
        dx, dy = max(width - w, 0) / 2, max(width - h, 0) / 2
        props: dict[str, Any] = {
            "lineColor": colour,
            "lineWidth": width,
            "points": [{"x": px - x0 + dx, "y": py - y0 + dy} for px, py in points],
            "arrowStart": False,
            "arrowEnd": arrow,
        }
        if tooltip is not None:
            props[_TOOLTIP] = tooltip
        self.s.widget("line", x0 - dx, y0 - dy, w + 2 * dx, h + 2 * dy, props)

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

    def button(
        self, x, y, link: Link, *, label: str | None = None, width: float | None = None
    ) -> float:
        """A link to another screen, at most *width* wide; returns the right edge of what it drew."""
        text = link.label if label is None else label
        w = min(text_width(text, TEXT_SIZE) + BUTTON_PAD, MAX_BUTTON_WIDTH)
        if width is not None:
            w = min(w, width)
        # Neutral: page text on the field background, outlined in the border
        # role. The format's navigation buttons take no colour, so the text is
        # a label and the outline the box that opens the screen.
        self.text(x, y, text, w=w, bg=FIELD_BACKGROUND, align="center")
        self.nav(x, y, w, ROW, replace(link, label=text), outline=True)
        right = x + w
        if link.basis == "inferred":
            right = self.mark(right + GAP, y, symbols.INFERRED_MARK)
        if link.standing == _CONTESTED:
            right = self.text(right + GAP, y, _CONTESTED, size=SMALL_SIZE)
        return right

    def quiet_link(self, right: float, y: float, link: Link) -> None:
        """A link as muted text ending at *right*: no box, a box that opens it on hover."""
        w = text_width(link.label, SMALL_SIZE)
        self.text(right - w, y, link.label, size=SMALL_SIZE, w=w, fg=MUTED_TEXT)
        self.nav(right - w, y, w, ROW, link)

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
        tooltip: str | None = None,
    ) -> float:
        """One light over *lights*; returns its right edge (after an inferred mark).

        A light whose rule is too long for the format is the aggregate look; a
        *tile* is one in an aggregate's mosaic and carries no inferred mark.
        """
        if rule is None:
            rule = ladder(self.graph, lights)
        look = "aggregate" if rule == "aggregate state" else "unknown"
        shape, mark = symbols.light(x, y + (ROW - size) / 2, size, look)
        self.attach(shape, mark, lights, rule, tile=tile, tooltip=tooltip)
        right = x + size
        if not tile and any(item.inferred for item in lights):
            right = self.mark(right + GAP, y, symbols.INFERRED_MARK)
        return right

    def attach(
        self,
        shape: dict[str, Any],
        mark: dict[str, Any],
        lights: Sequence[Light],
        rule: Ladder | str,
        *,
        tile: bool = False,
        tooltip: str | None = None,
    ) -> None:
        """Add a light's *shape* and *mark*, drawn in the look *rule* starts in, with
        the rules that turn them to the state of *lights*, or the tags saying why
        none can."""
        aggregate = rule == "aggregate state"
        names = ", ".join(item.channel for item in lights[:TOOLTIP_CHANNELS])
        if len(lights) > TOOLTIP_CHANNELS:
            names = f"{len(lights)} channels"
        shape["props"][_TOOLTIP] = names if tooltip is None else tooltip
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

    def no_rule(self, x, y, size: float = LIGHT_SIZE) -> float:
        """The look of a thing with no rule: a bare outline, no rules, no mark; its right edge."""
        shape, _mark = symbols.light(x, y + (ROW - size) / 2, size, symbols.NONE)
        shape["props"][_TOOLTIP] = _NO_RULE
        self.s.add(shape)
        return x + size

    def rules(self, shape_rules: list, mark_rules: list, names: str, rule: Ladder) -> None:
        """A light's rules: one per shape prop and one for its mark, all from *rule*."""
        looks = {state: symbols.look(state) for state in symbols.STATES}
        for prop in symbols.LIGHT_PROPS:
            values = {state: looks[state][prop] for state in symbols.STATES}
            shape_rules.append(self.s.rule(f"{names} {prop}", prop, rule, values))
        values = {state: looks[state][symbols.MARK_PROP] for state in symbols.STATES}
        mark_rules.append(self.s.rule(f"{names} mark", symbols.MARK_PROP, rule, values))

    def field(
        self, x, y, channel: str, *, drawn_as: str | None = None, writable=False, w=VALUE_WIDTH
    ) -> float:
        """A channel's value in the widget :data:`WIDGETS` names for its ``drawn_as``.

        A writable widget only on a device card, and only for a writable
        channel; anywhere else the value is a label. Returns the right edge.
        """
        kind = drawn_as if drawn_as not in (None, "untyped") else None
        if kind is None:
            kind = "setpoint" if writable else "untyped"
        on_card, elsewhere = WIDGETS[kind]
        widget = on_card if self.on_card else elsewhere
        if widget in _WRITABLE_WIDGETS and not (self.on_card and writable):
            widget = elsewhere
        common = {"pv": channel, "fontSize": TEXT_SIZE, "foregroundColor": PAGE_TEXT}
        if widget == "pv-text-input":
            props = {**common, "backgroundColor": FIELD_BACKGROUND, "alarmSensitiveBorder": False}
            self.s.widget(widget, x, y, w, ROW, props)
        elif widget == "pv-button":
            props = {
                **common,
                "label": COMMAND_LABEL,
                "pressValue": COMMAND_VALUE,
                "backgroundColor": FIELD_BACKGROUND,
            }
            self.s.widget(widget, x, y, w, ROW, props)
        elif widget == "waveform-plot":
            curve = {"name": channel, "y_channel": channel, "color": PAGE_TEXT}
            props = {"title": channel, "curves": [curve]}
            self.s.widget(widget, x, y, WAVEFORM_WIDTH, WAVEFORM_HEIGHT, props)
            self.y += WAVEFORM_HEIGHT - ROW
            return x + WAVEFORM_WIDTH
        else:
            props = {
                **common,
                "backgroundColor": FIELD_BACKGROUND,
                "showUnits": False,  # the unit is a label of its own: a tile never cuts it
                "alarmSensitive": True,
                "alarmSensitiveBorder": False,  # the viewer frames it once; the light says state
                "align": "right",
            }
            self.s.widget("pv-label", x, y, w, ROW, props)
        return x + w

    def unit(self, x, y, unit: str | None) -> float:
        """A value's unit, its own label beside the value; the right edge (*x* with none)."""
        if not unit:
            return x
        return self.text(x + UNIT_GAP, y, unit, size=SMALL_SIZE, w=UNIT_WIDTH)

    def next_row(self, height: float = ROW) -> None:
        self.y += height + ROW_GAP

    # -- the view -------------------------------------------------------------

    def draw(self) -> dict[str, Any]:
        self.header()
        self.y = HEADER_HEIGHT + SECTION_GAP
        if self.view.title != self.view.id.id and self.title_rest() is None:
            # else the header says it already
            self.text(MARGIN, self.y, self.view.title, size=TITLE_SIZE, w=STRIP_WIDTH, h=ROW + GAP)
            self.y += ROW + GAP
        sections = list(self.view.sections)
        if self.on_card:
            sections = self.card(sections)
        beside: Group | None = None
        arcs: Group | None = None
        if self.view.id.view == "rollup":
            for section in list(sections):
                group = section.items[0] if section.items else None
                if not isinstance(group, Group):
                    continue
                if group.form == "ring":
                    arcs = group
                elif section.title == _BY_SYSTEM_TITLE:
                    beside = group
                else:
                    continue
                sections.remove(section)
        for section in sections:
            tree = section.items[0] if section.items else None
            if self.view.id.view == "rollup" and isinstance(tree, Node):
                self.rollup(tree, beside, arcs)
            else:
                self.section(section)
        return self.s.document(self.y + SECTION_GAP)

    # -- the roll-up ----------------------------------------------------------

    def rollup(self, root: Node, beside: Group | None, arcs: Group | None = None) -> None:
        """A roll-up in one frame: the root's state, then its children as large tiles
        and, right of them, the rows *beside* (the place by system).

        A tile is a child's light, its short name and what it reads, opening its
        screen; all tiles have one height. The budget is spent as :meth:`plan`
        says: a child that fits one rule has its light; any other is ``open``:
        the aggregate mark, its count and a link down to its own screen, where
        its lights are. A tile never holds small lights of its parts. With
        *arcs*, those children are the ring's arcs (:meth:`ring`) and the
        others are tiles under it.
        """
        first = len(self.s.nodes)
        top = self.y
        self.y += PANEL_PAD
        width = ROLLUP_LEFT - COLUMN_GAP if beside is not None else STRIP_WIDTH
        self.x0, self.right = MARGIN, MARGIN + width
        self.heading(f"by {self.view.id.by or 'place'}")
        plan, decisions = self.plan(root)
        self.rollup_root(root, plan)
        on_ring = {node.label for node in arcs.nodes} if arcs is not None else set()
        if arcs is not None:
            self.ring(root, arcs, decisions)
        columns = max(1, int((width + GAP) // (ROLLUP_TILE_WIDTH + GAP)))
        w = (width - (columns - 1) * GAP) / columns
        children = [(i, c) for i, c in enumerate(root.children) if c.label not in on_ring]
        y = self.y
        for start in range(0, len(children), columns):
            for column, (i, child) in enumerate(children[start : start + columns]):
                x = MARGIN + column * (w + GAP)
                plan_i = "none" if child.no_rule else decisions.get(i, "open")
                self.rollup_tile(x, y, w, ROLLUP_TILE_HEIGHT, child, root.label, plan_i)
            y += ROLLUP_TILE_HEIGHT + GAP
        bottom = y
        if beside is not None:
            self.x0, self.right = MARGIN + ROLLUP_LEFT, MARGIN + STRIP_WIDTH
            self.y = top + PANEL_PAD
            self.heading(beside.title)
            self.summary(beside, self.x0, self.right - self.x0, 1)
            bottom = max(bottom, self.y)
        self.x0, self.right = MARGIN, PAGE_WIDTH - MARGIN + PANEL_PAD
        self.y = bottom + PANEL_PAD - ROW_GAP
        self.s.widget(
            "rectangle",
            MARGIN - PANEL_PAD,
            top,
            STRIP_WIDTH + 2 * PANEL_PAD,
            self.y - top,
            {
                "lineColor": BOX_BORDER,
                "lineWidth": LINE_WIDTH,
                "fill": True,
                "fillColor": PANEL_BACKGROUND,
            },
        )
        self.s.nodes.insert(first, self.s.nodes.pop())
        self.y += PANEL_GAP

    def ring(self, root: Node, arcs: Group, decisions: Mapping[int, str]) -> None:
        """The ring: one arc per child on it, clockwise from twelve o'clock in beam
        order, each in the look of its tile (:meth:`rollup_tile`): its light's
        rule, the aggregate mark when over the rule limit, a dashed outline with
        no rule; its short name and mark on it, and a link to its screen. The
        root's id stands in the middle."""
        index = {node.label: i for i, node in enumerate(root.children)}
        cx = (self.x0 + self.right) / 2
        cy = self.y + RING_PAD + RING_OUTER
        share = TURN / len(arcs.nodes)
        for number, node in enumerate(arcs.nodes):
            start, end = number * share + RING_GAP, (number + 1) * share - RING_GAP
            i = index.get(node.label)
            plan = "none" if node.no_rule or i is None else decisions.get(i, "open")
            self.arc(cx, cy, start, end, node, root.label, plan)
        w = text_width(root.label, TITLE_SIZE)
        self.text(cx - w / 2, cy - ROW / 2, root.label, size=TITLE_SIZE, w=w)
        self.y = cy + RING_OUTER + RING_PAD

    def arc(
        self, cx: float, cy: float, start: float, end: float, node: Node, parent: str, plan: str
    ) -> None:
        """One child on the ring: its arc, its short name and mark at the arc's middle,
        and a box there that opens its screen."""
        lights = [describe_light(self.graph, c) for c in node.channels]
        middle = math.radians((start + end) / 2)
        r = (RING_INNER + RING_OUTER) / 2
        mx, my = cx + r * math.sin(middle), cy - r * math.cos(middle)
        name = self.graph.thing(node.label, "place").short_name  # an arc is a child place
        name = name or node.label.removeprefix(parent + "/")
        name = _cut(name, RING_LABEL_WIDTH, TEXT_SIZE)
        left, top = mx - RING_LABEL_WIDTH / 2, my - ROW
        if plan == "none":
            shape = symbols.arc(cx, cy, RING_INNER, RING_OUTER, start, end, symbols.NONE)
            shape["props"]["lineStyle"] = "dashed"
            shape["props"][_TOOLTIP] = _NO_RULE
            self.s.add(shape)
        else:
            rule = ladder(self.graph, lights, node=True) if plan == "live" else "aggregate state"
            look = "aggregate" if rule == "aggregate state" else "unknown"
            shape = symbols.arc(cx, cy, RING_INNER, RING_OUTER, start, end, look)
            _light, mark = symbols.light(left, top + ROW, RING_LABEL_WIDTH, look)
            mark["geometry"]["height"] = ROW
            mark["props"]["fontSize"] = RING_MARK
            self.attach(shape, mark, lights, rule)
        self.text(left, top, name, w=RING_LABEL_WIDTH, align="center")
        if node.link is not None:
            self.nav(left, top, RING_LABEL_WIDTH, 2 * ROW, replace(node.link, label=node.label))

    def rollup_root(self, root: Node, plan: str | None) -> None:
        """The root's line: its light and what it reads, its parts, or that its summary
        needs a server-side aggregate."""
        count = len(root.channels)
        lights = [describe_light(self.graph, c) for c in root.channels]
        if plan == "summary":
            text = _SUMMARY_TEXT.format(count)
            tags = (_need("server-side aggregate"),)
            self.text(self.x0, self.y, text, size=SMALL_SIZE, tags=tags)
            self.next_row()
            return
        if root.no_rule:
            x = self.no_rule(self.x0, self.y)
            note = _NO_RULE_YET
        elif plan == "live":
            x = self.light(self.x0, self.y, lights, rule=ladder(self.graph, lights, node=True))
            note = _READS.format(count, "" if count == 1 else "s")
        else:
            x = self.light(self.x0, self.y, lights, rule="aggregate state")
            note = _AGGREGATE_TEXT.format(count)
        x = self.text(x + GAP, self.y, f"{root.label} · {note}", size=SMALL_SIZE, fg=MUTED_TEXT)
        height = ROW
        if plan == "parts":
            height = self.parts(chunks(self.graph, lights), x + GAP, self.x0 + INDENT)
        self.next_row(height)

    def rollup_tile(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        node: Node,
        parent: str,
        plan: str,
    ) -> None:
        """One child: its light top left with its short name and what it reads under
        it; a child over the rule limit has the aggregate mark, its count and
        ``open``."""
        self.box(x, y, w, h, fill=FIELD_BACKGROUND)
        inner = x + TILE_PAD
        top = y + TILE_PAD / 2
        lights = [describe_light(self.graph, c) for c in node.channels]
        count = len(node.channels)
        if plan == "none":
            right = self.no_rule(inner, top)
            note = _NO_RULE_YET
        elif plan == "live":
            right = self.light(inner, top, lights, rule=ladder(self.graph, lights, node=True))
            note = _READS.format(count, "" if count == 1 else "s")
        else:
            right = self.light(inner, top, lights, rule="aggregate state")
            note = _TILE_OPEN.format(count)
        name = node.label.removeprefix(parent + "/")
        room = x + w - TILE_PAD - (right + GAP)
        name = _cut(name, room, TEXT_SIZE)
        if name:
            self.text(right + GAP, top, name, w=min(text_width(name, TEXT_SIZE), room))
        lines = _wrap(note, w - 2 * TILE_PAD, SMALL_SIZE)
        for i, line in enumerate(lines):
            self.small(inner, top + ROW + i * SMALL_LINE, line, fg=MUTED_TEXT)
        if node.link is not None:
            self.nav(x, y, w, h, replace(node.link, label=node.label))

    # -- the device card ------------------------------------------------------

    def card(self, sections: list[Section]) -> list[Section]:
        """A device card's two columns in one frame: signals, settings, the health
        not shown in the table and the sources left; the fed-by chain, the other
        relations as chips under "also", and where it is right. Returns the other
        sections (a picture), drawn full width under it."""
        relations = {title for pair in RELATION_TITLES.values() for title in pair}
        left = [s for s in sections if s.title in _CARD_LEFT]
        right = [s for s in sections if s.title in relations | _CARD_RIGHT]
        below = [s for s in sections if s not in left and s not in right]
        if not left and not right:
            return below
        first = len(self.s.nodes)
        top = self.y
        start = top + PANEL_PAD
        self.y, self.x0, self.right = start, MARGIN, MARGIN + CARD_LEFT - COLUMN_GAP
        for section in left:
            self.card_section(section)
        bottom = self.y
        self.y, self.x0, self.right = start, MARGIN + CARD_LEFT, MARGIN + STRIP_WIDTH
        chips: list[tuple[str, Link | None]] = []
        for section in right:
            if section.title == _FED_BY:
                self.chain(section)
            elif section.title in relations:
                chips += [_chip(section.title, item) for item in section.items]
            else:
                self.heading(section.title)
                self.items(section)
                self.y += ROW_GAP
        if chips:
            self.also(chips)
        bottom = max(bottom, self.y)
        self.x0, self.right = MARGIN, PAGE_WIDTH - MARGIN + PANEL_PAD
        self.y = bottom + PANEL_PAD - ROW_GAP
        self.s.widget(
            "rectangle",
            MARGIN - PANEL_PAD,
            top,
            STRIP_WIDTH + 2 * PANEL_PAD,
            self.y - top,
            {
                "lineColor": BOX_BORDER,
                "lineWidth": LINE_WIDTH,
                "fill": True,
                "fillColor": PANEL_BACKGROUND,
            },
        )
        self.s.nodes.insert(first, self.s.nodes.pop())  # under what it frames
        self.y += PANEL_GAP
        return below

    def card_section(self, section: Section) -> None:
        """One block of a card's left column."""
        items = list(section.items)
        if section.title == "health":  # a light the table drew is not repeated
            items = [i for i in items if not (isinstance(i, Light) and i.channel in self.lit)]
            if not items:
                return
        self.heading(section.title)
        if section.title in ("signals", "settings"):
            if section.title == "signals":
                self.signal_heads()
            rows = [i for i in items if isinstance(i, ValueRow)]
            for row in rows[:CARD_ROWS]:
                self.signal_row(row)
            rest = Counter(row.drawn_as or "untyped" for row in rows[CARD_ROWS:])
            for kind, count in sorted(rest.items()):
                text = _MORE_CHANNELS.format(count, kind)
                self.text(self.x0, self.y, text, size=SMALL_SIZE, fg=MUTED_TEXT)
                self.next_row()
            for item in items:
                if not isinstance(item, ValueRow):
                    self.item(item)
        else:
            for item in items[:CARD_ROWS]:
                self.item(item)
            if len(items) > CARD_ROWS:
                text = _MORE_ROWS.format(len(items) - CARD_ROWS)
                self.text(self.x0, self.y, text, size=SMALL_SIZE, fg=MUTED_TEXT)
                self.next_row()
        self.y += ROW_GAP

    def signal_heads(self) -> None:
        """The table's column heads, over the name, the setpoint and the readback."""
        for word, x in zip(_SIGNAL_HEADS, self.signal_columns(), strict=True):
            self.text(x, self.y, word.upper(), size=SMALL_SIZE, fg=MUTED_TEXT)
        self.y += ROW

    def signal_columns(self) -> tuple[float, float, float]:
        name = self.x0
        setpoint = name + CARD_LABEL_WIDTH + GAP
        return name, setpoint, setpoint + VALUE_WIDTH + GAP

    def signal_row(self, row: ValueRow) -> None:
        """A row of the signals table: the name with its unit, the setpoint under its
        head, the readback (or the readable channel) under its, then the light of the
        deviation or of the readback's rule."""
        name, setpoint, readback = self.signal_columns()
        units = " / ".join(dict.fromkeys(u for u in (row.unit, row.readback_unit) if u))
        raw = row.label in (row.channel, row.readback)  # no signal name: the channel's own
        word = _short(row.label) if raw else row.label
        label = f"{word} ({units})" if units else word
        label = _cut(label, CARD_LABEL_WIDTH, TEXT_SIZE)
        self.text(name, self.y, label, w=min(text_width(label, TEXT_SIZE), CARD_LABEL_WIDTH))
        if raw:  # the full name, muted, under the short one
            full = _cut(row.label, CARD_LABEL_WIDTH, SMALL_SIZE)
            self.small(name, self.y + ROW, full, fg=MUTED_TEXT)
        shown = row.readback
        if row.readback is not None:
            self.field(setpoint, self.y, row.channel, drawn_as=row.drawn_as, writable=True)
            self.field(readback, self.y, row.readback)
        elif row.writable:
            self.field(setpoint, self.y, row.channel, drawn_as=row.drawn_as, writable=True)
        else:
            shown = row.channel
            self.field(readback, self.y, row.channel, drawn_as=row.drawn_as)
        x = readback + VALUE_WIDTH + GAP
        deviation = _tolerance_ladder(row)
        if deviation is not None:
            x = self.light(x, self.y, [Light(row.channel)], rule=deviation)
        elif shown is not None and self.graph.rule(shown):
            x = self.light(x, self.y, [describe_light(self.graph, shown)])
            self.lit.add(shown)
        if row.inferred:
            self.mark(x + GAP, self.y, symbols.INFERRED_MARK)
        self.next_row(ROW + SMALL_LINE if raw else ROW)

    def chain(self, section: Section) -> None:
        """The fed-by chain: the device's box, and each feeder's box right of it with an
        arrow into the device, the way the power flows; the arrows of several feeders
        join on one bus before the device."""
        self.heading(section.title)
        x = self.x0
        top = self.y
        name = self.view.alias or self.view.id.id
        self.chain_box(x, top, _THIS_DEVICE, name, None)
        feeder_x = x + CHAIN_BOX + CHAIN_ARROW
        y = top
        for item in section.items:
            label = item.label if isinstance(item, Link) else getattr(item, "text", "")
            link = item if isinstance(item, Link) else None
            kind = ""
            if link is not None:
                try:
                    kind = self.graph.thing(link.target.id, link.target.kind).cls or ""
                except KeyError:
                    kind = ""
            if getattr(item, "standing", None) == _CONTESTED:
                kind = f"{kind} · {_CONTESTED}" if kind else _CONTESTED
            if getattr(item, "basis", None) == "inferred":
                kind = f"{kind} {symbols.INFERRED_MARK}".strip()
            self.chain_box(feeder_x, y, kind, label, link)
            bus = x + CHAIN_BOX + CHAIN_ARROW / 2
            mid, here = top + CHAIN_HEIGHT / 2, y + CHAIN_HEIGHT / 2
            points = [(feeder_x, here), (bus, here), (bus, mid), (x + CHAIN_BOX, mid)]
            self.line(points if here != mid else [points[0], points[-1]], MUTED_TEXT, arrow=True)
            y += CHAIN_HEIGHT + GAP
        self.y = max(y, top + CHAIN_HEIGHT + GAP)
        self.text(self.x0, self.y, _CHAIN_CAPTION, size=SMALL_SIZE, fg=MUTED_TEXT)
        self.y += ROW + ROW_GAP

    def chain_box(self, x: float, y: float, kind: str, name: str, link: Link | None) -> None:
        """One box of the chain: what it is, small, over its name; opening its card."""
        self.box(x, y, CHAIN_BOX, CHAIN_HEIGHT)
        inner = CHAIN_BOX - GAP
        if kind:
            kind = _cut(kind, inner, SMALL_SIZE)
            self.text(x + GAP / 2, y, kind, size=SMALL_SIZE, w=inner, h=SMALL_LINE, fg=MUTED_TEXT)
        name = _cut(name, inner, SMALL_SIZE)
        w = text_width(name, SMALL_SIZE)
        self.text(x + GAP / 2, y + CHAIN_HEIGHT - ROW, name, size=SMALL_SIZE, w=w)
        if link is not None:
            self.nav(x, y, CHAIN_BOX, CHAIN_HEIGHT, link)

    def also(self, chips: Sequence[tuple[str, Link | None]]) -> None:
        """Every other relation as a chip, "<relation>: <thing>", wrapping in the column."""
        self.heading(_ALSO)
        x = self.x0
        for label, link in chips:
            room = self.right - self.x0 - GAP
            label = _cut(label, room, SMALL_SIZE)
            w = text_width(label, SMALL_SIZE) + GAP
            if x > self.x0 and x + w > self.right:
                x = self.x0
                self.y += CHIP_HEIGHT + CHIP_GAP
            x = self.chip(x, self.y, label, link) + CHIP_GAP
        self.y += CHIP_HEIGHT + CHIP_GAP

    def header(self) -> None:
        """The bar at the top: the screen's thing, the way out to the places around it,
        and the model the screen was made from, right-aligned."""
        self.s.widget(
            "rectangle",
            0,
            0,
            PAGE_WIDTH,
            HEADER_HEIGHT,
            {
                "lineColor": BOX_BORDER,
                "lineWidth": LINE_WIDTH,
                "fill": True,
                "fillColor": FIELD_BACKGROUND,
            },
        )
        y = HEADER_PAD
        x = MARGIN - GAP
        for crumb in self.view.crumbs:
            x = self.button(x + GAP, y, crumb)
            x = self.text(x + GAP, y, "›", w=MARK_WIDTH, fg=MUTED_TEXT)
        x = self.text(x + GAP, y, self.view.id.id)
        if self.view.alias is not None:  # the stated short name, as a chip
            w = text_width(self.view.alias, SMALL_SIZE)
            self.box(x + GAP, y, w, ROW, fill=PANEL_BACKGROUND)
            x = self.text(x + GAP, y, self.view.alias, size=SMALL_SIZE, w=w, fg=MUTED_TEXT)
        stamp = f"model {self.graph.content_hash[:STAMP_DIGITS]}"
        w = text_width(stamp, SMALL_SIZE)
        self.text(PAGE_WIDTH - MARGIN - w, y, stamp, size=SMALL_SIZE, w=w, fg=MUTED_TEXT)
        rest = self.title_rest()
        if rest:  # what the screen shows of it ("health by place"), muted
            self.text(x + GAP, y, rest, fg=MUTED_TEXT)

    def title_rest(self) -> str | None:
        """What the title says after the screen's id, when it starts with the id; the
        header bar carries it. ``None`` when the title is something else."""
        title, thing = self.view.title, self.view.id.id
        if title.startswith(thing + " "):
            return title.removeprefix(thing).strip()
        return None

    def heading(self, text: str) -> None:
        """A section's heading: small capitals in the muted role."""
        self.text(self.x0, self.y, text.upper(), size=HEADING_SIZE, fg=MUTED_TEXT)
        self.y += ROW + ROW_GAP

    def section(self, section: Section) -> None:
        """A section in its frame: the frame stands in the margin around what it holds."""
        first = len(self.s.nodes)
        top = self.y
        self.y += PANEL_PAD
        if not (section.items and isinstance(section.items[0], Strip | Band)):
            pictures = [i for i in section.items if isinstance(i, Picture)]
            plain = pictures and not any(_outlined(i) for i in pictures)
            self.heading(_PARTS if plain else section.title)
        self.items(section)
        self.y += PANEL_PAD - ROW_GAP
        self.s.widget(
            "rectangle",
            MARGIN - PANEL_PAD,
            top,
            STRIP_WIDTH + 2 * PANEL_PAD,
            self.y - top,
            {
                "lineColor": BOX_BORDER,
                "lineWidth": LINE_WIDTH,
                "fill": True,
                "fillColor": PANEL_BACKGROUND,
            },
        )
        self.s.nodes.insert(first, self.s.nodes.pop())  # under what it frames
        self.y += PANEL_GAP

    def items(self, section: Section) -> None:
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
        self.text(self.x0, self.y, text, w=STRIP_WIDTH, tags=(_need("table"),))
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
            Group: self.group,
            Band: self.band,
            CheckRow: self.check_row,
            Totals: self.totals,
            Calibrations: self.calibrations,
        }[type(item)]
        draw(item)

    # -- rows -----------------------------------------------------------------

    def value_row(self, row: ValueRow) -> None:
        x = self.text(self.x0, self.y, row.label, w=LABEL_WIDTH)
        x = self.field(x + GAP, self.y, row.channel, drawn_as=row.drawn_as, writable=row.writable)
        x = self.unit(x, self.y, row.unit)
        if row.readback is not None:
            x = self.field(x + GAP, self.y, row.readback)
            x = self.unit(x, self.y, row.readback_unit)
        deviation = _tolerance_ladder(row)
        if deviation is not None:
            x = self.light(x + GAP, self.y, [Light(row.channel)], rule=deviation)
        if row.inferred:
            self.mark(x + GAP, self.y, symbols.INFERRED_MARK)
        self.next_row()

    def light_row(self, item: Light) -> None:
        x = self.light(self.x0, self.y, [item])
        self.text(x + GAP, self.y, item.channel)
        self.next_row()

    def link_row(self, link: Link) -> None:
        self.button(self.x0, self.y, link, width=self.right - self.x0)
        self.next_row()

    def fact_row(self, fact: Fact) -> None:
        x = self.text(self.x0, self.y, fact.label, w=LABEL_WIDTH)
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
        x = self.button(self.x0, self.y, item.link, width=self.right - self.x0)
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
        """A set's members as an equal grid: a row per place under its short name, the
        rows grouped under the top-level place they are in (a heading each, a gap
        between), the members in beam order under column heads ``#1`` … ``#n``,
        and a line over the heads saying what a cell shows, with its unit. A cell
        is the member's readback of the set's primary pair or its short name, and
        one light: the deviation light when the pair has one, else the member's.
        It opens the member's card and nothing in it is writable. Few columns
        widen to at most :data:`SET_CELL_MAX`; columns past the page's width go
        on in a block below, :data:`SET_BLOCK_GAP` under the one before, with
        only the rows that have members in it. A group whose members all sit in
        one row too long for a block is no grid: it is a flow (:meth:`set_flow`)
        after the grid groups. Cells are drawn while fewer than
        :data:`SET_MEMBERS` are; the members of the rest are counted in a line
        under them."""
        tags = (_need("table"),)
        count = sum(len(row.cells) for row in grid.rows)
        line = " · ".join((f"{count} members", f"{len(grid.rows)} rows"))
        self.text(MARGIN, self.y, line.upper(), size=SMALL_SIZE, fg=MUTED_TEXT, tags=tags)
        self.next_row()
        room = STRIP_WIDTH - SET_ROW_HEAD
        fit = max(1, int((room + GAP) // (SET_CELL_WIDTH + GAP)))
        groups = self.grid_groups(grid.rows)
        flows = [(t, rows[0]) for t, rows in groups if len(rows) == 1 and len(rows[0].cells) > fit]
        flowing = {t for t, _row in flows}
        gridded = [(t, rows) for t, rows in groups if t not in flowing]
        width = max((len(r.cells) for _t, rows in gridded for r in rows), default=0)
        shown = max(1, min(fit, width or fit))
        self.cell_w = min(SET_CELL_MAX, (room - (shown - 1) * GAP) / shown)
        what = self.grid_what(grid)
        w = text_width(what, SMALL_SIZE)
        x_what = self.set_x(0) if gridded else MARGIN
        self.text(x_what, self.y, what, size=SMALL_SIZE, w=w, fg=MUTED_TEXT, tags=tags)
        self.y += ROW
        self.drawn = 0
        left_out = 0
        for first in range(0, width, fit):
            here = [(t, [r for r in rows if r.cells[first : first + fit]]) for t, rows in gridded]
            here = [(t, rows) for t, rows in here if rows]  # a row with none here is not drawn
            if self.drawn >= SET_MEMBERS:
                left_out += sum(len(r.cells[first : first + fit]) for _t, rs in here for r in rs)
                continue
            if first:
                self.y += SET_BLOCK_GAP - ROW_GAP
            for column in range(first, min(first + fit, width)):
                x = self.set_x(column - first)
                head = f"#{column + 1}"
                w = text_width(head, SMALL_SIZE)
                x_head = x + (self.cell_w - w) / 2
                self.text(x_head, self.y, head, size=SMALL_SIZE, w=w, fg=MUTED_TEXT, tags=tags)
            self.y += ROW
            for title, rows in here:
                self.y += ROW_GAP
                self.text(MARGIN, self.y, title.upper(), size=HEADING_SIZE, fg=MUTED_TEXT)
                self.y += ROW
                for row in rows:
                    cells = row.cells[first : first + fit]
                    if isinstance(row.label, Link):
                        short = _short(row.label.label)
                        w = min(text_width(short, SMALL_SIZE), SET_ROW_HEAD - GAP)
                        self.text(MARGIN, self.y, _cut(short, w, SMALL_SIZE), size=SMALL_SIZE, w=w)
                        self.nav(MARGIN, self.y, w, ROW, row.label)
                    for column, cell in enumerate(cells):
                        self.grid_cell(self.set_x(column), self.y, cell)
                        self.drawn += 1
                    self.y += ROW + ROW_GAP
        for title, row in flows:
            left_out += self.set_flow(title, row)
        if left_out:
            self.y += ROW_GAP
            self.text(MARGIN, self.y, _SET_MORE.format(left_out), size=SMALL_SIZE, fg=MUTED_TEXT)
            self.next_row()

    def set_flow(self, title: str, row: GridRow) -> int:
        """A group whose members all sit in one row too long for a block: its heading,
        then its cells in lines at full width, no column heads, no row head. Cells
        past :data:`SET_MEMBERS` drawn on the screen are not drawn; returns how many."""
        self.y += SET_BLOCK_GAP - ROW_GAP
        self.text(MARGIN, self.y, title.upper(), size=HEADING_SIZE, fg=MUTED_TEXT)
        if isinstance(row.label, Link):
            w = text_width(title.upper(), HEADING_SIZE)
            self.nav(MARGIN, self.y, w, ROW, row.label)
        self.y += ROW
        per = max(1, int((STRIP_WIDTH + GAP) // (self.cell_w + GAP)))
        room = max(0, SET_MEMBERS - self.drawn)
        cells = row.cells[:room]
        for k, cell in enumerate(cells):
            if k and not k % per:
                self.y += ROW + ROW_GAP
            self.grid_cell(MARGIN + (k % per) * (self.cell_w + GAP), self.y, cell)
            self.drawn += 1
        if cells:
            self.y += ROW + ROW_GAP
        return len(row.cells) - len(cells)

    def grid_what(self, grid: Grid) -> str:
        """What a set overview's cells show: the readback of the primary pair's signal
        with the readback's unit, or the members' short names."""
        pairs = [
            cell.pair for row in grid.rows for cell in row.cells
            if cell.pair is not None and cell.pair.readback is not None
        ]  # fmt: skip
        if grid.signal is None or not pairs:
            return _SET_NAMES
        unit = self.graph.channel(pairs[0].readback).unit  # type: ignore[arg-type]
        what = _SET_READBACK.format(grid.signal.replace("_", " "))
        return f"{what} ({unit})" if unit else what

    def grid_groups(self, rows: Sequence[GridRow]) -> list[tuple[str, list[GridRow]]]:
        """The rows grouped by the top-level place of each row's place, in row order; the
        rows of no place under their own fact."""
        groups: dict[str, list[GridRow]] = {}
        for row in rows:
            if isinstance(row.label, Link):
                top = row.label.target.id
                with contextlib.suppress(KeyError):
                    while (up := self.graph.thing(top, "place").place) is not None:
                        top = up
            else:
                top = row.label.text
            groups.setdefault(top, []).append(row)
        return list(groups.items())

    def set_x(self, column: int) -> float:
        """The left edge of a set overview's *column* within its block."""
        return MARGIN + SET_ROW_HEAD + column * (self.cell_w + GAP)

    def grid_cell(self, x0: float, y: float, cell: GridCell) -> None:
        """One member: the readback in its field, or the short name in a box, and one
        light at the right end: the deviation light when the pair has one, else the
        member's; the whole cell opens its card."""
        w = self.cell_w
        right = x0 + w - CHIP_LIGHT
        pair = cell.pair
        deviation = _tolerance_ladder(pair) if pair is not None else None
        if pair is not None and deviation is not None:
            self.light(right, y, [Light(pair.channel)], rule=deviation, size=CHIP_LIGHT, tile=True)
        elif cell.channels:
            lights = [describe_light(self.graph, c) for c in cell.channels]
            rule = ladder(self.graph, lights, node=True)
            self.light(right, y, lights, rule=rule, size=CHIP_LIGHT, tile=True)
        else:
            self.no_rule(right, y, CHIP_LIGHT)
        end = right - GAP / 2
        if pair is not None and pair.readback is not None:
            self.field(x0, y, pair.readback, w=end - x0)  # the viewer frames it
        else:
            self.box(x0, y, end - x0, ROW, fill=FIELD_BACKGROUND)
            name = self.member_name(cell.member)
            name = _cut(name, end - x0 - GAP, SMALL_SIZE)
            nw = text_width(name, SMALL_SIZE)
            self.text(x0 + GAP / 2, y, name, size=SMALL_SIZE, w=min(nw, end - x0 - GAP))
        if isinstance(cell.member, Link):
            self.nav(x0, y, w, ROW, cell.member)

    def calibrations(self, table: Calibrations) -> None:
        """A device's calibration block: column heads, then a row per kind (see
        :meth:`calibration_row`). Read-only."""
        for word, dx in zip(_CAL_HEADS, CAL_COLUMNS, strict=True):
            self.text(MARGIN + dx, self.y, word.upper(), size=SMALL_SIZE, fg=MUTED_TEXT)
        self.y += ROW
        for row in table.rows:
            self.calibration_row(row)

    def calibration_row(self, row: CalibrationRow) -> None:
        """One kind: its title (a link to the kind's view over a set) with its
        conditions muted under it; the value in force with a note under it, or the
        live channel; the basis chip; date, source and history."""
        x = [MARGIN + dx for dx in CAL_COLUMNS]
        widths = [b - a - GAP for a, b in itertools.pairwise([*x, MARGIN + STRIP_WIDTH])]
        top = self.y
        title = _cut(row.title, widths[0], TEXT_SIZE)
        self.text(x[0], top, title, w=min(text_width(title, TEXT_SIZE), widths[0]))
        if row.link is not None:
            self.nav(x[0], top, min(text_width(title, TEXT_SIZE), widths[0]), ROW, row.link)
        second = False
        if row.conditions:
            self.small(x[0], top + ROW, _cut(row.conditions, widths[0], SMALL_SIZE), fg=MUTED_TEXT)
            second = True
        if row.channel is not None:
            self.field(x[1], top, row.channel)
        elif row.shown is None:
            self.small(x[1], top + (ROW - SMALL_LINE) / 2, _NO_VALUE_YET, fg=MUTED_TEXT)
        else:
            value = _cut(row.shown, widths[1], TEXT_SIZE)
            self.text(x[1], top, value, w=min(text_width(value, TEXT_SIZE), widths[1]))
        if row.note:
            self.small(x[1], top + ROW, _cut(row.note, widths[1], SMALL_SIZE), fg=MUTED_TEXT)
            second = True
        _name_x, _value_x, chip_x, *rest_x = x
        self.basis_chip(chip_x, top, row)
        _w0, _w1, _w2, *rest_w = widths
        for left, width, text in zip(
            rest_x, rest_w, (row.date, row.source, row.history), strict=True
        ):
            if text:
                self.small(left, top + (ROW - SMALL_LINE) / 2, _cut(text, width, SMALL_SIZE))
        self.y = top + ROW + (SMALL_LINE if second else 0) + ROW_GAP

    def basis_chip(self, x: float, y: float, row: CalibrationRow) -> None:
        """The basis chip: an outline in the basis role (the disagree role when
        contested), dashed and muted for a value still wanted, page text for a
        value held live."""
        if row.channel is not None:
            colour, dashed = PAGE_TEXT, False
        elif row.shown is None:
            colour, dashed = MUTED_TEXT, True
        else:
            colour, dashed = symbols.BASIS_ROLES.get(row.state, MUTED_TEXT), False
        text = _cut(row.chip, CAL_CHIP_WIDTH - GAP, SMALL_SIZE)
        w = min(text_width(text, SMALL_SIZE) + GAP, CAL_CHIP_WIDTH)
        props = {
            "lineColor": colour,
            "lineWidth": LINE_WIDTH,
            "fill": False,
            "lineStyle": "dashed" if dashed else "solid",
        }
        self.s.widget("rectangle", x, y + CHIP_INSET, w, ROW - 2 * CHIP_INSET, props)
        self.text(x + GAP / 2, y, text, size=SMALL_SIZE, w=w - GAP, fg=colour)

    def member_name(self, member: Link | Fact) -> str:
        """A member's short name: what ``names.csv`` states, else its id's last part."""
        if isinstance(member, Fact):
            return member.text
        try:
            short = self.graph.thing(member.target.id, member.target.kind).short_name
        except KeyError:
            short = None
        return short or _short(member.label)

    def check_header(self) -> None:
        tags = (_need("table"),)
        x = self.text(MARGIN, self.y, "channel", size=SMALL_SIZE, w=NAME_WIDTH, tags=tags)
        x = self.text(x, self.y, "thing", size=SMALL_SIZE, w=NAME_WIDTH, tags=tags)
        for title in ("on device", "typed", "rule", "source"):
            x = self.text(x, self.y, title, size=SMALL_SIZE, w=CELL_WIDTH, tags=tags)
        self.next_row()

    def check_row(self, row: CheckRow) -> None:
        x = self.text(MARGIN, self.y, row.channel, w=NAME_WIDTH)
        x = self.text(x, self.y, row.thing or "", w=NAME_WIDTH)
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

    def node(self, node: Node, depth: int = 0, plan: str | None = None) -> None:
        """A roll-up node's row; the root's direct children follow it, one level only.

        The screen has one :class:`Budget`, spent by the root's row first
        (:meth:`plan`): a row is ``live`` (one light), drawn in ``parts``
        (one small light per part, each within the limit), or ``open``: the
        aggregate mark, its channel count and a link to its own screen. A
        root that is not live with children shows them instead, under a
        line that says its summary needs a server-side aggregate.
        """
        x = MARGIN + depth * INDENT
        lights = [describe_light(self.graph, c) for c in node.channels]
        rule: Ladder | str | None = None
        decisions: dict[int, str] = {}
        if depth == 0:
            plan, decisions = self.plan(node)
        if node.no_rule:
            x = self.no_rule(x, self.y)
        elif plan == "summary":
            tags = (_need("server-side aggregate"),)
            self.s.widget(
                "text-label",
                x,
                self.y,
                STRIP_WIDTH - x + MARGIN,
                ROW,
                {"text": _SUMMARY_TEXT.format(len(node.channels)), "fontSize": SMALL_SIZE},
                tags=tags,
            )
            self.next_row()
            for i, child in enumerate(node.children):
                self.node(child, 1, decisions.get(i))
            return
        else:
            rule = ladder(self.graph, lights, node=True) if plan == "live" else "aggregate state"
            x = self.light(x, self.y, lights, rule=rule)
        if node.link is not None:
            x = self.button(x + GAP, self.y, node.link, label=node.label)
        else:
            x = self.text(x + GAP, self.y, node.label)
        height = ROW
        if not node.no_rule and plan != "live":
            text = _AGGREGATE_TEXT.format(len(node.channels))
            if plan == "open" and node.link is not None:
                x = self.button(x + GAP, self.y, Link(text, node.link.target))
            else:
                x = self.text(x + GAP, self.y, text, size=SMALL_SIZE)
        if plan == "parts":
            height = self.parts(chunks(self.graph, lights), x + GAP, MARGIN + (depth + 1) * INDENT)
        self.next_row(height)
        for i, child in enumerate(node.children if depth == 0 else ()):
            self.node(child, 1, decisions.get(i))

    def plan(self, root: Node) -> tuple[str, dict[int, str]]:
        """How a roll-up screen spends its budget: the root's plan and each child's.

        In order: the root's own light when it fits one rule; then each child
        that fits one rule gets its light; every other child is ``open``, a
        link down to its own screen. A root that is not live is in
        parts when it has no children and they fit, else a ``summary``.
        """
        budget = self.budget  # the screen's: what is left goes to the rows beside

        def ladder_of(node: Node) -> Ladder | str:
            lights = [describe_light(self.graph, c) for c in node.channels]
            return ladder(self.graph, lights, node=True)

        def part_costs(node: Node) -> list[tuple[int, frozenset[str]]]:
            lights = [describe_light(self.graph, c) for c in node.channels]
            return [
                light_cost(ladder(self.graph, part, node=len(part) > 1))
                for part in chunks(self.graph, lights)
            ]

        decisions: dict[int, str] = {}
        if root.no_rule:
            plan = "none"
        else:
            rule = ladder_of(root)
            if isinstance(rule, Ladder) and budget.take([light_cost(rule)]):
                plan = "live"
            elif isinstance(rule, str) and rule != "aggregate state":
                plan = "live"  # drawn unknown and counted: it binds nothing
            elif not root.children and budget.take(part_costs(root)):
                plan = "parts"
            else:
                plan = "summary" if root.children else "open"
        for i, child in enumerate(root.children):
            if child.no_rule:
                continue
            rule = ladder_of(child)
            if (isinstance(rule, Ladder) and budget.take([light_cost(rule)])) or (
                isinstance(rule, str) and rule != "aggregate state"
            ):
                decisions[i] = "live"  # a rule that cannot be written is drawn unknown
            else:
                decisions[i] = "open"  # its own screen has its lights
        return plan, decisions

    def parts(self, parts: Sequence[Sequence[Light]], x0: float, line_x: float) -> float:
        """The parts of a node too long for one rule: a small light each, part i of n.

        The first line follows the row's label; the next lines start at
        *line_x* below it. Returns the row's height.
        """
        places, height = tile_positions(len(parts), x0, line_x, ROW)
        for number, (part, (x, dy)) in enumerate(zip(parts, places, strict=True), 1):
            rule = ladder(self.graph, part, node=len(part) > 1)
            top = self.y + dy - (ROW - MOSAIC_LIGHT) / 2 if dy else self.y
            tooltip = _PART.format(number, len(parts))
            self.light(x, top, part, rule=rule, size=MOSAIC_LIGHT, tile=True, tooltip=tooltip)
        return max(ROW, height)

    def nav(self, x, y, w, h, link: Link, *, outline: bool = False) -> None:
        """A box over what is drawn there that opens *link*'s screen.

        Every link is one: a button's text is a label under an *outline* in
        the border role, a symbol or a tile has a transparent box that shows
        the link's label while the pointer is on it.
        """
        props: dict[str, Any] = {
            "lineColor": BOX_BORDER if outline else TRANSPARENT,
            "lineWidth": LINE_WIDTH if outline else 0,
            "fill": False,
            _TOOLTIP: link.label,
            "actions": [
                {
                    "type": "open_display",
                    "file": self.screens(link.target),
                    "target": _NAVIGATION_TARGET,
                }
            ],
        }
        if not outline:
            props[_HOVER_LABEL] = link.label
        self.s.widget("rectangle", x, y, w, h, props, tags=(_need("clickable text"),))

    # -- the strip ------------------------------------------------------------

    def strip(self, item: Strip) -> None:
        """A place in beam order: the title line, each row of symbols with its locator,
        the legend of the symbols drawn, the tray, and the count of the rest."""
        self.strip_title()
        if item.elements:
            widths = {kind: units * SYMBOL_UNIT for kind, units in symbols.WIDTH_UNITS.items()}
            for row in _strip_rows(item.elements, widths):
                self.strip_row(row, widths, item.extent_m)
            self.values_line()
            self.legend(item.elements)
        if item.tray:
            self.tray(item)
        elif item.others is not None:
            self.quiet_link(MARGIN + STRIP_WIDTH, self.y, item.others)
            self.next_row()

    def strip_title(self) -> None:
        """What the strip is, left; what its looks mean, right."""
        self.text(MARGIN, self.y, _STRIP_TITLE)
        w = text_width(_LOOKS_LEGEND, SMALL_SIZE)
        self.text(
            MARGIN + STRIP_WIDTH - w, self.y, _LOOKS_LEGEND, size=SMALL_SIZE, w=w, fg=MUTED_TEXT
        )
        self.next_row()

    def strip_row(
        self,
        row: Sequence[StripElement],
        widths: Mapping[str, float],
        extent: tuple[float, float] | None,
    ) -> None:
        drawn = [
            e if e.symbol in symbols.WIDTH_UNITS else replace(e, symbol="generic") for e in row
        ]
        placed, _scale = strip.arrange_scaled(drawn, widths, STRIP_WIDTH, fill=True)
        beam = [p for p in placed if p.lane != "above"]
        badges = [p for p in placed if p.lane == "above"]
        levels = stem_levels([p.x for p in badges])
        # the upper lane only when something stands in it
        tags = stem_levels([p.x for p in beam], TAG_WIDTH + STRIP_LABEL_GAP)
        above = STRIP_AXIS  # room over the axis: rings' words, then badges with theirs
        if any(p.channels for p in beam):
            above += RING_PAD + (max(tags) + 1) * TAG_HEIGHT
        if badges:
            stem = STEM_HIGH if any(levels) else STEM_LOW
            ruled = any(p.channels for p in badges)
            above = max(above, stem + BADGE / 2 + (RING_PAD + TAG_HEIGHT if ruled else 0))
        axis = self.y + above + GAP
        self.line([(MARGIN, axis), (MARGIN + STRIP_WIDTH, axis)], PAGE_TEXT, AXIS_WIDTH)
        self.taken = []  # boxes a badge's name and values must clear: tags, badges
        for p, tag_level in zip(beam, tags, strict=True):
            x = MARGIN + p.x
            self.symbol(p, x, axis, p.width / symbols.WIDTH_UNITS[p.symbol], tag_level)
            if p.link is not None:
                half = p.width / 2
                link = replace(p.link, label=p.thing)
                self.nav(x - half, axis - STRIP_AXIS, p.width, 2 * STRIP_AXIS, link)
        centres = []
        for p, level in zip(badges, levels, strict=True):
            x = MARGIN + p.x
            cy = axis - (STEM_HIGH if level else STEM_LOW)
            self.line([(x, cy + BADGE / 2), (x, axis)], MUTED_TEXT)
            self.badge(p, x, cy)
            self.taken.append((x - BADGE / 2, cy - BADGE / 2, BADGE, BADGE))
            centres.append((p, x, cy))
        for p, x, cy in centres:
            self.badge_text(p, x, cy)
        lines = 0
        for text, left, w, level in strip_labels(beam):
            top = axis + STRIP_LABEL_TOP + level * STRIP_LABEL_STEP
            self.text(MARGIN + left, top, text, size=STRIP_LABEL_SIZE, w=w, h=STRIP_LABEL_STEP)
            lines = max(lines, level + 1)
        names = axis + STRIP_LABEL_TOP + lines * STRIP_LABEL_STEP
        below = names
        # one width, centred: bare text, so two close neighbours' boxes may overlap
        # where their numbers do not
        w = STRIP_VALUE_WIDTH
        for p in beam:
            for i, channel in enumerate(self.affordable(p)):
                top = names + i * VALUE_STEP  # row i: every symbol's i-th value
                left = min(max(MARGIN + p.x - w / 2, MARGIN), MARGIN + STRIP_WIDTH - w)
                self.value(left, top, w, channel)
                below = max(below, top + VALUE_STEP)
        self.y = max(axis + STRIP_AXIS, below) + LEADER_GAP
        if extent is not None:
            self.locator(beam, extent)
        else:
            self.y += ROW_GAP

    def affordable(self, element: strip.Placed) -> list[str]:
        """The channels of *element*'s values, when the screen's budget allows them all;
        none, counted in :attr:`dropped`, when it does not."""
        channels = [channel for channel, _unit in element.annotations]
        if not channels:
            return []
        if not self.budget.take([(0, frozenset(channels))]):
            self.dropped += len(channels)
            return []
        for channel, unit in element.annotations:
            units = self.shown.setdefault(element.symbol, {})
            signal = self.graph.channel(channel).signal
            if units.get(signal) is None:  # the first unit stated for the signal
                units[signal] = unit
        return channels

    def value(self, x: float, y: float, w: float, channel: str, align: str = "center") -> None:
        """A channel's live value on the strip: small, muted, no ground, no alarm
        border, no unit (the strip's values line names the units)."""
        props = {
            "pv": channel,
            "fontSize": SMALL_SIZE,
            "foregroundColor": MUTED_TEXT,
            "backgroundColor": TRANSPARENT,
            "showUnits": False,
            "alarmSensitive": False,
            "alarmBorder": False,
            "alarmSensitiveBorder": False,  # the viewer's name for the alarm border
            "align": align,
        }
        self.s.widget("pv-label", x, y, w, VALUE_STEP, props)

    def values_line(self) -> None:
        """What the values on the strip are, and how many the budget left out."""
        if self.shown:
            parts = []
            for kind in symbols.KINDS:
                if kind in self.shown:
                    signals = ", ".join(
                        f"{signal} ({unit})" if unit else str(signal)
                        for signal, unit in self.shown[kind].items()
                    )
                    parts.append(f"{SYMBOL_WORDS[kind]}: {signals}")
            line = _VALUES
            for part in parts:  # one line per what fits, a part never split
                joined = part if line == _VALUES else f"{line} · {part}"
                if line != _VALUES and text_width(joined, SMALL_SIZE) > STRIP_WIDTH:
                    self.text(MARGIN, self.y, line, size=SMALL_SIZE, fg=MUTED_TEXT)
                    self.y += ROW
                    line = part
                else:
                    line = _VALUES + part if line == _VALUES else joined
            self.text(MARGIN, self.y, line, size=SMALL_SIZE, fg=MUTED_TEXT)
            self.next_row()
        if self.dropped:
            note = _DROPPED.format(self.dropped)
            self.text(MARGIN, self.y, note, size=SMALL_SIZE, fg=MUTED_TEXT)
            self.next_row()

    def badge(self, element: strip.Placed, cx: float, cy: float) -> None:
        """A device off the axis: a circle with its kind's letter, filled with its
        type colour, its state ringed (:meth:`marked`)."""
        circle, letter = symbols.badge(cx, cy, BADGE, element.symbol, SMALL_SIZE)
        fill = self.colours.get(element.symbol)
        circle["props"].update(
            {
                "fillColor": fill if fill is not None else PANEL_BACKGROUND,
                "lineColor": PAGE_TEXT,
                "lineStyle": "dashed" if not element.channels else "solid",
                _TOOLTIP: element.thing,
            }
        )
        if fill is not None:  # a letter on a type colour reads in the page's ground
            letter["props"]["foregroundColor"] = PAGE_BACKGROUND
        self.marked(element, [circle])
        self.s.add(letter)
        if element.link is not None:
            link = replace(element.link, label=element.thing)
            self.nav(cx - BADGE / 2, cy - BADGE / 2, BADGE, BADGE, link)

    def badge_text(self, element: strip.Placed, cx: float, cy: float) -> None:
        """A badge's short name right of it, its values under the name; each left out
        only when it would touch a tag, a badge or another badge's text, or leave
        the strip."""
        x = cx + BADGE / 2 + GAP / 2
        name = element.short if element.short is not None else _short(element.thing)
        w = text_width(name, SMALL_SIZE)
        box = (x, cy - VALUE_STEP, w, VALUE_STEP)
        if self.free(box):
            self.text(x, cy - VALUE_STEP, name, size=SMALL_SIZE, w=w, h=VALUE_STEP)
            self.taken.append(box)
        for i, channel in enumerate(self.affordable(element)):
            top = cy + i * VALUE_STEP
            box = (x, top, BADGE_VALUE_WIDTH, VALUE_STEP)
            if self.free(box):
                self.value(x, top, BADGE_VALUE_WIDTH, channel, align="left")
                self.taken.append(box)

    def free(self, box: tuple[float, float, float, float]) -> bool:
        """Whether *box* stays inside the strip and touches nothing in :attr:`taken`."""
        x, y, w, h = box
        if x + w > MARGIN + STRIP_WIDTH:
            return False
        return all(
            x + w <= ox or ox + ow <= x or y + h <= oy or oy + oh <= y
            for ox, oy, ow, oh in self.taken
        )

    def symbol_rule(self, element: strip.Placed) -> tuple[Ladder | None, list[str]]:
        """The live rule of a strip element (the worst of its channels, as a roll-up
        node reads them) and the needs it hits; ``None`` when it has no live rule."""
        if not element.channels:
            return None, []
        lights = [describe_light(self.graph, c) for c in element.channels]
        rule = ladder(self.graph, lights, node=True)
        if isinstance(rule, Ladder) and rule.arms and not self.budget.take([ring_cost(rule)]):
            rule = "aggregate state"
        tags = [_need("state light")]
        if rule == "aggregate state":
            return None, [*tags, _need("aggregate state"), _AGGREGATE_SYMBOL]
        if isinstance(rule, str):
            self.s.unknown += 1
            return None, [*tags, _need(rule), _UNKNOWN_TAG + rule]
        if rule.unwritten:
            tags.append(_need("label condition"))
        return (rule if rule.arms else None), tags

    def symbol(
        self, element: strip.Placed, x: float, y: float, unit: float, level: int = 0
    ) -> None:
        """A device's symbol: filled with its type colour, never with its state; its
        state is a ring around it and a word above (:meth:`marked`)."""
        shapes = symbols.coloured(
            element.symbol,
            x,
            y,
            unit=unit,
            fill=self.colours.get(element.symbol),
            dashed=not element.channels,
        )
        for node in shapes:
            node["props"][_TOOLTIP] = element.thing
        self.marked(element, shapes, level)

    def marked(self, element: strip.Placed, shapes: list[dict], level: int = 0) -> None:
        """*shapes*, and around them the ring and the tag that show the worst state of
        the element's channels live. A thing with no rule gets neither (its outline is
        dashed); one whose rule cannot be written or is past the budget gets them in
        the aggregate or unknown look, with no rules."""
        for node in shapes:
            self.s.add(node)
        if not element.channels:
            return
        left = min(n["geometry"]["x"] for n in shapes)
        top = min(n["geometry"]["y"] for n in shapes)
        right = max(n["geometry"]["x"] + n["geometry"]["width"] for n in shapes)
        bottom = max(n["geometry"]["y"] + n["geometry"]["height"] for n in shapes)
        live, tags = self.symbol_rule(element)
        static = "aggregate" if _AGGREGATE_SYMBOL in tags else "unknown"
        outline = symbols.ring(left, top, right, bottom, RING_PAD, static)
        outline["props"][_TOOLTIP] = element.thing
        above = top - RING_PAD - level * TAG_HEIGHT
        low, high = MARGIN + TAG_WIDTH / 2, MARGIN + STRIP_WIDTH - TAG_WIDTH / 2
        centre = min(max((left + right) / 2, low), high)  # the word stays on the page
        word = symbols.tag(centre, above, TAG_WIDTH, TAG_HEIGHT, SMALL_SIZE, static)
        g = word["geometry"]
        self.taken.append((g["x"], g["y"], g["width"], g["height"]))
        rules: list[dict] = []
        word_rules: list[dict] = []
        if live is not None:
            looks = {state: symbols.ring_look(state) for state in symbols.STATES}
            rules = [
                self.s.rule(
                    f"{element.thing} ring {prop}",
                    prop,
                    live,
                    {st: looks[st][prop] for st in symbols.STATES},
                )
                for prop in symbols.RING_PROPS
            ]
            prop = symbols.TAG_PROP
            word_rules = [
                self.s.rule(
                    f"{element.thing} tag",
                    prop,
                    live,
                    {st: looks[st][prop] for st in symbols.STATES},
                )
            ]
        self.s.add(outline, rules=rules, tags=tags)
        self.s.add(word, rules=word_rules)

    def locator(self, beam: Sequence[strip.Placed], extent: tuple[float, float]) -> None:
        """The to-scale bar under a row: its caption, a mark per symbol where the
        symbol truly is, a leader from each symbol's name down to its mark, and
        the metres."""
        lo, hi = extent
        span = hi - lo
        self.text(MARGIN, self.y, _LOCATOR_CAPTION, size=SMALL_SIZE, fg=MUTED_TEXT)
        leaders_top = self.y + ROW
        bar = leaders_top + LEADER_DROP + LOCATOR_BIG
        self.line([(MARGIN, bar), (MARGIN + STRIP_WIDTH, bar)], MUTED_TEXT, LOCATOR_BAR)
        shifted = [StripElement(p.thing, None, p.s_m - lo, p.symbol, p.lane, None) for p in beam]
        marks = strip.locator(shifted, span, STRIP_WIDTH)
        for p, (thing, x) in zip(beam, marks, strict=True):
            big = p.symbol in _MAGNETS
            reach = LOCATOR_BIG if big else LOCATOR_MARK
            colour, width = (PAGE_TEXT, AXIS_WIDTH) if big else (MUTED_TEXT, LINE_WIDTH)
            self.line([(MARGIN + x, bar - reach), (MARGIN + x, bar)], colour, width, tooltip=thing)
            self.line([(MARGIN + p.x, leaders_top), (MARGIN + x, bar - reach)], MUTED_TEXT)
        top = bar + LOCATOR_BAR
        for metres, x in _ticks(lo, hi):
            self.line([(MARGIN + x, top), (MARGIN + x, top + TICK)], MUTED_TEXT)
            label = f"{metres:g}"
            w = text_width(label, SMALL_SIZE)
            self.text(MARGIN + x - w / 2, top + TICK, label, size=SMALL_SIZE, w=w, fg=MUTED_TEXT)
        self.y = top + TICK + ROW + ROW_GAP

    def legend(self, elements: Sequence[StripElement]) -> None:
        """Each symbol the strip draws, as it is drawn, and its name."""
        present = {e.symbol if e.symbol in symbols.WIDTH_UNITS else "generic" for e in elements}
        above = {e.symbol for e in elements if e.lane == "above"}
        x = MARGIN
        cy = self.y + LEGEND_ROW / 2
        for kind in symbols.KINDS:
            if kind not in present:
                continue
            word = SYMBOL_WORDS[kind]
            w = BADGE if kind in above else symbols.WIDTH_UNITS[kind] * LEGEND_SYMBOL
            room = w + GAP + text_width(word, SMALL_SIZE)
            if x + room > MARGIN + STRIP_WIDTH:
                x = MARGIN
                self.y += LEGEND_ROW
                cy = self.y + LEGEND_ROW / 2
            if kind in above:
                drawn = symbols.badge(x + w / 2, cy, BADGE, kind, SMALL_SIZE)
                fill = self.colours.get(kind)
                drawn[0]["props"].update(
                    {
                        "fillColor": fill if fill is not None else PANEL_BACKGROUND,
                        "lineColor": PAGE_TEXT,
                    }
                )
                if fill is not None:
                    drawn[1]["props"]["foregroundColor"] = PAGE_BACKGROUND
            else:
                drawn = symbols.coloured(
                    kind, x + w / 2, cy, unit=LEGEND_SYMBOL, fill=self.colours.get(kind)
                )
                top = min(n["geometry"]["y"] for n in drawn)
                bottom = max(n["geometry"]["y"] + n["geometry"]["height"] for n in drawn)
                for node in drawn:  # a quadrupole sits off its axis: centre it on the row
                    node["geometry"]["y"] += cy - (top + bottom) / 2
            for node in drawn:
                self.s.add(node)
            right = self.text(x + w + GAP, cy - ROW / 2, word, size=SMALL_SIZE, fg=MUTED_TEXT)
            x = right + LEGEND_GAP
        self.y += LEGEND_ROW + ROW_GAP

    def chip(
        self, x: float, y: float, label: str, link: Link | None, node: Node | None = None
    ) -> float:
        """A name in a box, with its node's light when given, opening *link*; its right edge."""
        lit = node is not None
        w = text_width(label, SMALL_SIZE) + (CHIP_LIGHT + GAP if lit else 0) + GAP
        self.s.widget(
            "rectangle",
            x,
            y,
            w,
            CHIP_HEIGHT,
            {
                "lineColor": BOX_BORDER,
                "lineWidth": LINE_WIDTH,
                "fill": True,
                "fillColor": FIELD_BACKGROUND,
            },
        )
        inner = x + GAP / 2
        if node is not None:
            light_y = y + (CHIP_HEIGHT - ROW) / 2
            if node.no_rule:
                self.no_rule(inner, light_y, CHIP_LIGHT)
            else:
                self.node_light(inner, light_y, node, CHIP_LIGHT)
            inner += CHIP_LIGHT + GAP / 2
        self.text(inner, y, label, size=SMALL_SIZE, h=CHIP_HEIGHT)
        if link is not None:
            self.nav(x, y, w, CHIP_HEIGHT, replace(link, label=link.label))
        return x + w

    def node_light(self, x: float, y: float, node: Node, size: float = LIGHT_SIZE) -> float:
        """One light over a node's channels, as a roll-up reads them; the aggregate look
        when the rule is too long for the format or past the screen's budget."""
        lights = [describe_light(self.graph, c) for c in node.channels]
        rule = ladder(self.graph, lights, node=True)
        if isinstance(rule, Ladder) and rule.arms and not self.budget.take([light_cost(rule)]):
            rule = "aggregate state"
        return self.light(x, y, lights, rule=rule, size=size, tile=True)

    def chips(self, x0: float, entries: Sequence[tuple[str, Link | None, Node | None]]) -> None:
        """Chips that wrap across the content width, from *x0*."""
        x = x0
        for label, link, node in entries:
            w = text_width(label, SMALL_SIZE) + (CHIP_LIGHT + GAP if node is not None else 0) + GAP
            if x > x0 and x + w > MARGIN + STRIP_WIDTH:
                x = x0
                self.y += CHIP_HEIGHT + CHIP_GAP
            x = self.chip(x, self.y, label, link, node) + CHIP_GAP
        self.y += CHIP_HEIGHT + CHIP_GAP

    def tray(self, item: Strip) -> None:
        """What our lists hold and the strip cannot place, in a dashed frame of chips."""
        top = self.y
        self.y += TRAY_PAD
        self.text(MARGIN + TRAY_PAD, self.y, _TRAY_TITLE.upper(), size=SMALL_SIZE, fg=MUTED_TEXT)
        if item.others is not None:
            self.quiet_link(MARGIN + STRIP_WIDTH - TRAY_PAD, self.y, item.others)
        self.y += ROW
        # the heading says "not placed": a reason only when it says more
        entries = [
            (e.thing if e.reason == NO_POSITION else f"{e.thing} · {e.reason}", e.link, None)
            for e in item.tray
        ]
        self.chips(MARGIN + TRAY_PAD, entries)
        self.y += TRAY_PAD - CHIP_GAP
        self.box(MARGIN, top, STRIP_WIDTH, self.y - top, line=MUTED_TEXT, dashed=True)
        self.y += GAP

    def band(self, band: Band) -> None:
        """The key numbers: an even grid of tiles across the content width, in the fewest
        rows whose tiles are :data:`MIN_TILE_WIDTH` wide, the earlier rows one tile
        longer when the count does not divide; each row spans the full width."""
        n = len(band.tiles)
        rows = next(
            r for r in range(1, n + 1) if _tile_width(math.ceil(n / r)) >= MIN_TILE_WIDTH or r == n
        )
        base, extra = divmod(n, rows)
        sizes = [base + 1] * extra + [base] * (rows - extra)
        widths = [_tile_width(size) for size in sizes]
        inner = min(widths) - 2 * TILE_PAD
        titles = max(len(_wrap(t.title.upper(), inner, SMALL_SIZE, caps=True)) for t in band.tiles)
        captions = max(len(_wrap(self.caption(t), inner, SMALL_SIZE)) for t in band.tiles)
        height = 2 * TILE_PAD + (titles + captions) * SMALL_LINE + NUMBER_HEIGHT + 2 * ROW_GAP
        i = 0
        for size, w in zip(sizes, widths, strict=True):
            for column in range(size):
                x = MARGIN + column * (w + GAP)
                self.tile(x, self.y, w, height, band.tiles[i], f"m{i}", titles)
                i += 1
            self.y += height + GAP

    @staticmethod
    def caption(item: Tile) -> str:
        """The line under a tile's number as the data gives it (:meth:`tile` may replace it)."""
        if not item.channels and not item.pairs:
            return _NO_DATA
        return item.caption or ""

    def tile(
        self, x: float, y: float, w: float, h: float, item: Tile, name: str, titles: int
    ) -> None:
        """One key number: its title, the number (a channel, or a screen formula over
        the channels its row selects) with "of N" or its unit on its baseline, and up to
        two lines under it. A count of losses turns the tile's border bad while it is
        above zero. Text is wrapped, never left to the viewer to shrink."""
        tags = (f"tile: {item.title}",)
        frame = {
            "lineColor": BOX_BORDER,
            "lineWidth": LINE_WIDTH,
            "fill": True,
            "fillColor": FIELD_BACKGROUND,
        }
        left = x + TILE_PAD
        top = y + TILE_PAD
        inner = w - 2 * TILE_PAD
        number_top = top + titles * SMALL_LINE + ROW_GAP
        caption = self.caption(item)
        pv: str | None = None
        total = len(item.pairs) if item.measure == "count_in_tolerance" else len(item.channels)
        formula = None
        extra_tags: tuple[str, ...] = ()
        if not item.channels and not item.pairs:
            pass
        elif item.measure == "value":
            if len(item.channels) == 1:
                pv = item.channels[0]
                if not self.budget.take([(0, frozenset(item.channels))]):
                    pv, caption = None, _NEEDS_SERVER
                    extra_tags = (_need("server-side aggregate"),)
            else:
                caption = _MANY_VALUES.format(len(item.channels))
        else:
            formula = metric_formula(name, item)
            reads = frozenset(formula["bindings"].values())
            fits = len(formula["expression"]) <= RULE_EXPRESSION_LIMIT
            if fits and self.budget.take([(1, reads)]):
                self.s.formulas.append(formula)
                pv = _FORMULA_PREFIX + name
            else:
                formula, caption = None, _NEEDS_SERVER
                extra_tags = (_need("server-side aggregate"),)
        rules: list[dict] = []
        if formula is not None and item.measure in ("count_false", "count_in_tolerance"):
            lost = "{0} > 0" if item.measure == "count_false" else f"{{0}} < {total}"
            rule = {
                "pvs": [{"name": pv, "trigger": True}],
                "conditions": [{"expression": lost, "value": symbols.ALARM_MAJOR}],
                "default": BOX_BORDER,
            }
            if len(engine_expression(rule, {name: formula})) <= RULE_EXPRESSION_LIMIT:
                self.s._rules += 1
                rules = [
                    {
                        "id": f"r-{self.s._rules:04d}",
                        "name": f"{item.title} border",
                        "targetProperty": "lineColor",
                        **rule,
                    }
                ]
        self.s.widget("rectangle", x, y, w, h, frame, rules=rules, tags=(*tags, *extra_tags))
        for i, line in enumerate(_wrap(item.title.upper(), inner, SMALL_SIZE, caps=True)):
            w_caps = caps_width(line, SMALL_SIZE)
            self.text(
                left, top + i * SMALL_LINE, line, size=SMALL_SIZE, w=w_caps, h=SMALL_LINE,
                fg=MUTED_TEXT, tags=tags,
            )  # fmt: skip
        counted = item.measure in _COUNTS
        beside = f"of {total}" if counted else (item.unit or "")
        beside_w = text_width(beside, SMALL_SIZE) if beside else 0
        if pv is None:
            dash_w = text_width(_DASH, BIG_SIZE)
            self.text(
                left, number_top, _DASH, size=BIG_SIZE, w=dash_w, h=NUMBER_HEIGHT, fg=MUTED_TEXT,
                tags=tags,
            )  # fmt: skip
        else:
            digits = max(len(str(total)), COUNT_DIGITS) if counted else VALUE_DIGITS
            number_w = min(digits * CHAR_WIDTH * BIG_SIZE + NUMBER_PAD, inner - beside_w - GAP)
            props = {
                "pv": pv,
                "fontSize": BIG_SIZE,
                "foregroundColor": PAGE_TEXT,
                "backgroundColor": TRANSPARENT,
                "showUnits": False,
                "alarmSensitive": False,
                "alarmBorder": False,
                "alarmSensitiveBorder": False,
            }
            if counted:
                props["precision"] = 0
            self.s.widget("pv-label", left, number_top, number_w, NUMBER_HEIGHT, props, tags=tags)
            if beside:  # on the number's baseline: its line's middle below the number's
                base = number_top + NUMBER_HEIGHT - SMALL_LINE - ROW_GAP
                self.small(left + number_w + ROW_GAP, base, beside, fg=MUTED_TEXT, tags=tags)
        below = number_top + NUMBER_HEIGHT + ROW_GAP
        for i, line in enumerate(_wrap(caption, inner, SMALL_SIZE)):
            self.small(left, below + i * SMALL_LINE, line, fg=MUTED_TEXT, tags=tags)

    def small(self, x: float, y: float, text: str, *, fg=PAGE_TEXT, tags=()) -> float:
        """One line of small text, its box as wide as the text."""
        w = text_width(text, SMALL_SIZE)
        return self.text(x, y, text, size=SMALL_SIZE, w=w, h=SMALL_LINE, fg=fg, tags=tags)

    def group(self, group: Group) -> None:
        """Things of one kind: chips that wrap, or summary rows in columns."""
        if group.form == "rows":
            self.summary(group)
            return
        self.text(MARGIN, self.y, group.title.upper(), size=SMALL_SIZE, fg=MUTED_TEXT)
        self.y += ROW
        entries = [(n.label, n.link, n) for n in group.nodes]
        if group.more is not None:
            entries.append((group.more.label, group.more, None))
        self.chips(MARGIN, entries)

    def summary(
        self,
        group: Group,
        x0: float = MARGIN,
        width: float = STRIP_WIDTH,
        columns: int = SUMMARY_COLUMNS,
    ) -> None:
        """One row per node: its light, its name, and right-aligned what it reads."""
        column = (width - (columns - 1) * GAP) / columns
        per = max(1, math.ceil(len(group.nodes) / columns))
        top = self.y
        for i, node in enumerate(group.nodes):
            x = x0 + (i // per) * (column + GAP)
            y = top + (i % per) * SUMMARY_ROW
            h = SUMMARY_ROW - ROW_GAP
            self.s.widget(
                "rectangle",
                x,
                y,
                column,
                h,
                {
                    "lineColor": BOX_BORDER,
                    "lineWidth": LINE_WIDTH,
                    "fill": True,
                    "fillColor": FIELD_BACKGROUND,
                },
            )
            row_y = y + (h - ROW) / 2
            if node.no_rule:
                self.no_rule(x + GAP, row_y)
            else:
                self.node_light(x + GAP, row_y, node)
            self.text(x + GAP + LIGHT_SIZE + GAP, row_y, node.label)
            count = len(node.channels)
            note = _NO_RULE_YET if node.no_rule else _READS.format(count, "" if count == 1 else "s")
            w = text_width(note, SMALL_SIZE)
            self.text(x + column - GAP - w, row_y, note, size=SMALL_SIZE, w=w, fg=MUTED_TEXT)
            if node.link is not None:
                self.nav(x, y, column, h, node.link)
        self.y = top + per * SUMMARY_ROW + ROW_GAP

    # -- picture and elevation ------------------------------------------------

    def picture(self, picture: Picture) -> None:
        """A device class's picture: its outline as the table states it (rows of kind
        :data:`OUTLINE_KIND`, joined in file order) and its parts where the table
        puts them, each named in a column on its side, joined to it by a leader
        line; no two leaders cross (:func:`picture_labels`). Without an outline in
        the data there is no picture: the parts are listed (:meth:`part_list`)."""
        if not _outlined(picture):
            self.part_list(picture)
            return
        top = self.y + ROW
        self.text(MARGIN, self.y, f"{picture.view} view", size=SMALL_SIZE)
        left_col = MARGIN
        x0 = MARGIN + PART_LABEL_WIDTH + PICTURE_PAD
        right_col = x0 + PICTURE_SIZE + PICTURE_PAD
        outline = [p for p in picture.parts if p.kind == OUTLINE_KIND]
        parts = [p for p in picture.parts if p.kind != OUTLINE_KIND]
        coords = [_picture_xy(picture.view, part) for part in picture.parts]
        reach = max([max(abs(a), abs(b)) for a, b in coords] or [0]) or 1
        scale = (PICTURE_SIZE / 2 - PICTURE_PAD) / reach
        cx, cy = x0 + PICTURE_SIZE / 2, top + PICTURE_SIZE / 2

        def at(part) -> tuple[float, float]:
            a, b = _picture_xy(picture.view, part)
            return cx + a * scale, cy - b * scale

        points = [at(part) for part in outline]
        self.line([*points, points[0]], PAGE_TEXT)
        placed = [(part.part, *at(part)) for part in parts]
        anchors = (left_col + PART_LABEL_WIDTH, right_col)
        labels = picture_labels(placed, cx, top, top + PICTURE_SIZE, anchors)
        for part, (name, px, py), (side, label_y) in zip(parts, placed, labels, strict=True):
            channel = part.channel  # stated by the pictures row; no table states one yet
            if channel is not None and self.graph.rule(channel):
                self.light(px - LIGHT_SIZE / 2, py - ROW / 2, [describe_light(self.graph, channel)])
            else:
                half = PART_MARKER / 2
                props = {"lineColor": PAGE_TEXT, "lineWidth": LINE_WIDTH, "fill": False}
                props[_TOOLTIP] = part.part
                self.s.widget("ellipse", px - half, py - half, PART_MARKER, PART_MARKER, props)
            lx = left_col if side == "left" else right_col
            anchor = anchors[0] if side == "left" else anchors[1]
            self.line([(px, py), (anchor, label_y + ROW / 2)], MUTED_TEXT)
            align = "right" if side == "left" else "left"
            self.text(lx, label_y, name, size=SMALL_SIZE, w=PART_LABEL_WIDTH, align=align)
        bottom = max([top + PICTURE_SIZE, *(y + ROW for _side, y in labels)])
        self.y = bottom + ROW + ROW_GAP

    def part_list(self, picture: Picture) -> None:
        """A class's parts with no outline in the data: chips grouped by the kind the
        table states, a small heading each, each chip with its part's light (the
        no-rule look while no row names the part's channel). A part listed by an
        earlier view of the class is not listed again."""
        kinds: dict[str, list[tuple[str, Link | None, Node | None]]] = {}
        for part in picture.parts:
            if part.part in self.parts_listed:
                continue
            self.parts_listed.add(part.part)
            channel = part.channel
            ruled = channel is not None and bool(self.graph.rule(channel))
            node = Node(part.part, None, (channel,) if ruled else (), not ruled)  # type: ignore[arg-type]
            kinds.setdefault(part.kind, []).append((part.part, None, node))
        for kind, entries in kinds.items():
            self.text(self.x0, self.y, kind.replace("_", " "), size=SMALL_SIZE, fg=MUTED_TEXT)
            self.y += ROW
            self.chips(self.x0, entries)

    def elevation(self, elevation: Elevation) -> None:
        """A rack as it stands: its frame, each chassis a bar at its true unit spanning
        the rack, the unit numbers left, the name inside the bar and one light over
        the unit's channels at its right end, opening the card. Right of it the
        classes counted and the sources; under it the devices not at a unit."""
        tops = [_unit_top(unit) for unit in elevation.units]
        height = max(tops, default=0)
        frame_x = MARGIN + RACK_LABEL_WIDTH
        bars_x = frame_x + RACK_PAD
        top = self.y
        rack_top = top + RACK_PAD
        self.s.widget(
            "rectangle",
            frame_x,
            top,
            RACK_WIDTH + 2 * RACK_PAD,
            height * RACK_U + 2 * RACK_PAD,
            {"lineColor": PAGE_TEXT, "lineWidth": AXIS_WIDTH, "fill": False},
        )
        labelled: set[int | None] = set()
        lanes = rack_lanes(elevation.units)
        for unit, unit_top, (lane, count) in zip(elevation.units, tops, lanes, strict=True):
            w = RACK_WIDTH / count  # chassis whose units overlap share the rack's width
            x = bars_x + lane * w
            y = rack_top + (height - unit_top) * RACK_U
            h = (unit.size_u or 1) * RACK_U
            if unit.position not in labelled:  # at the unit stated: the bar's lowest
                labelled.add(unit.position)
                label = f"U{unit.position}"
                self.text(
                    MARGIN, y + h - RACK_U, label, size=SMALL_SIZE, w=RACK_LABEL_WIDTH - GAP,
                    h=RACK_U,
                )  # fmt: skip
            self.chassis(x, y, w, h, unit)
        bottom = top + height * RACK_U + 2 * RACK_PAD
        self.rack_facts(elevation, top)
        self.y = max(bottom, self.facts_bottom) + SECTION_GAP
        if elevation.unpositioned:
            self.not_at_a_unit(elevation.unpositioned)

    def chassis(self, x: float, y: float, w: float, h: float, unit: RackUnit) -> None:
        """One bar: its name inside, the light at its right end, a link over it."""
        self.box(x, y, w, h, fill=FIELD_BACKGROUND, dashed=unit.conflicting)
        light_x = x + w - CHIP_LIGHT - GAP / 2
        light_y = y + (min(h, RACK_U) - ROW) / 2
        self.unit_light(light_x, light_y, unit)
        name = unit.link.label if isinstance(unit.link, Link) else unit.link.text
        if unit.standing == _CONTESTED:
            name = f"{name} · {_CONTESTED}"
        x_text = x + GAP / 2
        if unit.conflicting:
            x_text = self.text(x_text, y, symbols.CONFLICT_MARK, size=SMALL_SIZE, h=RACK_U)
        room = light_x - GAP / 2 - x_text
        name = _cut(name, room, SMALL_SIZE)
        if name:
            w_name = min(text_width(name, SMALL_SIZE), room)
            self.text(x_text, y, name, size=SMALL_SIZE, w=w_name, h=RACK_U)
        if isinstance(unit.link, Link):
            self.nav(x, y, w, h, unit.link)

    def unit_light(self, x: float, y: float, unit: RackUnit) -> None:
        """One light over a unit's channels (and its parts'), the no-rule look with none."""
        if not unit.lights:
            self.no_rule(x, y, CHIP_LIGHT)
        else:
            self.light(x, y, unit.lights, size=CHIP_LIGHT)

    def rack_facts(self, elevation: Elevation, top: float) -> None:
        """Right of the rack: the classes of its devices counted, how many stand at a
        unit, and the documents the units come from."""
        x0, right = self.x0, self.right
        self.x0, self.right = MARGIN + RACK_COLUMN, MARGIN + STRIP_WIDTH
        self.y = top
        every = [*elevation.units, *elevation.unpositioned]
        classes: Counter[str] = Counter()
        for device in dict.fromkeys(u.link.target.id for u in every if isinstance(u.link, Link)):
            try:
                cls = self.graph.thing(device, "device").cls
            except KeyError:
                cls = None
            classes[cls or _NO_CLASS] += 1
        if classes:
            self.heading(_BY_CLASS)
            for cls, count in sorted(classes.items(), key=lambda kv: (-kv[1], kv[0])):
                self.text(self.x0, self.y, cls, size=SMALL_SIZE, h=SMALL_LINE)
                n = str(count)
                w = text_width(n, SMALL_SIZE)
                self.text(self.right - w, self.y, n, size=SMALL_SIZE, w=w, h=SMALL_LINE)
                self.y += SMALL_LINE
            self.y += ROW_GAP
        counts = _RACK_COUNTS.format(len(elevation.units), len(elevation.unpositioned))
        self.text(self.x0, self.y, counts, size=SMALL_SIZE, fg=MUTED_TEXT)
        self.next_row()
        documents: Counter[str] = Counter()
        for unit in every:  # a document per unit once, however many of its rows say it
            documents.update(
                list(
                    dict.fromkeys(
                        " ".join(p for p in (c.document, c.version) if p) for c in unit.citations
                    )
                )
            )
        if documents:
            self.heading(_SOURCES)
            ranked = sorted(documents.items(), key=lambda kv: (-kv[1], kv[0]))
            for document, count in ranked[:RACK_SOURCES]:
                text = _cut(_UNITS_FROM.format(document, count), self.right - self.x0, SMALL_SIZE)
                self.text(self.x0, self.y, text, size=SMALL_SIZE, h=SMALL_LINE, fg=MUTED_TEXT)
                self.y += SMALL_LINE
            if len(ranked) > RACK_SOURCES:
                more = _MORE_ROWS.format(len(ranked) - RACK_SOURCES)
                self.text(self.x0, self.y, more, size=SMALL_SIZE, h=SMALL_LINE, fg=MUTED_TEXT)
                self.y += SMALL_LINE
        self.facts_bottom = self.y
        self.x0, self.right = x0, right

    def not_at_a_unit(self, entries: Sequence[RackUnit]) -> None:
        """The devices in the rack with no unit: a framed list of chips, each with its
        light and the unit text a claim gives, under the rack."""
        first = len(self.s.nodes)
        top = self.y
        self.y += PANEL_PAD
        self.heading(_NOT_AT_A_UNIT)
        x = MARGIN
        for unit in entries:
            name = unit.link.label if isinstance(unit.link, Link) else unit.link.text
            label = f"{name} · unit {unit.unit_text}" if unit.unit_text else name
            w = text_width(label, SMALL_SIZE) + CHIP_LIGHT + 2 * GAP
            if x > MARGIN and x + w > MARGIN + STRIP_WIDTH:
                x = MARGIN
                self.y += CHIP_HEIGHT + CHIP_GAP
            self.box(x, self.y, w, CHIP_HEIGHT, fill=FIELD_BACKGROUND)
            self.unit_light(x + GAP / 2, self.y + (CHIP_HEIGHT - ROW) / 2, unit)
            self.text(x + CHIP_LIGHT + GAP, self.y, label, size=SMALL_SIZE, h=CHIP_HEIGHT)
            if isinstance(unit.link, Link):
                self.nav(x, self.y, w, CHIP_HEIGHT, unit.link)
            x += w + CHIP_GAP
        self.y += CHIP_HEIGHT + PANEL_PAD
        self.s.widget(
            "rectangle",
            MARGIN - PANEL_PAD / 2,
            top,
            STRIP_WIDTH + PANEL_PAD,
            self.y - top,
            {"lineColor": BOX_BORDER, "lineWidth": LINE_WIDTH, "fill": False},
        )
        self.s.nodes.insert(first, self.s.nodes.pop())
        self.y += GAP


def rack_lanes(units: Sequence[RackUnit]) -> list[tuple[int, int]]:
    """Each unit's lane and the lanes of its cluster: units whose spans overlap form a
    cluster, each takes the first lane free over its span, in order."""
    spans = [(unit.position or 0, _unit_top(unit)) for unit in units]
    order = sorted(range(len(units)), key=lambda i: (spans[i][0], i))
    lanes = [0] * len(units)
    cluster = [0] * len(units)
    clusters: list[list[int]] = []
    reach = -math.inf
    for i in order:
        low, high = spans[i]
        if low > reach:
            clusters.append([])
        reach = max(reach, high) if low <= reach else high
        members = clusters[-1]
        taken = {lanes[j] for j in members if spans[j][1] >= low}
        lanes[i] = next(n for n in itertools.count() if n not in taken)
        members.append(i)
        cluster[i] = len(clusters) - 1
    width = [max(lanes[j] for j in members) + 1 for members in clusters]
    return [(lanes[i], width[cluster[i]]) for i in range(len(units))]


def _unit_top(unit: RackUnit) -> int:
    """The highest unit a chassis takes: its stated unit is its lowest."""
    return (unit.position or 0) + int(unit.size_u or 1) - 1


def picture_labels(
    parts: Sequence[tuple[str, float, float]],
    centre: float,
    top: float,
    bottom: float,
    anchors: tuple[float, float] | None = None,
) -> list[tuple[str, float]]:
    """Where each part's name goes: ``(side, y)`` per ``(name, x, y)`` part, in order.

    A part left of *centre* is named in the left column, any other in the
    right one. In each column the names go in order of their part's ``y``,
    each at its part's height or just under the name above it, so no two
    overlap; a column that runs past *bottom* is pushed back up, never
    above *top*. With *anchors* (the x where the left and the right
    column's leaders end), two leaders of a column that cross swap their
    names' places until none cross: each swap shortens the leaders, so it ends.
    """
    slots: list[tuple[str, float]] = [("", 0.0)] * len(parts)
    for side in ("left", "right"):
        mine = [i for i, (_n, x, _y) in enumerate(parts) if (x < centre) == (side == "left")]
        mine.sort(key=lambda i: (parts[i][2], i))
        ys: list[float] = []
        for i in mine:
            want = parts[i][2] - ROW / 2
            ys.append(max(want, ys[-1] + ROW + ROW_GAP) if ys else want)
        for k in range(len(ys) - 1, -1, -1):  # push back up from the bottom
            limit = bottom - ROW if k == len(ys) - 1 else ys[k + 1] - ROW - ROW_GAP
            ys[k] = min(ys[k], limit)
        if ys and ys[0] < top:  # a column taller than the picture runs on downward
            ys[0] = top
            for k in range(1, len(ys)):
                ys[k] = max(ys[k], ys[k - 1] + ROW + ROW_GAP)
        if anchors is not None:
            ax = anchors[0] if side == "left" else anchors[1]
            _uncross(mine, ys, parts, ax)
        for i, y in zip(mine, ys, strict=True):
            slots[i] = (side, y)
    return slots


def _uncross(
    mine: list[int], ys: list[float], parts: Sequence[tuple[str, float, float]], ax: float
) -> None:
    """Swap the label heights *ys* of the parts *mine* until no two leaders from a part to
    (*ax*, its label's middle) cross."""

    def leader(k: int) -> tuple[tuple[float, float], tuple[float, float]]:
        _n, px, py = parts[mine[k]]
        return (px, py), (ax, ys[k] + ROW / 2)

    for _round in range(len(mine) ** 2 + 1):
        swapped = False
        for a, b in itertools.combinations(range(len(mine)), 2):
            if _crossing(*leader(a), *leader(b)):
                ys[a], ys[b] = ys[b], ys[a]
                swapped = True
        if not swapped:
            return


def _crossing(p1, p2, q1, q2) -> bool:
    """Whether the segments p1-p2 and q1-q2 cross at a point inside both."""

    def turn(a, b, c) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = turn(q1, q2, p1), turn(q1, q2, p2)
    d3, d4 = turn(p1, p2, q1), turn(p1, p2, q2)
    return d1 * d2 < 0 and d3 * d4 < 0


def _outlined(picture: Picture) -> bool:
    """Whether the table states an outline for *picture*."""
    return any(part.kind == OUTLINE_KIND for part in picture.parts)


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


def _short(thing: str) -> str:
    """A symbol's name: the last component of its id."""
    parts = [part for part in _ID_PARTS.split(thing) if part]
    return parts[-1] if parts else thing


def _cut(text: str, width: float, size: float) -> str:
    """*text* cut to fit *width* at *size* by :func:`text_width`, an ellipsis marking the cut."""
    if text_width(text, size) <= width:
        return text
    room = math.floor((width - TEXT_PAD) / (size * CHAR_WIDTH))
    return text[: room - 1] + _ELLIPSIS if room > 1 else ""


def _chip(title: str, item: object) -> tuple[str, Link | None]:
    """A card's chip for a related thing: "<relation>: <thing>", its card when it has one."""
    if isinstance(item, Link):
        return f"{title}: {item.label}", item
    if isinstance(item, Placement):
        return f"{title}: {item.link.label}", item.link
    if isinstance(item, Fact):
        return f"{title}: {item.text}", None
    return title, None


def caps_width(text: str, size: float) -> float:
    """:func:`text_width` of *text* written in capitals."""
    return text_width(text, size) * CAPS_WIDTH / CHAR_WIDTH


def _wrap(
    text: str, width: float, size: float, lines: int = TILE_LINES, caps: bool = False
) -> list[str]:
    """*text* in at most *lines* lines that each fit *width* at *size*, broken at
    spaces; the last line cut with an ellipsis when the rest does not fit. *caps*:
    measured as capitals (:func:`caps_width`)."""
    fit = width * CHAR_WIDTH / CAPS_WIDTH if caps else width  # in text_width's measure
    out: list[str] = []
    words = text.split()
    while words and len(out) < lines:
        line = words.pop(0)
        while words and text_width(f"{line} {words[0]}", size) <= fit:
            line = f"{line} {words.pop(0)}"
        if len(out) == lines - 1 and words:
            line = _cut(" ".join([line, *words]), fit, size)
            words = []
        out.append(_cut(line, fit, size))
    return out or [""]


def _tile_width(per_row: int) -> float:
    """The width of each of *per_row* tiles spanning the content width."""
    return (STRIP_WIDTH - (per_row - 1) * GAP) / per_row


def strip_labels(placed: Sequence[strip.Placed]) -> list[tuple[str, float, float, int]]:
    """Each symbol's name under it: ``(text, left, width, level)``, in beam order.

    The name is the last component of the id, cut to the symbol's budget
    (:data:`STRIP_LABEL_WIDTH`, or its width when wider) and centred on it. It
    goes on the first of two levels where it clears the last name there by
    :data:`STRIP_LABEL_GAP`; when neither is clear, on the level that ends
    first, cut to the room left of its centre's reach. A name with no room is
    left out (the symbol's box still shows the id). By :func:`text_width`,
    so no two names of one level overlap.
    """
    ends = [-math.inf, -math.inf]
    found: list[tuple[str, float, float, int]] = []
    for p in placed:
        name = p.short if p.short is not None else _short(p.thing)
        text = _cut(name, max(p.width, STRIP_LABEL_WIDTH), STRIP_LABEL_SIZE)
        w = text_width(text, STRIP_LABEL_SIZE)
        left = p.x - w / 2
        level = next((n for n in (0, 1) if left >= ends[n] + STRIP_LABEL_GAP), None)
        if level is None:
            level = 0 if ends[0] <= ends[1] else 1
            left = ends[level] + STRIP_LABEL_GAP
            text = _cut(name, p.x + w / 2 - left, STRIP_LABEL_SIZE)
            if not text:
                continue
            w = text_width(text, STRIP_LABEL_SIZE)
        ends[level] = left + w
        found.append((text, left, w, level))
    return found


def metric_formula(name: str, item: Tile) -> dict[str, Any]:
    """The screen formula of a key number: its expression over ``{key}`` tokens and the
    channel each key binds. Counts are sums of ``1 * (test)``, so a true test counts one
    whatever type the engine gives a comparison."""
    bindings: dict[str, str] = {}
    if item.measure == "count_in_tolerance":
        terms = []
        for i, (setpoint, readback, term, amount) in enumerate(item.pairs):
            bindings[f"s{i}"], bindings[f"r{i}"] = setpoint, readback
            gap = f"abs({{s{i}}} - {{r{i}}})"
            bound = _number(amount) if term == "absolute" else f"{_number(amount)} * abs({{s{i}}})"
            terms.append(f"1 * ({gap} <= {bound})")
        expression = " + ".join(terms)
    else:
        keys = [f"c{i}" for i in range(len(item.channels))]
        bindings = dict(zip(keys, item.channels, strict=True))
        tokens = [f"{{{key}}}" for key in keys]
        if item.measure == "count_true":
            expression = " + ".join(f"1 * ({t} != 0)" for t in tokens)
        elif item.measure == "count_false":
            expression = " + ".join(f"1 * ({t} == 0)" for t in tokens)
        elif item.measure == "rms":
            expression = f"sqrt(({' + '.join(f'{t} ** 2' for t in tokens)}) / {len(tokens)})"
        else:  # max, min: a reduction over one tuple
            expression = f"{item.measure}(({', '.join(tokens)},))"
    return {"name": name, "kind": "scalar", "expression": expression, "bindings": bindings}


def stem_levels(xs: Sequence[float], room: float = BADGE + BADGE_GAP) -> list[int]:
    """The level of each of a row of things at *xs* (in order), each *room* wide: the
    low one, unless it would touch the last one there, then the high one; when both
    would, the level whose last thing is further away. Badges' stems, rings' tags."""
    last = [-math.inf, -math.inf]
    out = []
    for x in xs:
        level = next((n for n in (0, 1) if x - last[n] >= room), None)
        if level is None:
            level = 0 if last[0] <= last[1] else 1
        last[level] = x
        out.append(level)
    return out


def _strip_rows(
    elements: Sequence[StripElement], widths: Mapping[str, float]
) -> list[list[StripElement]]:
    """The strip in rows, each as long as fits at no less than :data:`MIN_STRIP_SCALE`."""

    def fits(chunk: Sequence[StripElement]) -> bool:
        return strip.arrange_scaled(chunk, widths, STRIP_WIDTH, fill=True)[1] >= MIN_STRIP_SCALE

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
    doc_id: str,
    colours: Mapping[str, str] = MappingProxyType({}),
) -> dict[str, Any]:
    """The view as a Screen-IR 1.1 document.

    Args:
        view: What the screen shows.
        graph: The panel graph, for the health rows the lights' rules are written from.
        screens: The screen file a link to a target opens, relative to this
            screen's directory.
        doc_id: The document's id; the caller names it, as it names the files
            links open (:func:`osprey.facility.panels.emit.screen_ids`).
        colours: The fill of each symbol on a strip, by symbol name (the
            drawing settings' ``colours``); a symbol not named is filled with
            the panel's ground.

    Returns:
        The document, as JSON-ready data. Equal inputs give equal documents.
    """
    return _Page(view, graph, screens, doc_id, colours).draw()
