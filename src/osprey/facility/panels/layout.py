"""Layout: a view description drawn as a Canopy Screen-IR 1.1 document.

This is the only module of the panels with geometry. :func:`render` walks a
:class:`~osprey.facility.panels.view.View` top to bottom, one section after
another, and draws every item with the pinned viewer's widgets. Colours are
the theme roles of :mod:`~osprey.facility.panels.symbols`.

States are the viewer's to compute. A light's rule over its channels, written from
their health rows (see :func:`ladder`), is a list of arms, each a screen formula
(:func:`arm_formulas`) that holds when the light is in that arm's state; each
prop of the light has a rule (:func:`light_rule`) whose conditions read those
formulas, first that holds wins, and set the prop to the state's look. The viewer
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

import itertools
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
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
    GridCell,
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
    "OUTLINE_KIND",
    "arm_formulas",
    "light_rule",
    "WIDGETS",
    "Ladder",
    "ladder",
    "aggregates",
    "needs",
    "render",
    "picture_labels",
    "engine_expression",
    "engine_length",
    "strip_labels",
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
UNIT_WIDTH = 44  # a value's unit, beside it
UNIT_GAP = 2
BUTTON_PAD = 16
MAX_BUTTON_WIDTH = 420
LIGHT_SIZE = 14
MARK_WIDTH = 14
LINE_WIDTH = 1
AXIS_WIDTH = 2
CELL_WIDTH = 110  # a checkout's column
NAME_WIDTH = 260  # a checkout's first column
GRID_CELL_WIDTH = 240  # one member of a set overview
STRIP_WIDTH = PAGE_WIDTH - 2 * MARGIN
MIN_STRIP_SCALE = 0.5  # below this a strip breaks into rows
SYMBOL_UNIT = 14  # one shape unit of a symbol, in px, at scale 1
STRIP_ABOVE = 40  # height of the lane above the beam, from the axis
STRIP_LEVEL = 16  # second level of the above lane, further up
STRIP_LABEL_TOP = 20  # a symbol's name, below the axis
STRIP_LABEL_STEP = 13  # the second level of names, further down
STRIP_LABEL_SIZE = 10
STRIP_LABEL_WIDTH = 72  # the most a symbol's name may take along the strip
STRIP_LABEL_GAP = 3  # between two names on one level
STRIP_BELOW = STRIP_LABEL_TOP + 2 * STRIP_LABEL_STEP + 4  # room under the axis
LOCATOR_GAP = 10
LOCATOR_BAR = 6
LOCATOR_MARK = 8  # how far a device's mark reaches above the bar
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
_PART = "part {} of {}"
_ID_PARTS = re.compile(r"[:/]")
_NO_OUTLINE = "outline not in the data"
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


def arm_formulas(rule: Ladder) -> list[tuple[str, dict[str, str]]]:
    """The screen formula of each arm of *rule*: its condition and its bindings.

    ``{i}`` of an arm is the variable ``v<i>``, bound to ``rule.channels[i]``;
    a formula binds the channels its arm reads, nothing else.
    """
    out = []
    for condition, _state in rule.arms:
        used = sorted({int(i) for i in _TOKEN.findall(condition)})
        out.append((_TOKEN.sub(r"v\1", condition), {f"v{i}": rule.channels[i] for i in used}))
    return out


def light_rule(rule: Ladder, values: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
    """One prop's rule over the arm formulas *names* (one per arm of *rule*).

    The first arm that holds gives its state's value, none holding gives
    ``unknown``'s, the default. The viewer inlines a formula once for each
    ``{N}`` that names it, so each arm is named once: successive arms of one
    value share a condition (``{0} or {1}``), and the arms after the last one
    whose value is not the default's are left out.
    """
    default = values["unknown"]
    groups: list[tuple[Any, list[int]]] = []
    for index, (_condition, state) in enumerate(rule.arms):
        value = values[state]
        if groups and groups[-1][0] == value:
            groups[-1][1].append(index)
        else:
            groups.append((value, [index]))
    while groups and groups[-1][0] == default:
        groups.pop()
    pvs: list[dict[str, Any]] = []
    conditions: list[dict[str, Any]] = []
    for value, arms in groups:
        tokens = []
        for arm in arms:
            tokens.append(f"{{{len(pvs)}}}")
            pvs.append({"name": _FORMULA_PREFIX + names[arm], "trigger": True})
        conditions.append({"expression": " or ".join(tokens), "value": value})
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


#: The width of a screen formula's name; every name has it, so a ladder's
#: expression has one length wherever it is drawn.
_FORMULA_NAME = "s{:05d}"


def engine_length(rule: Ladder) -> int:
    """The longest expression the engine receives for a light drawn with *rule*.

    The rules a light writes (one per prop, for every prop any look sets) are
    built by the writer's own functions and compiled by
    :func:`engine_expression`; the bound is their longest.
    """
    names = [_FORMULA_NAME.format(i + 1) for i in range(len(rule.arms))]
    formulas = {
        name: {"name": name, "expression": expression, "bindings": bindings}
        for name, (expression, bindings) in zip(names, arm_formulas(rule), strict=True)
    }
    looks = {state: symbols.look(state) for state in symbols.STATES}
    longest = 0
    for prop in (*symbols.LIGHT_PROPS, symbols.MARK_PROP):
        values = {state: looks[state][prop] for state in symbols.STATES}
        written = light_rule(rule, values, names)
        longest = max(longest, len(engine_expression(written, formulas)))
    return longest


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
    _formula_names: dict[tuple, str] = field(default_factory=dict)
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

    def formulas_of(self, ladder: Ladder) -> list[str]:
        """The names of *ladder*'s arm formulas, each distinct formula declared once."""
        names = []
        for expression, bindings in arm_formulas(ladder):
            key = (expression, tuple(bindings.items()))
            name = self._formula_names.get(key)
            if name is None:
                name = _FORMULA_NAME.format(len(self._formula_names) + 1)
                if len(name) != len(_FORMULA_NAME.format(1)):
                    raise ValueError("a screen holds more formulas than their names can number")
                self._formula_names[key] = name
                self.formulas.append(
                    {"name": name, "kind": "scalar", "expression": expression, "bindings": bindings}
                )
            names.append(name)
        return names

    def rule(self, name: str, target: str, ladder: Ladder, values: Mapping[str, Any]) -> dict:
        """One prop's rule: the value of the first arm that holds, else unknown's."""
        self._rules += 1
        return {
            "id": f"r-{self._rules:04d}",
            "name": name,
            "targetProperty": target,
            **light_rule(ladder, values, self.formulas_of(ladder)),
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
            "formulas": self.formulas,
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
        # writable widgets only on a device card's own signals and settings
        self.on_card = view.id.view == "card" and view.id.kind == "device"

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
        """A text label; returns its right edge."""
        w = text_width(text, size) if w is None else w
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
        self, points: Sequence[tuple[float, float]], colour, width=LINE_WIDTH, *, tooltip=None
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
            "arrowEnd": False,
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
        aggregate = rule == "aggregate state"
        look = "aggregate" if aggregate else "unknown"
        shape, mark = symbols.light(x, y + (ROW - size) / 2, size, look)
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
        right = x + size
        if not tile and any(item.inferred for item in lights):
            right = self.mark(right + GAP, y, symbols.INFERRED_MARK)
        return right

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
            self.s.widget(widget, x, y, w, ROW, {**common, "backgroundColor": FIELD_BACKGROUND})
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
                "alarmBorder": True,
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
        """A set's members: a row per place, its members across the page in beam order.

        A cell is the member's light, its name as a link and, for the set's
        primary pair, the readback and a deviation light; nothing in a cell
        is writable. At most :data:`MAX_TABLE_ROWS` cells are drawn.
        """
        tags = (_need("table"),)
        if grid.signal is not None:
            heading = f"{grid.signal.replace('_', ' ')}: readback and deviation"
            self.text(MARGIN, self.y, heading, size=SMALL_SIZE, w=STRIP_WIDTH, tags=tags)
            self.next_row()
        height = 2 * ROW + ROW_GAP if grid.signal is not None else ROW
        drawn = left_out = 0
        for row in grid.rows:
            self.ref(MARGIN, self.y, row.label)
            self.next_row()
            x = MARGIN + INDENT
            for cell in row.cells:
                if drawn >= MAX_TABLE_ROWS:
                    left_out += 1
                    continue
                if x + GRID_CELL_WIDTH > MARGIN + STRIP_WIDTH:
                    self.y += height + ROW_GAP
                    x = MARGIN + INDENT
                self.grid_cell(x, self.y, cell)
                x += GRID_CELL_WIDTH + GAP
                drawn += 1
            self.y += height + ROW_GAP
        if left_out:
            self.left_out(left_out)

    def grid_cell(self, x0: float, y: float, cell: GridCell) -> None:
        x = x0
        if cell.channels:
            lights = [describe_light(self.graph, c) for c in cell.channels]
            x = self.light(x, y, lights, rule=ladder(self.graph, lights, node=True))
        else:
            x = self.no_rule(x, y)
        room = x0 + GRID_CELL_WIDTH - (x + GAP)
        if isinstance(cell.member, Link):
            self.button(x + GAP, y, cell.member, width=room)
        else:
            self.text(x + GAP, y, cell.member.text, w=room)
        if cell.pair is None or cell.pair.readback is None:
            return
        below = y + ROW + ROW_GAP
        x = self.field(x0, below, cell.pair.readback)
        x = self.unit(x, below, cell.pair.readback_unit)
        deviation = _tolerance_ladder(cell.pair)
        if deviation is not None:
            x = self.light(x + GAP, below, [Light(cell.pair.channel)], rule=deviation)
        if cell.pair.inferred:
            self.mark(x + GAP, below, symbols.INFERRED_MARK)

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

    def node(self, node: Node, depth: int = 0) -> None:
        """A roll-up node's row; the root's direct children follow it, one level only.

        A node whose rule is too long for the format is drawn as an aggregate:
        its mark and channel count. On the root that has children, a mosaic of
        its children's lights follows (a child that is itself too long is an
        aggregate tile that opens the child's screen). A node with no
        children to show is split into parts each within the limit, drawn as
        a row of small lights after its label, each a rule of its own.
        """
        x = MARGIN + depth * INDENT
        rule: Ladder | str | None = None
        lights: list[Light] = []
        if node.no_rule:
            x = self.no_rule(x, self.y)
        else:
            lights = [describe_light(self.graph, c) for c in node.channels]
            rule = ladder(self.graph, lights, node=True)
            x = self.light(x, self.y, lights, rule=rule)
        if node.link is not None:
            x = self.button(x + GAP, self.y, node.link, label=node.label)
        else:
            x = self.text(x + GAP, self.y, node.label)
        if rule == "aggregate state":
            x = self.text(
                x + GAP, self.y, _AGGREGATE_TEXT.format(len(node.channels)), size=SMALL_SIZE
            )
        shown = depth == 0 and bool(node.children)
        if rule == "aggregate state" and not shown:
            self.parts(chunks(self.graph, lights), x + GAP)
        self.next_row()
        if depth:
            return
        if rule == "aggregate state" and shown:
            self.mosaic(node.children, MARGIN + INDENT)
        for child in node.children:
            self.node(child, 1)

    def parts(self, parts: Sequence[Sequence[Light]], x0: float) -> None:
        """The parts of a node too long for one rule: a small light each, part i of n."""
        x = x0
        for number, part in enumerate(parts, 1):
            if x + MOSAIC_LIGHT > MARGIN + STRIP_WIDTH:
                self.y += MOSAIC_LIGHT + MOSAIC_GAP
                x = MARGIN + INDENT
            rule = ladder(self.graph, part, node=len(part) > 1)
            top = self.y - (ROW - MOSAIC_LIGHT) / 2
            tooltip = _PART.format(number, len(parts))
            self.light(x, top, part, rule=rule, size=MOSAIC_LIGHT, tile=True, tooltip=tooltip)
            x += MOSAIC_LIGHT + MOSAIC_GAP

    def mosaic(self, children: Sequence[Node], x0: float) -> None:
        """One small light per child that reads a rule, in rows across the page.

        A child too long for one rule is an aggregate tile; it opens the
        child's screen, where its parts are drawn.
        """
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
            if rule == "aggregate state" and child.link is not None:
                self.nav(x, top + (ROW - MOSAIC_LIGHT) / 2, MOSAIC_LIGHT, MOSAIC_LIGHT, child.link)
            x += MOSAIC_LIGHT + MOSAIC_GAP
        self.y += MOSAIC_LIGHT + ROW_GAP

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
        if item.elements:
            widths = {kind: units * SYMBOL_UNIT for kind, units in symbols.WIDTH_UNITS.items()}
            for row in _strip_rows(item.elements, widths):
                self.strip_row(row, widths, item.extent_m)
        if item.tray:
            self.tray(item)
        if item.others is not None:
            self.button(MARGIN, self.y, item.others)
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
                reach = STRIP_LEVEL if p.lane == "above" else STRIP_LABEL_TOP
                self.nav(x - half, y - reach, p.width, 2 * reach, replace(p.link, label=p.thing))
        for text, left, w, level in strip_labels(placed):
            top = axis + STRIP_LABEL_TOP + level * STRIP_LABEL_STEP
            self.text(MARGIN + left, top, text, size=STRIP_LABEL_SIZE, w=w, h=STRIP_LABEL_STEP)
        self.y = axis + STRIP_BELOW + LOCATOR_GAP
        if extent is not None:
            self.locator(row, extent)

    def symbol(self, element: strip.Placed, x: float, y: float, unit: float) -> None:
        """A device's symbol, in the live state of its rule channels.

        The symbol reads the ladder a roll-up node of the device would read: the
        worst of its channels. With no rule channel it is in the ``none`` look
        and carries no mark; over the format's limit it is drawn in the
        aggregate look.
        """
        if not element.channels:
            for node in symbols.draw(element.symbol, x, y, symbols.NONE, unit=unit):
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
        for thing, x in strip.locator(shifted, span, STRIP_WIDTH):
            mark = [(MARGIN + x, bar - LOCATOR_MARK), (MARGIN + x, bar + LOCATOR_BAR)]
            self.line(mark, PAGE_TEXT, AXIS_WIDTH, tooltip=thing)
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
        """A device class's picture: its parts where the table puts them, each named.

        The outline is what the table states (rows of kind :data:`OUTLINE_KIND`,
        joined in file order); with none, the bounding frame of the parts is
        drawn dashed and captioned, never an invented shape. Every other kind
        is the generic marker. Names sit in a column on each side, joined to
        their part by a leader line (:func:`picture_labels`).
        """
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

        if outline:
            points = [at(part) for part in outline]
            self.line([*points, points[0]], PAGE_TEXT)
        elif parts:
            xs = [at(part)[0] for part in parts]
            ys = [at(part)[1] for part in parts]
            fx, fy = min(xs) - PART_MARKER, min(ys) - PART_MARKER
            fw, fh = max(xs) - min(xs) + 2 * PART_MARKER, max(ys) - min(ys) + 2 * PART_MARKER
            self.box(fx, fy, fw, fh, dashed=True)
            self.text(fx, fy + fh + ROW_GAP, _NO_OUTLINE, size=SMALL_SIZE)
        placed = [(part.part, *at(part)) for part in parts]
        labels = picture_labels(placed, cx, top, top + PICTURE_SIZE)
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
            anchor = lx + PART_LABEL_WIDTH if side == "left" else lx
            self.line([(px, py), (anchor, label_y + ROW / 2)], BOX_BORDER)
            align = "right" if side == "left" else "left"
            self.text(lx, label_y, name, size=SMALL_SIZE, w=PART_LABEL_WIDTH, align=align)
        bottom = max([top + PICTURE_SIZE, *(y + ROW for _side, y in labels)])
        self.y = bottom + ROW + ROW_GAP

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
            if unit.unit_text is not None:
                x = self.text(x + GAP, y_side, f"unit {unit.unit_text}", size=SMALL_SIZE)
            for item in unit.lights:
                x = self.light(x + GAP, y_side, [item])
            if not unit.lights:
                self.no_rule(x + GAP, y_side)
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
        if not unit.lights:
            x = self.no_rule(x + GAP, y)
        if unit.standing == _CONTESTED:
            self.text(x + GAP, y, _CONTESTED, size=SMALL_SIZE)
        return y + max(h, ROW)


def picture_labels(
    parts: Sequence[tuple[str, float, float]], centre: float, top: float, bottom: float
) -> list[tuple[str, float]]:
    """Where each part's name goes: ``(side, y)`` per ``(name, x, y)`` part, in order.

    A part left of *centre* is named in the left column, any other in the
    right one. In each column the names go in order of their part's ``y``,
    each at its part's height or just under the name above it, so no two
    overlap; a column that runs past *bottom* is pushed back up, never
    above *top*.
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
        for i, y in zip(mine, ys, strict=True):
            slots[i] = (side, y)
    return slots


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
        name = _short(p.thing)
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
    doc_id: str,
) -> dict[str, Any]:
    """The view as a Screen-IR 1.1 document.

    Args:
        view: What the screen shows.
        graph: The panel graph, for the health rows the lights' rules are written from.
        screens: The screen file a link to a target opens, relative to this
            screen's directory.
        doc_id: The document's id; the caller names it, as it names the files
            links open (:func:`osprey.facility.panels.emit.screen_ids`).

    Returns:
        The document, as JSON-ready data. Equal inputs give equal documents.
    """
    return _Page(view, graph, screens, doc_id).draw()
