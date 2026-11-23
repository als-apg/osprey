"""Health: the state of a channel, of a thing and of a roll-up node.

A channel's state comes from the rows of ``health.csv`` that name its value.
A channel with no row, a missing value, or a value no supported row names is
``unknown``; rows whose form the engine cannot evaluate are never matched. Rows from different sources that match the value and give
different states make the channel ``disagree``. The row's ``basis`` does not
change the state.

A roll-up reads the rule-bearing channels of a node (its own, plus the
:meth:`~osprey_panels.model.PanelGraph.contents` of a place, or the
members of a group) and takes the
:func:`worst`. A thing with no rule-bearing channel contributes nothing, and a
roll-up where nothing contributed is ``None``: whoever draws it decides how to
show "no rule here".
"""

from __future__ import annotations

import functools
import math
import operator
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from osprey_panels.model import PanelGraph, ThingKind

State = Literal["normal", "warn", "bad", "unknown", "disagree"]

#: Rank of each state; the highest wins a roll-up.
_RANK: dict[str, int] = {"normal": 0, "unknown": 1, "disagree": 2, "warn": 3, "bad": 4}


_NUMBER = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
_INTEGER = re.compile(r"[+-]?[0-9]+", re.ASCII)
_NUMERIC = re.compile(_NUMBER, re.ASCII)
_COMPARE = re.compile(rf"^(!=|<=|>=|<|>)\s*({_NUMBER})$", re.ASCII)
_BIT = re.compile(r"^bit([0-9]*)\s*=\s*([0-9]+)$", re.ASCII)
# A value that starts like an operator or a bit test but fits no form above is
# unsupported, and so is anything marked ``expr:``. A row starting with ``<``,
# ``>``, ``!`` or ``=`` therefore cannot be a literal label. Everything else is
# literal text.
_LOOKS_LIKE_FORM = re.compile(r"^(expr:|[<>!=]|bit[0-9]*\s*=)", re.ASCII)
_DIGITS = re.compile(r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", re.ASCII)

Number = int | float

_COMPARE_OPS: dict[str, Callable[[Number, Number], bool]] = {
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


def _parse_number(text: str) -> Number | None:
    """A number from text: an integer stays an integer; NaN and infinities are not numbers."""
    text = text.strip()
    try:
        if _INTEGER.fullmatch(text):
            return int(text)
        if _NUMERIC.fullmatch(text):
            number = float(text)
            return number if math.isfinite(number) else None
    except (ValueError, OverflowError):
        return None
    return None


def _number(value: Any) -> Number | None:
    """The live value as a finite number, or ``None``.

    An int, or a string that parses as an int, stays an int, so a large
    integer keeps its precision. A bool, a NaN or an infinity is not a number.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return _parse_number(value)
    return None


def _text(value: Any) -> str | None:
    try:
        return str(value)
    except Exception:
        return None


def _literal(row_value: str) -> Callable[[Any], bool]:
    expected = _parse_number(row_value)

    def match(value: Any) -> bool:
        if _text(value) == row_value:
            return True
        live = _number(value)
        return expected is not None and live is not None and live == expected

    return match


def _compare(op: str, limit: Number) -> Callable[[Any], bool]:
    compare = _COMPARE_OPS[op]

    def match(value: Any) -> bool:
        live = _number(value)
        return live is not None and compare(live, limit)

    return match


def _bit(index: int, wanted: int) -> Callable[[Any], bool]:
    def match(value: Any) -> bool:
        live = _number(value)
        if live is None or live != int(live):
            return False
        return (int(live) >> index) & 1 == wanted

    return match


@dataclass(frozen=True)
class Form:
    """The parsed form of a health row's value.

    Attributes:
        kind: ``literal``, ``compare`` or ``bit``.
        text: The value, stripped; a literal's text.
        op: A comparison's operator (``!=``, ``<``, ``<=``, ``>``, ``>=``).
        number: A literal's number when it parses as one, or a comparison's limit.
        bit: A bit test's bit index.
        wanted: A bit test's wanted value, 0 or 1.
    """

    kind: Literal["literal", "compare", "bit"]
    text: str
    op: str | None = None
    number: Number | None = None
    bit: int | None = None
    wanted: int | None = None


@functools.cache
def form(row_value: str) -> Form | None:
    """The form of a health row's value, or ``None`` for one the engine cannot evaluate.

    The grammar is :func:`row_matches`'s; whoever writes the rule in another
    language (a screen format's rule expressions) reads it from here.
    """
    text = row_value.strip()
    if found := _COMPARE.match(text):
        limit = _parse_number(found.group(2))
        return None if limit is None else Form("compare", text, op=found.group(1), number=limit)
    if found := _BIT.match(text):
        wanted = int(found.group(2))
        if wanted not in (0, 1):
            return None
        return Form("bit", text, bit=int(found.group(1) or 0), wanted=wanted)
    if _LOOKS_LIKE_FORM.match(text):
        return None
    return Form("literal", text, number=_parse_number(text))


@functools.cache
def _parse(row_value: str) -> Callable[[Any], bool] | None:
    """The test a health row's value stands for, built once per distinct text.

    Returns:
        ``None`` for a form the engine cannot evaluate.
    """
    found = form(row_value)
    if found is None:
        return None
    if found.kind == "compare":
        assert found.op is not None and found.number is not None
        return _compare(found.op, found.number)
    if found.kind == "bit":
        assert found.bit is not None and found.wanted is not None
        return _bit(found.bit, found.wanted)
    return _literal(found.text)


def row_matches(row_value: str, value: Any) -> bool:
    """Whether a health row's value names the live value.

    The grammar of *row_value*: literal text or number (equal as text, or as
    numbers when both parse); ``!=N``, ``<N``, ``<=N``, ``>N``, ``>=N`` (numeric
    comparison; a live value that is not a number does not match); ``bit=N`` and
    ``bitK=N`` with N 0 or 1 (bit K of the live value as an integer, bit 0 for plain
    ``bit``). A live value that is a NaN or an infinity is not a number for any form.
    Any other form, ``expr:`` rows among them, is unsupported and never matches.
    """
    test = _parse(row_value)
    return test is not None and test(value)


def supported(row_value: str) -> bool:
    """Whether the engine can evaluate a health row's value (see :func:`row_matches`)."""
    return _parse(row_value) is not None


def unsupported(graph: PanelGraph) -> dict[str, int]:
    """The health row forms the engine cannot evaluate, with how many rows use each.

    Digits are replaced by ``N``, so ``expr:a>3`` and ``expr:a>7`` count as one form.
    """
    counts: dict[str, int] = {}
    for row in graph.rules():
        if _parse(row.value) is None:
            form = _DIGITS.sub("N", row.value.strip())
            counts[form] = counts.get(form, 0) + 1
    return counts


def disagree(states: Iterable[str]) -> bool:
    """Whether rows that name one value disagree: they give it more than one state.

    The one definition of ``disagree``: :func:`channel_state` applies it to
    the rows that match a live value, :func:`disagrees` to the rows of one
    stated value, and the layout's rules to the conditions they write.
    """
    return len(set(states)) > 1


def disagrees(graph: PanelGraph, channel: str) -> bool:
    """Whether some value the channel's rows state is given more than one state by them."""
    by_value: dict[str, set[str]] = {}
    for row in graph.rule(channel):
        by_value.setdefault(row.value, set()).add(row.state)
    return any(disagree(states) for states in by_value.values())


def channel_state(graph: PanelGraph, channel: str, value: Any) -> State:
    """The state of one channel at one value.

    Args:
        graph: The panel graph.
        channel: The channel name.
        value: The current value, of any type, or ``None`` when it is missing.

    Returns:
        ``unknown`` without a rule, without a value, or when no row matches;
        ``disagree`` when matching rows give different states; otherwise the
        state they agree on.
    """
    rows = graph.rule(channel)
    if not rows or value is None:
        return "unknown"
    states = {row.state for row in rows if row_matches(row.value, value)}
    if not states:
        return "unknown"
    if disagree(states):
        return "disagree"
    return states.pop()


def worst(states: Iterable[str]) -> State | None:
    """The most severe state: ``bad > warn > disagree > unknown > normal``.

    Returns:
        ``None`` for no states.
    """
    found: State | None = None
    rank = -1
    for state in states:
        if _RANK[state] > rank:
            rank, found = _RANK[state], state  # type: ignore[assignment]
    return found


def rule_channels(graph: PanelGraph, node_id: str, kind: ThingKind | None = None) -> list[str]:
    """The channels a roll-up of the node reads, each once, in graph order.

    Args:
        graph: The panel graph.
        node_id: A device, a place or a group.
        kind: Picks between things that share an id.

    Raises:
        KeyError: No such node, or it is excluded.
    """
    node = graph.thing(node_id, kind)
    holders = [node]
    if node.kind == "place":
        holders += graph.contents(node.id)
    elif node.kind == "group":
        # The facility build refuses a group member that is not a device or a
        # place, so a group cannot hold a group and no cycle can occur here.
        holders = []
        for member in graph.members(node.id):
            holders.append(member)
            if member.kind == "place":
                holders += graph.contents(member.id)
    seen: set[str] = set()
    found: list[str] = []
    for holder in holders:
        if holder.kind == "group":
            continue
        for channel in graph.channels(holder.id, holder.kind):
            if channel.name not in seen and graph.rule(channel.name):
                seen.add(channel.name)
                found.append(channel.name)
    return found


def unplaced_channels(graph: PanelGraph) -> list[str]:
    """The rule-bearing channels in no place, each once: those of every device in no
    place in graph order, then those on no thing (:meth:`PanelGraph.unheld`)."""
    seen: set[str] = set()
    found: list[str] = []
    held = [graph.channels(thing.id, "device") for thing in graph.unplaced()]
    for channel in [*(c for channels in held for c in channels), *graph.unheld()]:
        if channel.name not in seen and graph.rule(channel.name):
            seen.add(channel.name)
            found.append(channel.name)
    return found


def rollup(
    graph: PanelGraph,
    node_id: str,
    values: Mapping[str, Any],
    kind: ThingKind | None = None,
) -> State | None:
    """The worst state over the node's rule-bearing channels.

    Args:
        graph: The panel graph.
        node_id: A device, a place or a group.
        values: Current value by channel name; a missing name is a missing value.
        kind: Picks between things that share an id.

    Returns:
        ``None`` when no channel contributed.

    Raises:
        KeyError: No such node, or it is excluded.
    """
    return worst(
        channel_state(graph, name, values.get(name)) for name in rule_channels(graph, node_id, kind)
    )
