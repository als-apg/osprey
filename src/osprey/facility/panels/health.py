"""Health: the state of a channel, of a thing and of a roll-up node.

A channel's state comes from the rows of ``health.csv`` that name its value.
A channel with no row, a missing value, or a value no supported row names is
``unknown``; rows whose form the engine cannot evaluate are never matched. Rows from different sources that match the value and give
different states make the channel ``disagree``. The row's ``basis`` does not
change the state.

A roll-up reads the rule-bearing channels of a node (its own, plus everything
within a place at any depth, or the members of a group) and takes the
:func:`worst`. A thing with no rule-bearing channel contributes nothing, and a
roll-up where nothing contributed is ``None``: whoever draws it decides how to
show "no rule here".
"""

from __future__ import annotations

import functools
import operator
import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Literal

from osprey.facility.panels.model import PanelGraph, ThingKind

State = Literal["normal", "warn", "bad", "unknown", "disagree"]

#: Rank of each state; the highest wins a roll-up.
_RANK: dict[str, int] = {"normal": 0, "unknown": 1, "disagree": 2, "warn": 3, "bad": 4}


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_COMPARE = re.compile(rf"^(!=|<=|>=|<|>)\s*({_NUMBER})$")
_BIT = re.compile(r"^bit(\d*)=(\d+)$")
# a value that starts like an operator or a bit test but fits no form above is
# unsupported; so is anything marked ``expr:``. Everything else is literal text.
_LOOKS_LIKE_FORM = re.compile(r"^(expr:|[<>!=]|bit\d*=)")
_DIGITS = re.compile(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")

_COMPARE_OPS: dict[str, Callable[[float, float], bool]] = {
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


def _number(value: Any) -> float | None:
    """The live value as a number, or ``None`` (a bool is not a number here)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _literal(row_value: str) -> Callable[[Any], bool]:
    expected = _number(row_value)

    def match(value: Any) -> bool:
        if str(value) == row_value:
            return True
        live = _number(value)
        return expected is not None and live is not None and live == expected

    return match


def _compare(op: str, limit: float) -> Callable[[Any], bool]:
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


@functools.cache
def _parse(row_value: str) -> Callable[[Any], bool] | None:
    """The test a health row's value stands for, built once per distinct text.

    Returns:
        ``None`` for a form the engine cannot evaluate.
    """
    text = row_value.strip()
    if found := _COMPARE.match(text):
        return _compare(found.group(1), float(found.group(2)))
    if found := _BIT.match(text):
        return _bit(int(found.group(1) or 0), int(found.group(2)))
    if _LOOKS_LIKE_FORM.match(text):
        return None
    return _literal(text)


def row_matches(row_value: str, value: Any) -> bool:
    """Whether a health row's value names the live value.

    The grammar of *row_value*: literal text or number (equal as text, or as
    numbers when both parse); ``!=N``, ``<N``, ``<=N``, ``>N``, ``>=N`` (numeric
    comparison; a live value that is not a number does not match); ``bit=N`` and
    ``bitK=N`` (bit K of the live value as an integer, bit 0 for plain ``bit``).
    Any other form, ``expr:`` rows among them, is unsupported and never matches.
    """
    test = _parse(row_value)
    return test is not None and test(value)


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
    if len(states) > 1:
        return "disagree"
    return states.pop()  # type: ignore[return-value]


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
        holders += graph.within(node.id, deep=True)
    elif node.kind == "group":
        holders = []
        for member in graph.members(node.id):
            holders.append(member)
            if member.kind == "place":
                holders += graph.within(member.id, deep=True)
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
