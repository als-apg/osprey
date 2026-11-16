"""Health: the state of a channel, of a thing and of a roll-up node.

A channel's state comes from the rows of ``health.csv`` that name its value.
A channel with no row, a missing value, or a value no row names is
``unknown``. Rows from different sources that match the value and give
different states make the channel ``disagree``. The row's ``basis`` does not
change the state.

A roll-up reads the rule-bearing channels of a node (its own, plus everything
within a place at any depth, or the members of a group) and takes the
:func:`worst`. A thing with no rule-bearing channel contributes nothing, and a
roll-up where nothing contributed is ``None``: whoever draws it decides how to
show "no rule here".
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal

from osprey.facility.panels.model import PanelGraph, ThingKind

State = Literal["normal", "warn", "bad", "unknown", "disagree"]

#: Rank of each state; the highest wins a roll-up.
_RANK: dict[str, int] = {"normal": 0, "unknown": 1, "disagree": 2, "warn": 3, "bad": 4}


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _same(row_value: str, text: str) -> bool:
    if row_value == text:
        return True
    left, right = _number(row_value), _number(text)
    return left is not None and right is not None and left == right


def _matches(row_value: str, text: str) -> bool:
    """Whether a health row's value (``"0"`` or ``"!=0"``) names the text."""
    if row_value.startswith("!="):
        return not _same(row_value[2:], text)
    return _same(row_value, text)


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
    text = str(value)
    states = {row.state for row in rows if _matches(row.value, text)}
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
