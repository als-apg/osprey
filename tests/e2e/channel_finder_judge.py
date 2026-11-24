"""The verdict on a channel-finder delegation run: what the agent answered, and how.

The run asks for the addresses of the storage ring's horizontal beam position
monitors. It passes when:

- no tool erred, save the hierarchical finder's level-order refusal
  (:func:`designed_level_refusal`): an options request whose selections leave
  out a required level above the one asked for;
- each such refusal is followed by the repaired request (options at the refused
  level whose selections name the missing level) and a successful
  ``build_channels``;
- the final answer names every address of the expected set and no other address
  of the facility file.

The expected set is read from the facility definition the render holds
(:func:`expected_horizontal_bpms`), never typed. A refusal is recognised by
replaying the call on the render's own index and catching the product's
:class:`~osprey.services.channel_finder.databases.hierarchical.UnselectedLevelError`,
so both its type and its text come from the product.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from osprey.services.channel_finder.databases.hierarchical import (
    HierarchicalChannelDatabase,
    UnselectedLevelError,
)
from osprey_facility.records import ChannelSelection, FacilityRecords
from osprey_facility.schema._generated.core import SignalRoleEnum
from tests.e2e.tool_errors import assert_no_tool_errors

CHANNEL_FINDER_TOOLS = "mcp__channel-finder__"
GET_OPTIONS = f"{CHANNEL_FINDER_TOOLS}get_options"
BUILD_CHANNELS = f"{CHANNEL_FINDER_TOOLS}build_channels"

#: The place the run's question names, as the facility file's ``names`` holds it.
RING = "storage ring"

#: A run of characters an address token is made of; dots at its ends are punctuation.
_TOKEN = re.compile(r"[\w:.\-]+")


def expected_horizontal_bpms(records: FacilityRecords) -> frozenset[str]:
    """The horizontal beam position readback addresses at the place named :data:`RING`."""
    places = [record.id for _match, record in records.lookup(RING) if record.kind == "place"]
    assert len(places) == 1, f"the facility file names {len(places)} places {RING!r}: {places}"
    selection = ChannelSelection(place=places[0], signal=SignalRoleEnum.position_x_readback.value)
    return frozenset(str(channel["id"]) for channel in records.select_channels(selection))


def _envelope(result: str | None) -> dict[str, Any] | None:
    """The error envelope a tool result carries, or ``None``."""
    if not result:
        return None
    start = result.find("{")
    if start < 0:
        return None
    try:
        payload, _end = json.JSONDecoder().raw_decode(result[start:])
    except ValueError:
        return None
    return payload if isinstance(payload, dict) and payload.get("error") is True else None


def _replayed_refusal(
    database: HierarchicalChannelDatabase, trace: Any
) -> UnselectedLevelError | None:
    """The level-order refusal the product makes for *trace*'s call, or ``None``."""
    name = trace.name or ""
    if not (name.startswith(CHANNEL_FINDER_TOOLS) and name.endswith("__get_options")):
        return None
    if not trace.is_error:
        return None
    envelope = _envelope(trace.result)
    if envelope is None or envelope.get("error_type") != "validation_error":
        return None
    arguments = trace.input or {}
    level = arguments.get("level")
    selections = arguments.get("selections") or {}
    if not isinstance(level, str) or not isinstance(selections, dict):
        return None
    try:
        database.get_options_at_level(level, selections)
    except UnselectedLevelError as refusal:
        return refusal if str(refusal) == envelope.get("error_message") else None
    except ValueError:
        return None
    return None


def designed_level_refusal(database: HierarchicalChannelDatabase) -> Callable[[Any], bool]:
    """A predicate accepting a trace that is the product's level-order refusal of its call.

    True only for a channel-finder ``get_options`` call marked as an error whose
    result is a ``validation_error`` envelope, and whose call, replayed on
    *database*, raises :class:`UnselectedLevelError` with the envelope's message.
    """
    return lambda trace: _replayed_refusal(database, trace) is not None


def _names(selections: Any, level: str) -> bool:
    """Whether *selections* name a value at *level*; an empty value is no selection."""
    if not isinstance(selections, dict):
        return False
    value = selections.get(level)
    values = value if isinstance(value, list) else [value]
    return any(values)


def judge_channel_finder_run(
    result: Any,
    database: HierarchicalChannelDatabase,
    records: FacilityRecords,
    expected: frozenset[str],
) -> list[str]:
    """The reasons the channel-finder run *result* fails; empty when it passes."""
    failures: list[str] = []
    try:
        assert_no_tool_errors(result, exempt=designed_level_refusal(database))
    except AssertionError as exc:
        failures.append(str(exc))

    traces = list(result.tool_traces)
    for index, trace in enumerate(traces):
        refusal = _replayed_refusal(database, trace)
        if refusal is None:
            continue
        later = [t for t in traces[index + 1 :] if not t.is_error]
        repaired = any(
            t.name == trace.name
            and (t.input or {}).get("level") == refusal.requested
            and _names((t.input or {}).get("selections"), refusal.level)
            for t in later
        )
        if not repaired:
            failures.append(
                f"the refused {refusal.requested!r} options request leaving out "
                f"{refusal.level!r} is followed by no successful {refusal.requested!r} "
                f"options request naming {refusal.level!r}"
            )
        if not any(t.name == BUILD_CHANNELS for t in later):
            failures.append(
                f"the refused {refusal.requested!r} options request is followed by "
                f"no successful {BUILD_CHANNELS}"
            )

    final = getattr(result, "result", None)
    if final is None:
        failures.append("no ResultMessage received")
        return failures
    text = final.result or ""
    if not text.strip():
        failures.append("the run gave no final answer")
        return failures
    named = {token.strip(".") for token in _TOKEN.findall(text)}
    missing = sorted(expected - named)
    extra = sorted((named & records.ids("channel")) - expected)
    if missing:
        failures.append(f"the answer leaves out {len(missing)} expected address(es): {missing}")
    if extra:
        failures.append(f"the answer names {len(extra)} address(es) outside the set: {extra}")
    return failures
