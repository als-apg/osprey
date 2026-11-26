"""A scenario's start marks: which simulator instances start in the scenario.

A deployment serves one simulator instance, the ``va`` target, or two: the
second stands in for the live machine and is the ``standin`` target. Each has
an active scenario set of its own. ``start: true`` puts a scenario in the start
set of every instance a deployment runs; ``start_on: [standin]`` (or ``[va]``)
puts it in the named instances' start sets alone. A scenario states one of the
two. A mark for an instance a deployment does not run has no effect there.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from osprey_connectors.types import TARGET_STANDIN, TARGET_VA

__all__ = ["START_SPELLINGS", "start_error", "start_targets"]

#: The simulator targets ``start_on`` may name, in instance order.
_START_TARGETS: tuple[str, ...] = (TARGET_VA, TARGET_STANDIN)

#: The valid spellings, as a refusal's remedy lists them.
START_SPELLINGS = "`start: true` for every simulator instance, or " + " or ".join(
    f"`start_on: [{target}]`" for target in _START_TARGETS
)


def start_error(scenario: Mapping[str, Any]) -> str | None:
    """Why a scenario's start marks are refused, or ``None`` when they are valid."""
    start, start_on = scenario.get("start"), scenario.get("start_on")
    if start is not None and not isinstance(start, bool):
        return f"`start` is {start!r}, neither true nor false"
    if start_on is None:
        return None
    if not isinstance(start_on, list) or not start_on:
        return f"`start_on` is {start_on!r}, not a list of simulator targets"
    unknown = [entry for entry in start_on if entry not in _START_TARGETS]
    if unknown:
        return f"`start_on` names {unknown[0]!r}, which is no simulator target"
    if len(set(start_on)) != len(start_on):
        return f"`start_on` names a simulator target twice: {start_on!r}"
    if start is not None:
        return (
            f"`start: {str(start).lower()}` is stated beside `start_on`, "
            "which already says where the scenario starts"
        )
    return None


def start_targets(scenario: Mapping[str, Any]) -> tuple[str, ...]:
    """The simulator targets a scenario starts on, in instance order.

    ``start: true`` is every target and a valid ``start_on`` its members.
    Marks :func:`start_error` refuses start the scenario nowhere.
    """
    if start_error(scenario) is not None:
        return ()
    if scenario.get("start") is True:
        return _START_TARGETS
    named = scenario.get("start_on") or ()
    return tuple(target for target in _START_TARGETS if target in named)
