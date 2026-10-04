"""The active scenario set and the composition rule scenarios must obey.

Every reader of the active set — the engine, ``sim apply``, the archive
composite and the stand-in — resolves it and checks it here, so they cannot
disagree on which scenarios run or on which sets compose.

Scenarios compose only when they write disjoint targets: two scenarios writing
one target would apply in an order-dependent way, so such a set is refused as
an :class:`Overlap` naming the target. A process that serves without a
scenario because of an overlap logs one :func:`overlap_record`, and readers
print it with :func:`format_overlap_record`.

The module imports neither numpy nor lume.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from osprey_connectors.config import get_facility_timezone
from osprey_connectors.logger import get_logger
from osprey_connectors.simulation.machine import DEFAULT_SCENARIO

__all__ = [
    "ACTIVE_SCENARIOS_FILENAME",
    "OVERLAP_EVENT",
    "Overlap",
    "format_overlap_record",
    "overlap_record",
    "parse_active_state",
    "resolve_active_scenarios",
    "validate_composition",
]

#: Name of the plain-text file holding the active scenario set.
ACTIVE_SCENARIOS_FILENAME = "active_scenarios"

#: The ``event`` an overlap record carries in a simulator log.
OVERLAP_EVENT = "scenario-overlap"

logger = get_logger("simulation_state")


def parse_active_state(text: str) -> tuple[list[str], float | None]:
    """The scenario names and the anchor an ``active_scenarios`` file records.

    Blank lines and ``#`` comments are skipped. A ``key=value`` line is
    metadata, of which only ``anchor=<ISO 8601>`` is read; every other line is
    a scenario name, kept in file order. A naive anchor is read in the
    facility timezone; a malformed one is logged and ignored.

    Args:
        text: The file's contents.

    Returns:
        The names, and the anchor as epoch seconds or ``None`` when the file
        records none.
    """
    names: list[str] = []
    anchor_epoch: float | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" in stripped:
            key, _, value = stripped.partition("=")
            if key.strip() == "anchor":
                try:
                    parsed = datetime.fromisoformat(value.strip())
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=get_facility_timezone())
                    anchor_epoch = parsed.timestamp()
                except ValueError:
                    logger.warning(f"Ignoring malformed anchor in state file: {stripped!r}")
            continue
        names.append(stripped)
    return names, anchor_epoch


def resolve_active_scenarios(names: Sequence[str]) -> list[str]:
    """The active scenario set a request for ``names`` really means.

    ``nominal`` is the machine's baseline, not a fault: it is always active, so
    it is prepended whether or not the caller named it. The rest keep the
    caller's order and are deduplicated, because activating a scenario twice
    would compose its physics twice.

    One function for a rule two callers depend on — activation writes the state
    file from it, and physics-fault rendering derives ``VA_*`` variables from
    it — so the environment a project is built with and the scenarios its
    engine runs cannot describe different machines.
    """
    resolved: list[str] = [DEFAULT_SCENARIO]
    for name in names:
        if name != DEFAULT_SCENARIO and name not in resolved:
            resolved.append(name)
    return resolved


@dataclass(frozen=True)
class Overlap:
    """One target written by two scenarios of one set.

    Attributes:
        target: The address or variable both scenarios write.
        first: The scenario that claimed the target first, in set order.
        second: The scenario that writes it again.
    """

    target: str
    first: str
    second: str

    def __str__(self) -> str:
        return (
            f"Channel {self.target!r} is touched by both {self.first!r} and {self.second!r}; "
            f"active scenarios must touch disjoint channel sets"
        )


def validate_composition(
    scenarios_view: Mapping[str, Collection[str]], names: Sequence[str]
) -> list[Overlap]:
    """Return the overlaps of a scenario set; an empty list means it composes.

    ``nominal`` writes nothing, so it is skipped when the view does not list
    it.

    Args:
        scenarios_view: Each scenario's name mapped to the targets it writes.
        names: Scenario names to check, in set order.

    Returns:
        One :class:`Overlap` per target a later scenario writes again, sorted
        by target within each scenario.

    Raises:
        ValueError: If a name is not in ``scenarios_view``.
    """
    unknown = [n for n in names if n not in scenarios_view and n != DEFAULT_SCENARIO]
    if unknown:
        raise ValueError(f"Unknown scenarios {unknown!r}. Available: {sorted(scenarios_view)}")
    overlaps: list[Overlap] = []
    owner: dict[str, str] = {}
    for name in names:
        for target in sorted(scenarios_view.get(name, ())):
            if target in owner and owner[target] != name:
                overlaps.append(Overlap(target=target, first=owner[target], second=name))
            else:
                owner[target] = name
    return overlaps


def overlap_record(overlap: Overlap, *, instance: str, pid: int) -> dict[str, Any]:
    """The log record a process writes when it serves without a scenario.

    Args:
        overlap: The overlap that kept the scenario out.
        instance: The serving instance that logs it.
        pid: The process id of that instance.

    Returns:
        The record, one JSON line in the simulator log.
    """
    return {"instance": instance, "pid": pid, "event": OVERLAP_EVENT, "target": overlap.target}


def format_overlap_record(model: str, record: Mapping[str, Any]) -> str:
    """One printable line for an overlap record read back from a model's log.

    The line names where it came from, so a reader never takes it for the
    model's status.

    Args:
        model: The model whose log holds the record.
        record: The record as :func:`overlap_record` wrote it.

    Returns:
        ``<model> (log, instance <i>, pid <p>): <event> on <target>``.
    """
    return (
        f"{model} (log, instance {record['instance']}, pid {record['pid']}): "
        f"{record['event']} on {record['target']}"
    )
