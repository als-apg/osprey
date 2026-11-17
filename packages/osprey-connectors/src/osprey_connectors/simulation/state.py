"""The active scenario set and the composition rule scenarios must obey.

Every reader of the active set — ``sim apply``, the archive, the composite
and the stand-in — resolves it and checks it here, so they cannot
disagree on which scenarios run or on which sets compose; ``sim apply`` writes
it here too, with :func:`write_active_state`.

Scenarios compose only when they write disjoint targets: two scenarios writing
one target would apply in an order-dependent way, so such a set is refused as
an :class:`Overlap` naming the target. A scenario's ``still`` meets only the
motion other scenarios set: it overlaps the ``noise`` or ``couple`` of the
readings it stills, never another target and never another ``still``. An
``archiver`` entry writes no target: events several scenarios script on one
channel compose in set order in the archive, so they never refuse a set. A
process that serves without a scenario because of an overlap logs one
:class:`OverlapLogRecord`, and readers print it with
:meth:`OverlapLogRecord.describe`.

:func:`compose` is the one composition of an active set: the composite serves
it, the archive synthesizes its history and the envelope bounds its motion
from the same :class:`ComposedSet`, so the three cannot describe different
machines.

The module imports neither numpy nor lume.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Final, Literal

from pydantic import AwareDatetime, ValidationError, field_validator
from typing_extensions import override

from osprey_connectors.errors import require_aware
from osprey_connectors.logger import get_logger
from osprey_connectors.model import FrozenModel, FrozenTuple, ReadOnlyMapping, parse_instant
from osprey_connectors.safety_file import atomic_replace
from osprey_connectors.simulation.records import (
    ArchiverEvent,
    ChannelFaultKind,
    CouplingTerm,
    DriveRecord,
    NoiseRecord,
    ResolvedCoupling,
    ScenarioRecord,
)

__all__ = [
    "ACTIVE_SCENARIOS_FILENAME",
    "DEFAULT_SCENARIO",
    "STILL_SCENARIO",
    "ActiveState",
    "ComposedSet",
    "Overlap",
    "OverlapLogRecord",
    "ServingInstance",
    "compose",
    "parse_active_state",
    "read_active_state",
    "resolve_active_scenarios",
    "validate_composition",
    "write_active_state",
]

#: Name of the plain-text file holding the active scenario set.
ACTIVE_SCENARIOS_FILENAME = "active_scenarios"

#: The baseline scenario, active in every set.
DEFAULT_SCENARIO = "nominal"

#: The scenario every view lists, which stills every reading.
STILL_SCENARIO = "still"

#: The ``event`` an overlap record carries in a simulator log.
_OVERLAP_EVENT: Final = "scenario-overlap"

#: The serving instance a model log record names: the served virtual
#: accelerator, the served live stand-in, or a composite in process.
ServingInstance = Literal["virtual_accelerator", "live_standin", "inprocess"]

logger = get_logger("simulation_state")


class ActiveState(FrozenModel):
    """An active scenario set and its anchor, as an ``active_scenarios`` file records them.

    Attributes:
        names: The set :func:`resolve_active_scenarios` makes of the names it
            is given: ``nominal`` first, then the rest in their order, each
            once. A name no view lists is kept; :func:`compose` drops it.
        anchor_at: The instant the set was applied at, or ``None`` when none
            is recorded.
    """

    names: FrozenTuple[str] = (DEFAULT_SCENARIO,)
    anchor_at: AwareDatetime | None = None

    @field_validator("names")
    @classmethod
    def _resolved(cls, names: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(resolve_active_scenarios(names))


def parse_active_state(text: str) -> ActiveState:
    """The scenario names and the anchor an ``active_scenarios`` file records.

    Blank lines and ``#`` comments are skipped. A ``key=value`` line is
    metadata, of which only ``anchor=<ISO 8601>`` is read; every other line is
    a scenario name, resolved as :class:`ActiveState` resolves its names. An
    anchor must state its UTC offset (:func:`write_active_state` always writes
    one): an anchor without one, like a malformed one, is logged and ignored,
    because no zone is guessed for it.

    Args:
        text: The file's contents.

    Returns:
        The set and the anchor; the anchor is ``None`` when the file records
        none.
    """
    names: list[str] = []
    anchor_at: datetime | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" in stripped:
            key, _, value = stripped.partition("=")
            if key.strip() == "anchor":
                try:
                    anchor_at = parse_instant(value.strip())
                except ValueError:
                    logger.warning(
                        "Ignoring an anchor that is malformed or without a UTC offset"
                        f" in state file: {stripped!r}"
                    )
            continue
        names.append(stripped)
    return ActiveState(names=tuple(names), anchor_at=anchor_at)


def read_active_state(state_dir: Path) -> ActiveState:
    """The scenario names and the anchor the ``active_scenarios`` file in *state_dir* records.

    Args:
        state_dir: The directory holding the file.

    Returns:
        What :func:`parse_active_state` reads from the file, or ``nominal``
        alone without an anchor when the file is absent.
    """
    path = state_dir / ACTIVE_SCENARIOS_FILENAME
    if not path.is_file():
        return ActiveState()
    return parse_active_state(path.read_text(encoding="utf-8"))


def write_active_state(
    state_dir: Path,
    scenarios: Mapping[str, ScenarioRecord],
    names: Sequence[str],
    *,
    anchor_at: datetime | None = None,
) -> ActiveState:
    """Activate a scenario set by writing the ``active_scenarios`` file in *state_dir*.

    The file holds an ``anchor=<ISO 8601>`` line when ``anchor_at`` is given,
    then the set's scenarios other than ``nominal``, or ``nominal`` alone. It
    is replaced whole through ``safety_file.atomic_replace``, so a reader
    polling it never reads a half-written set.

    Args:
        state_dir: The directory holding the file; made when missing.
        scenarios: The simulator view's scenarios, by name.
        names: The requested scenario names; ``nominal`` is always active.
        anchor_at: The instant the set is applied at, written with its offset.

    Returns:
        The set written, ``nominal`` first, and its anchor.

    Raises:
        NaiveDatetimeError: ``anchor_at`` is naive; nothing is written.
        ValueError: If a name is unknown or the set does not compose; nothing
            is written.
    """
    if anchor_at is not None:
        require_aware(anchor_at, name="anchor_at")
    active = ActiveState(names=tuple(names), anchor_at=anchor_at)
    try:
        overlaps = validate_composition(scenarios, active.names)
    except ValueError as exc:
        raise ValueError(f"Cannot activate scenarios: {exc}") from None
    if overlaps:
        raise ValueError("Cannot activate scenarios: " + "; ".join(map(str, overlaps)))

    body = [] if anchor_at is None else [f"anchor={anchor_at.isoformat()}"]
    body.extend([name for name in active.names if name != DEFAULT_SCENARIO] or [DEFAULT_SCENARIO])
    path = state_dir / ACTIVE_SCENARIOS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_replace(path, ("\n".join(body) + "\n").encode("utf-8"), mode=0o644, sync="best_effort")
    return active


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


class Overlap(FrozenModel):
    """One target written by two scenarios of one set.

    Attributes:
        target: The address or variable both scenarios write.
        first: The scenario that claimed the target first, in set order.
        second: The scenario that writes it again.
        motion: True when one scenario stills the reading at ``target`` and
            the other sets its motion.
    """

    target: str
    first: str
    second: str
    motion: bool = False

    @override
    def __str__(self) -> str:
        if self.motion:
            return (
                f"Scenarios {self.first!r} and {self.second!r} both set the motion of "
                f"{self.target!r}; active scenarios must not"
            )
        return (
            f"Channel {self.target!r} is touched by both {self.first!r} and {self.second!r}; "
            f"active scenarios must touch disjoint channel sets"
        )


class _Targets(FrozenModel):
    """What one scenario writes, as :func:`validate_composition` reads it apart.

    Attributes:
        writes: The addresses and engine variables it writes: the keys of its
            ``overrides``, of every fault's ``writes``, of its
            ``channel_faults`` and of its ``couple`` and ``noise``.
        moves: The readings whose motion it sets: the keys of its ``couple``
            and ``noise``.
        stills: The readings it stills.
    """

    writes: frozenset[str] = frozenset()
    moves: frozenset[str] = frozenset()
    stills: frozenset[str] = frozenset()


def _scenario_targets(scenario: ScenarioRecord) -> _Targets:
    """The targets a scenario writes: addresses, engine variables and coupled channels.

    Args:
        scenario: One entry of the simulator view's scenario list.

    Returns:
        Its targets; a block the scenario does not state contributes nothing.
        ``archiver`` entries are not targets: the archive merges the events
        of every active scenario on a channel in set order.
    """
    moves = frozenset(scenario.couple) | frozenset(scenario.noise)
    writes: set[str] = set(scenario.overrides)
    for fault in scenario.faults.values():
        writes.update(fault.writes)
    writes.update(scenario.channel_faults)
    return _Targets(writes=frozenset(writes) | moves, moves=moves, stills=frozenset(scenario.still))


def validate_composition(
    scenarios: Mapping[str, ScenarioRecord], names: Sequence[str]
) -> list[Overlap]:
    """Return the overlaps of a scenario set; an empty list means it composes.

    ``nominal`` writes nothing, so it is skipped when the view does not list
    it.

    Args:
        scenarios: The simulator view's scenarios, by name.
        names: Scenario names to check, in set order.

    Returns:
        One :class:`Overlap` per target a later scenario writes again, sorted
        by target within each scenario, then one per reading and scenario pair
        where one scenario stills the reading and the other sets its motion.

    Raises:
        ValueError: If a name is not in ``scenarios``.
    """
    unknown = [n for n in names if n not in scenarios and n != DEFAULT_SCENARIO]
    if unknown:
        raise ValueError(f"Unknown scenarios {unknown!r}. Available: {sorted(scenarios)}")
    targets = {name: _scenario_targets(scenarios[name]) for name in names if name in scenarios}
    overlaps: list[Overlap] = []
    owner: dict[str, str] = {}
    for name in names:
        stated = targets.get(name)
        for target in sorted(stated.writes) if stated is not None else ():
            if target in owner and owner[target] != name:
                overlaps.append(Overlap(target=target, first=owner[target], second=name))
            else:
                owner[target] = name
    return overlaps + _motion_overlaps(targets, names)


def _motion_overlaps(targets: Mapping[str, _Targets], names: Sequence[str]) -> list[Overlap]:
    """One motion :class:`Overlap` per reading one scenario stills and another moves."""
    stated = [(name, targets[name]) for name in names if name in targets]
    movers = [(name, target.moves) for name, target in stated if target.moves]
    stillers = [(name, target.stills) for name, target in stated if target.stills]
    order = {name: index for index, name in enumerate(names)}
    found: dict[tuple[str, str, str], Overlap] = {}
    for mover, addresses in movers:
        for stiller, readings in stillers:
            if stiller == mover:
                continue
            first, second = sorted((mover, stiller), key=order.__getitem__)
            for address in sorted(addresses & readings):
                found.setdefault(
                    (first, second, address),
                    Overlap(target=address, first=first, second=second, motion=True),
                )
    return sorted(found.values(), key=lambda o: (order[o.second], order[o.first], o.target))


class ComposedSet(FrozenModel):
    """What an active set serves: its scenarios, and their blocks merged in set order.

    A later scenario's entry replaces an earlier one's in every merged block
    but ``couple``, whose terms add up, and ``archiver``, whose events on one
    channel run in set order. A set that does not compose is served as
    ``nominal`` alone.

    Attributes:
        served: The served scenarios the view lists, in set order.
        overlaps: The overlaps that kept the set's scenarios out; empty when
            it composes.
        overrides: The active writes, by address.
        faults: Each model's fault seeds, from its active (not ``inactive``)
            faults.
        couplings: Each coupled address's terms, each with its driver's
            ``drive``; a term coupling to a driver no served scenario
            declares is left out, so an address may hold none.
        noise: The noise records standing in for the seeds', by address.
        still: The readings served without motion.
        channel_faults: The channel faults, by address.
        archiver_events: Each channel's archiver events, in set order; a
            channel an ``archiver`` entry names with no events holds none.
    """

    served: FrozenTuple[ScenarioRecord] = ()
    overlaps: FrozenTuple[Overlap] = ()
    overrides: ReadOnlyMapping[str, object]
    faults: ReadOnlyMapping[str, ReadOnlyMapping[str, object]]
    couplings: ReadOnlyMapping[str, FrozenTuple[ResolvedCoupling]]
    noise: ReadOnlyMapping[str, NoiseRecord]
    still: frozenset[str] = frozenset()
    channel_faults: ReadOnlyMapping[str, ChannelFaultKind]
    archiver_events: ReadOnlyMapping[str, FrozenTuple[ArchiverEvent]]

    @property
    def names(self) -> tuple[str, ...]:
        """The served set's names, ``nominal`` first whether or not the view lists it."""
        return (
            DEFAULT_SCENARIO,
            *(scenario.name for scenario in self.served if scenario.name != DEFAULT_SCENARIO),
        )


def compose(scenarios: Mapping[str, ScenarioRecord], active: ActiveState) -> ComposedSet:
    """The set a reader serves for ``active``: its scenarios merged, or ``nominal`` alone.

    A name the view does not list is logged and left out. A set whose
    scenarios write one target twice does not compose, and a reader serves
    the machine without its scenarios rather than in an order-dependent state.

    Args:
        scenarios: The simulator view's scenarios, by name.
        active: The active set.

    Returns:
        The served set and its merged blocks.
    """
    names: list[str] = []
    for name in active.names:
        if name in scenarios or name == DEFAULT_SCENARIO:
            names.append(name)
        else:
            logger.warning(f"Unknown scenario {name!r} in the active set; ignoring")
    overlaps = validate_composition(scenarios, names)
    if overlaps:
        names = [DEFAULT_SCENARIO]
    served = [scenarios[name] for name in names if name in scenarios]

    overrides: dict[str, object] = {}
    faults: dict[str, dict[str, object]] = {}
    drivers: dict[str, DriveRecord] = {}
    couple: dict[str, list[CouplingTerm]] = {}
    noise: dict[str, NoiseRecord] = {}
    still: set[str] = set()
    channel_faults: dict[str, ChannelFaultKind] = {}
    archiver_events: dict[str, list[ArchiverEvent]] = {}
    for scenario in served:
        overrides.update(scenario.overrides)
        for model, fault in scenario.faults.items():
            if not fault.inactive:
                faults.setdefault(model, {}).update(fault.writes)
        drivers.update(scenario.drivers)
        for address, terms in scenario.couple.items():
            couple.setdefault(address, []).extend(terms)
        noise.update(scenario.noise)
        still.update(scenario.still)
        channel_faults.update(scenario.channel_faults)
        for entry in scenario.archiver:
            archiver_events.setdefault(entry.channel, []).extend(entry.events)

    couplings: dict[str, tuple[ResolvedCoupling, ...]] = {}
    for address, merged in couple.items():
        resolved: list[ResolvedCoupling] = []
        for term in merged:
            drive = drivers.get(term.driver)
            if drive is None:
                logger.warning(f"{address} couples to undeclared driver {term.driver!r}")
                continue
            resolved.append(
                ResolvedCoupling(
                    driver=term.driver, gain=term.gain, gain_wander=term.gain_wander, drive=drive
                )
            )
        couplings[address] = tuple(resolved)
    return ComposedSet(
        served=tuple(served),
        overlaps=tuple(overlaps),
        overrides=overrides,
        faults=faults,
        couplings=couplings,
        noise=noise,
        still=frozenset(still),
        channel_faults=channel_faults,
        archiver_events={address: tuple(events) for address, events in archiver_events.items()},
    )


class OverlapLogRecord(FrozenModel):
    """The model log record a process writes when it serves without a set's scenarios.

    One JSON line in the simulator log, its keys sorted; ``osprey sim status``
    reads it back.

    Attributes:
        event: Always ``scenario-overlap``.
        instance: The serving instance that logs it.
        pid: The process id of that instance.
        model: The model whose log holds it.
        target: The target the overlap is on.
    """

    event: Literal["scenario-overlap"] = _OVERLAP_EVENT
    instance: ServingInstance
    pid: int
    model: str
    target: str

    @classmethod
    def from_line(cls, line: str) -> OverlapLogRecord | None:
        """The overlap record one model log line holds.

        Args:
            line: One line of a model log.

        Returns:
            The record, or ``None`` when the line is not an overlap record
            (another event, not JSON, or not the record's shape).
        """
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(record, dict) or record.get("event") != _OVERLAP_EVENT:
            return None
        try:
            return cls.model_validate(record)
        except ValidationError:
            return None

    def describe(self) -> str:
        """One printable line naming where the record came from, never taken for a status.

        Returns:
            ``<model> (log, instance <i>, pid <p>): scenario-overlap on <target>``.
        """
        return (
            f"{self.model} (log, instance {self.instance}, pid {self.pid}): "
            f"{self.event} on {self.target}"
        )
