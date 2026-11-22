"""The shapes of the records a simulator view's documents carry.

``osprey build`` validates the facility's YAML and writes the simulator view
from it; :class:`~osprey_connectors.simulation.view.SimulatorView` refuses a
view file of another schema. A record read from a view of the schema this
OSPREY reads therefore has the shape stated here, and the reader states it
where it takes a document's top-level fields. A number is typed ``float``,
which admits the ``int`` a JSON number may parse to.

A view document is frozen as the reader serves it: its lists are tuples and
its mappings read-only, so a list is typed as a ``Sequence`` and a mapping as a
``Mapping``.

The module imports only the standard library. Its types are public so that
code outside the simulation package can name the records it passes in; they
exist only for the type checker, and nothing checks a record against them at
runtime.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, NotRequired, Required, TypedDict

__all__ = [
    "AddressesDocument",
    "ArchiverEvent",
    "ChannelRecord",
    "CouplingTerm",
    "DriveRecord",
    "GateRecord",
    "LinearTerm",
    "ModelEntry",
    "NoiseRecord",
    "PulseRecord",
    "ResolvedCoupling",
    "RespondsRecord",
    "ScenarioRecord",
    "ScenariosDocument",
    "SeedRecord",
    "SeedsDocument",
    "ServedModelsDocument",
    "TopoffRecord",
    "VariablesDocument",
    "WiringRecord",
]


class NoiseRecord(TypedDict, total=False):
    """A ``noise`` record: one term, a sigma in the channel's unit or a fraction of the reading."""

    absolute: float
    relative: float


class _WanderRecord(TypedDict):
    """A seed's ``drift`` or a coupling's ``gain_wander``: a bounded slow wander."""

    amplitude: float
    period_s: float


class DriveRecord(TypedDict):
    """A scenario driver's ``drive``, the shared latent signal its couplings read."""

    kind: str
    amplitude: float
    period_s: float


class CouplingTerm(TypedDict):
    """One term of a scenario's ``couple`` block."""

    driver: str
    gain: float
    gain_wander: NotRequired[_WanderRecord]


class ResolvedCoupling(CouplingTerm):
    """A coupling term with its driver's ``drive``."""

    drive: DriveRecord


class TopoffRecord(TypedDict):
    """A ``topoff`` seed: a stored current that decays and refills."""

    ceiling: float
    lifetime_h: float
    period_s: float


class _LifetimeRecord(TypedDict):
    """A ``topoff_lifetime`` seed: the channel whose ``topoff`` lifetime it holds."""

    of: str


class PulseRecord(TypedDict):
    """A ``pulse`` seed: the driver and enable channels a pulsed waveform reads."""

    driver: str
    enable: str


class LinearTerm(TypedDict):
    """One input of a ``linear`` seed stated as a record."""

    coefficient: float


class GateRecord(TypedDict):
    """A ``gate`` seed: its inputs' blocking labels and the gated value."""

    inputs: Mapping[str, object]
    gated: float | str


class RespondsRecord(TypedDict):
    """A ``responds`` seed: the value a channel reads for ``for_s`` seconds after each write to ``to``."""

    to: str
    for_s: float
    value: str | int


class SeedRecord(TypedDict, total=False):
    """One channel's seed in ``seeds.json``."""

    nominal: object
    noise: NoiseRecord
    drift: _WanderRecord
    clamp: Sequence[float | None]
    linear: Mapping[str, float | LinearTerm]
    gate: GateRecord
    pulse: PulseRecord
    topoff: TopoffRecord
    topoff_lifetime: _LifetimeRecord
    responds: RespondsRecord


class ChannelRecord(TypedDict):
    """One channel entry of ``variables.json``.

    ``value_range`` is ``[min, max]`` when the channel's limits record states
    both, else ``None``. ``started_by`` names the command whose write moves a
    setpoint's readback to the setpoint's held value.
    """

    address: str
    role: str
    pair: str | None
    value_type: str
    unit: str | None
    description: str | None
    writable: bool
    value_range: Sequence[float] | None
    owner: str
    on: Mapping[str, str] | None
    options: NotRequired[Sequence[str]]
    shape: NotRequired[Sequence[int]]
    precision: NotRequired[int]
    started_by: NotRequired[str]


class WiringRecord(TypedDict, total=False):
    """One wiring record of a model in ``variables.json``."""

    id: str
    address: Required[str]
    also: Sequence[str]
    element: str
    direction: str
    default: object
    engine_role: str
    plane: str | None
    refresh: str
    marks: Sequence[str]


class ModelEntry(TypedDict):
    """One model entry of ``variables.json``."""

    name: str
    engine: str | None
    served: bool
    settings: Mapping[str, object]
    deck: str | None
    wiring: Sequence[WiringRecord]
    inputs: NotRequired[Sequence[Mapping[str, object]]]


class _When(TypedDict):
    """A calendar placement, ``{days_ago, time}``, read in the facility timezone.

    ``days_ago`` is an ``int`` of 0 or more, never a ``bool``; ``time`` is an
    ``HH:MM[:SS]`` time of day without a timezone offset.
    """

    days_ago: int
    time: str


class ArchiverEvent(TypedDict, total=False):
    """One event of a scenario's ``archiver`` entry.

    The facility schema leaves a scenario's ``archiver`` block free-form;
    ``osprey build`` checks each event with
    :func:`osprey.facility.scenarios.check_scenario_events` and writes it to
    ``scenarios.json`` as stated, so an event of the view has this shape:

    * ``shape`` is one of the five shapes, and the event states exactly one
      position key: ``at`` (a window fraction from 0 to 1), ``at_offset``
      (seconds from the activation anchor), ``at_time`` (an ``HH:MM[:SS]``
      time of day) or ``at_when``.
    * A ``step`` or ``ramp`` states ``to``; a ``spike`` ``amplitude`` and
      ``width``; a ``burst`` ``amplitude`` and ``decay_s``, and ``tail_share``
      with ``tail_decay_s`` or neither; a ``loss`` ``amount``.
    * A ``ramp`` placed by ``at`` states ``until``, one placed by ``at_offset``
      ``until_offset``; a ``step`` may state the end its position pairs with
      (``until``, ``until_offset``, ``until_when``), but not on an ``at_time``
      step. ``until`` is a fraction from 0 to 1.
    * Every key read as a number is an ``int`` or a ``float``, never a
      ``bool``; ``width``, ``decay_s``, ``tail_decay_s`` and ``amount`` are
      above 0 and ``tail_share`` lies between 0 and 1.
    * ``to`` is whatever the channel's value type coerces: a number on a
      ``float`` or ``int`` channel; a label, an option index or a ``bool`` on a
      ``bool`` or ``enum`` channel; a string on a ``string`` channel; on a
      ``waveform`` channel a list of numbers, nested to the channel's
      ``shape``. A shape other than ``step`` is on a ``float`` channel only.
    """

    shape: Required[Literal["step", "ramp", "spike", "burst", "loss"]]
    at: float
    at_offset: float
    at_when: _When
    at_time: str
    until: float
    until_offset: float
    until_when: _When
    to: float | str | Sequence[object]
    amplitude: float
    width: float
    decay_s: float
    tail_share: float
    tail_decay_s: float
    amount: float


class _ArchiverEntry(TypedDict):
    """One entry of a scenario's ``archiver`` block: a channel and its events."""

    channel: str
    events: NotRequired[Sequence[ArchiverEvent]]


class _FaultRecord(TypedDict):
    """One model's fault in a scenario's ``faults`` block."""

    writes: Mapping[str, object]
    inactive: NotRequired[str]


class ScenarioRecord(TypedDict, total=False):
    """One scenario of ``scenarios.json``."""

    name: Required[str]
    description: str
    overrides: Mapping[str, object]
    archiver: Sequence[_ArchiverEntry]
    logbook: Sequence[Mapping[str, object]]
    sessions: object
    drivers: Mapping[str, DriveRecord]
    couple: Mapping[str, Sequence[CouplingTerm]]
    noise: Mapping[str, NoiseRecord]
    channel_faults: Mapping[str, str]
    stopped_servers: Sequence[str]
    still: Sequence[str]
    faults: Mapping[str, _FaultRecord]


class ServedModelsDocument(TypedDict):
    """``served_models.json``: the models the render serves, in serving order."""

    schema: str
    models: Sequence[str]


class AddressesDocument(TypedDict):
    """``addresses.json``: every channel address, sorted, and the simulator's own addresses.

    ``status`` lists each served physics model's status address and
    ``server_health`` each served model's server health addresses.
    ``ioc_health`` is the key that list was first written under, read when a
    document has no ``server_health``; a build that wrote neither lists none.
    """

    schema: str
    channels: Sequence[str]
    status: Sequence[str]
    server_health: NotRequired[Sequence[str]]
    ioc_health: NotRequired[Sequence[str]]


class VariablesDocument(TypedDict):
    """``variables.json``: the facility's code, its models sorted by name, its channels sorted."""

    schema: str
    code: str
    models: Sequence[ModelEntry]
    channels: Sequence[ChannelRecord]


class SeedsDocument(TypedDict):
    """``seeds.json``: each seeded channel's seed, by address."""

    schema: str
    seeds: Mapping[str, SeedRecord]


class ScenariosDocument(TypedDict):
    """``scenarios.json``: every scenario, sorted by name."""

    schema: str
    scenarios: Sequence[ScenarioRecord]
