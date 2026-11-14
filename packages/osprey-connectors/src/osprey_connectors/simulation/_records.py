"""The shapes of the records a simulator view's documents carry, for the type checker.

``osprey build`` validates the facility's YAML and writes the simulator view
from it; :class:`~osprey_connectors.simulation.view.SimulatorView` refuses a
view file of another schema. A record read from a view of the schema this
OSPREY reads therefore has the shape stated here, and the reader states it
where it takes a document's top-level fields. A number is typed ``float``,
which admits the ``int`` a JSON number may parse to.

A view document is frozen as the reader serves it: its lists are tuples and
its mappings read-only, so a list is typed as a ``Sequence`` and a mapping as a
``Mapping``.

The module imports only the standard library.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NotRequired, Required, TypedDict


class NoiseRecord(TypedDict, total=False):
    """A ``noise`` record: one term, a sigma in the channel's unit or a fraction of the reading."""

    absolute: float
    relative: float


class WanderRecord(TypedDict):
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
    gain_wander: NotRequired[WanderRecord]


class ResolvedCoupling(CouplingTerm):
    """A coupling term with its driver's ``drive``."""

    drive: DriveRecord


class TopoffRecord(TypedDict):
    """A ``topoff`` seed: a stored current that decays and refills."""

    ceiling: float
    lifetime_h: float
    period_s: float


class LifetimeRecord(TypedDict):
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


class SeedRecord(TypedDict, total=False):
    """One channel's seed in ``seeds.json``."""

    nominal: object
    noise: NoiseRecord
    drift: WanderRecord
    clamp: Sequence[float | None]
    linear: Mapping[str, float | LinearTerm]
    gate: GateRecord
    pulse: PulseRecord
    topoff: TopoffRecord
    topoff_lifetime: LifetimeRecord


class ChannelRecord(TypedDict):
    """One channel entry of ``variables.json``."""

    address: str
    role: str
    pair: str | None
    value_type: str
    unit: str | None
    description: str | None
    writable: bool
    value_range: Sequence[float | None] | None
    owner: str
    on: Mapping[str, str] | None
    options: NotRequired[Sequence[str]]
    shape: NotRequired[Sequence[int]]
    precision: NotRequired[int]


class WiringRecord(TypedDict, total=False):
    """One wiring record of a model in ``variables.json``."""

    id: str
    address: Required[str]
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


class When(TypedDict):
    """A calendar placement, ``{days_ago, time}``, read in the facility timezone."""

    days_ago: int
    time: str


class ArchiverEvent(TypedDict, total=False):
    """One event of a scenario's ``archiver`` entry."""

    shape: Required[str]
    at: float
    at_offset: float
    at_when: When
    at_time: str
    until: float
    until_offset: float
    until_when: When
    to: float | str
    amplitude: float
    width: float
    decay_s: float
    tail_share: float
    tail_decay_s: float
    amount: float


class ArchiverEntry(TypedDict):
    """One entry of a scenario's ``archiver`` block: a channel and its events."""

    channel: str
    events: NotRequired[Sequence[ArchiverEvent]]


class FaultRecord(TypedDict):
    """One model's fault in a scenario's ``faults`` block."""

    writes: Mapping[str, object]
    inactive: NotRequired[str]


class ScenarioRecord(TypedDict, total=False):
    """One scenario of ``scenarios.json``."""

    name: Required[str]
    description: str
    overrides: Mapping[str, object]
    archiver: Sequence[ArchiverEntry]
    logbook: Sequence[Mapping[str, object]]
    sessions: object
    drivers: Mapping[str, DriveRecord]
    couple: Mapping[str, Sequence[CouplingTerm]]
    noise: Mapping[str, NoiseRecord]
    channel_faults: Mapping[str, str]
    still: Sequence[str]
    faults: Mapping[str, FaultRecord]
