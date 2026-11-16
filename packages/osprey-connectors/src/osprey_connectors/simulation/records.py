"""The records a simulator view's documents carry, as models.

``osprey build`` validates the facility's YAML and writes the simulator view
from it; :class:`~osprey_connectors.simulation.view.SimulatorView` refuses a
view file of another schema, and validates each file it reads against its
document model here, so a record read from a view has the shape stated here or
the view is refused with a :class:`~osprey_connectors.simulation.view.ViewSchemaError`.

A number is an ``int`` or a ``float``, never a ``bool``, and keeps the type the
document wrote, so a record writes back the bytes it was read from. A record is
frozen as the reader serves it: its lists are tuples and its mappings
read-only, and a value the view carries verbatim (a model's ``settings``, a
scenario's ``overrides``, a seed's ``nominal``) is served the same way, its
lists as tuples and its mappings read-only. A list given to a field that holds
a tuple is taken as that tuple, so a record also builds from a mapping read
from YAML.

A record states an optional key by a default meaning "not stated". Where a
reader tells a key the document leaves out apart from one it states, the
reader asks the record's ``model_fields_set``.

The closed vocabularies the records carry — a channel's :data:`ChannelRole`, a
wiring record's :data:`Direction`, :data:`EngineRole`, :data:`Plane`,
:data:`Refresh` and :data:`Mark`, a scenario's :data:`ChannelFaultKind` — are
named here, the lowest module every reader of a view imports, and a word
outside one refuses the view like any other shape it breaks.

The module imports only the standard library, pydantic and the package's
model bases.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Literal, TypeAlias

from pydantic import (
    AfterValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    field_validator,
    model_validator,
)

from osprey_connectors.model import FrozenModel, FrozenTuple, ReadOnlyMapping, empty_mapping

__all__ = [
    "AddressesDocument",
    "ArchiverEvent",
    "ChannelFaultKind",
    "ChannelRecord",
    "ChannelRole",
    "CouplingTerm",
    "Direction",
    "DriveRecord",
    "EngineRole",
    "GateRecord",
    "LinearTerm",
    "Mark",
    "ModelEntry",
    "NoiseRecord",
    "Plane",
    "PulseRecord",
    "Refresh",
    "ResolvedCoupling",
    "ScenarioRecord",
    "ScenariosDocument",
    "SeedRecord",
    "SeedsDocument",
    "ServedModelsDocument",
    "TopoffRecord",
    "VariablesDocument",
    "WiringRecord",
]

#: What a program does with a channel, the facility schema's ``role_enum``:
#: written (``setpoint``), read (``readback``), or neither (``none``).
ChannelRole = Literal["setpoint", "readback", "none"]

#: Which way a wiring record moves its value, computed from its channel's role.
Direction = Literal["read", "write"]

#: What a model's engine says a wiring record is, the schema's ``engine_role_enum``.
EngineRole = Literal["setpoint", "readback", "monitor", "output"]

#: The transverse plane a wiring record steers or reads, the schema's ``plane_enum``.
Plane = Literal["x", "y"]

#: When a wiring record's value is recomputed, the schema's ``refresh_enum``:
#: ``pass`` with each write's solve, ``periodic`` with the periodic solve alone,
#: ``fast`` at the fast period, ``event`` with each event the engine raises.
Refresh = Literal["pass", "periodic", "fast", "event"]

#: A mark an engine sets on a wiring record, the schema's ``wiring_mark_enum``.
Mark = Literal["command"]

#: A fault of a simulated channel itself, the word a scenario's
#: ``channel_faults`` maps an address to; its semantics are
#: :data:`~osprey_connectors.simulation.channel_faults.CHANNEL_FAULTS`.
ChannelFaultKind = Literal["stuck", "frozen", "disconnected"]


def _freeze(value: object) -> object:
    """A read-only copy of a JSON value: mappings as proxies, lists as tuples."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: object) -> object:
    """A JSON value served read-only, as the plain dicts and lists it was read from."""
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


#: A number a document states: an ``int`` or a ``float``, kept as written.
_Real: TypeAlias = int | float

#: A value the view carries verbatim, served read-only and written back as read.
_Verbatim: TypeAlias = Annotated[object, AfterValidator(_freeze), PlainSerializer(_thaw)]


class _Record(FrozenModel):
    """A view document's record: a :class:`FrozenModel` that writes a non-finite number as read.

    ``model_dump(mode="json")`` keeps an infinite or not-a-number float a float,
    as :func:`json.loads` read it, rather than writing it as ``None``.
    """

    model_config = ConfigDict(ser_json_inf_nan="constants")


class NoiseRecord(_Record):
    """A ``noise`` record: one term, a sigma in the channel's unit or a fraction of the reading.

    A reader tells which term the record states from its ``model_fields_set``.
    """

    absolute: _Real | None = None
    relative: _Real | None = None


class _WanderRecord(_Record):
    """A seed's ``drift`` or a coupling's ``gain_wander``: a bounded slow wander."""

    amplitude: _Real
    period_s: _Real


class DriveRecord(_Record):
    """A scenario driver's ``drive``, the shared latent signal its couplings read."""

    kind: Literal["wander"]
    amplitude: _Real
    period_s: _Real


class CouplingTerm(_Record):
    """One term of a scenario's ``couple`` block; ``gain_wander`` is ``None`` when not stated."""

    driver: str
    gain: _Real
    gain_wander: _WanderRecord | None = None


class ResolvedCoupling(CouplingTerm):
    """A coupling term with its driver's ``drive``."""

    drive: DriveRecord


class TopoffRecord(_Record):
    """A ``topoff`` seed: a stored current that decays and refills."""

    ceiling: _Real
    lifetime_h: _Real
    period_s: _Real


class _LifetimeRecord(_Record):
    """A ``topoff_lifetime`` seed: the channel whose ``topoff`` lifetime it holds."""

    of: str


class PulseRecord(_Record):
    """A ``pulse`` seed: the driver and enable channels a pulsed waveform reads."""

    driver: str
    enable: str


class LinearTerm(_Record):
    """One input of a ``linear`` seed stated as a record."""

    coefficient: _Real


class GateRecord(_Record):
    """A ``gate`` seed: its inputs' blocking labels and the gated value."""

    inputs: ReadOnlyMapping[str, _Verbatim]
    gated: bool | _Real | str


class SeedRecord(_Record):
    """One channel's seed in ``seeds.json``; a key the seed does not state is ``None``."""

    nominal: _Verbatim = None
    noise: NoiseRecord | None = None
    drift: _WanderRecord | None = None
    clamp: FrozenTuple[_Real | None] | None = None
    linear: ReadOnlyMapping[str, _Real | LinearTerm] | None = None
    gate: GateRecord | None = None
    pulse: PulseRecord | None = None
    topoff: TopoffRecord | None = None
    topoff_lifetime: _LifetimeRecord | None = None


class ChannelRecord(_Record):
    """One channel entry of ``variables.json``.

    ``value_range`` is ``[min, max]`` when the channel's limits record states
    both, else ``None``; a reader takes a ``None`` side as unbounded.
    ``options``, ``shape`` and ``precision`` are ``None`` on a channel that
    states none.
    """

    address: str
    role: ChannelRole
    pair: str | None
    value_type: str
    unit: str | None
    description: str | None
    writable: bool
    value_range: FrozenTuple[_Real | None] | None
    owner: str
    on: ReadOnlyMapping[str, str] | None
    options: FrozenTuple[str] | None = None
    shape: FrozenTuple[int] | None = None
    precision: int | None = None


class WiringRecord(_Record):
    """One wiring record of a model in ``variables.json``.

    The record holds the wiring record's authored keys and the slots the build
    filled; a key the record does not state is ``None``, and a reader telling
    a stated ``None`` (a ``plane`` of no plane, say) apart from a key left out
    asks ``model_fields_set``. The keys an engine reads and the view does not
    are carried verbatim.
    """

    id: str | None = None
    address: str
    element: str | None = None
    slices: _Verbatim = None
    engine: _Verbatim = None
    calibration: _Verbatim = None
    direction: Direction | None = None
    unit: _Verbatim = None
    default: _Verbatim = None
    value_range: _Verbatim = None
    engine_role: EngineRole | None = None
    plane: Plane | None = None
    refresh: Refresh | None = None
    marks: FrozenTuple[Mark] | None = None
    value_type: str | None = None
    shape: FrozenTuple[int] | None = None


class ModelEntry(_Record):
    """One model entry of ``variables.json``; ``inputs`` is empty when the model declares none."""

    name: str
    engine: str | None = None
    served: bool = False
    settings: ReadOnlyMapping[str, _Verbatim] = Field(default_factory=empty_mapping)
    deck: str | None = None
    wiring: FrozenTuple[WiringRecord] = ()
    inputs: FrozenTuple[ReadOnlyMapping[str, _Verbatim]] = ()


class _When(_Record):
    """A calendar placement, ``{days_ago, time}``, read in the facility timezone.

    ``days_ago`` is an ``int`` of 0 or more, never a ``bool``; ``time`` is an
    ``HH:MM[:SS]`` time of day without a timezone offset.
    """

    days_ago: int
    time: str


class ArchiverEvent(_Record):
    """One event of a scenario's ``archiver`` entry; a key the event does not state is ``None``.

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

    The model checks the types, and refuses a key stated as ``null``: a
    reader tells the position and end keys apart by which the event states.
    The rules between keys are the build's.
    """

    shape: Literal["step", "ramp", "spike", "burst", "loss"]
    at: _Real | None = None
    at_offset: _Real | None = None
    at_when: _When | None = None
    at_time: str | None = None
    until: _Real | None = None
    until_offset: _Real | None = None
    until_when: _When | None = None
    to: bool | _Real | str | FrozenTuple[_Verbatim] | None = None
    amplitude: _Real | None = None
    width: _Real | None = None
    decay_s: _Real | None = None
    tail_share: _Real | None = None
    tail_decay_s: _Real | None = None
    amount: _Real | None = None

    @model_validator(mode="before")
    @classmethod
    def _stated(cls, event: object) -> object:
        """An event states a key with a value; ``None`` is a key the event leaves out."""
        if isinstance(event, Mapping):
            stated: Mapping[object, object] = event
            nulls = sorted(str(key) for key, value in stated.items() if value is None)
            if nulls:
                raise ValueError(f"an event states {', '.join(nulls)} as null")
        return event


class _ArchiverEntry(_Record):
    """One entry of a scenario's ``archiver`` block: a channel and its events."""

    channel: str
    events: FrozenTuple[ArchiverEvent] = ()


class _FaultRecord(_Record):
    """One model's fault in a scenario's ``faults`` block; ``inactive`` is ``None`` when active."""

    writes: ReadOnlyMapping[str, _Verbatim]
    inactive: str | None = None


class ScenarioRecord(_Record):
    """One scenario of ``scenarios.json``; a block the scenario does not state is empty.

    ``description`` and ``sessions`` are ``None`` when not stated. A scenario
    writes back only the blocks it states (``model_dump(exclude_unset=True)``).
    """

    name: str
    description: str | None = None
    overrides: ReadOnlyMapping[str, _Verbatim] = Field(default_factory=empty_mapping)
    archiver: FrozenTuple[_ArchiverEntry] = ()
    logbook: FrozenTuple[ReadOnlyMapping[str, _Verbatim]] = ()
    sessions: _Verbatim = None
    drivers: ReadOnlyMapping[str, DriveRecord] = Field(default_factory=empty_mapping)
    couple: ReadOnlyMapping[str, FrozenTuple[CouplingTerm]] = Field(default_factory=empty_mapping)
    noise: ReadOnlyMapping[str, NoiseRecord] = Field(default_factory=empty_mapping)
    channel_faults: ReadOnlyMapping[str, ChannelFaultKind] = Field(default_factory=empty_mapping)
    still: tuple[str, ...] = ()
    faults: ReadOnlyMapping[str, _FaultRecord] = Field(default_factory=empty_mapping)

    @field_validator("still", mode="before")
    @classmethod
    def _addresses(cls, still: object) -> object:
        """``still`` is a list of addresses: the build resolves its ``all`` and selections."""
        if isinstance(still, list | tuple) and all(isinstance(address, str) for address in still):
            return tuple(still)
        raise ValueError(f"`still` is {still!r}, not a list of addresses")


class ServedModelsDocument(_Record):
    """``served_models.json``: the models the render serves, in serving order."""

    schema_id: str = Field(alias="schema")
    models: FrozenTuple[str]


class AddressesDocument(_Record):
    """``addresses.json``: every channel address, sorted, and the simulator's own addresses.

    ``status`` lists each served physics model's status address and
    ``ioc_health`` each served model's IOC health addresses; a build before
    ``ioc_health`` was written lists none.
    """

    schema_id: str = Field(alias="schema")
    channels: FrozenTuple[str]
    status: FrozenTuple[str]
    ioc_health: FrozenTuple[str] = ()


class VariablesDocument(_Record):
    """``variables.json``: the facility's code, its models sorted by name, its channels sorted."""

    schema_id: str = Field(alias="schema")
    code: str
    models: FrozenTuple[ModelEntry]
    channels: FrozenTuple[ChannelRecord]


class SeedsDocument(_Record):
    """``seeds.json``: each seeded channel's seed, by address."""

    schema_id: str = Field(alias="schema")
    seeds: ReadOnlyMapping[str, SeedRecord]


class ScenariosDocument(_Record):
    """``scenarios.json``: every scenario, sorted by name."""

    schema_id: str = Field(alias="schema")
    scenarios: FrozenTuple[ScenarioRecord]
