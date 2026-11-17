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

Each view file is a :class:`ViewDocument`, which names the file
(``FILE``), the schema it carries (``SCHEMA``) and the older schemas this
OSPREY still reads (``ALSO_READS``); :data:`VIEW_DOCUMENTS` lists the five.
The reader serves these records themselves: a channel as its
:class:`ChannelRecord`, a model as its :class:`ModelRecord`, a model's wiring
as its :class:`WiringRecord` items.

The closed vocabularies the records carry — a channel's :data:`ChannelRole`, a
wiring record's :data:`Direction`, :data:`EngineRole`, :data:`Plane`,
:data:`Refresh` and :data:`Mark`, a scenario's :data:`ChannelFaultKind` — are
named here, the lowest module every reader of a view imports, and a word
outside one refuses the view like any other shape it breaks.

The module imports only the standard library, pydantic, the package's model
bases and :mod:`osprey_connectors.simulation.values`, which imports only the
standard library.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, ClassVar, Literal, Self, TypeAlias

from pydantic import (
    AfterValidator,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    field_validator,
    model_validator,
)

from osprey_connectors.model import FrozenModel, FrozenTuple, ReadOnlyMapping, empty_mapping
from osprey_connectors.simulation import values
from osprey_connectors.simulation.values import DEFAULT_BOOL_OPTIONS, StoredValue, ValueType

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
    "ModelRecord",
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
    "VIEW_DOCUMENTS",
    "VariablesDocument",
    "ViewDocument",
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


def _as_pair(value: object) -> object:
    """A list as the tuple a pair field holds; any other value as it is, for the field to judge."""
    return tuple(value) if isinstance(value, list) else value


#: A ``[min, max]`` pair a document states, both sides numbers.
_Bounds: TypeAlias = Annotated[tuple[_Real, _Real], BeforeValidator(_as_pair)]


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

    Attributes:
        address: The channel's address.
        role: The facility file's role, ``readback`` when it states none.
        pair: A setpoint's readback, itself when it names none; ``None`` on
            any other role.
        value_type: The channel's :data:`~osprey_connectors.simulation.values.ValueType`.
        unit: The channel's unit, or ``None``.
        description: The channel's description, or ``None``.
        writable: Whether a write to the channel is allowed.
        value_range: ``(min, max)`` when the channel's limits record states
            both, else ``None``.
        owner: The model wiring the address, else ``texture``.
        on: The node the channel is on, ``{device: id}`` or ``{place: id}``,
            or ``None``.
        options: A ``bool`` or ``enum`` channel's labels, or ``None``.
        shape: A ``waveform`` channel's shape.
        precision: A ``float`` channel's display precision, or ``None``.

    A channel states ``options`` only on a ``bool`` or ``enum`` channel,
    ``shape`` exactly on a ``waveform`` one and ``precision`` only on a
    ``float`` one, as the build checks them; any other channel is refused.
    """

    address: str
    role: ChannelRole
    pair: str | None
    value_type: ValueType
    unit: str | None
    description: str | None
    writable: bool
    value_range: _Bounds | None
    owner: str
    on: ReadOnlyMapping[str, str] | None
    options: FrozenTuple[str] | None = None
    shape: FrozenTuple[int] | None = None
    precision: int | None = None

    @model_validator(mode="after")
    def _value_keys(self) -> Self:
        """``options``, ``shape`` and ``precision`` belong to the value types that take them."""
        value_type = self.value_type
        if self.options is not None and value_type not in values.LABELLED_TYPES:
            raise ValueError(f"a {value_type} channel carries `options`")
        if values.is_waveform(value_type):
            if not self.shape or any(size <= 0 for size in self.shape):
                raise ValueError("a waveform channel needs `shape`, a list of positive ints")
        elif self.shape is not None:
            raise ValueError(f"a {value_type} channel carries `shape`")
        if self.precision is not None and value_type != "float":
            raise ValueError(f"a {value_type} channel carries `precision`")
        return self

    def coerce(self, value: object) -> StoredValue:
        """``value`` in the stored representation of the channel's value type.

        Raises:
            ValueError: The channel's value type refuses ``value``.
        """
        return values.coerce(value, self.value_type, self.options, self.shape)

    def zero(self) -> StoredValue:
        """The zero of the channel's value type, in its stored representation.

        Raises:
            ValueError: An ``enum`` channel states no labels.
        """
        return values.zero(self.value_type, self.options, self.shape)

    @property
    def labels(self) -> tuple[str, ...]:
        """A ``bool`` or ``enum`` channel's labels: its ``options``, or the default bool labels.

        Raises:
            ValueError: The channel is of another value type, which carries no labels.
        """
        if self.value_type not in values.LABELLED_TYPES:
            raise ValueError(
                f"channel {self.address} is a {self.value_type} channel, with no labels"
            )
        return self.options or DEFAULT_BOOL_OPTIONS


class WiringRecord(_Record):
    """One wiring record of a model in ``variables.json``.

    The record holds the wiring record's authored keys and the slots the build
    filled. ``id``, ``address``, ``direction``, ``engine_role``, ``plane`` and
    ``refresh`` are stated on every record, ``plane`` as ``None`` for a record
    steering or reading no plane; another key the record does not state is
    ``None`` (``marks`` empty). The keys an engine reads and the view does not
    are carried verbatim.

    Attributes:
        id: The wiring record's id.
        address: The channel address the record serves.
        element: The element the record names, or ``None`` (a record naming
            slices, or none).
        direction: ``write`` for a setpoint channel, else ``read``.
        default: The engine's start value for the address, or ``None``.
        engine_role: What the model's engine says the record is. A
            ``readback`` reads an element's setting, paired with a setpoint
            or not; the setpoint channel's ``pair`` names its readback.
        plane: ``x`` or ``y`` for a record steering or reading one transverse
            plane, else ``None``.
        refresh: When the record's value is recomputed.
        marks: The marks the record carries; empty when it carries none.
    """

    id: str
    address: str
    element: str | None = None
    slices: _Verbatim = None
    engine: _Verbatim = None
    calibration: _Verbatim = None
    direction: Direction
    unit: _Verbatim = None
    default: _Verbatim = None
    value_range: _Verbatim = None
    engine_role: EngineRole
    plane: Plane | None
    refresh: Refresh
    marks: FrozenTuple[Mark] = ()
    value_type: ValueType | None = None
    shape: FrozenTuple[int] | None = None

    def engine_entry(self) -> dict[str, object]:
        """The record as a model's engine takes it: the keys it states, as plain JSON values."""
        entry: dict[str, object] = self.model_dump(mode="json", exclude_unset=True)
        return entry


class ModelRecord(_Record):
    """One model entry of ``variables.json``.

    Attributes:
        name: The model's name.
        engine: The engine the model names.
        served: Whether the render serves the model.
        settings: The model's settings, read-only.
        deck: The path of the view's copy of the model's deck, relative to
            the view (:meth:`~osprey_connectors.simulation.view.SimulatorView.deck_path`),
            or ``None``.
        wiring: The model's wiring records, in wiring order; empty for a
            model without wiring.
        inputs: The model's inputs, in declared order, each a read-only
            ``{address, engine}`` record: the producer channel's address and
            the engine block the consuming engine reads; empty when the model
            declares none.
    """

    name: str
    engine: str | None = None
    served: bool = False
    settings: ReadOnlyMapping[str, _Verbatim] = Field(default_factory=empty_mapping)
    deck: str | None = None
    wiring: FrozenTuple[WiringRecord] = ()
    inputs: FrozenTuple[ReadOnlyMapping[str, _Verbatim]] = ()

    def engine_settings(self) -> dict[str, object]:
        """The model's settings as its engine takes them: plain JSON values."""
        return {key: _thaw(value) for key, value in self.settings.items()}


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


class ViewDocument(_Record):
    """One file of a simulator view: its document, and the file and schema it is.

    ``FILE`` is the file's name under the view, ``SCHEMA`` the schema this
    OSPREY writes and ``ALSO_READS`` the older schemas it still reads.
    """

    FILE: ClassVar[str]
    SCHEMA: ClassVar[str]
    ALSO_READS: ClassVar[frozenset[str]] = frozenset()

    schema_id: str = Field(alias="schema")


class ServedModelsDocument(ViewDocument):
    """``served_models.json``: the models the render serves, in serving order."""

    FILE: ClassVar[str] = "served_models.json"
    SCHEMA: ClassVar[str] = "osprey.facility.served_models/1"

    models: FrozenTuple[str]


class AddressesDocument(ViewDocument):
    """``addresses.json``: every channel address, sorted, and the simulator's own addresses.

    ``status`` lists each served physics model's status address and
    ``ioc_health`` each served model's IOC health addresses; a build before
    ``ioc_health`` was written lists none.
    """

    FILE: ClassVar[str] = "addresses.json"
    SCHEMA: ClassVar[str] = "osprey.facility.addresses/1"

    channels: FrozenTuple[str]
    status: FrozenTuple[str]
    ioc_health: FrozenTuple[str] = ()


class VariablesDocument(ViewDocument):
    """``variables.json``: the facility's code, its models sorted by name, its channels sorted.

    A released ``/3`` file reads as one whose models declare no inputs.
    """

    FILE: ClassVar[str] = "variables.json"
    SCHEMA: ClassVar[str] = "osprey.facility.simulator/4"
    ALSO_READS: ClassVar[frozenset[str]] = frozenset({"osprey.facility.simulator/3"})

    code: str
    models: FrozenTuple[ModelRecord]
    channels: FrozenTuple[ChannelRecord]


class SeedsDocument(ViewDocument):
    """``seeds.json``: each seeded channel's seed, by address."""

    FILE: ClassVar[str] = "seeds.json"
    SCHEMA: ClassVar[str] = "osprey.facility.seeds/2"

    seeds: ReadOnlyMapping[str, SeedRecord]


class ScenariosDocument(ViewDocument):
    """``scenarios.json``: every scenario, sorted by name."""

    FILE: ClassVar[str] = "scenarios.json"
    SCHEMA: ClassVar[str] = "osprey.facility.scenarios/3"

    scenarios: FrozenTuple[ScenarioRecord]


#: Every document of a simulator view, in the order the build writes them.
VIEW_DOCUMENTS: tuple[type[ViewDocument], ...] = (
    ServedModelsDocument,
    AddressesDocument,
    VariablesDocument,
    SeedsDocument,
    ScenariosDocument,
)
