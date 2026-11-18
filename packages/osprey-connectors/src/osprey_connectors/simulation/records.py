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

A scenario's archiver event is one of five records told apart by its
``shape`` (:data:`ArchiverEvent`), each holding the rules between its keys
and refusing an event that breaks one with an :class:`EventRuleError`; a
noise record is one of two told apart by the term it states (:data:`Noise`).

The module imports only the standard library, pydantic, the package's model
bases, its errors, :mod:`~osprey_connectors.relative_time` and
:mod:`osprey_connectors.simulation.values`, which imports only the standard
library.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import time as dtime
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, Self, TypeAlias

from pydantic import (
    AfterValidator,
    BeforeValidator,
    ConfigDict,
    Discriminator,
    Field,
    ModelWrapValidatorHandler,
    PlainSerializer,
    Tag,
    ValidationError,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    WrapValidator,
    field_validator,
    model_validator,
)
from typing_extensions import override

from osprey_connectors.errors import ConnectorsError
from osprey_connectors.model import FrozenModel, FrozenTuple, ReadOnlyMapping, empty_mapping
from osprey_connectors.relative_time import RelativeTimestamp, TimeOfDay
from osprey_connectors.simulation import values
from osprey_connectors.simulation.values import DEFAULT_BOOL_OPTIONS, StoredValue, ValueType

if TYPE_CHECKING:  # pragma: no cover - typing only
    from osprey_connectors.simulation.logbook import ScenarioLogEntry

__all__ = [
    "AbsoluteNoise",
    "AddressesDocument",
    "AnchorOffset",
    "ArchiverEntry",
    "ArchiverEvent",
    "BurstEvent",
    "CalendarDay",
    "ChannelFaultKind",
    "ChannelRecord",
    "ChannelRole",
    "CouplingTerm",
    "DailyTime",
    "Direction",
    "DriveRecord",
    "EngineRole",
    "EventRuleError",
    "GateRecord",
    "LinearTerm",
    "LogEntryRecord",
    "LossEvent",
    "Mark",
    "ModelFaultRecord",
    "ModelRecord",
    "Noise",
    "PictureRef",
    "Plane",
    "PlotRef",
    "PulseRecord",
    "RampEvent",
    "Refresh",
    "RelativeNoise",
    "ResolvedCoupling",
    "ScenarioRecord",
    "ScenariosDocument",
    "SeedRecord",
    "SeedsDocument",
    "ServedModelsDocument",
    "SpikeEvent",
    "StepEvent",
    "TopoffLifetimeRecord",
    "TopoffRecord",
    "VIEW_DOCUMENTS",
    "VariablesDocument",
    "ViewDocument",
    "WanderRecord",
    "WindowFraction",
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


class AbsoluteNoise(_Record):
    """A ``noise`` record stating a sigma in the channel's unit."""

    absolute: _Real


class RelativeNoise(_Record):
    """A ``noise`` record stating a sigma as a fraction of the reading."""

    relative: _Real


def _stated_term(noise: object) -> str | None:
    """The term a ``noise`` record states, by the key it carries; ``None`` when it carries neither."""
    if isinstance(noise, AbsoluteNoise | RelativeNoise):
        return "absolute" if isinstance(noise, AbsoluteNoise) else "relative"
    if isinstance(noise, Mapping):
        return next((term for term in ("absolute", "relative") if term in noise), None)
    return None


#: A ``noise`` record, ``{absolute: <sigma>}`` or ``{relative: <fraction>}``,
#: told apart by the key it states. A record stating both is refused (the
#: ``absolute`` record carries no ``relative``), as is one stating neither.
Noise: TypeAlias = Annotated[
    Annotated[AbsoluteNoise, Tag("absolute")] | Annotated[RelativeNoise, Tag("relative")],
    Discriminator(
        _stated_term,
        custom_error_type="noise_term",
        custom_error_message="a noise record states `absolute` or `relative`",
    ),
]


class WanderRecord(_Record):
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
    gain_wander: WanderRecord | None = None


class ResolvedCoupling(CouplingTerm):
    """A coupling term with its driver's ``drive``."""

    drive: DriveRecord


class TopoffRecord(_Record):
    """A ``topoff`` seed: a stored current that decays and refills."""

    ceiling: _Real
    lifetime_h: _Real
    period_s: _Real


class TopoffLifetimeRecord(_Record):
    """A ``topoff_lifetime`` seed: the channel whose ``topoff`` lifetime it holds."""

    of: str


class PulseRecord(_Record):
    """A ``pulse`` seed: the driver and enable channels a pulsed waveform reads."""

    driver: str
    enable: str


class LinearTerm(_Record):
    """One input of a ``linear`` seed stated as a record."""

    coefficient: _Real


#: A label a ``gate`` reads or writes: a ``bool``, a number or a string.
_Label: TypeAlias = bool | _Real | str


class _BlockingLabel(_Record):
    """A ``gate`` input stated as a record, ``{blocking_label: <label>}``."""

    blocking_label: _Label


class GateRecord(_Record):
    """A ``gate`` seed: its inputs' blocking labels and the gated value.

    An input's label is stated as itself or as the record
    ``{blocking_label: <label>}``; the record writes back the form it was read
    in, and :meth:`blocking_labels` reads either.
    """

    inputs: ReadOnlyMapping[str, _Label | _BlockingLabel]
    gated: _Label

    def blocking_labels(self) -> tuple[tuple[str, _Label], ...]:
        """``(input address, blocking label)`` of every input, sorted by address."""
        return tuple(
            (address, label.blocking_label if isinstance(label, _BlockingLabel) else label)
            for address, label in sorted(self.inputs.items())
        )


class SeedRecord(_Record):
    """One channel's seed in ``seeds.json``; a key the seed does not state is ``None``.

    An input of ``linear`` states its coefficient as a number or as a
    :class:`LinearTerm`; the record writes back the form it was read in, and
    :meth:`linear_terms` reads either.
    """

    nominal: _Verbatim = None
    noise: Noise | None = None
    drift: WanderRecord | None = None
    clamp: FrozenTuple[_Real | None] | None = None
    linear: ReadOnlyMapping[str, _Real | LinearTerm] | None = None
    gate: GateRecord | None = None
    pulse: PulseRecord | None = None
    topoff: TopoffRecord | None = None
    topoff_lifetime: TopoffLifetimeRecord | None = None

    def linear_terms(self) -> tuple[tuple[str, float], ...]:
        """``(input address, coefficient)`` of every ``linear`` input, sorted; empty without any."""
        return tuple(
            (address, float(term.coefficient if isinstance(term, LinearTerm) else term))
            for address, term in sorted((self.linear or {}).items())
        )


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


class EventRuleError(ConnectorsError, ValueError):
    """An archiver event breaks one of its rules.

    The event's validators raise it, so a view or a build that reads the event
    is refused with its words: ``detail`` names the rule the event breaks, as
    the end of a sentence whose subject is the event ("lacks `width`"), and
    ``remedy`` says how to correct it.

    Attributes:
        detail: What the event states that breaks the rule.
        remedy: How to correct it.
    """

    def __init__(self, detail: str, remedy: str) -> None:
        super().__init__(detail)
        self.detail = detail
        self.remedy = remedy


#: The keys that place an event; an event states exactly one.
_POSITIONS = ("at", "at_offset", "at_time", "at_when")

#: The keys that place an event's end; a ``step`` or ``ramp`` states at most one.
_ENDS = ("until", "until_offset", "until_when")

#: A ``burst``'s optional slow tail: both keys or neither.
_TAIL = ("tail_share", "tail_decay_s")

#: The calendar placements, ``{days_ago, time}``.
_CALENDAR_KEYS = ("at_when", "until_when")


def _quoted(keys: Iterable[str]) -> str:
    """Keys as an event's rule names them: each in backticks, comma-separated."""
    return ", ".join(f"`{key}`" for key in keys)


def _unreadable(key: str, value: object) -> EventRuleError:
    """The refusal of a value that is not of its key's kind: a number, a time of day, a day."""
    if key == "at_time":
        return EventRuleError(
            f"has `at_time` {value!r}, not an HH:MM[:SS] time of day", "write `at_time` as HH:MM:SS"
        )
    if key in _CALENDAR_KEYS:
        return EventRuleError(
            f"has `{key}` {value!r}, not {{days_ago: <int>, time: HH:MM[:SS]}}",
            f"write `{key}` as {{days_ago: <int>, time: HH:MM:SS}}",
        )
    return EventRuleError(f"has `{key}` {value!r}, not a number", f"write `{key}` as a number")


def _number(value: object, info: ValidationInfo[object]) -> object:
    """An event's number: an ``int`` or a ``float``, never a ``bool``."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _unreadable(str(info.field_name), value)
    return value


def _fraction(value: float, info: ValidationInfo[object]) -> float:
    if not 0.0 <= value <= 1.0:
        key = info.field_name
        raise EventRuleError(
            f"has `{key}` {value!r}, outside 0 to 1",
            f"write `{key}` as a window fraction from 0 to 1",
        )
    return value


def _positive(value: float, info: ValidationInfo[object]) -> float:
    if value <= 0:
        key = info.field_name
        raise EventRuleError(
            f"has `{key}` {value!r}, not greater than 0",
            f"write `{key}` as a number greater than 0",
        )
    return value


def _share(value: float) -> float:
    if not 0.0 < value < 1.0:
        raise EventRuleError(
            f"has `tail_share` {value!r}, not between 0 and 1",
            "write `tail_share` as the slow tail's fraction of the amplitude, above 0 and below 1",
        )
    return value


#: A number an event reads.
_EventReal: TypeAlias = Annotated[_Real, BeforeValidator(_number)]

#: A position or an end on the reader's window, a fraction from 0 to 1.
_Fraction: TypeAlias = Annotated[_Real, BeforeValidator(_number), AfterValidator(_fraction)]

#: A duration or a size, greater than 0.
_PositiveReal: TypeAlias = Annotated[_Real, BeforeValidator(_number), AfterValidator(_positive)]

#: A burst's slow-tail share, above 0 and below 1.
_OpenUnit: TypeAlias = Annotated[_Real, BeforeValidator(_number), AfterValidator(_share)]


def _daily_time(value: object, handler: ValidatorFunctionWrapHandler) -> object:
    """``at_time`` as a time of day, refused in the event's words."""
    try:
        return handler(value)
    except ValidationError as refusal:
        if any(error["type"] == "time_offset" for error in refusal.errors()):
            raise EventRuleError(
                f"has `at_time` {value!r}; time is local time and must not carry a timezone offset",
                "write `at_time` as HH:MM:SS without an offset",
            ) from None
        raise _unreadable("at_time", value) from None


def _calendar_day(
    value: object, handler: ValidatorFunctionWrapHandler, info: ValidationInfo[object]
) -> object:
    """``at_when`` or ``until_when`` as a :class:`RelativeTimestamp`, refused in the event's words."""
    try:
        return handler(value)
    except ValidationError as refusal:
        key = str(info.field_name)
        broken = {(error["loc"][:1], error["type"]) for error in refusal.errors()}
        negative = (("days_ago",), "greater_than_equal")
        offset = (("time",), "time_offset")
        if broken - {negative, offset}:
            raise _unreadable(key, value) from None
        if negative in broken:
            raise EventRuleError(
                f"has `{key}` {value!r}; days_ago must be a non-negative integer",
                "write `days_ago` as 0 or more",
            ) from None
        raise EventRuleError(
            f"has `{key}` {value!r}; time is local time and must not carry a timezone offset",
            f"write `{key}.time` as HH:MM:SS without an offset",
        ) from None


#: A daily time of day an event recurs at.
_DailyTime: TypeAlias = Annotated[TimeOfDay, WrapValidator(_daily_time)]

#: A calendar day and time an event is placed at, counted back from the anchor.
_CalendarDay: TypeAlias = Annotated[RelativeTimestamp, WrapValidator(_calendar_day)]


class WindowFraction(FrozenModel):
    """An event placed on the reader's window: ``at`` is a fraction from 0 to 1 of it."""

    at: _Real


class AnchorOffset(FrozenModel):
    """An event placed ``offset_s`` seconds from the activation anchor; negative is before it."""

    offset_s: _Real


class CalendarDay(FrozenModel):
    """An event placed on a calendar day before the anchor's, at a time of day, in the facility zone."""

    when: RelativeTimestamp


class DailyTime(FrozenModel):
    """An event placed at a time of day on every calendar day, in the facility zone."""

    time: dtime


#: Where an event is placed, or where a ``step`` or ``ramp`` ends.
_Placement: TypeAlias = WindowFraction | AnchorOffset | CalendarDay | DailyTime


class _Event(_Record):
    """What every archiver event states: exactly one position key.

    ``at`` is a window fraction from 0 to 1, ``at_offset`` seconds from the
    activation anchor, ``at_when`` a ``{days_ago, time}`` calendar placement
    and ``at_time`` an ``HH:MM[:SS]`` daily time of day; :attr:`placement` is
    the one the event states. A key stated as ``null`` is refused, as a value
    not of the key's kind is.
    """

    at: _Fraction | None = None
    at_offset: _EventReal | None = None
    at_when: _CalendarDay | None = None
    at_time: _DailyTime | None = None

    #: The keys a shape states besides its position, in the order a missing one is named.
    _VALUES: ClassVar[tuple[str, ...]] = ()

    @model_validator(mode="wrap")
    @classmethod
    def _rules(cls, event: object, handler: ModelWrapValidatorHandler[Self]) -> Self:
        """The rules between an event's keys, judged on the keys it states."""
        if not isinstance(event, Mapping):
            return handler(event)
        positions = [key for key in _POSITIONS if key in event]
        if len(positions) != 1:
            stated = f" ({_quoted(positions)})" if positions else ""
            raise EventRuleError(
                f"states {len(positions)} position keys{stated}",
                f"state exactly one of {_quoted(_POSITIONS)}",
            )
        (position,) = positions
        required = [*cls._VALUES, *cls._paired(position, event)]
        missing = [key for key in required if key not in event]
        if missing:
            shape = event.get("shape")
            raise EventRuleError(
                f"lacks {_quoted(missing)}", f"give the `{shape}` event {_quoted(missing)}"
            )
        record = handler(event)
        record._ordered(event)
        return record

    @field_validator(*_POSITIONS, *_ENDS, *_TAIL, mode="before", check_fields=False)
    @classmethod
    def _stated(cls, value: object, info: ValidationInfo[object]) -> object:
        """A key the event states holds a value: a ``null`` is refused as the key's kind refuses it."""
        if value is None:
            raise _unreadable(str(info.field_name), value)
        return value

    @classmethod
    def _paired(cls, position: str, event: Mapping[object, object]) -> list[str]:
        """The keys the event's position makes it state, besides its shape's; none by default."""
        del position, event
        return []

    def _ordered(self, event: Mapping[object, object]) -> None:
        """Refuse an end that does not lie after the event's position; none by default."""
        del event

    @property
    def placement(self) -> _Placement:
        """Where the event is placed: the one position key it states."""
        if self.at is not None:
            return WindowFraction(at=self.at)
        if self.at_offset is not None:
            return AnchorOffset(offset_s=self.at_offset)
        if self.at_when is not None:
            return CalendarDay(when=self.at_when)
        if self.at_time is not None:
            return DailyTime(time=self.at_time)
        # The validator refuses an event that states no position.
        raise EventRuleError(
            "states 0 position keys", f"state exactly one of {_quoted(_POSITIONS)}"
        )


def _ends(event: Mapping[object, object]) -> list[str]:
    return [key for key in _ENDS if key in event]


#: A step's value: whatever the channel's value type coerces.
_StepValue: TypeAlias = bool | _Real | str | FrozenTuple[_Verbatim]

#: The ``step`` end key each position key pairs with; a daily ``at_time`` step has none.
_STEP_UNTIL = {"at": "until", "at_offset": "until_offset", "at_when": "until_when"}

#: The ``ramp`` end key each position key pairs with; a ramp is placed by these alone.
_RAMP_UNTIL = {"at": "until", "at_offset": "until_offset"}


class StepEvent(_Event):
    """A ``step`` to ``to``, held from its position on, or until its end when it states one.

    The end is the key its position pairs with: ``until`` (a window fraction)
    with ``at``, ``until_offset`` with ``at_offset``, ``until_when`` with
    ``at_when``; it lies after the position. A daily ``at_time`` step never
    ends. ``to`` is whatever the channel's value type coerces, which the build
    checks against the channel.
    """

    shape: Literal["step"]
    to: _StepValue
    until: _Fraction | None = None
    until_offset: _EventReal | None = None
    until_when: _CalendarDay | None = None

    _VALUES: ClassVar[tuple[str, ...]] = ("to",)

    @override
    @classmethod
    def _paired(cls, position: str, event: Mapping[object, object]) -> list[str]:
        ends = _ends(event)
        if ends and position == "at_time":
            raise EventRuleError(
                f"ends an `at_time` step with {_quoted(ends)}",
                "drop the end, or place the step by `at`, `at_offset` or `at_when`",
            )
        if ends and ends != [_STEP_UNTIL[position]]:
            raise EventRuleError(
                f"ends a `step` placed by `{position}` with {_quoted(ends)}",
                f"end a step placed by `{position}` with `{_STEP_UNTIL[position]}` only",
            )
        return ends

    @override
    def _ordered(self, event: Mapping[object, object]) -> None:
        (position,) = (key for key in _POSITIONS if key in event)
        until = _STEP_UNTIL.get(position)
        if until is None or until not in event:
            return
        start, end = self.placement, self.end
        if end is not None and not _after(start, end):
            raise EventRuleError(
                f"ends at `{until}` {event[until]!r}, not after `{position}` {event[position]!r}",
                f"write `{until}` later than `{position}`",
            )

    @property
    def end(self) -> WindowFraction | AnchorOffset | CalendarDay | None:
        """Where the step ends, on its position's axis; ``None`` when it holds to the end."""
        if self.until is not None:
            return WindowFraction(at=self.until)
        if self.until_offset is not None:
            return AnchorOffset(offset_s=self.until_offset)
        if self.until_when is not None:
            return CalendarDay(when=self.until_when)
        return None


def _after(start: _Placement, end: _Placement) -> bool:
    """Whether ``end`` lies after ``start``, two placements on one axis."""
    match start, end:
        case WindowFraction(at=low), WindowFraction(at=high):
            return high > low
        case AnchorOffset(offset_s=low), AnchorOffset(offset_s=high):
            return high > low
        case CalendarDay(when=low), CalendarDay(when=high):
            # More days ago is earlier; on the same day the later time of day is later.
            return (-low.days_ago, low.time) < (-high.days_ago, high.time)
        case _:
            return False


class RampEvent(_Event):
    """A ``ramp`` to ``to``: linear from the value at its position to ``to`` at its end.

    A ramp is placed by ``at`` and ends at ``until`` (window fractions), or is
    placed by ``at_offset`` and ends at ``until_offset`` (seconds from the
    anchor).
    """

    shape: Literal["ramp"]
    to: _Real
    until: _Fraction | None = None
    until_offset: _EventReal | None = None

    _VALUES: ClassVar[tuple[str, ...]] = ("to",)

    @override
    @classmethod
    def _paired(cls, position: str, event: Mapping[object, object]) -> list[str]:
        until = _RAMP_UNTIL.get(position)
        if until is None:
            raise EventRuleError(
                f"places a `ramp` by `{position}`",
                "place the `ramp` by `at` with `until`, or by `at_offset` with `until_offset`",
            )
        ends = _ends(event)
        if ends and ends != [until]:
            raise EventRuleError(
                f"ends a `ramp` placed by `{position}` with {_quoted(ends)}",
                f"end a ramp placed by `{position}` with `{until}` only",
            )
        return [until]

    @property
    def end(self) -> WindowFraction | AnchorOffset:
        """Where the ramp reaches ``to``, on its position's axis."""
        if self.until_offset is not None:
            return AnchorOffset(offset_s=self.until_offset)
        if self.until is not None:
            return WindowFraction(at=self.until)
        # The validator refuses a ramp that states no end.
        raise EventRuleError("lacks its end", "give the `ramp` event its end")


class SpikeEvent(_Event):
    """A ``spike``: a Gaussian bump of ``amplitude`` with sigma ``width``.

    ``width`` is a window fraction on an event placed by ``at``, else seconds.
    """

    shape: Literal["spike"]
    amplitude: _EventReal
    width: _PositiveReal

    _VALUES: ClassVar[tuple[str, ...]] = ("amplitude", "width")


class BurstEvent(_Event):
    """A ``burst``: a jump of ``amplitude`` that pumps down with ``decay_s``.

    ``tail_share`` with ``tail_decay_s``, stated together or neither, carries
    that share of the amplitude on a slower decay.
    """

    shape: Literal["burst"]
    amplitude: _EventReal
    decay_s: _PositiveReal
    tail_share: _OpenUnit | None = None
    tail_decay_s: _PositiveReal | None = None

    _VALUES: ClassVar[tuple[str, ...]] = ("amplitude", "decay_s")

    @override
    @classmethod
    def _paired(cls, position: str, event: Mapping[object, object]) -> list[str]:
        del position
        tail = [key for key in _TAIL if key in event]
        if len(tail) == 1:
            raise EventRuleError(
                f"states {_quoted(tail)} without the rest of its tail",
                "state `tail_share`, `tail_decay_s` together, or neither",
            )
        return tail


class LossEvent(_Event):
    """A ``loss``: a drop of ``amount`` that only the channel's top-off refills."""

    shape: Literal["loss"]
    amount: _PositiveReal

    _VALUES: ClassVar[tuple[str, ...]] = ("amount",)


#: One event of a scenario's ``archiver`` entry, told apart by its ``shape``.
#:
#: The facility schema leaves a scenario's ``archiver`` block free-form, and
#: the event keeps the flat form it is authored in: ``shape``, one position
#: key, the end key a ``step`` or ``ramp`` pairs with it, and the shape's
#: values. Each shape's record holds the rules between its keys and refuses an
#: event that breaks one with an :class:`EventRuleError`; ``osprey build``
#: checks the rules that need the channel (a ``to`` the channel's value type
#: coerces, a shape other than ``step`` on a ``float`` channel only, a bounded
#: ``step`` on a ``float`` channel only).
ArchiverEvent: TypeAlias = Annotated[
    StepEvent | RampEvent | SpikeEvent | BurstEvent | LossEvent, Field(discriminator="shape")
]


class ArchiverEntry(_Record):
    """One entry of a scenario's ``archiver`` block: a channel and its events."""

    channel: str
    events: FrozenTuple[ArchiverEvent] = ()


class ModelFaultRecord(_Record):
    """One model's fault in a scenario's ``faults`` block; ``inactive`` is ``None`` when active."""

    writes: ReadOnlyMapping[str, _Verbatim]
    inactive: str | None = None


class PictureRef(_Record):
    """A logbook entry's shipped picture, ``{path: <picture file>}``, relative to its scenario's directory."""

    path: str


class PlotRef(_Record):
    """A logbook entry's plot spec, ``{plot: <plot spec .json>}``, relative to its scenario's directory."""

    plot: str


def _attachment_key(item: object) -> str | None:
    """Which attachment an item is, by the key it states; ``None`` when it states neither."""
    if isinstance(item, PictureRef | PlotRef):
        return "path" if isinstance(item, PictureRef) else "plot"
    if isinstance(item, Mapping):
        return next((key for key in ("path", "plot") if key in item), None)
    return None


#: One picture a logbook entry attaches, told apart by the key it states; an
#: item stating both is refused (a ``path`` item carries no ``plot``).
_Attachment: TypeAlias = Annotated[
    Annotated[PictureRef, Tag("path")] | Annotated[PlotRef, Tag("plot")],
    Discriminator(
        _attachment_key,
        custom_error_type="attachment_key",
        custom_error_message="an attachment states `path` or `plot`",
    ),
]


class LogEntryRecord(_Record):
    """One entry of a scenario's ``logbook`` block, as the scenario states it.

    ``when`` is the ``{days_ago, time}`` the entry is written at, counted back
    from the activation anchor; ``attachments`` name its pictures relative to
    the scenario's own directory ``scenarios/<name>/``. :meth:`resolve` reads
    the pictures and gives the entry the logbook seed writes.
    """

    entry_id: str = Field(min_length=1)
    when: RelativeTimestamp
    author: str
    title: str
    text: str
    tags: FrozenTuple[str] = ()
    categories: FrozenTuple[str] = ()
    loto_tag: str | None = None
    extra: ReadOnlyMapping[str, _Verbatim] = Field(default_factory=empty_mapping)
    attachments: FrozenTuple[_Attachment] = ()

    def resolve(self, scenario_dir: Path, *, scenario: str) -> ScenarioLogEntry:
        """The entry the logbook seed writes, its pictures read from ``scenario_dir``.

        Args:
            scenario_dir: The scenario's own directory, which the entry's
                ``attachments`` are relative to.
            scenario: The scenario's name, which a refusal names.

        Returns:
            The entry: a shipped picture is its absolute path inside
            ``scenario_dir``, a plot spec its parsed spec, and ``extra`` the
            plain mapping it states.

        Raises:
            ValueError: An attachment leaves ``scenario_dir``, names a missing
                file, holds no picture of its suffix or no valid plot spec, or
                shares its file name with another; the message names the
                scenario, the entry and the attachment.
        """
        from osprey_connectors.simulation.logbook import ScenarioLogEntry, _parse_log_attachments

        attachments: list[tuple[Literal["path", "plot"], str]] = [
            ("path", item.path) if isinstance(item, PictureRef) else ("plot", item.plot)
            for item in self.attachments
        ]
        return ScenarioLogEntry(
            entry_id=self.entry_id,
            when=self.when,
            author=self.author,
            title=self.title,
            text=self.text,
            tags=self.tags,
            categories=self.categories,
            loto_tag=self.loto_tag,
            extra={key: _thaw(value) for key, value in self.extra.items()},
            attachments=_parse_log_attachments(
                f"Scenario {scenario!r} logbook entry {self.entry_id!r}", attachments, scenario_dir
            ),
        )


class ScenarioRecord(_Record):
    """One scenario of ``scenarios.json``; a block the scenario does not state is empty.

    ``description`` and ``sessions`` are ``None`` when not stated. A scenario
    writes back only the blocks it states (``model_dump(exclude_unset=True)``).
    """

    name: str
    description: str | None = None
    overrides: ReadOnlyMapping[str, _Verbatim] = Field(default_factory=empty_mapping)
    archiver: FrozenTuple[ArchiverEntry] = ()
    logbook: FrozenTuple[LogEntryRecord] = ()
    sessions: _Verbatim = None
    drivers: ReadOnlyMapping[str, DriveRecord] = Field(default_factory=empty_mapping)
    couple: ReadOnlyMapping[str, FrozenTuple[CouplingTerm]] = Field(default_factory=empty_mapping)
    noise: ReadOnlyMapping[str, Noise] = Field(default_factory=empty_mapping)
    channel_faults: ReadOnlyMapping[str, ChannelFaultKind] = Field(default_factory=empty_mapping)
    still: tuple[str, ...] = ()
    faults: ReadOnlyMapping[str, ModelFaultRecord] = Field(default_factory=empty_mapping)

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
