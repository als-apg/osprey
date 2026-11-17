"""The simulator view a build writes, and the one reader of it.

``osprey build`` writes each render's simulator view into
``<render>/data/simulator/``; this module names its files and their schemas
and opens it. Every reader of a view opens it through :class:`SimulatorView`,
which refuses a file whose ``schema`` is not the one this OSPREY reads, so a
render from an older OSPREY is rebuilt rather than misread.

A wiring record of a physics model carries what its engine's ``describe()``
stated at build time: its ``engine_role`` (one of :data:`ENGINE_ROLES`), the transverse
``plane`` it steers or reads (``x``, ``y`` or ``None``), its ``refresh``
(one of :data:`REFRESH`) and, when it carries any, its ``marks`` (of
:data:`MARKS`). The words are the facility schema's. A ``readback`` is any read of an element's setting, paired
with a setpoint or not; a reader wanting the readback of a setpoint reads that
setpoint channel's ``pair``.

Each file is validated, when first read, against its document model in
:mod:`osprey_connectors.simulation.records`; a file whose shape breaks its model
is refused as a :class:`ViewSchemaError` naming the key and the reason.

The module imports only the standard library, pydantic and ``osprey_connectors``,
so a reader that must not load a model can open a view.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final, Literal, get_args, overload

from pydantic import BaseModel, ValidationError

from osprey_connectors.errors import ConnectorsError
from osprey_connectors.model import FrozenModel, ReadOnlyMapping
from osprey_connectors.simulation import TEXTURE
from osprey_connectors.simulation.records import (
    AddressesDocument,
    ChannelRecord,
    ChannelRole,
    Direction,
    EngineRole,
    Mark,
    ModelEntry,
    Plane,
    Refresh,
    ScenarioRecord,
    ScenariosDocument,
    SeedRecord,
    SeedsDocument,
    ServedModelsDocument,
    VariablesDocument,
    WiringRecord,
    _thaw,
)

__all__ = [
    "ADDRESSES_FILE",
    "ADDRESSES_SCHEMA",
    "COMMAND_MARK",
    "DECKS_DIR",
    "ENGINEER",
    "ENGINE_ROLES",
    "HEARTBEAT",
    "IOC_HEALTH_SIGNALS",
    "IocHealthSignal",
    "MARKS",
    "MONITOR",
    "OUTPUT",
    "PERIODIC",
    "PERIODIC_IOC_HEALTH",
    "PLANES",
    "READBACK",
    "REFRESH",
    "SCENARIOS_DIR",
    "SCENARIOS_FILE",
    "SCENARIOS_SCHEMA",
    "SERVED_SIGNAL_PROPERTY",
    "SCHEMAS",
    "SEEDS_FILE",
    "SEEDS_SCHEMA",
    "SERVED_MODELS_FILE",
    "SERVED_MODELS_SCHEMA",
    "SETPOINT",
    "STATUS_SIGNAL",
    "UPTIME",
    "VARIABLES_FILE",
    "VARIABLES_SCHEMA",
    "VIEW_RELPATH",
    "Binding",
    "Channel",
    "Model",
    "NoSimulatorViewError",
    "SimulatorViewError",
    "SimulatorView",
    "ViewSchemaError",
]

#: The view's directory under a render.
VIEW_RELPATH = PurePosixPath("data/simulator")

#: The file names are ``Final`` so that :meth:`SimulatorView.document` types
#: each one's document.
SERVED_MODELS_FILE: Final = "served_models.json"
SERVED_MODELS_SCHEMA = "osprey.facility.served_models/1"
ADDRESSES_FILE: Final = "addresses.json"
ADDRESSES_SCHEMA = "osprey.facility.addresses/1"
VARIABLES_FILE: Final = "variables.json"
VARIABLES_SCHEMA = "osprey.facility.simulator/4"
SEEDS_FILE: Final = "seeds.json"
SEEDS_SCHEMA = "osprey.facility.seeds/2"
SCENARIOS_FILE: Final = "scenarios.json"
SCENARIOS_SCHEMA = "osprey.facility.scenarios/3"
#: The directory holding a byte copy of each deck-bearing model's deck.
DECKS_DIR = "decks"
#: The directory holding each scenario's attached files, ``scenarios/<name>/``,
#: under the facility tree and under the view alike.
SCENARIOS_DIR = "scenarios"

#: Each view file and the schema it carries.
SCHEMAS: Mapping[str, str] = MappingProxyType(
    {
        SERVED_MODELS_FILE: SERVED_MODELS_SCHEMA,
        ADDRESSES_FILE: ADDRESSES_SCHEMA,
        VARIABLES_FILE: VARIABLES_SCHEMA,
        SEEDS_FILE: SEEDS_SCHEMA,
        SCENARIOS_FILE: SCENARIOS_SCHEMA,
    }
)

#: Each view file and the model its document validates against.
_DOCUMENTS: Mapping[str, type[FrozenModel]] = MappingProxyType(
    {
        SERVED_MODELS_FILE: ServedModelsDocument,
        ADDRESSES_FILE: AddressesDocument,
        VARIABLES_FILE: VariablesDocument,
        SEEDS_FILE: SeedsDocument,
        SCENARIOS_FILE: ScenariosDocument,
    }
)

#: The schemas this OSPREY reads besides the one it writes, per file: a released
#: ``/3`` facility file reads as one whose models declare no inputs.
_ALSO_READ: Mapping[str, frozenset[str]] = MappingProxyType(
    {VARIABLES_FILE: frozenset({"osprey.facility.simulator/3"})}
)

SETPOINT: Final = "setpoint"
READBACK: Final = "readback"
MONITOR: Final = "monitor"
OUTPUT: Final = "output"
#: The engine roles a wiring record's description names, every
#: :data:`~osprey_connectors.simulation.records.EngineRole`. A channel's
#: facility role is a :data:`~osprey_connectors.simulation.records.ChannelRole`.
ENGINE_ROLES: tuple[EngineRole, ...] = get_args(EngineRole)
#: The planes a wiring record's description names; ``None`` is no plane.
PLANES: tuple[Plane, ...] = get_args(Plane)
PERIODIC: Final = "periodic"
#: The refresh a wiring record's description names, every
#: :data:`~osprey_connectors.simulation.records.Refresh`.
REFRESH: tuple[Refresh, ...] = get_args(Refresh)
#: The mark of a record whose write is a command; only a ``write`` record
#: carries it.
COMMAND_MARK: Final = "command"
#: The marks a wiring record's description may set.
MARKS: tuple[Mark, ...] = get_args(Mark)

#: The signal of a physics model's status address, ``<code>:SIM:<model>:STATUS``.
STATUS_SIGNAL = "STATUS"
#: The signal of one of a model's IOC health addresses, ``<code>:SIM:<model>:<signal>``.
IocHealthSignal = Literal["HEARTBEAT", "UPTIME", "ENGINEER"]
HEARTBEAT: Final = "HEARTBEAT"
UPTIME: Final = "UPTIME"
ENGINEER: Final = "ENGINEER"
#: Every :data:`IocHealthSignal`, in the order a view lists them.
IOC_HEALTH_SIGNALS: tuple[IocHealthSignal, ...] = get_args(IocHealthSignal)
#: The IOC health signals each periodic pass publishes.
PERIODIC_IOC_HEALTH: tuple[IocHealthSignal, ...] = (HEARTBEAT, UPTIME)
#: The directory property the simulator sets on each status and IOC health row,
#: valued with that row's signal; channel rows carry none.
SERVED_SIGNAL_PROPERTY = "signal"

#: The text a missing view reports.
_NO_VIEW_MESSAGE = "no simulator view at {path}: run osprey build"

#: The keys a wiring record's description adds.
_DESCRIPTION_KEYS = ("engine_role", "plane", "refresh")


class SimulatorViewError(ConnectorsError):
    """A simulator view could not be read.

    Attributes:
        path: The view file or directory at fault, or ``None`` when the refusal
            is about one record and names no file.
    """

    def __init__(self, message: str, *, path: Path | None = None) -> None:
        self.path = path
        super().__init__(message)


class NoSimulatorViewError(SimulatorViewError, LookupError):
    """The place holds no simulator view, or the view lacks a file it lists.

    ``path`` is the missing file: the place's ``addresses.json`` when there is
    no view there at all.
    """


class ViewSchemaError(SimulatorViewError, ValueError):
    """A view file is not JSON, its ``schema`` is not one this OSPREY reads, or its shape breaks it.

    Attributes:
        path: The view file, or ``None`` for a refusal of one record.
        key: The dotted key refused, or ``None`` when the whole file is.
        found: What the file holds there (``None`` when nothing could be read).
        expected: What this OSPREY reads there, or ``None`` when the refusal
            names no one value.
    """

    def __init__(
        self,
        message: str,
        *,
        path: Path | None = None,
        key: str | None = None,
        found: object = None,
        expected: str | None = None,
    ) -> None:
        super().__init__(message, path=path)
        self.key = key
        self.found = found
        self.expected = expected


def _schema_error(path: Path, found: object, expected: str) -> ViewSchemaError:
    return ViewSchemaError(
        f"{path} is schema {found!r}; this OSPREY reads {expected!r}: rebuild with osprey build",
        path=path,
        key="schema",
        found=found,
        expected=expected,
    )


def _shape_error(where: object, refusal: ValidationError) -> ViewSchemaError:
    """The first key a view file's model refuses, and why, as the view's error.

    ``where`` is the view file, or a phrase naming the one record refused.
    """
    error = refusal.errors()[0]
    key = ".".join(str(part) for part in error["loc"]) or "the document"
    return ViewSchemaError(
        f"{where}: {key}: {error['msg']} (found {error['input']!r}): rebuild with osprey build",
        path=where if isinstance(where, Path) else None,
        key=key if error["loc"] else None,
        found=error["input"],
        expected=error["msg"],
    )


class Channel(FrozenModel):
    """One channel of the view, as ``variables.json`` states it.

    Attributes:
        address: The channel's address.
        role: The facility file's role, ``readback`` when it states none.
        pair: A setpoint's readback, itself when it names none; ``None`` on
            any other role.
        value_type: The channel's value type.
        unit: The channel's unit, or ``None``.
        description: The channel's description, or ``None``.
        writable: Whether a write to the channel is allowed.
        value_range: ``(min, max)`` from the channel's limits record, or
            ``None``.
        owner: The model wiring the address, else ``texture``.
        on: The node the channel is on, ``{device: id}`` or ``{place: id}``,
            or ``None``.
        options: An enum channel's options, or ``None``.
        shape: A waveform channel's shape, or ``None``.
        precision: A float channel's display precision, or ``None``.
    """

    address: str
    role: ChannelRole
    pair: str | None
    value_type: str
    unit: str | None
    description: str | None
    writable: bool
    value_range: tuple[int | float, int | float] | None
    owner: str
    on: ReadOnlyMapping[str, str] | None
    options: tuple[str, ...] | None
    shape: tuple[int, ...] | None
    precision: int | None

    @classmethod
    def from_entry(cls, entry: ChannelRecord) -> Channel:
        """Read one channel entry of ``variables.json``.

        Raises:
            ViewSchemaError: the entry's ``value_range`` is not ``[min, max]``.
        """
        # ``value_range`` reaches the model as the entry states it, and the
        # model holds it to ``(min, max)``: ``model_validate`` takes it untyped.
        try:
            return cls.model_validate(
                {
                    "address": entry.address,
                    "role": entry.role,
                    "pair": entry.pair,
                    "value_type": entry.value_type,
                    "unit": entry.unit,
                    "description": entry.description,
                    "writable": entry.writable,
                    "value_range": entry.value_range,
                    "owner": entry.owner,
                    "on": entry.on,
                    "options": entry.options,
                    "shape": entry.shape,
                    "precision": entry.precision,
                }
            )
        except ValidationError as refusal:
            raise _shape_error(f"channel {entry.address!r}", refusal) from None


class Binding(FrozenModel):
    """One wiring record of a physics model: the address it serves and what it is.

    Attributes:
        id: The wiring record's id.
        model: The model the record belongs to.
        address: The channel address the record serves.
        direction: ``write`` or ``read``.
        engine_role: One of :data:`ENGINE_ROLES`.
            A ``readback`` reads an element's setting, paired with a setpoint
            or not; the setpoint channel's ``pair`` names its readback.
        plane: ``x`` or ``y`` for a record steering or reading one transverse
            plane, else ``None``.
        refresh: One of :data:`REFRESH`.
        element: The element the record names, or ``None`` (a record naming
            slices, or none).
        record: The wiring record as the view states it, read-only; the
            engine's ``build`` takes it verbatim.
        marks: The :data:`MARKS` the record carries; empty when it carries
            none.
    """

    id: str
    model: str
    address: str
    direction: Direction
    engine_role: EngineRole
    plane: Plane | None
    refresh: Refresh
    element: str | None
    record: ReadOnlyMapping[str, object]
    marks: frozenset[Mark] = frozenset()

    @classmethod
    def from_record(cls, model: str, record: WiringRecord) -> Binding:
        """Read one wiring record of ``variables.json``.

        Raises:
            ViewSchemaError: the record lacks ``engine_role``, ``plane`` or
                ``refresh``, or states ``None`` for ``engine_role``,
                ``refresh`` or ``direction``.
        """
        missing = [key for key in _DESCRIPTION_KEYS if key not in record.model_fields_set]
        if missing:
            raise ViewSchemaError(
                f"wiring record {record.address!r} of model {model!r} lacks "
                f"{', '.join(missing)}: rebuild with osprey build",
                key=missing[0],
            )
        engine_role, refresh, direction = record.engine_role, record.refresh, record.direction
        if engine_role is None or refresh is None or direction is None:
            stated = {"engine_role": engine_role, "refresh": refresh, "direction": direction}
            unstated = next(key for key, value in stated.items() if value is None)
            raise ViewSchemaError(
                f"wiring record {record.address!r} of model {model!r} states no "
                f"{unstated}: rebuild with osprey build",
                key=unstated,
            )
        return cls(
            id=str(record.id),
            model=model,
            address=record.address,
            direction=direction,
            engine_role=engine_role,
            plane=record.plane,
            refresh=refresh,
            element=record.element,
            record=record.model_dump(mode="json", exclude_unset=True),
            marks=frozenset(record.marks or ()),
        )


class Model(FrozenModel):
    """One model of the view.

    Attributes:
        name: The model's name.
        engine: The engine the model names.
        served: Whether the render serves the model.
        settings: The model's settings, read-only.
        deck: The path of the view's copy of the model's deck, or ``None``.
        bindings: The model's wiring records, in wiring order; empty for a
            ``texture`` model or one without wiring.
        inputs: The model's inputs, in declared order, each a read-only
            ``{address, engine}`` record: the producer channel's address and
            the engine block the consuming engine reads; empty when the model
            declares none.
    """

    name: str
    engine: str | None
    served: bool
    settings: ReadOnlyMapping[str, object]
    deck: Path | None
    bindings: tuple[Binding, ...]
    inputs: tuple[ReadOnlyMapping[str, object], ...] = ()


class SimulatorView:
    """A render's simulator view, opened from its directory.

    ``addresses.json`` is read when the view opens; every other file is read
    once, on first use, and each is refused when its ``schema`` is not the
    one this OSPREY reads.

    Attributes:
        path: The view's directory.
        render_root: The render the view sits in, when the view is at
            ``<render>/data/simulator``; else ``None``.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        parts = PurePosixPath(*self.path.parts[-len(VIEW_RELPATH.parts) :])
        self.render_root = self.path.parents[1] if parts == VIEW_RELPATH else None
        self._documents: dict[str, BaseModel] = {}
        self._models: tuple[Model, ...] | None = None
        self._channel_records: Mapping[str, Channel] | None = None
        self._envelopes: dict[tuple[str, ...], dict[str, float]] = {}
        if not (self.path / ADDRESSES_FILE).is_file():
            raise NoSimulatorViewError(
                _NO_VIEW_MESSAGE.format(path=self.path), path=self.path / ADDRESSES_FILE
            )
        self.document(ADDRESSES_FILE)

    @classmethod
    def open(cls, view_dir: Path | str) -> SimulatorView:
        """Open the view in ``view_dir``.

        Raises:
            NoSimulatorViewError: ``view_dir`` has no ``addresses.json``.
            ViewSchemaError: its ``addresses.json`` is not schema
                :data:`ADDRESSES_SCHEMA`.
        """
        return cls(Path(view_dir))

    @classmethod
    def of_render(cls, render_root: Path | str) -> SimulatorView:
        """Open the view of the render rooted at ``render_root``."""
        return cls.open(Path(render_root) / VIEW_RELPATH)

    @classmethod
    def path_for_project(cls, project_dir: Path | str) -> Path:
        """Where the view of ``project_dir``'s render is, read or not.

        A deployment repo keeps its render under ``build/``, beside the
        rendered ``config.yml``; a container's project directory is the
        render itself.
        """
        from osprey_connectors.workspace import rendered_config_path

        project = Path(project_dir)
        rendered = rendered_config_path(project)
        render = rendered.parent if rendered.is_file() else project
        return render / VIEW_RELPATH

    @classmethod
    def of_project(cls, project_dir: Path | str) -> SimulatorView:
        """Open the view of ``project_dir``'s render (:meth:`path_for_project`)."""
        return cls.open(cls.path_for_project(project_dir))

    @classmethod
    def find(cls, project_dir: Path | str) -> SimulatorView | None:
        """Open the view of ``project_dir``'s render, or ``None`` when it has none.

        Raises:
            ViewSchemaError: the view is there but from another schema.
        """
        try:
            return cls.of_project(project_dir)
        except NoSimulatorViewError:
            return None

    # A file of the view reaches a reader only through :meth:`document`, which
    # refuses one whose ``schema`` is not a schema this OSPREY reads and
    # validates the rest against the file's document model: the overloads below
    # name that model, of :mod:`osprey_connectors.simulation.records`, for each
    # file by name. The facility schema leaves a scenario's ``archiver`` events
    # free-form; the build checks each one with
    # ``osprey.facility.scenarios.check_scenario_events`` before it writes
    # ``scenarios.json``, and ``ArchiverEvent`` states what that check
    # guarantees.
    @overload
    def document(self, name: Literal["served_models.json"]) -> ServedModelsDocument: ...
    @overload
    def document(self, name: Literal["addresses.json"]) -> AddressesDocument: ...
    @overload
    def document(self, name: Literal["variables.json"]) -> VariablesDocument: ...
    @overload
    def document(self, name: Literal["seeds.json"]) -> SeedsDocument: ...
    @overload
    def document(self, name: Literal["scenarios.json"]) -> ScenariosDocument: ...
    @overload
    def document(self, name: str) -> BaseModel: ...
    def document(self, name: str) -> BaseModel:
        """One view file, parsed and validated once, read-only.

        Args:
            name: The file's name, one of :data:`SCHEMAS`.

        Returns:
            The file's document, the model of
            :mod:`osprey_connectors.simulation.records` for the file.

        Raises:
            NoSimulatorViewError: the file is not there.
            ViewSchemaError: the file is not JSON, its ``schema`` is absent or
                neither ``SCHEMAS[name]`` nor an older one this OSPREY still
                reads, or its shape breaks the file's document model.
        """
        if name not in self._documents:
            path = self.path / name
            try:
                parsed: object = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                raise NoSimulatorViewError(
                    f"no simulator view file at {path}: run osprey build", path=path
                ) from None
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ViewSchemaError(
                    f"{path} is not JSON ({exc}): rebuild with osprey build", path=path
                ) from None
            expected = SCHEMAS[name]
            if not isinstance(parsed, dict):
                raise _schema_error(path, None, expected)
            found = parsed.get("schema")
            if found != expected and found not in _ALSO_READ.get(name, ()):
                raise _schema_error(path, found, expected)
            try:
                self._documents[name] = _DOCUMENTS[name].model_validate(parsed)
            except ValidationError as refusal:
                raise _shape_error(path, refusal) from None
        return self._documents[name]

    @property
    def code(self) -> str:
        """The facility's identity code."""
        return self.document(VARIABLES_FILE).code

    def models(self) -> tuple[Model, ...]:
        """Every model of the facility file, sorted by name."""
        if self._models is None:
            entries = self.document(VARIABLES_FILE).models
            self._models = tuple(self._model(entry) for entry in entries)
        return self._models

    def _model(self, entry: ModelEntry) -> Model:
        return Model(
            name=entry.name,
            engine=entry.engine,
            served=entry.served,
            settings={key: _thaw(value) for key, value in entry.settings.items()},
            deck=self.path / entry.deck if entry.deck else None,
            bindings=()
            if entry.engine == TEXTURE
            else tuple(Binding.from_record(entry.name, record) for record in entry.wiring),
            inputs=entry.inputs,
        )

    def model(self, name: str) -> Model:
        """The model named ``name``.

        Raises:
            KeyError: the view has no such model.
        """
        for model in self.models():
            if model.name == name:
                return model
        raise KeyError(f"the simulator view has no model {name!r}")

    def served(self) -> tuple[str, ...]:
        """The names of the models the render serves, in ``served_models.json`` order."""
        return self.document(SERVED_MODELS_FILE).models

    def physics_models(self) -> tuple[Model, ...]:
        """The served models whose engine is not ``texture``, sorted by name."""
        return tuple(model for model in self.models() if model.served and model.engine != TEXTURE)

    def channels(self) -> tuple[str, ...]:
        """Every channel address of the facility file, sorted."""
        return self.document(ADDRESSES_FILE).channels

    def status_addresses(self) -> Mapping[str, str]:
        """Each served physics model's status address, ``<code>:SIM:<model>:STATUS``."""
        listed = set(self.document(ADDRESSES_FILE).status)
        return MappingProxyType(
            {
                model.name: address
                for model in self.physics_models()
                if (address := f"{self.code}:SIM:{model.name}:{STATUS_SIGNAL}") in listed
            }
        )

    def ioc_health_by_model(self) -> Mapping[str, Mapping[IocHealthSignal, str]]:
        """Each listed model's IOC health addresses by signal, ``{model: {signal: address}}``.

        An address is ``<code>:SIM:<model>:<signal>`` with ``<signal>`` one of
        :data:`IOC_HEALTH_SIGNALS`; each model's signals keep the list's order.
        ``ioc_health`` is optional under :data:`ADDRESSES_SCHEMA`: a view from
        a build without it yields an empty mapping. Unlike the status
        addresses, it lists the ``texture`` model too.

        Raises:
            ViewSchemaError: a listed entry is not such an address.
        """
        listed = self.document(ADDRESSES_FILE).ioc_health
        if not listed:
            return MappingProxyType({})
        prefix = f"{self.code}:SIM:"
        by_model: dict[str, dict[IocHealthSignal, str]] = {}
        for address in listed:
            model, _, word = address.removeprefix(prefix).rpartition(":")
            signal = next((known for known in IOC_HEALTH_SIGNALS if known == word), None)
            if not model or not address.startswith(prefix) or signal is None:
                raise ViewSchemaError(
                    f"{self.path / ADDRESSES_FILE} lists {address!r} under ioc_health, "
                    f"which is not {prefix}<model>:<signal> with <signal> one of "
                    f"{', '.join(IOC_HEALTH_SIGNALS)}: rebuild with osprey build",
                    path=self.path / ADDRESSES_FILE,
                    key="ioc_health",
                    found=address,
                    expected=f"{prefix}<model>:<signal>",
                )
            by_model.setdefault(model, {})[signal] = address
        return MappingProxyType(
            {model: MappingProxyType(signals) for model, signals in by_model.items()}
        )

    def _channels(self) -> Mapping[str, Channel]:
        if self._channel_records is None:
            entries = self.document(VARIABLES_FILE).channels
            self._channel_records = MappingProxyType(
                {
                    channel.address: channel
                    for channel in (Channel.from_entry(entry) for entry in entries)
                }
            )
        return self._channel_records

    def channel(self, address: str) -> Channel:
        """The channel at ``address``.

        Raises:
            KeyError: the view has no channel at ``address``.
        """
        return self._channels()[address]

    def bindings(
        self,
        *,
        model: str | None = None,
        engine_role: EngineRole | None = None,
        plane: Plane | None = None,
        node: str | None = None,
        served_only: bool = True,
    ) -> tuple[Binding, ...]:
        """The physics models' wiring records, model by name, then in wiring order.

        Args:
            model: Only this model's records.
            engine_role: Only records of this engine role.
            plane: Only records of this plane.
            node: Only records whose channel is on this node (a value of the
                channel's ``on``).
            served_only: Only the served models' records.
        """
        channels = self._channels()
        return tuple(
            binding
            for binding in self._all_bindings(served_only)
            if (model is None or binding.model == model)
            and (engine_role is None or binding.engine_role == engine_role)
            and (plane is None or binding.plane == plane)
            and (node is None or node in _on_nodes(channels.get(binding.address)))
        )

    def _all_bindings(self, served_only: bool) -> Iterator[Binding]:
        for model in self.models():
            if model.engine == TEXTURE or (served_only and not model.served):
                continue
            yield from model.bindings

    def binding(self, address: str) -> Binding | None:
        """The served physics model's wiring record of ``address``, or ``None``."""
        for binding in self._all_bindings(served_only=True):
            if binding.address == address:
                return binding
        return None

    def seed(self, address: str) -> SeedRecord | None:
        """The seed record of the channel at ``address``, or ``None``."""
        return self.document(SEEDS_FILE).seeds.get(address)

    def scenarios(self) -> tuple[ScenarioRecord, ...]:
        """Every scenario of ``scenarios.json``, sorted by name.

        Raises:
            ViewSchemaError: A scenario's shape breaks :class:`ScenarioRecord`,
                its ``still`` not a list of addresses among them.
        """
        return self.document(SCENARIOS_FILE).scenarios

    def motion_envelope(self, address: str, active: Sequence[str] = ()) -> float:
        """The band ``address``'s served motion keeps it within around its held value.

        Args:
            address: A channel address.
            active: The active scenario names, in order; ``nominal`` always
                counts, as the composite counts it.

        Returns:
            The channel's envelope under the set the composite serves for
            ``active`` (:func:`osprey_connectors.simulation.state.compose`,
            :func:`osprey_connectors.simulation.envelope.active_envelopes`):
            ``nominal``'s alone for a set that does not compose; 0.0 for a
            channel that declares no motion.
        """
        from osprey_connectors.simulation.envelope import active_envelopes
        from osprey_connectors.simulation.state import ActiveState, compose

        requested = ActiveState(names=tuple(active))
        if requested.names not in self._envelopes:
            scenarios = {scenario.name: scenario for scenario in self.scenarios()}
            self._envelopes[requested.names] = active_envelopes(
                self.document(SEEDS_FILE).seeds, compose(scenarios, requested)
            )
        return self._envelopes[requested.names].get(address, 0.0)

    def scenario_dir(self, name: str) -> Path:
        """The view's copy of a scenario's attached files, ``scenarios/<name>/``."""
        return self.path / SCENARIOS_DIR / name


def _on_nodes(channel: Channel | None) -> tuple[str, ...]:
    if channel is None or channel.on is None:
        return ()
    return tuple(channel.on.values())
