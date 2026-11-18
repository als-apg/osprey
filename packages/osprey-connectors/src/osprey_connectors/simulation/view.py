"""The simulator view a build writes, and the one reader of it.

``osprey build`` writes each render's simulator view into
``<render>/data/simulator/``; this module names its files and their schemas
and opens it. Every reader of a view opens it through :class:`SimulatorView`,
which refuses a file whose ``schema`` is not the one this OSPREY reads, so a
render from an older OSPREY is rebuilt rather than misread.

A wiring record of a physics model carries what its engine's ``describe()``
stated at build time: its ``engine_role`` (an
:data:`~osprey_connectors.simulation.records.EngineRole`), the transverse
``plane`` it steers or reads (``x``, ``y`` or ``None``), its ``refresh``
(one of :data:`REFRESH`) and, when it carries any, its ``marks`` (of
:data:`~osprey_connectors.simulation.records.Mark`). The words are the facility schema's. A ``readback`` is any read of an element's setting, paired
with a setpoint or not; a reader wanting the readback of a setpoint reads that
setpoint channel's ``pair``.

Each file is validated, when first read, against its document model in
:mod:`osprey_connectors.simulation.records`, a
:class:`~osprey_connectors.simulation.records.ViewDocument`; a file whose shape
breaks its model is refused as a :class:`ViewSchemaError` naming the key and
the reason. The view serves the records themselves: a channel as its
:class:`~osprey_connectors.simulation.records.ChannelRecord`, a model as its
:class:`~osprey_connectors.simulation.records.ModelRecord` and a model's wiring
as its :class:`~osprey_connectors.simulation.records.WiringRecord` items.
:func:`document_bytes` is the one writer of a view file's bytes.

The module imports only the standard library, pydantic and ``osprey_connectors``,
so a reader that must not load a model can open a view.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final, Literal, TypeVar, get_args

from pydantic import ValidationError

from osprey_connectors.errors import ConnectorsError
from osprey_connectors.simulation import TEXTURE
from osprey_connectors.simulation.records import (
    VIEW_DOCUMENTS,
    AddressesDocument,
    ChannelRecord,
    EngineRole,
    Mark,
    ModelRecord,
    Plane,
    Refresh,
    ScenarioRecord,
    ScenariosDocument,
    SeedRecord,
    SeedsDocument,
    ServedModelsDocument,
    VariablesDocument,
    ViewDocument,
    WiringRecord,
)

__all__ = [
    "DECKS_DIR",
    "ENGINEER",
    "HEARTBEAT",
    "IOC_HEALTH_SIGNALS",
    "IocHealthSignal",
    "MONITOR",
    "OUTPUT",
    "PERIODIC",
    "PERIODIC_IOC_HEALTH",
    "PLANES",
    "READBACK",
    "REFRESH",
    "SCENARIOS_DIR",
    "SERVED_SIGNAL_PROPERTY",
    "SETPOINT",
    "STATUS_SIGNAL",
    "UPTIME",
    "VIEW_RELPATH",
    "NoSimulatorViewError",
    "SimulatorViewError",
    "SimulatorView",
    "ViewSchemaError",
    "document_bytes",
    "sim_address",
    "view_path",
]

#: The view's directory under a render.
VIEW_RELPATH = PurePosixPath("data/simulator")

#: The directory holding a byte copy of each deck-bearing model's deck.
DECKS_DIR = "decks"
#: The directory holding each scenario's attached files, ``scenarios/<name>/``,
#: under the facility tree and under the view alike.
SCENARIOS_DIR = "scenarios"

SETPOINT: Final = "setpoint"
READBACK: Final = "readback"
MONITOR: Final = "monitor"
OUTPUT: Final = "output"
#: The engine roles a wiring record's description names, every
#: :data:`~osprey_connectors.simulation.records.EngineRole`. A channel's
#: facility role is a :data:`~osprey_connectors.simulation.records.ChannelRole`.
_ENGINE_ROLES: tuple[EngineRole, ...] = get_args(EngineRole)
#: The planes a wiring record's description names; ``None`` is no plane.
PLANES: tuple[Plane, ...] = get_args(Plane)
PERIODIC: Final = "periodic"
#: The refresh a wiring record's description names, every
#: :data:`~osprey_connectors.simulation.records.Refresh`.
REFRESH: tuple[Refresh, ...] = get_args(Refresh)
#: The mark of a record whose write is a command; only a ``write`` record
#: carries it.
_COMMAND_MARK: Final = "command"
#: The marks a wiring record's description may set.
_MARKS: tuple[Mark, ...] = get_args(Mark)

#: The signal of a physics model's status address, ``<code>:SIM:<model>:STATUS``.
STATUS_SIGNAL: Final = "STATUS"
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

_Document = TypeVar("_Document", bound=ViewDocument)


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


def _shape_error(path: Path, refusal: ValidationError) -> ViewSchemaError:
    """The first key a view file's model refuses, and why, as the view's error."""
    error = refusal.errors()[0]
    key = ".".join(str(part) for part in error["loc"]) or "the document"
    return ViewSchemaError(
        f"{path}: {key}: {error['msg']} (found {error['input']!r}): rebuild with osprey build",
        path=path,
        key=key if error["loc"] else None,
        found=error["input"],
        expected=error["msg"],
    )


def document_bytes(document: ViewDocument) -> bytes:
    """A view file's bytes: its document's keys as stated, sorted, two-space indented.

    The one serialization of a view file; ``osprey build`` writes each file
    through it, so a file reads back as the document it was written from.

    Raises:
        ValueError: The document holds a non-finite number, which JSON cannot state.
    """
    stated = document.model_dump(mode="json", by_alias=True, exclude_unset=True)
    text = json.dumps(stated, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return (text + "\n").encode("utf-8")


def view_path(project_dir: Path | str) -> Path:
    """Where the view of ``project_dir``'s render is, read or not.

    A deployment repo keeps its render under ``build/``, beside the rendered
    ``config.yml``; a container's project directory is the render itself.
    """
    from osprey_connectors.workspace import rendered_config_path

    project = Path(project_dir)
    rendered = rendered_config_path(project)
    render = rendered.parent if rendered.is_file() else project
    return render / VIEW_RELPATH


#: The infix of every address the simulator serves for a model itself.
_SIM = "SIM"


def sim_address(code: str, model: str, signal: Literal["STATUS"] | IocHealthSignal) -> str:
    """An address the simulator serves for a model itself, ``<code>:SIM:<model>:<signal>``.

    Args:
        code: The facility's identity code.
        model: The model's name.
        signal: :data:`STATUS_SIGNAL` for a physics model's status, else one
            of :data:`IOC_HEALTH_SIGNALS`.
    """
    return f"{code}:{_SIM}:{model}:{signal}"


def _sim_signal(code: str, address: str) -> tuple[str, IocHealthSignal] | None:
    """The model and IOC health signal of a :func:`sim_address`; ``None`` for any other address."""
    prefix = f"{code}:{_SIM}:"
    model, _, word = address.removeprefix(prefix).rpartition(":")
    signal = next((known for known in IOC_HEALTH_SIGNALS if known == word), None)
    if not model or not address.startswith(prefix) or signal is None:
        return None
    return model, signal


class SimulatorView:
    """A render's simulator view, opened from its directory.

    A view is opened by :meth:`open` (or :meth:`of_render`, :meth:`of_project`
    or :meth:`find`), which reads ``addresses.json``; constructing one reads
    nothing. Every other file is read once, on first use, and each is refused
    when its ``schema`` is not one this OSPREY reads; :meth:`check` reads them
    all at once.

    Attributes:
        path: The view's directory.
        render_root: The render the view sits in, when the view is at
            ``<render>/data/simulator``; else ``None``.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        parts = PurePosixPath(*self.path.parts[-len(VIEW_RELPATH.parts) :])
        self.render_root = self.path.parents[1] if parts == VIEW_RELPATH else None
        self._documents: dict[type[ViewDocument], ViewDocument] = {}
        self._channel_records: Mapping[str, ChannelRecord] | None = None
        self._envelopes: dict[tuple[str, ...], dict[str, float]] = {}

    @classmethod
    def open(cls, view_dir: Path | str) -> SimulatorView:
        """Open the view in ``view_dir``.

        Raises:
            NoSimulatorViewError: ``view_dir`` has no ``addresses.json``.
            ViewSchemaError: its ``addresses.json`` is not schema
                ``AddressesDocument.SCHEMA``, or its shape breaks the document.
        """
        view = cls(Path(view_dir))
        if not (view.path / AddressesDocument.FILE).is_file():
            raise NoSimulatorViewError(
                _NO_VIEW_MESSAGE.format(path=view.path), path=view.path / AddressesDocument.FILE
            )
        view.document(AddressesDocument)
        return view

    @classmethod
    def of_render(cls, render_root: Path | str) -> SimulatorView:
        """Open the view of the render rooted at ``render_root``."""
        return cls.open(Path(render_root) / VIEW_RELPATH)

    @classmethod
    def of_project(cls, project_dir: Path | str) -> SimulatorView:
        """Open the view of ``project_dir``'s render (:func:`view_path`)."""
        return cls.open(view_path(project_dir))

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
    # validates the rest against the file's document model. The facility
    # schema leaves a scenario's ``archiver`` events free-form; each event's
    # record (``ArchiverEvent``) holds the rules between its keys, so an event
    # that breaks one refuses the view, and the build checks the rest against
    # the event's channel with ``osprey.facility.scenarios.check_scenario_events``
    # before it writes ``scenarios.json``.
    def document(self, kind: type[_Document]) -> _Document:
        """One view file, parsed and validated once, read-only.

        Args:
            kind: The file's document, one of
                :data:`~osprey_connectors.simulation.records.VIEW_DOCUMENTS`.

        Returns:
            The file's document.

        Raises:
            NoSimulatorViewError: the file is not there.
            ViewSchemaError: the file is not JSON, its ``schema`` is absent or
                neither ``kind.SCHEMA`` nor one of ``kind.ALSO_READS``, or its
                shape breaks the document.
        """
        held = self._documents.get(kind)
        if isinstance(held, kind):
            return held
        path = self.path / kind.FILE
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
        if not isinstance(parsed, dict):
            raise _schema_error(path, None, kind.SCHEMA)
        found = parsed.get("schema")
        if found != kind.SCHEMA and found not in kind.ALSO_READS:
            raise _schema_error(path, found, kind.SCHEMA)
        try:
            document = kind.model_validate(parsed)
        except ValidationError as refusal:
            raise _shape_error(path, refusal) from None
        self._documents[kind] = document
        return document

    def check(self) -> None:
        """Read every file of the view now, so one this OSPREY cannot read is refused here.

        Raises:
            NoSimulatorViewError: a file is not there.
            ViewSchemaError: a file is not JSON, is from another schema, or its
                shape breaks its document.
        """
        for kind in VIEW_DOCUMENTS:
            self.document(kind)

    @property
    def code(self) -> str:
        """The facility's identity code."""
        return self.document(VariablesDocument).code

    def models(self) -> tuple[ModelRecord, ...]:
        """Every model of the facility file, sorted by name."""
        return self.document(VariablesDocument).models

    def model(self, name: str) -> ModelRecord:
        """The model named ``name``.

        Raises:
            KeyError: the view has no such model.
        """
        for model in self.models():
            if model.name == name:
                return model
        raise KeyError(f"the simulator view has no model {name!r}")

    def deck_path(self, model: ModelRecord) -> Path | None:
        """The view's copy of ``model``'s deck, or ``None`` for a model with no deck."""
        return self.path / model.deck if model.deck else None

    def served(self) -> tuple[str, ...]:
        """The names of the models the render serves, in ``served_models.json`` order."""
        return self.document(ServedModelsDocument).models

    def physics_models(self) -> tuple[ModelRecord, ...]:
        """The served models whose engine is not ``texture``, sorted by name."""
        return tuple(model for model in self.models() if model.served and model.engine != TEXTURE)

    def channels(self) -> tuple[str, ...]:
        """Every channel address of the facility file, sorted."""
        return self.document(AddressesDocument).channels

    def status_addresses(self) -> Mapping[str, str]:
        """Each served physics model's status address, ``<code>:SIM:<model>:STATUS``."""
        listed = set(self.document(AddressesDocument).status)
        return MappingProxyType(
            {
                model.name: address
                for model in self.physics_models()
                if (address := sim_address(self.code, model.name, STATUS_SIGNAL)) in listed
            }
        )

    def ioc_health_by_model(self) -> Mapping[str, Mapping[IocHealthSignal, str]]:
        """Each listed model's IOC health addresses by signal, ``{model: {signal: address}}``.

        An address is :func:`sim_address` of a model and one of
        :data:`IOC_HEALTH_SIGNALS`; each model's signals keep the list's order.
        ``ioc_health`` is optional under ``AddressesDocument.SCHEMA``: a view from
        a build without it yields an empty mapping. Unlike the status
        addresses, it lists the ``texture`` model too.

        Raises:
            ViewSchemaError: a listed entry is not such an address.
        """
        listed = self.document(AddressesDocument).ioc_health
        if not listed:
            return MappingProxyType({})
        code = self.code
        by_model: dict[str, dict[IocHealthSignal, str]] = {}
        for address in listed:
            parsed = _sim_signal(code, address)
            if parsed is None:
                prefix = f"{code}:{_SIM}:"
                raise ViewSchemaError(
                    f"{self.path / AddressesDocument.FILE} lists {address!r} under ioc_health, "
                    f"which is not {prefix}<model>:<signal> with <signal> one of "
                    f"{', '.join(IOC_HEALTH_SIGNALS)}: rebuild with osprey build",
                    path=self.path / AddressesDocument.FILE,
                    key="ioc_health",
                    found=address,
                    expected=f"{prefix}<model>:<signal>",
                )
            model, signal = parsed
            by_model.setdefault(model, {})[signal] = address
        return MappingProxyType(
            {model: MappingProxyType(signals) for model, signals in by_model.items()}
        )

    def _channels(self) -> Mapping[str, ChannelRecord]:
        if self._channel_records is None:
            self._channel_records = MappingProxyType(
                {channel.address: channel for channel in self.document(VariablesDocument).channels}
            )
        return self._channel_records

    def channel(self, address: str) -> ChannelRecord:
        """The channel at ``address``.

        Raises:
            KeyError: the view has no channel at ``address``.
        """
        return self._channels()[address]

    def wiring(
        self,
        *,
        model: str | None = None,
        engine_role: EngineRole | None = None,
        plane: Plane | None = None,
        node: str | None = None,
        served_only: bool = True,
    ) -> tuple[WiringRecord, ...]:
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
            record
            for entry in self._wired(served_only)
            if model is None or entry.name == model
            for record in entry.wiring
            if (engine_role is None or record.engine_role == engine_role)
            and (plane is None or record.plane == plane)
            and (node is None or node in _on_nodes(channels.get(record.address)))
        )

    def _wired(self, served_only: bool) -> Iterator[ModelRecord]:
        """The models whose wiring a physics engine serves: every one but ``texture``."""
        for model in self.models():
            if model.engine == TEXTURE or (served_only and not model.served):
                continue
            yield model

    def wiring_of(self, address: str) -> WiringRecord | None:
        """The served physics model's wiring record of ``address``, or ``None``."""
        for model in self._wired(served_only=True):
            for record in model.wiring:
                if record.address == address:
                    return record
        return None

    def seed(self, address: str) -> SeedRecord | None:
        """The seed record of the channel at ``address``, or ``None``."""
        return self.document(SeedsDocument).seeds.get(address)

    def scenarios(self) -> tuple[ScenarioRecord, ...]:
        """Every scenario of ``scenarios.json``, sorted by name.

        Raises:
            ViewSchemaError: A scenario's shape breaks :class:`ScenarioRecord`,
                its ``still`` not a list of addresses among them.
        """
        return self.document(ScenariosDocument).scenarios

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
                self.document(SeedsDocument).seeds, compose(scenarios, requested)
            )
        return self._envelopes[requested.names].get(address, 0.0)

    def scenario_dir(self, name: str) -> Path:
        """The view's copy of a scenario's attached files, ``scenarios/<name>/``."""
        return self.path / SCENARIOS_DIR / name


def _on_nodes(channel: ChannelRecord | None) -> tuple[str, ...]:
    if channel is None or channel.on is None:
        return ()
    return tuple(channel.on.values())
