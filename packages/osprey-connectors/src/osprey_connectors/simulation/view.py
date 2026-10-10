"""The simulator view a build writes, and the one reader of it.

``osprey build`` writes each render's simulator view into
``<render>/data/simulator/``; this module names its files and their schemas
and opens it. Every reader of a view opens it through :class:`SimulatorView`,
which refuses a file whose ``schema`` is not the one this OSPREY reads, so a
render from an older OSPREY is rebuilt rather than misread.

A wiring record of a physics model carries what its engine's ``describe()``
stated at build time: its ``role`` (``setpoint``, ``readback``, ``monitor``
or ``output``), the transverse ``plane`` it steers or reads (``x``, ``y`` or
``None``) and its ``refresh`` (``pass`` or ``periodic``). A ``readback`` is
any read of an element's setting, paired with a setpoint or not; a reader
wanting the readback of a setpoint reads that setpoint channel's ``pair``.

The module imports only the standard library and ``osprey_connectors``, so a
reader that must not load a model can open a view.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

__all__ = [
    "ADDRESSES_FILE",
    "ADDRESSES_SCHEMA",
    "DECKS_DIR",
    "NO_VIEW_MESSAGE",
    "PLANES",
    "REFRESH",
    "ROLES",
    "SCENARIOS_DIR",
    "SCENARIOS_FILE",
    "SCENARIOS_SCHEMA",
    "SCHEMAS",
    "SEEDS_FILE",
    "SEEDS_SCHEMA",
    "SERVED_MODELS_FILE",
    "SERVED_MODELS_SCHEMA",
    "TEXTURE",
    "VARIABLES_FILE",
    "VARIABLES_SCHEMA",
    "VIEW_RELPATH",
    "Binding",
    "Channel",
    "Model",
    "NoSimulatorView",
    "SimulatorView",
    "ViewSchemaError",
]

#: The view's directory under a render.
VIEW_RELPATH = PurePosixPath("data/simulator")

SERVED_MODELS_FILE = "served_models.json"
SERVED_MODELS_SCHEMA = "osprey.facility.served_models/1"
ADDRESSES_FILE = "addresses.json"
ADDRESSES_SCHEMA = "osprey.facility.addresses/1"
VARIABLES_FILE = "variables.json"
VARIABLES_SCHEMA = "osprey.facility.simulator/2"
SEEDS_FILE = "seeds.json"
SEEDS_SCHEMA = "osprey.facility.seeds/2"
SCENARIOS_FILE = "scenarios.json"
SCENARIOS_SCHEMA = "osprey.facility.scenarios/2"
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

#: The engine of the model that serves every channel no physics model wires.
TEXTURE = "texture"

#: The roles a wiring record's description names.
ROLES = ("setpoint", "readback", "monitor", "output")
#: The planes a wiring record's description names; ``None`` is no plane.
PLANES = ("x", "y")
#: The refresh a wiring record's description names: ``pass`` with each
#: write's solve, ``periodic`` with the periodic solve alone.
REFRESH = ("pass", "periodic")

#: The text a missing view reports.
NO_VIEW_MESSAGE = "no simulator view at {path}: run osprey build"

#: The keys a wiring record's description adds.
_DESCRIPTION_KEYS = ("role", "plane", "refresh")


class NoSimulatorView(LookupError):
    """The place holds no simulator view: it has no ``addresses.json``."""


class ViewSchemaError(ValueError):
    """A view file's ``schema`` is absent or not the one this OSPREY reads."""


def _freeze(value: Any) -> Any:
    """A read-only copy of a JSON value: mappings as proxies, lists as tuples."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _schema_error(path: Path, found: Any, expected: str) -> ViewSchemaError:
    return ViewSchemaError(
        f"{path} is schema {found!r}; this OSPREY reads {expected!r}: rebuild with osprey build"
    )


@dataclass(frozen=True)
class Channel:
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
    role: str
    pair: str | None
    value_type: str
    unit: str | None
    description: str | None
    writable: bool
    value_range: tuple[Any, ...] | None
    owner: str
    on: Mapping[str, str] | None
    options: tuple[Any, ...] | None
    shape: tuple[int, ...] | None
    precision: int | None

    @classmethod
    def from_entry(cls, entry: Mapping[str, Any]) -> Channel:
        """Read one channel entry of ``variables.json``.

        Raises:
            ViewSchemaError: the entry lacks a key every channel carries.
        """
        try:
            return cls(
                address=str(entry["address"]),
                role=entry["role"],
                pair=entry["pair"],
                value_type=entry["value_type"],
                unit=entry["unit"],
                description=entry["description"],
                writable=bool(entry["writable"]),
                value_range=_freeze(entry["value_range"]),
                owner=entry["owner"],
                on=_freeze(entry["on"]),
                options=_freeze(entry.get("options")),
                shape=_freeze(entry.get("shape")),
                precision=entry.get("precision"),
            )
        except KeyError as missing:
            raise ViewSchemaError(
                f"channel {entry.get('address')!r} lacks {missing}: rebuild with osprey build"
            ) from None


@dataclass(frozen=True)
class Binding:
    """One wiring record of a physics model: the address it serves and what it is.

    Attributes:
        id: The wiring record's id.
        model: The model the record belongs to.
        address: The channel address the record serves.
        direction: ``write`` or ``read``.
        role: ``setpoint``, ``readback``, ``monitor`` or ``output``. A
            ``readback`` reads an element's setting, paired with a setpoint
            or not; the setpoint channel's ``pair`` names its readback.
        plane: ``x`` or ``y`` for a record steering or reading one transverse
            plane, else ``None``.
        refresh: ``pass`` or ``periodic``.
        element: The element the record names, or ``None`` (a record naming
            slices, or none).
        record: The wiring record as the view states it, read-only; the
            engine's ``build`` takes it verbatim.
    """

    id: str
    model: str
    address: str
    direction: str
    role: str
    plane: str | None
    refresh: str
    element: str | None
    record: Mapping[str, Any]

    @classmethod
    def from_record(cls, model: str, record: Mapping[str, Any]) -> Binding:
        """Read one wiring record of ``variables.json``.

        Raises:
            ViewSchemaError: the record lacks ``role``, ``plane`` or
                ``refresh``.
        """
        missing = [key for key in _DESCRIPTION_KEYS if key not in record]
        if missing:
            raise ViewSchemaError(
                f"wiring record {record.get('address')!r} of model {model!r} lacks "
                f"{', '.join(missing)}: rebuild with osprey build"
            )
        return cls(
            id=str(record.get("id")),
            model=model,
            address=str(record["address"]),
            direction=str(record.get("direction")),
            role=record["role"],
            plane=record["plane"],
            refresh=record["refresh"],
            element=record.get("element"),
            record=MappingProxyType(copy.deepcopy(dict(record))),
        )


@dataclass(frozen=True)
class Model:
    """One model of the view.

    Attributes:
        name: The model's name.
        engine: The engine the model names.
        served: Whether the render serves the model.
        settings: The model's settings, read-only.
        deck: The path of the view's copy of the model's deck, or ``None``.
        bindings: The model's wiring records, in wiring order; empty for a
            ``texture`` model or one without wiring.
    """

    name: str
    engine: str | None
    served: bool
    settings: Mapping[str, Any]
    deck: Path | None
    bindings: tuple[Binding, ...]


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
        self._raw: dict[str, dict[str, Any]] = {}
        self._documents: dict[str, Mapping[str, Any]] = {}
        self._models: tuple[Model, ...] | None = None
        self._channel_records: Mapping[str, Channel] | None = None
        self._envelopes: dict[tuple[str, ...], dict[str, float]] = {}
        if not (self.path / ADDRESSES_FILE).is_file():
            raise NoSimulatorView(NO_VIEW_MESSAGE.format(path=self.path))
        self.document(ADDRESSES_FILE)

    @classmethod
    def open(cls, view_dir: Path | str) -> SimulatorView:
        """Open the view in ``view_dir``.

        Raises:
            NoSimulatorView: ``view_dir`` has no ``addresses.json``.
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
        except NoSimulatorView:
            return None

    def document(self, name: str) -> Mapping[str, Any]:
        """One view file, parsed once and read-only.

        Args:
            name: The file's name, one of :data:`SCHEMAS`.

        Raises:
            ViewSchemaError: the file's ``schema`` is absent or not
                ``SCHEMAS[name]``.
        """
        if name not in self._documents:
            self._documents[name] = _freeze(self._parsed(name))
        return self._documents[name]

    def _parsed(self, name: str) -> dict[str, Any]:
        if name not in self._raw:
            path = self.path / name
            parsed = json.loads(path.read_text(encoding="utf-8"))
            expected = SCHEMAS[name]
            found = parsed.get("schema") if isinstance(parsed, dict) else None
            if found != expected:
                raise _schema_error(path, found, expected)
            self._raw[name] = parsed
        return self._raw[name]

    @property
    def code(self) -> str:
        """The facility's identity code."""
        return str(self.document(VARIABLES_FILE)["code"])

    def models(self) -> tuple[Model, ...]:
        """Every model of the facility file, sorted by name."""
        if self._models is None:
            self._models = tuple(
                self._model(entry) for entry in self._parsed(VARIABLES_FILE)["models"]
            )
        return self._models

    def _model(self, entry: Mapping[str, Any]) -> Model:
        name = str(entry["name"])
        engine = entry.get("engine")
        deck = entry.get("deck")
        wiring = entry.get("wiring") or ()
        return Model(
            name=name,
            engine=engine,
            served=bool(entry.get("served")),
            settings=MappingProxyType(copy.deepcopy(entry.get("settings") or {})),
            deck=self.path / deck if deck else None,
            bindings=()
            if engine == TEXTURE
            else tuple(Binding.from_record(name, record) for record in wiring),
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
        return tuple(str(name) for name in self.document(SERVED_MODELS_FILE)["models"])

    def physics_models(self) -> tuple[Model, ...]:
        """The served models whose engine is not ``texture``, sorted by name."""
        return tuple(model for model in self.models() if model.served and model.engine != TEXTURE)

    def channels(self) -> tuple[str, ...]:
        """Every channel address of the facility file, sorted."""
        return tuple(str(address) for address in self.document(ADDRESSES_FILE)["channels"])

    def status_addresses(self) -> Mapping[str, str]:
        """Each served physics model's status address, ``<code>:SIM:<model>:STATUS``."""
        listed = set(self.document(ADDRESSES_FILE)["status"])
        return MappingProxyType(
            {
                model.name: address
                for model in self.physics_models()
                if (address := f"{self.code}:SIM:{model.name}:STATUS") in listed
            }
        )

    def _channels(self) -> Mapping[str, Channel]:
        if self._channel_records is None:
            self._channel_records = MappingProxyType(
                {
                    channel.address: channel
                    for channel in (
                        Channel.from_entry(entry)
                        for entry in self._parsed(VARIABLES_FILE)["channels"]
                    )
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
        role: str | None = None,
        plane: str | None = None,
        node: str | None = None,
        served_only: bool = True,
    ) -> tuple[Binding, ...]:
        """The physics models' wiring records, model by name, then in wiring order.

        Args:
            model: Only this model's records.
            role: Only records of this role.
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
            and (role is None or binding.role == role)
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

    def seed(self, address: str) -> Mapping[str, Any] | None:
        """The seed record of the channel at ``address``, or ``None``."""
        seed: Mapping[str, Any] | None = self.document(SEEDS_FILE)["seeds"].get(address)
        return seed

    def scenarios(self) -> tuple[Mapping[str, Any], ...]:
        """Every scenario of ``scenarios.json``, sorted by name."""
        scenarios: tuple[Mapping[str, Any], ...] = self.document(SCENARIOS_FILE)["scenarios"]
        return scenarios

    def motion_envelope(self, address: str, active: Sequence[str] = ()) -> float:
        """The band ``address``'s served motion keeps it within around its held value.

        Args:
            address: A channel address.
            active: The active scenario names, in order; none for the seeds'
                own motion.

        Returns:
            The channel's envelope under ``active``
            (:func:`osprey_connectors.simulation.envelope.active_envelopes`);
            0.0 for a channel that declares no motion.
        """
        from osprey_connectors.simulation.envelope import active_envelopes

        key = tuple(active)
        if key not in self._envelopes:
            self._envelopes[key] = active_envelopes(
                self.document(SEEDS_FILE)["seeds"],
                {str(scenario["name"]): scenario for scenario in self.scenarios()},
                key,
            )
        return self._envelopes[key].get(address, 0.0)

    def scenario_dir(self, name: str) -> Path:
        """The view's copy of a scenario's attached files, ``scenarios/<name>/``."""
        return self.path / SCENARIOS_DIR / name


def _on_nodes(channel: Channel | None) -> tuple[str, ...]:
    if channel is None or channel.on is None:
        return ()
    return tuple(str(node) for node in channel.on.values())
