"""The composite: one LUME model serving a simulator view's every channel.

:class:`Composite` reads the simulator view a render writes under
``data/simulator/`` and serves its channels through two kinds of child:

* one **physics child** per served model whose engine is not ``texture``,
  built by the engine plug-in the ``osprey.simulation.engines`` entry-point
  group names, through ``build(model, wiring, deck, settings, active=...)``;
* the **texture**, a :class:`~osprey_connectors.simulation.texture.TextureModel`
  holding every other channel and the motion of all of them.

``supported_variables`` is the channel addresses of ``addresses.json`` and its
status addresses, ``<code>:SIM:<model>:STATUS``, one per physics child. Each
child's own variables that no channel names (faults, optics) are reached as
``<model>/<name>`` through :meth:`Composite.model_get` and
:meth:`Composite.model_set` only.

**Reads.** A physics readback of type float reads ``truth + motion(address,
t)``, passed through the child's ``readout`` when the engine exports one, then
clamped; the readout always sees both readings of a monitor, so a read of one
plane reads its partner too. A texture channel reads as the texture serves it.
:meth:`Composite.held` reads every channel without motion or readout.

**Writes.** :meth:`Composite.set` coerces each value to its channel's
``value_type``, then sets the physics children in name order and the texture
last. When any child refuses, every earlier child gets its previous inputs back
and the refusal is raised. A setpoint the active scenarios mark ``stuck``
accepts a write and does not forward it.

**A failed child.** A child whose engine raises while it is built or read is
failed, with the engine's error text as its status (``ok`` otherwise), capped
at :data:`STATUS_MAX_BYTES` UTF-8 bytes. It holds the inputs written to it and
never raises: its setpoints read those inputs, its float outputs NaN, any other
output its last good value or its type's zero, and
:meth:`Composite.output_severity` names each of its channels. A refused write
leaves a child as it was. A failed child is rebuilt on :meth:`Composite.reset`
and on a change of the active scenarios.

**Scenarios.** The active set is the ``active_scenarios`` file of the state
directory, re-read when its modification time changes. A change rebuilds every
physics child with the active scenarios' writes and fault seeds as its start
state, and hands the texture the active writes it owns and the active drivers,
couplings and noise replacements; session writes are dropped. A set whose
scenarios touch one target twice is served without its scenarios.

**The model log.** Each physics child appends JSON lines ``{instance, pid,
...}`` to ``var/simulator/<model>.log`` under the repo root of the loaded
config, one ``os.write`` per record of under :data:`LOG_RECORD_MAX_BYTES`
bytes. Without a loaded config the records go to the process logger only.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
from lume.exceptions import ReadOnlyError
from lume.model import LUMEModel
from lume.variables import NDVariable, StrVariable, Variable

from osprey_connectors.config import default_config_path
from osprey_connectors.logger import get_logger
from osprey_connectors.simulation import values
from osprey_connectors.simulation.state import (
    ACTIVE_SCENARIOS_FILENAME,
    overlap_record,
    parse_active_state,
    resolve_active_scenarios,
    validate_composition,
)
from osprey_connectors.simulation.texture import TEXTURE_OWNER, TextureModel, channel_variable
from osprey_connectors.workspace import repo_root_for_config

__all__ = [
    "ENGINE_GROUP",
    "INSTANCES",
    "LOG_RECORD_MAX_BYTES",
    "STATUS_MAX_BYTES",
    "STATUS_OK",
    "STUCK",
    "UDF",
    "Composite",
    "cap_status",
    "log_dir",
]

logger = get_logger("simulation_composite")

#: The entry-point group every engine plug-in registers under.
ENGINE_GROUP = "osprey.simulation.engines"

#: The serving instances a model log record names.
INSTANCES = ("virtual_accelerator", "live_standin", "inprocess")

#: A physics child's status while its engine serves it.
STATUS_OK = "ok"

#: The longest status text, in UTF-8 bytes.
STATUS_MAX_BYTES = 1023

#: Every model log record, newline included, is shorter than this many bytes.
LOG_RECORD_MAX_BYTES = 4096

#: The fault value that makes a setpoint accept writes without forwarding them.
STUCK = "stuck"

#: The condition :meth:`Composite.output_severity` names for a failed child's channel.
UDF = "udf"

_ELLIPSIS = "…"
_SETPOINT = "setpoint"
_FAULT_SEPARATOR = "/"
_MODEL_SEPARATOR = "/"
_LOG_MODE = 0o664
_MS_PER_S = 1000.0

_SERVED_MODELS_FILE = "served_models.json"
_ADDRESSES_FILE = "addresses.json"
_VARIABLES_FILE = "variables.json"
_SEEDS_FILE = "seeds.json"
_SCENARIOS_FILE = "scenarios.json"


def cap_status(text: str, max_bytes: int = STATUS_MAX_BYTES) -> str:
    """``text`` cut to at most ``max_bytes`` UTF-8 bytes, ending in ``…`` when cut.

    Args:
        text: The status text.
        max_bytes: The bound, in UTF-8 bytes.

    Returns:
        ``text`` itself when it fits; otherwise its longest prefix of whole
        characters that fits with ``…`` appended.
    """
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    room = max_bytes - len(_ELLIPSIS.encode("utf-8"))
    if room <= 0:
        return ""
    return raw[:room].decode("utf-8", errors="ignore") + _ELLIPSIS


def log_dir() -> Path | None:
    """The directory the model logs are appended in, or ``None`` without a loaded config.

    Returns:
        ``var/simulator`` under the repo root of the config this process
        loaded.
    """
    config_path = default_config_path()
    if config_path is None:
        return None
    return repo_root_for_config(config_path) / "var" / "simulator"


def _read_json(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


def _plain(value: Any) -> Any:
    """A child's output in its stored representation: a waveform as a flat list."""
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return [float(item) for item in value.reshape(-1)]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _for_variable(variable: Variable, value: Any) -> Any:
    """A stored value as ``variable`` takes it: a waveform reshaped to the variable's shape."""
    if isinstance(variable, NDVariable):
        return np.asarray(value, dtype=np.float64).reshape(variable.shape)
    return value


@dataclass
class _Child:
    """One physics child and what the composite keeps for it."""

    name: str
    record: Mapping[str, Any]
    owned: list[str]
    defaults: dict[str, Any]
    setpoints: frozenset[str]
    partners: dict[str, tuple[str, ...]]
    engine: Any = None
    model: LUMEModel | None = None
    status: str = STATUS_OK
    active: dict[str, Any] = field(default_factory=dict)
    stuck: frozenset[str] = frozenset()
    inputs: dict[str, Any] = field(default_factory=dict)
    last_good: dict[str, Any] = field(default_factory=dict)


class Composite(LUMEModel):
    """Serve one simulator view through its physics children and the texture.

    Args:
        view_dir: The simulator view, ``<render>/data/simulator``.
        state_dir: The directory holding the ``active_scenarios`` file; ``None``
            serves ``nominal`` alone.
        instance: The serving instance the model log names, one of
            :data:`INSTANCES`.
        clock: Returns the current instant in epoch seconds.
        model_log: ``False`` sends the model log records to the process logger
            only, for a composite no instance serves.

    Raises:
        ValueError: ``instance`` is not one of :data:`INSTANCES`.
    """

    def __init__(
        self,
        view_dir: Path | str,
        *,
        state_dir: Path | str | None = None,
        instance: str = "inprocess",
        clock: Callable[[], float] = time.time,
        model_log: bool = True,
    ) -> None:
        if instance not in INSTANCES:
            raise ValueError(f"instance is {instance!r}; use one of {list(INSTANCES)}")
        self._view_dir = Path(view_dir)
        self._instance = instance
        self._clock = clock
        self._state_path = (
            None if state_dir is None else Path(state_dir) / ACTIVE_SCENARIOS_FILENAME
        )
        self._log_dir = log_dir() if model_log else None
        self._active: list[str] = []

        variables = _read_json(self._view_dir / _VARIABLES_FILE)
        seeds = _read_json(self._view_dir / _SEEDS_FILE)
        addresses = _read_json(self._view_dir / _ADDRESSES_FILE)
        served = _read_json(self._view_dir / _SERVED_MODELS_FILE)["models"]
        self._scenarios: dict[str, Mapping[str, Any]] = {
            str(scenario["name"]): scenario
            for scenario in _read_json(self._view_dir / _SCENARIOS_FILE)["scenarios"]
        }
        self._channels: dict[str, Mapping[str, Any]] = {
            str(channel["address"]): channel for channel in variables["channels"]
        }
        records = {str(model["name"]): model for model in variables["models"]}
        physics = sorted(
            name
            for name in served
            if name in records and records[name].get("engine") != TEXTURE_OWNER
        )
        code = str(variables["code"])

        self._texture = TextureModel(variables, seeds, clock=clock)
        self._owner: dict[str, str] = {
            address: (owner if (owner := str(record.get("owner"))) in physics else TEXTURE_OWNER)
            for address, record in self._channels.items()
        }
        self._children: dict[str, _Child] = {
            name: self._child(name, records[name]) for name in physics
        }
        self._status: dict[str, str] = {}
        for name in physics:
            address = f"{code}:SIM:{name}:STATUS"
            if address in addresses["status"]:
                self._status[address] = name

        self._variables: dict[str, Variable] = {}
        for address in addresses["channels"]:
            if self._owner[address] == TEXTURE_OWNER:
                self._variables[address] = self._texture.supported_variables[address]
            else:
                child = self._children[self._owner[address]]
                self._variables[address] = channel_variable(
                    self._channels[address], self._start_value(child, address)
                )
        for address in addresses["status"]:
            self._variables[address] = StrVariable(
                name=address, default_value=STATUS_OK, read_only=True
            )

        self._signature: tuple[str, int] | None | bool = False
        self._refresh()

    # -- construction --------------------------------------------------------

    def _child(self, name: str, record: Mapping[str, Any]) -> _Child:
        owned = sorted(address for address, owner in self._owner.items() if owner == name)
        wiring = list(record.get("wiring") or [])
        defaults = {
            str(entry["address"]): entry.get("default")
            for entry in wiring
            if str(entry["address"]) in owned
        }
        setpoints = frozenset(
            str(entry["address"])
            for entry in wiring
            if entry.get("direction") == "write" and str(entry["address"]) in owned
        )
        groups: dict[str, list[str]] = {}
        for entry in wiring:
            engine = entry.get("engine") or {}
            if (
                entry.get("direction") == "read"
                and entry.get("element") is not None
                and engine.get("axis") is not None
                and engine.get("attribute") is None
            ):
                groups.setdefault(str(entry["element"]), []).append(str(entry["address"]))
        partners = {address: tuple(sorted(group)) for group in groups.values() for address in group}
        return _Child(
            name=name,
            record=record,
            owned=owned,
            defaults=defaults,
            setpoints=setpoints,
            partners=partners,
        )

    def _start_value(self, child: _Child, address: str) -> Any:
        """A physics channel's start value: its active write, its wiring default, or zero."""
        channel = self._channels[address]
        value_type = channel.get("value_type")
        if address in child.active:
            return values.coerce(
                child.active[address], value_type, channel.get("options"), channel.get("shape")
            )
        default = child.defaults.get(address)
        if default is not None:
            return values.coerce(default, value_type, channel.get("options"), channel.get("shape"))
        return values.zero(value_type, channel.get("options"), channel.get("shape"))

    @staticmethod
    def _engine(name: str) -> Any:
        """The plug-in module registered for engine ``name``.

        Raises:
            LookupError: No plug-in is registered under that name.
        """
        found = metadata.entry_points(group=ENGINE_GROUP, name=name)
        for entry in found:
            return entry.load()
        raise LookupError(f"no simulation engine {name!r} is registered in {ENGINE_GROUP}")

    def _build(self, child: _Child) -> None:
        """Build a child at its active state; a raise leaves it failed with the engine's text."""
        record = child.record
        child.model = None
        child.engine = None
        child.inputs = {address: self._start_value(child, address) for address in child.setpoints}
        try:
            child.engine = self._engine(str(record.get("engine")))
            deck = record.get("deck")
            child.model = child.engine.build(
                child.name,
                list(record.get("wiring") or []),
                None if deck is None else self._view_dir / deck,
                record.get("settings"),
                active=dict(child.active),
            )
        except Exception as exc:
            self._fail(child, exc)
            return
        child.status = STATUS_OK
        self._log(child.name, {"event": "built", "status": STATUS_OK})

    def _fail(self, child: _Child, exc: BaseException) -> None:
        """Mark a child failed with the text its engine gives for ``exc``."""
        error_text = getattr(child.engine, "error_text", None)
        text = error_text(exc) if callable(error_text) else ""
        if not text:
            text = str(exc) or type(exc).__name__
        child.model = None
        child.status = cap_status(text)
        self._log(child.name, {"event": "failed", "error": text})

    # -- the active scenarios ------------------------------------------------

    def _state_signature(self) -> tuple[str, int] | None:
        if self._state_path is None:
            return None
        try:
            return (str(self._state_path), self._state_path.stat().st_mtime_ns)
        except FileNotFoundError:
            return None

    def _refresh(self) -> None:
        """Rebuild when the ``active_scenarios`` file changed since the last look."""
        signature = self._state_signature()
        if signature == self._signature:
            return
        self._signature = signature
        names: list[str] = []
        if signature is not None and self._state_path is not None:
            try:
                text = self._state_path.read_text(encoding="utf-8")
            except FileNotFoundError:
                text = ""
            for name in parse_active_state(text)[0]:
                if name in self._scenarios:
                    names.append(name)
                else:
                    logger.warning(f"Unknown scenario {name!r} in {self._state_path}; ignoring")
        self._rebuild(resolve_active_scenarios(names))

    def _targets(self, scenario: Mapping[str, Any]) -> set[str]:
        """The targets a scenario writes: addresses, engine variables and coupled channels."""
        targets: set[str] = set(scenario.get("overrides") or {})
        for fault in (scenario.get("faults") or {}).values():
            targets.update(fault.get("writes") or {})
        for entry in scenario.get("archiver") or []:
            targets.add(str(entry["channel"]))
        targets.update(scenario.get("couple") or {})
        targets.update(scenario.get("noise") or {})
        return targets

    def _rebuild(self, active: Sequence[str]) -> None:
        """Start every child at the composed state of the ``active`` scenarios."""
        view = {name: self._targets(scenario) for name, scenario in self._scenarios.items()}
        overlaps = validate_composition(view, list(active))
        if overlaps:
            for overlap in overlaps:
                logger.error(str(overlap))
                for name in self._children:
                    self._log(
                        name, overlap_record(overlap, instance=self._instance, pid=os.getpid())
                    )
            active = resolve_active_scenarios([])
        self._active = list(active)

        overrides: dict[str, Any] = {}
        faults: dict[str, dict[str, Any]] = {}
        drivers: dict[str, Mapping[str, Any]] = {}
        couple: dict[str, list[Mapping[str, Any]]] = {}
        noise: dict[str, Mapping[str, Any]] = {}
        for name in active:
            scenario = self._scenarios.get(name, {})
            overrides.update(scenario.get("overrides") or {})
            for model, fault in (scenario.get("faults") or {}).items():
                if not fault.get("inactive"):
                    faults.setdefault(str(model), {}).update(fault.get("writes") or {})
            drivers.update(scenario.get("drivers") or {})
            for address, terms in (scenario.get("couple") or {}).items():
                couple.setdefault(str(address), []).extend(terms)
            noise.update(scenario.get("noise") or {})

        resolved: dict[str, list[Mapping[str, Any]]] = {}
        for address, terms in couple.items():
            for term in terms:
                drive = drivers.get(str(term["driver"]))
                if drive is None:
                    logger.warning(f"{address} couples to undeclared driver {term['driver']!r}")
                    continue
                resolved.setdefault(address, []).append({**term, "drive": drive})
        self._texture.set_motion(resolved, noise)
        self._texture.set_active(
            {
                address: value
                for address, value in overrides.items()
                if self._owner.get(address) == TEXTURE_OWNER
                and address in self._texture.supported_variables
            }
        )

        for child in self._children.values():
            child.active, child.stuck = self._child_active(child, overrides, faults)
            child.last_good = {}
            self._build(child)

    def _child_active(
        self,
        child: _Child,
        overrides: Mapping[str, Any],
        faults: Mapping[str, Mapping[str, Any]],
    ) -> tuple[dict[str, Any], frozenset[str]]:
        """A child's active writes and fault seeds, and the setpoints it holds stuck."""
        active = {
            address: value for address, value in overrides.items() if address in child.setpoints
        }
        stuck: set[str] = set()
        for key, value in (faults.get(child.name) or {}).items():
            if isinstance(value, Mapping):
                for name, seed in value.items():
                    active[f"{key}{_FAULT_SEPARATOR}{name}"] = seed
            elif value == STUCK:
                stuck.add(str(key))
            else:
                active[str(key)] = value
        return dict(sorted(active.items())), frozenset(stuck)

    # -- the LUME contract ---------------------------------------------------

    @property
    def supported_variables(self) -> dict[str, Variable]:
        """Every channel address and status address of the view, by address."""
        return self._variables

    @property
    def models(self) -> list[str]:
        """The served physics models, sorted by name."""
        return list(self._children)

    @property
    def active(self) -> list[str]:
        """The scenarios served: the active set, or ``nominal`` alone when its scenarios overlap."""
        self._refresh()
        return list(self._active)

    def get(self, names: list[str] | str) -> dict[str, Any] | Any:
        """Read channels; a waveform reads as a flat list.

        Args:
            names: One address or a list of addresses.

        Returns:
            The value of a single address, else the values by address.

        Raises:
            ValueError: A name is not a channel or status address of the view.
        """
        single = isinstance(names, str)
        wanted = [names] if isinstance(names, str) else list(names)
        self._require(wanted)
        outputs = self._get(wanted)
        return outputs[wanted[0]] if single else outputs

    def _require(self, names: Sequence[str]) -> None:
        for name in names:
            if name not in self._variables:
                raise ValueError(f"Variable '{name}' is not supported by the model.")

    def _get(self, names: list[str]) -> dict[str, Any]:
        self._refresh()
        t_s = float(self._clock())
        outputs: dict[str, Any] = {}
        texture: list[str] = []
        physics: dict[str, list[str]] = {}
        for name in names:
            if name in self._status:
                outputs[name] = self._children[self._status[name]].status
            elif self._owner[name] == TEXTURE_OWNER:
                texture.append(name)
            else:
                physics.setdefault(self._owner[name], []).append(name)
        if texture:
            outputs.update(
                {name: _plain(value) for name, value in self._texture.get(texture).items()}
            )
        for model, owned in physics.items():
            outputs.update(self._read(self._children[model], owned, t_s))
        return {name: outputs[name] for name in names}

    def _is_moving_readback(self, address: str) -> bool:
        channel = self._channels[address]
        value_type = channel.get("value_type") or values.DEFAULT_VALUE_TYPE
        return channel.get("role") != _SETPOINT and value_type == "float"

    def _read(self, child: _Child, names: list[str], t_s: float) -> dict[str, Any]:
        """A physics child's channels at ``t_s``: readbacks with motion, readout and clamp."""
        readbacks = sorted(
            {
                partner
                for name in names
                if self._is_moving_readback(name)
                for partner in child.partners.get(name, (name,))
            }
        )
        truth = self._plain_get(child, sorted(set(names) | set(readbacks)))
        if truth is None:
            return self._failed_values(child, names)
        moved = {
            address: float(truth[address])
            + float(self._texture.motion(address, t_s, base=float(truth[address])))
            for address in readbacks
        }
        readout = getattr(child.engine, "readout", None)
        if moved and callable(readout):
            moved = readout(child.model, moved, int(round(t_s * _MS_PER_S)))
        for address in readbacks:
            truth[address] = self._texture.clamp(address, float(moved[address]))
        return {name: truth[name] for name in names}

    def _plain_get(self, child: _Child, names: list[str]) -> dict[str, Any] | None:
        """A built child's plain reads, or ``None`` once a read fails the child."""
        if child.model is None:
            return None
        try:
            read = {name: _plain(value) for name, value in child.model.get(names).items()}
        except Exception as exc:
            self._fail(child, exc)
            return None
        child.last_good.update(
            {name: value for name, value in read.items() if not isinstance(value, float)}
        )
        return read

    def _failed_values(self, child: _Child, names: Sequence[str]) -> dict[str, Any]:
        """What a failed child's channels read: inputs, NaN, or the last good value."""
        outputs: dict[str, Any] = {}
        for name in names:
            channel = self._channels[name]
            value_type = channel.get("value_type") or values.DEFAULT_VALUE_TYPE
            if name in child.inputs:
                outputs[name] = child.inputs[name]
            elif value_type == "float":
                outputs[name] = math.nan
            elif name in child.last_good:
                outputs[name] = child.last_good[name]
            else:
                outputs[name] = values.zero(
                    value_type, channel.get("options"), channel.get("shape")
                )
        return outputs

    def readout_group(self, address: str) -> tuple[str, ...]:
        """The channels a read of ``address`` reads with it, ``address`` among them.

        A monitor plane of a physics child is read with its partner planes;
        any other channel alone.

        Raises:
            ValueError: ``address`` is not a channel of the view.
        """
        self._require([address])
        owner = self._owner[address]
        if owner == TEXTURE_OWNER or not self._is_moving_readback(address):
            return (address,)
        return self._children[owner].partners.get(address, (address,))

    def readings(self, levels: Mapping[str, Any], t_s: Any) -> dict[str, np.ndarray]:
        """Float channels as served at epoch seconds, from levels in place of held values.

        A moving channel, a texture float or a physics float readback, reads
        its level plus its motion on that level, through its child's
        ``readout`` when the engine exports one, then clamped; a physics
        setpoint reads its level. A failed child's readbacks read NaN.

        Args:
            levels: Float channel levels by address, each a scalar or an array
                shaped like ``t_s``; a monitor plane needs every channel of its
                :meth:`readout_group`.
            t_s: Epoch seconds, one-dimensional.

        Returns:
            One float64 array shaped like ``t_s`` per address of ``levels``.

        Raises:
            ValueError: A name is not a float channel of the view, or a
                monitor plane's partner has no level.
        """
        names = sorted(levels)
        self._require(names)
        for name in names:
            channel = self._channels.get(name)
            if (
                channel is None
                or (channel.get("value_type") or values.DEFAULT_VALUE_TYPE) != "float"
            ):
                raise ValueError(f"{name} is not a float channel of the view")
        self._refresh()
        times = np.asarray(t_s, dtype=np.float64).reshape(-1)
        out: dict[str, np.ndarray] = {}
        readbacks: dict[str, list[str]] = {}
        unread: set[str] = set()
        for name in names:
            level = np.broadcast_to(np.asarray(levels[name], dtype=np.float64), times.shape)
            owner = self._owner[name]
            if owner != TEXTURE_OWNER and not self._is_moving_readback(name):
                out[name] = np.array(level, dtype=np.float64)
                unread.add(name)
                continue
            out[name] = level + self._texture.motion(name, times, base=level)
            if owner != TEXTURE_OWNER:
                readbacks.setdefault(owner, []).append(name)
        for model, owned in readbacks.items():
            child = self._children[model]
            if child.model is None:
                for name in owned:
                    out[name] = np.full(times.shape, math.nan)
                unread.update(owned)
                continue
            readout = getattr(child.engine, "readout", None)
            if not callable(readout):
                continue
            for index, instant in enumerate(times):
                read = readout(
                    child.model,
                    {name: float(out[name][index]) for name in owned},
                    int(round(float(instant) * _MS_PER_S)),
                )
                for name in owned:
                    out[name][index] = float(read[name])
        for name in names:
            if name not in unread:
                out[name] = np.array(
                    [self._texture.clamp(name, float(value)) for value in out[name]],
                    dtype=np.float64,
                )
        return out

    def held(self, names: Sequence[str]) -> dict[str, Any]:
        """Each channel's held value: no motion, no readout, no clamp.

        A texture channel returns the value the texture holds; a physics
        channel its child's plain read; a status address its status.

        Args:
            names: Channel or status addresses.

        Returns:
            The value by address, in its stored representation.

        Raises:
            ValueError: A name is not a channel or status address of the view.
        """
        wanted = list(names)
        self._require(wanted)
        self._refresh()
        outputs: dict[str, Any] = {}
        physics: dict[str, list[str]] = {}
        texture: list[str] = []
        for name in wanted:
            if name in self._status:
                outputs[name] = self._children[self._status[name]].status
            elif self._owner[name] == TEXTURE_OWNER:
                texture.append(name)
            else:
                physics.setdefault(self._owner[name], []).append(name)
        outputs.update(self._texture.held(texture))
        for model, owned in physics.items():
            child = self._children[model]
            read = self._plain_get(child, owned)
            outputs.update(read if read is not None else self._failed_values(child, owned))
        return {name: outputs[name] for name in wanted}

    def output_severity(self, names: Sequence[str]) -> dict[str, dict[str, str]]:
        """The condition of each named channel whose child failed.

        Args:
            names: Channel or status addresses.

        Returns:
            ``{name: {"condition": "udf"}}`` for each name a failed child
            owns; a name of a serving child, of the texture or of a status
            address is absent.
        """
        self._refresh()
        return {
            name: {"condition": UDF}
            for name in names
            if (child := self._children.get(self._owner.get(name, TEXTURE_OWNER))) is not None
            and child.model is None
        }

    def status(self, model: str) -> str:
        """A served physics model's status: ``ok`` or its engine's capped error text.

        Raises:
            ValueError: ``model`` is not a served physics model; the message
                names the served ones.
        """
        if model not in self._children:
            raise ValueError(f"model {model!r} is not served; served: {self.models}")
        self._refresh()
        return self._children[model].status

    def set(self, values_by_name: dict[str, Any]) -> None:
        """Write channels: physics children first, in name order, the texture last.

        Args:
            values_by_name: Values by channel address; a waveform as a flat or
                nested list.

        Raises:
            ValueError: A name is not a channel of the view, a value is refused
                for its channel, or a child refuses the write; every earlier
                child has its previous inputs back.
            ReadOnlyError: A name is not a settable channel.
        """
        self._refresh()
        if not values_by_name:
            return
        coerced: dict[str, Any] = {}
        for name, value in values_by_name.items():
            variable = self._variables.get(name)
            if variable is None:
                raise ValueError(f"Variable '{name}' is not supported by the model.")
            if variable.read_only:
                raise ReadOnlyError(f"Variable '{name}' is read-only. Cannot be set.")
            channel = self._channels[name]
            coerced[name] = values.coerce(
                value.tolist() if isinstance(value, np.ndarray) else value,
                channel.get("value_type"),
                channel.get("options"),
                channel.get("shape"),
            )
            variable.validate_value(_for_variable(variable, coerced[name]))
        self._set(coerced)

    def _set(self, values_by_name: dict[str, Any]) -> None:
        batches: dict[str, dict[str, Any]] = {}
        for name, value in values_by_name.items():
            batches.setdefault(self._owner[name], {})[name] = value
        restores: list[Callable[[], None]] = []
        try:
            for model, child in self._children.items():
                if model in batches:
                    restores.append(self._set_child(child, batches[model]))
            texture = batches.get(TEXTURE_OWNER)
            if texture:
                self._texture.set(
                    {
                        name: _for_variable(self._texture.supported_variables[name], value)
                        for name, value in texture.items()
                    }
                )
        except Exception:
            for restore in reversed(restores):
                restore()
            raise

    def _set_child(self, child: _Child, batch: Mapping[str, Any]) -> Callable[[], None]:
        """Write one child's batch; returns what puts its previous inputs back."""
        forward = {name: value for name, value in batch.items() if name not in child.stuck}
        previous = {name: child.inputs[name] for name in forward if name in child.inputs}
        model = child.model
        if model is not None and forward:
            model.set(
                {
                    name: _for_variable(model.supported_variables[name], value)
                    for name, value in forward.items()
                }
            )
        child.inputs.update(forward)

        def restore() -> None:
            child.inputs.update(previous)
            if child.model is not None and previous:
                try:
                    child.model.set(
                        {
                            name: _for_variable(child.model.supported_variables[name], value)
                            for name, value in previous.items()
                        }
                    )
                except Exception as exc:
                    self._fail(child, exc)

        return restore

    def reset(self) -> None:
        """Return every child to its start state; a failed child is rebuilt.

        Setpoints return to their active writes or defaults, the texture to its
        nominals and active writes; session writes are dropped.
        """
        self._refresh()
        for child in self._children.values():
            if child.model is None:
                self._build(child)
                continue
            try:
                child.model.reset()
            except Exception as exc:
                self._fail(child, exc)
                continue
            child.inputs = {
                address: self._start_value(child, address) for address in child.setpoints
            }
        self._texture.reset()

    # -- the children's own variables ----------------------------------------

    def _model_variable(self, name: str) -> tuple[_Child, str]:
        model, separator, variable = name.partition(_MODEL_SEPARATOR)
        child = self._children.get(model)
        if not separator or child is None:
            raise ValueError(
                f"{name!r} names no model variable; use <model>/<name>, model in {self.models}"
            )
        if child.model is None:
            raise ValueError(f"model {model!r} has failed: {child.status}")
        if variable in self._channels or variable not in child.model.supported_variables:
            raise ValueError(f"model {model!r} has no variable {variable!r}")
        return child, variable

    def model_get(self, names: Sequence[str]) -> dict[str, Any]:
        """Read the children's own variables, each named ``<model>/<name>``.

        Raises:
            ValueError: A name names no served model, a channel, an unknown
                variable, or a variable of a failed model.
        """
        self._refresh()
        outputs: dict[str, Any] = {}
        for name in names:
            child, variable = self._model_variable(name)
            assert child.model is not None
            outputs[name] = child.model.get(variable)
        return outputs

    def model_set(self, values_by_name: Mapping[str, Any]) -> None:
        """Write the children's own variables, each named ``<model>/<name>``.

        Raises:
            ValueError: A name names no served model, a channel, an unknown
                variable, or a variable of a failed model; or the child
                refuses the write.
        """
        self._refresh()
        batches: dict[str, tuple[_Child, dict[str, Any]]] = {}
        for name, value in values_by_name.items():
            child, variable = self._model_variable(name)
            batches.setdefault(child.name, (child, {}))[1][variable] = value
        for child, batch in batches.values():
            assert child.model is not None
            child.model.set(batch)

    # -- the model log -------------------------------------------------------

    def _log(self, model: str, record: Mapping[str, Any]) -> None:
        """Append one record to ``model``'s log, and hand it to the process logger."""
        entry = {"instance": self._instance, "pid": os.getpid(), "model": model, **record}
        line = self._log_line(entry)
        if record.get("event") == "failed":
            logger.warning(line)
        else:
            logger.info(line)
        if self._log_dir is None:
            return
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            fd = os.open(
                self._log_dir / f"{model}.log", os.O_WRONLY | os.O_APPEND | os.O_CREAT, _LOG_MODE
            )
            try:
                if os.fstat(fd).st_uid == os.geteuid():
                    os.fchmod(fd, _LOG_MODE)
                os.write(fd, (line + "\n").encode("utf-8"))
            finally:
                os.close(fd)
        except OSError as exc:
            logger.warning(f"cannot append to the {model} log in {self._log_dir}: {exc}")

    @staticmethod
    def _log_line(entry: Mapping[str, Any]) -> str:
        """One JSON line, its ``error`` cut so the line and its newline stay under the bound."""
        line = json.dumps(entry, sort_keys=True, ensure_ascii=False)
        error = entry.get("error")
        while (
            isinstance(error, str)
            and error
            and len(line.encode("utf-8")) >= (LOG_RECORD_MAX_BYTES - 1)
        ):
            excess = len(line.encode("utf-8")) - (LOG_RECORD_MAX_BYTES - 2)
            error = cap_status(error, max(len(error.encode("utf-8")) - excess, 0))
            line = json.dumps({**entry, "error": error}, sort_keys=True, ensure_ascii=False)
        return line
