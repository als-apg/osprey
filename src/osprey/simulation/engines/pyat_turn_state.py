"""The turn state of a turn-by-turn ring, and the model that holds it (contract C8).

One :class:`TurnState` per turn-by-turn ring model: per monitor element an
armed flag and its three turn arrays (``x``, ``y``, ``sum``, each
``trigger_sample + turns`` samples), the turns of the last tracked shot, the
inputs last delivered and the ones before them, and the "arrays emitted, not
yet drained" flag. It is model state, not settings: ``set``, restore and
replay never reach it, and a ``reset`` returns it to its start values.

The variables of the ``armed``, ``turns_tracked`` and turn-array records
read it live (:class:`~osprey.simulation.engines.pyat_variables.TurnStateVariable`,
:class:`~osprey.simulation.engines.pyat_variables.TurnArrayVariable`); the
shot handling of the engine's time-contract functions writes it.
:class:`TurnByTurnModel` is the model a turn-by-turn ring is built as: it
creates the state from the variables its wiring declares, binds them to it
and serves them from it; :func:`turn_state_of` finds the state of any model.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

import numpy as np

from osprey.simulation.engines.pyat_model import PyATLatticeModel
from osprey.simulation.engines.pyat_variables import (
    TurnArrayVariable,
    TurnSetpointVariable,
    TurnStateVariable,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Mapping

    import at
    from lume.variables import Variable

    from osprey.simulation.engines.pyat_turn_settings import TurnSettings

#: The signal of a monitor's arm setting.
_ARM = "arm"


def _zeros(samples: int) -> np.ndarray:
    return np.zeros(samples, dtype=np.float64)


@dataclass
class MonitorTurns:
    """One monitor element's part of the turn state.

    Attributes:
        armed: Whether the monitor is armed for the next shot.
        x: Horizontal position per sample.
        y: Vertical position per sample.
        sum: Sum signal per sample.
    """

    armed: bool
    x: np.ndarray
    y: np.ndarray
    sum: np.ndarray

    @classmethod
    def at_start(cls, samples: int) -> MonitorTurns:
        """Unarmed, every array zeros of ``samples``."""
        return cls(armed=False, x=_zeros(samples), y=_zeros(samples), sum=_zeros(samples))

    def array(self, signal: str) -> np.ndarray:
        """The array a turn signal (``turn_x``, ``turn_y``, ``turn_sum``) names."""
        return {"turn_x": self.x, "turn_y": self.y, "turn_sum": self.sum}[signal]


@dataclass
class TurnState:
    """The turn state of one turn-by-turn ring model.

    Attributes:
        settings: The ring's turn-by-turn settings block.
        monitors: Each wired monitor element's armed flag and arrays, keyed by
            the element's name.
        turns_tracked: The turns of the last tracked shot; ``0`` at start.
        inputs: The inputs last delivered, input signal -> value; ``None``
            before the first delivery.
        previous_inputs: The delivery before that; ``None`` until there is
            one.
        emitted_pending: The arrays of a tracked shot were emitted and not
            yet drained.
        drop_pending: Those arrays were drained; the next advance drops the
            armed flags of the monitors that read the shot.
        delivery_pending: A delivery arrived that no advance has looked at
            for a shot yet.
        baseline: The last delivery was a starting point: nothing is
            compared against what was held before it.
        spent: The monitors whose arrays hold a shot whose armed flag has
            not dropped yet; a new shot does not reach them.
        shot_time: The epoch second of the advance that tracked the last
            shot; ``None`` before one.
        now: The latest epoch second an advance saw; ``None`` before one.
        emitted: The event channels changed since the last drain, in
            first-change order.
    """

    settings: TurnSettings
    monitors: dict[str, MonitorTurns] = field(default_factory=dict)
    turns_tracked: int = 0
    inputs: Mapping[str, float] | None = None
    previous_inputs: Mapping[str, float] | None = None
    emitted_pending: bool = False
    drop_pending: bool = False
    delivery_pending: bool = False
    baseline: bool = False
    spent: set[str] = field(default_factory=set)
    shot_time: float | None = None
    now: float | None = None
    emitted: dict[str, None] = field(default_factory=dict)

    @classmethod
    def at_start(cls, settings: TurnSettings, monitors: Iterable[str]) -> TurnState:
        """The start state of a ring whose wiring names ``monitors``."""
        samples = settings.samples
        return cls(
            settings=settings,
            monitors={name: MonitorTurns.at_start(samples) for name in sorted(set(monitors))},
        )

    def reset(self) -> None:
        """Return every field but the settings to its start value, in place.

        In place, so every variable bound to the state keeps reading it.
        """
        start = TurnState.at_start(self.settings, self.monitors)
        for state_field in fields(self):
            setattr(self, state_field.name, getattr(start, state_field.name))

    def take(self, inputs: Mapping[str, float], *, baseline: bool) -> None:
        """Store one delivery, input signal -> value; the next advance looks at it.

        The delivery before becomes :attr:`previous_inputs`. A baseline, or
        the first delivery, is a starting point.
        """
        self.previous_inputs = self.inputs
        self.inputs = dict(inputs)
        self.baseline = baseline or self.previous_inputs is None
        self.delivery_pending = True

    def shot(self) -> int | None:
        """The shot value the pending delivery carries, or ``None`` for no shot.

        A shot is a rise of the ``shot`` input against the previous delivery,
        outside a baseline (decision CF-10). The delivery is consumed: a
        second call finds none.
        """
        if not self.delivery_pending:
            return None
        self.delivery_pending = False
        if self.baseline or self.inputs is None or self.previous_inputs is None:
            return None
        now, before = self.inputs.get("shot"), self.previous_inputs.get("shot")
        if now is None or before is None or not float(now) > float(before):
            return None
        return int(now)

    def drop(self, armed_names: Mapping[str, Iterable[str]]) -> None:
        """Apply a pending drop: unarm every spent monitor, emitting each one that changes."""
        if not self.drop_pending:
            return
        self.drop_pending = False
        for name in sorted(self.spent):
            monitor = self.monitors[name]
            if monitor.armed:
                monitor.armed = False
                self.emit(armed_names.get(name, ()))
        self.spent = set()

    def emit(self, names: Iterable[str]) -> None:
        """Mark ``names`` changed, keeping the order they first changed in."""
        for name in names:
            self.emitted[name] = None

    def drain(self) -> list[str]:
        """The names changed since the last drain, cleared; a drained shot's drop is now pending."""
        names = list(self.emitted)
        self.emitted = {}
        if self.emitted_pending:
            self.emitted_pending = False
            self.drop_pending = True
        return names

    def deadline(self) -> float | None:
        """The shot's time while its arrays or its drop are pending, else ``None``."""
        if self.emitted_pending or self.drop_pending:
            return self.shot_time
        return None


def _monitor_of(variable: Any) -> str | None:
    """The monitor element a turn variable reads or arms, or ``None``."""
    if isinstance(variable, (TurnStateVariable, TurnArrayVariable)):
        return variable.monitor
    if isinstance(variable, TurnSetpointVariable) and variable.signal == _ARM:
        return variable.monitor
    return None


class TurnByTurnModel(PyATLatticeModel):
    """A turn-by-turn ring: a tolerant pyat model that holds a :class:`TurnState`.

    It is a :class:`~osprey.simulation.engines.pyat_model.PyATLatticeModel`
    in every respect, plus one turn state built from the turn records its
    channels declare -- one :class:`MonitorTurns` per monitor element an
    ``arm``, ``armed`` or turn-array record names. Every turn-state variable
    is bound to that state and read from it on every read, so a change the
    shot handling makes shows at once, with no solve. :meth:`reset` returns
    the state to its start values beside the settings.

    Attributes:
        turn_state: The ring's turn state.
    """

    def __init__(
        self,
        lattice: at.Lattice,
        channels: Iterable[Variable],
        *,
        turn_settings: TurnSettings,
        model_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Build the turn state, bind the turn variables to it, and adopt every channel.

        Args:
            lattice: As for ``PyATLatticeModel``.
            channels: As for ``PyATLatticeModel``, turn variables among them.
            turn_settings: The ring's turn-by-turn settings block.
            model_id: The model's id, which keys each shot's jitter.
            **kwargs: ``PyATLatticeModel``'s keywords; ``tolerant`` is
                always on.

        Raises:
            UnknownElementError: a turn record names a monitor the lattice
                does not hold exactly once.
            ValueError: a turn record's element is not an ``at.Monitor``;
                otherwise as ``PyATLatticeModel``.
        """
        channels = list(channels)
        monitors = [name for name in map(_monitor_of, channels) if name is not None]
        self.turn_state = TurnState.at_start(turn_settings, monitors)
        self._turn_names: frozenset[str] = frozenset(
            variable.name
            for variable in channels
            if isinstance(variable, (TurnStateVariable, TurnArrayVariable))
        )
        self.model_id = model_id
        self._arm_names: dict[str, str] = {}
        self._armed_names: dict[str, list[str]] = {}
        self._array_names: dict[str, dict[str, str]] = {}
        self._turns_names: list[str] = []
        for variable in channels:
            if isinstance(variable, (TurnStateVariable, TurnArrayVariable)):
                variable.bind(self.turn_state)
            if isinstance(variable, TurnSetpointVariable) and variable.signal == _ARM:
                self._arm_names[variable.name] = str(variable.monitor)
            elif isinstance(variable, TurnStateVariable) and variable.signal == "armed":
                self._armed_names.setdefault(str(variable.monitor), []).append(variable.name)
            elif isinstance(variable, TurnStateVariable):
                self._turns_names.append(variable.name)
            elif isinstance(variable, TurnArrayVariable):
                self._array_names.setdefault(variable.monitor, {})[variable.signal] = variable.name
        kwargs["tolerant"] = True
        super().__init__(lattice, channels, **kwargs)

    def reset(self) -> None:
        """Return every setting to its default and the turn state to its start."""
        super().reset()
        self.turn_state.reset()

    # -- the time contract (decisions Arm, CF-6, CF-10, M4, RF) --------------------

    def take_inputs(self, inputs: Mapping[str, float], *, baseline: bool) -> None:
        """Store one delivery, input signal -> value; only the next advance acts on it."""
        self.turn_state.take(inputs, baseline=baseline)

    def advance(self, t_s: float) -> None:
        """Apply a pending drop, then track the pending delivery's shot, if any.

        The drop of a drained shot comes first, so a shot in the same advance
        finds those monitors unarmed. A shot reaches only the monitors armed
        and not still holding an undrained shot; with none, it is consumed.
        One advance sees one delivery and tracks at most once.
        """
        state = self.turn_state
        if state.now is None or t_s > state.now:
            state.now = t_s
        state.drop(self._armed_names)
        shot = state.shot()
        if shot is None:
            return
        monitors = [
            name
            for name, turns in state.monitors.items()
            if turns.armed and name not in state.spent
        ]
        if not monitors:
            return
        self._take_shot(monitors, shot)

    def client_write(self, values: Mapping[str, Any], t_s: float) -> None:
        """Arm or disarm each monitor whose ``arm`` record ``values`` holds.

        ``set`` has already stored the values; only this call arms. A monitor
        whose armed flag changes emits its ``armed`` records.
        """
        self.advance(t_s)
        state = self.turn_state
        for name, value in values.items():
            monitor = self._arm_names.get(name)
            if monitor is None:
                continue
            armed = float(value) != 0.0
            turns = state.monitors[monitor]
            if turns.armed != armed:
                turns.armed = armed
                state.emit(self._armed_names.get(monitor, ()))

    def next_deadline(self) -> float | None:
        """The shot's time while its arrays await a drain or its drop is pending, else ``None``."""
        return self.turn_state.deadline()

    def emitted(self) -> list[str]:
        """The event channels changed since the last call, in first-change order, cleared."""
        return self.turn_state.drain()

    def _take_shot(self, monitors: list[str], shot: int) -> None:
        """Track ``shot``, write the monitors' arrays and the turns tracked, and emit them.

        A tracking or readout that raises leaves the monitors' arrays at their
        start values and the turns tracked at 0; the handshake completes either way.
        """
        state = self.turn_state
        settings = state.settings
        try:
            arrays = self._track(monitors, shot)
            turns = settings.turns
        except Exception:  # CF-10: a physics outcome never hangs the handshake
            samples = settings.samples
            arrays = {
                name: (_zeros(samples), _zeros(samples), _zeros(samples)) for name in monitors
            }
            turns = 0
        for name in monitors:
            turns_of = state.monitors[name]
            turns_of.x, turns_of.y, turns_of.sum = arrays[name]
            wired = self._array_names.get(name, {})
            state.emit(
                wired[signal] for signal in ("turn_x", "turn_y", "turn_sum") if signal in wired
            )
        state.turns_tracked = turns
        state.emit(self._turns_names)
        state.spent.update(monitors)
        state.emitted_pending = True
        state.shot_time = state.now

    def _track(
        self, monitors: list[str], shot: int
    ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Each monitor's ``(x, y, sum)`` arrays for one tracked shot."""
        from osprey.simulation.engines import pyat_turns
        from osprey.simulation.engines.pyat_turn_readout import monitor_turn_arrays
        from osprey.simulation.engines.pyat_variables import pilot_tone_on, rf_on

        state = self.turn_state
        settings = state.settings
        inputs = state.inputs or {}
        previous = state.previous_inputs or {}
        lattice = self.lattice.deepcopy()
        if settings.aperture is not None:
            pyat_turns.write_apertures(lattice, **settings.aperture.keywords)
        if any(
            (settings.emittance_x, settings.emittance_y, settings.sigma_delta, settings.sigma_z)
        ):
            shape = pyat_turns.declared_bunch(
                lattice,
                settings.injection_index,
                settings.particles,
                emittance_x=settings.emittance_x,
                emittance_y=settings.emittance_y,
                sigma_delta=settings.sigma_delta,
                sigma_z=settings.sigma_z,
            )
        else:
            shape = np.zeros((6, settings.particles), dtype=np.float64)
        reading = pyat_turns.track_shot(
            lattice,
            settings.injection_index,
            [self.simulator.unique_element_index(name) for name in monitors],
            settings.launch,
            settings.turns,
            shape=shape,
            key=self.model_id,
            shot=shot,
            jitter=settings.jitter,
        )
        charge = float(inputs.get("charge", 0.0))
        stored_current = float(previous.get("stored_current", 0.0))
        beam_on, tone_on = rf_on(self), pilot_tone_on(self)
        arrays = {}
        for row, name in enumerate(monitors):
            wired = self._array_names.get(name, {})
            x_address, y_address, sum_address = (
                wired.get(signal, f"{self.model_id}/{name}/{signal}")
                for signal in ("turn_x", "turn_y", "turn_sum")
            )
            addresses = (x_address, y_address, sum_address)
            arrays[name] = monitor_turn_arrays(
                alive=reading.alive[row],
                centroid_x=reading.x[row],
                centroid_y=reading.y[row],
                trigger_sample=settings.trigger_sample,
                charge=charge,
                stored_current=stored_current,
                rf_on=beam_on,
                tone_on=tone_on,
                addresses=addresses,
                shot=shot,
                readout=settings.readout,
            )
        return arrays

    def _validate_binding(self, name: str, variable: Variable) -> None:
        """Check a turn variable's monitor; defer everything else."""
        monitor = _monitor_of(variable)
        if monitor is not None:
            import at as pyat

            index = self.simulator.unique_element_index(monitor)
            if not isinstance(self.lattice[index], pyat.Monitor):
                raise ValueError(f"variable {name!r} names {monitor!r}, which is not a monitor")
        if isinstance(variable, TurnStateVariable):
            return
        super()._validate_binding(name, variable)

    def _get(self, names: list[str]) -> dict[str, Any]:
        """Serve turn-state variables from the state, live; the rest as the base does."""
        turn = [name for name in names if name in self._turn_names]
        if not turn:
            return super()._get(names)
        rest = super()._get([name for name in names if name not in self._turn_names])
        variables = self.supported_variables
        return {
            name: variables[name]._get(self.simulator) if name in self._turn_names else rest[name]
            for name in names
        }


def turn_state_of(model: Any) -> TurnState | None:
    """A model's turn state, or ``None`` for a model that is not a turn-by-turn ring."""
    return model.turn_state if isinstance(model, TurnByTurnModel) else None


__all__ = [
    "MonitorTurns",
    "TurnByTurnModel",
    "TurnState",
    "turn_state_of",
]
