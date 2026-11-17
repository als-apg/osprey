"""The texture model: the simulator's value for every channel no served model wires.

:class:`TextureModel` is a ``LUMEModel`` built from two documents of the
simulator view, ``variables.json`` and ``seeds.json``. It owns the VALUE of a
channel whose ``owner`` is ``texture`` or names a model the view marks
unserved, and the MOTION (keyed noise and drift) and clamp of every channel
the view declares, so a served model's readbacks can carry the same motion.

A texture channel reads ``clamp(held + drift(t) + couplings(t) + noise(t))``.
``held`` starts at the channel's nominal: the seed's ``nominal``, else the
unserved model's wiring ``default``, else the zero of the channel's
``value_type``. A paired readback starts at its setpoint's nominal. A
``linear`` channel holds the weighted sum of its inputs' held values, a
``linear`` input counting as its own sum. Noise and drift are pure functions
of the address and the epoch millisecond, so a live read and an archived
sample at the same instant agree. Writing a setpoint holds the
value and echoes it into the readback its ``pair`` names.

A ``noise`` record states one term: ``{absolute: sigma}`` adds ``sigma``
times a keyed normal draw, and ``{relative: f}`` adds ``f`` times
``held + drift + couplings`` times one. The active scenarios add motion
through :meth:`TextureModel.set_motion`: a coupled channel adds
``gain * (1 + gain_wander(t)) * driver(t)`` per coupling, where every channel
coupled to one driver sees the same ``driver(t)``, and a scenario's noise
record stands in for the seed's while it is held, drawing from its own
stream. A reading the active scenarios still serves its held value: no drift,
couplings or noise.

A waveform channel whose seed states ``pulse: {driver, enable}`` reads
:func:`~osprey_connectors.simulation.pulse.trace` at the read time, scaled by
the driver's held value and switched by the enable channel's held label, so
it moves on every read. Every other waveform channel reads its held array.

A float channel whose seed states ``topoff: {ceiling, lifetime_h, period_s}``
holds the level ``ceiling * m`` of :mod:`~osprey_connectors.simulation.topoff`
and carries the tooth ``level * (s(t) / m - 1)`` as motion on that level, so
it reads ``ceiling * s(t)`` at the held level and a stilled one serves the
level. A channel whose seed states ``topoff_lifetime: {of}`` holds the named
``topoff`` seed's ``lifetime_h``.

A channel whose seed states ``gate: {inputs, gated}`` holds its gated value
while any input holds its blocking label, and its value by the rules above
otherwise. ``gated`` is typed by the channel: a float holds its value times
the factor ``gated``, so a gated reading keeps a small residual and its
motion; a bool or enum holds the label ``gated``; an int holds the integer
``gated``. An input is read as its own held value, gate included, and an
input the model does not hold never blocks. The gate is part of the held
value, so a read, :meth:`TextureModel.held` and every reader built on it --
the archive seed, a subscription comparing held values -- see it alike.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
import numpy.typing as npt
from lume.model import LUMEModel
from lume.variables import (
    EnumVariable,
    IntVariable,
    NDVariable,
    ScalarVariable,
    StrVariable,
    Variable,
)
from typing_extensions import override

from osprey_connectors.simulation import TEXTURE, pulse, series, topoff, values
from osprey_connectors.simulation.envelope import declares_motion
from osprey_connectors.simulation.records import LinearTerm
from osprey_connectors.simulation.view import SETPOINT

if TYPE_CHECKING:  # pragma: no cover - typing only
    from osprey_connectors.simulation.records import (
        ChannelRecord,
        GateRecord,
        ModelRecord,
        NoiseRecord,
        PulseRecord,
        ResolvedCoupling,
        SeedRecord,
        SeedsDocument,
        VariablesDocument,
    )

__all__ = [
    "TextureModel",
    "channel_variable",
]

_MS_PER_S = 1000.0
_GAIN_WANDER_SUBKEY = b":gain_wander:"
_ABSOLUTE: Final = "absolute"
_RELATIVE: Final = "relative"
_SCENARIO_RELATIVE_SUBKEY = b":noise"
_SCENARIO_ABSOLUTE_SUBKEY = b":noise_abs"
_INT_UNBOUNDED = 2**63 - 1


def _int_range(channel: ChannelRecord) -> tuple[int, int] | None:
    """An ``int`` variable's range from the view's ``value_range``."""
    bounds = channel.value_range
    return None if bounds is None else (int(bounds[0]), int(bounds[1]))


def _float_range(channel: ChannelRecord) -> tuple[float, float] | None:
    """A scalar variable's range from the view's ``value_range``."""
    bounds = channel.value_range
    return None if bounds is None else (float(bounds[0]), float(bounds[1]))


def _linear_terms(linear: Mapping[str, float | LinearTerm]) -> list[tuple[str, float]]:
    """``(input address, coefficient)`` pairs of a seed's ``linear`` map, sorted."""
    terms: list[tuple[str, float]] = []
    for address in sorted(linear):
        term = linear[address]
        coefficient = term.coefficient if isinstance(term, LinearTerm) else term
        terms.append((address, float(coefficient)))
    return terms


def _gate_inputs(inputs: Mapping[str, object] | None) -> list[tuple[str, object]]:
    """``(input address, blocking label)`` pairs of a seed's ``gate.inputs``, sorted.

    ``inputs`` is ``{<address>: <blocking_label>}``; a value may also be the
    record ``{blocking_label: <label>}``.
    """
    pairs: list[tuple[str, object]] = []
    stated = inputs or {}
    for address in sorted(stated, key=str):
        label = stated[address]
        if isinstance(label, Mapping):
            label = label["blocking_label"]
        pairs.append((str(address), label))
    return pairs


def channel_variable(channel: ChannelRecord, nominal: values.StoredValue) -> Variable:
    """The LUME variable of one channel of the view's ``variables.json``.

    A setpoint the view marks writable is settable; every other channel is
    read-only. ``value_type`` picks the class: an ``EnumVariable`` over the
    channel's labels for ``bool`` and ``enum``, a ``StrVariable`` for
    ``string``, an ``NDVariable`` of the channel's ``shape`` for
    ``waveform``, an ``IntVariable`` for ``int`` and a ``ScalarVariable``
    otherwise, each bounded by the view's ``value_range``.

    Args:
        channel: The channel's record in ``variables.json``.
        nominal: Its start value, in its stored representation.

    Returns:
        The variable, named by the channel's address.
    """
    address = channel.address
    read_only = not (channel.role == SETPOINT and channel.writable)
    value_type = channel.value_type
    unit = channel.unit
    if value_type in values.LABELLED_TYPES:
        return EnumVariable(
            name=address, options=list(channel.labels), default_value=nominal, read_only=read_only
        )
    # The start value is in the stored representation of the channel's value
    # type, which the type checker does not see; ``model_validate`` validates
    # it exactly as the constructor would.
    if value_type == "string":
        return StrVariable.model_validate(
            {"name": address, "default_value": nominal, "read_only": read_only}
        )
    if value_type == "waveform":
        shape = channel.shape or ()
        return NDVariable(
            name=address,
            shape=shape,
            default_value=np.asarray(nominal, dtype=np.float64).reshape(shape),
            unit=unit,
            read_only=read_only,
        )
    if value_type == "int":
        return IntVariable.model_validate(
            {
                "name": address,
                "default_value": nominal,
                "value_range": _int_range(channel),
                "unit": unit,
                "read_only": read_only,
            }
        )
    return ScalarVariable.model_validate(
        {
            "name": address,
            "default_value": nominal,
            "value_range": _float_range(channel),
            "unit": unit,
            "read_only": read_only,
        }
    )


def _noise_term(record: NoiseRecord | None) -> tuple[Literal["absolute", "relative"], float]:
    """The term a noise record states and its value; no record is an absolute 0.0."""
    if record is not None and _RELATIVE in record.model_fields_set:
        return _RELATIVE, float(record.relative or 0.0)
    return _ABSOLUTE, float(0.0 if record is None else record.absolute or 0.0)


class TextureModel(LUMEModel):
    """Serve the texture-owned channels of a simulator view.

    Args:
        variables: The view's ``variables.json`` document (``models[]`` and
            ``channels[]``).
        seeds: The view's ``seeds.json`` document (``seeds`` by address).
        clock: Returns the current instant in epoch seconds.
    """

    def __init__(
        self,
        variables: VariablesDocument,
        seeds: SeedsDocument,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._clock = clock
        self._seeds: dict[str, SeedRecord] = dict(seeds.seeds)
        self._channels: dict[str, ChannelRecord] = {
            channel.address: channel for channel in variables.channels
        }
        models = variables.models
        unserved_defaults = self._unserved_wiring_defaults(models)
        unserved = {model.name for model in models if not model.served}

        owned = sorted(
            address
            for address, channel in self._channels.items()
            if channel.owner in (TEXTURE, *unserved)
        )
        self._echo: dict[str, str] = {
            address: pair
            for address in owned
            if self._channels[address].role == SETPOINT
            and (pair := self._channels[address].pair) not in (None, address)
            and pair in owned
        }

        self._nominals: dict[str, values.StoredValue] = {
            address: self._nominal(address, unserved_defaults.get(address)) for address in owned
        }
        for setpoint, readback in self._echo.items():
            self._nominals[readback] = self._coerce(readback, self._nominals[setpoint])
        start = {address: self._derived(address, self._nominals) for address in owned}
        self._variables: dict[str, Variable] = {
            address: channel_variable(self._channels[address], start[address]) for address in owned
        }
        self._held: dict[str, values.StoredValue] = {}
        self._couple: dict[str, list[ResolvedCoupling]] = {}
        self._noise: dict[str, NoiseRecord] = {}
        self._still: frozenset[str] = frozenset()
        self._active: dict[str, object] = {}
        self.reset()

    @staticmethod
    def _unserved_wiring_defaults(models: Sequence[ModelRecord]) -> dict[str, object]:
        defaults: dict[str, object] = {}
        for model in models:
            if model.served:
                continue
            for record in model.wiring:
                if record.default is not None:
                    defaults.setdefault(record.address, record.default)
        return defaults

    def _coerce(self, address: str, value: object) -> values.StoredValue:
        return self._channels[address].coerce(value)

    def _nominal(self, address: str, wiring_default: object) -> values.StoredValue:
        channel = self._channels[address]
        seed = self._seeds.get(address)
        held = topoff.held_value(seed, self._seeds.get)
        if held is not None:
            return self._coerce(address, held)
        if seed is not None and seed.nominal is not None:
            return self._coerce(address, seed.nominal)
        if wiring_default is not None:
            return self._coerce(address, wiring_default)
        return channel.zero()

    @property
    @override
    def supported_variables(self) -> dict[str, Variable]:
        """The texture-owned channels, by address."""
        return self._variables

    @override
    def reset(self) -> None:
        """Return every texture-owned channel to its nominal, then apply the active writes."""
        self._held = {address: self._nominals[address] for address in self._variables}
        self._hold(self._active)

    def set_active(self, writes: Mapping[str, object]) -> None:
        """Make the active scenarios' writes the start state, and reset to it.

        An active write is held whether or not its channel is settable, and a
        setpoint's write echoes into its readback as a set does.

        Args:
            writes: Values by texture-owned address; empty for none.

        Raises:
            ValueError: A value is refused for its channel's value_type; the
                model keeps its earlier start state and held values.
        """
        previous, held = self._active, self._held
        self._active = dict(writes)
        try:
            self.reset()
        except ValueError:
            self._active, self._held = previous, held
            raise

    def held(self, names: Sequence[str]) -> dict[str, values.StoredValue]:
        """Each channel's held value, without motion or clamp.

        A ``linear`` channel holds the weighted sum of its inputs' held values;
        an input the model does not hold counts as zero. A ``gate`` channel
        holds its gated value while any input holds its blocking label.

        Args:
            names: Texture-owned addresses.

        Returns:
            The held value by address, in its stored representation.
        """
        return {name: self._held_value(name) for name in names}

    def reach(self) -> dict[str, frozenset[str]]:
        """What a write to each settable channel can change, by address.

        The channel itself, the readback it echoes into, and every channel
        downstream of either: a ``linear`` channel summing it, a ``gate``
        channel it blocks, a ``pulse`` waveform it drives or enables.
        """
        downstream: dict[str, set[str]] = {}
        for address in self._variables:
            seed = self._seeds.get(address)
            linear = None if seed is None else seed.linear
            sources = [source for source, _ in _linear_terms(linear or {})]
            gate = None if seed is None else seed.gate
            sources += [source for source, _ in _gate_inputs(None if gate is None else gate.inputs)]
            record = self._pulse(address)
            if record is not None:
                sources += [record.driver, record.enable]
            for source in sources:
                downstream.setdefault(source, set()).add(address)
        reach: dict[str, frozenset[str]] = {}
        for address, variable in self._variables.items():
            if variable.read_only:
                continue
            found = {address, self._echo.get(address, address)}
            pending = list(found)
            while pending:
                for dependent in downstream.get(pending.pop(), ()):
                    if dependent not in found:
                        found.add(dependent)
                        pending.append(dependent)
            reach[address] = frozenset(found)
        return reach

    def _held_value(self, address: str) -> values.StoredValue:
        return self._derived(address, self._held)

    def _derived(self, address: str, held: Mapping[str, values.StoredValue]) -> values.StoredValue:
        """A channel's value from ``held``: its weighted sum, then its gate."""
        return self._gated(address, self._weighted_sum(address, held), held)

    def _weighted_sum(
        self, address: str, held: Mapping[str, values.StoredValue]
    ) -> values.StoredValue:
        """``held[address]``, or for a ``linear`` channel the weighted sum of its inputs."""
        seed = self._seeds.get(address)
        linear = None if seed is None else seed.linear
        if not linear:
            return held[address]
        return sum(
            coefficient * values._real(self._derived(source, held)) if source in held else 0.0
            for source, coefficient in _linear_terms(linear)
        )

    def _gated(
        self, address: str, value: values.StoredValue, held: Mapping[str, values.StoredValue]
    ) -> values.StoredValue:
        """``value``, or the seed's gated value while any ``gate`` input blocks."""
        seed = self._seeds.get(address)
        gate = None if seed is None else seed.gate
        if gate is None or not self._blocked(gate, held):
            return value
        gated = gate.gated
        value_type = self._channels[address].value_type
        if value_type == "float":
            return values._real(value) * values._real(gated)
        if value_type == "int":
            return int(gated)
        if value_type in values.LABELLED_TYPES:
            return self._coerce(address, gated)
        return value

    def _blocked(self, gate: GateRecord, held: Mapping[str, values.StoredValue]) -> bool:
        """Whether any input of ``gate`` holds its blocking label."""
        return any(
            source in held
            and source in self._channels
            and self._derived(source, held) == self._coerce(source, label)
            for source, label in _gate_inputs(gate.inputs)
        )

    def set_motion(
        self,
        couple: Mapping[str, Sequence[ResolvedCoupling]],
        noise: Mapping[str, NoiseRecord],
        *,
        still: frozenset[str] = frozenset(),
    ) -> None:
        """Hold the active scenarios' motion until the next call.

        Args:
            couple: Couplings by address, each ``{driver, gain, gain_wander?,
                drive}``: ``drive`` is the driver's ``{kind, amplitude,
                period_s}`` and ``gain_wander`` an optional ``{amplitude,
                period_s}``. Empty for no coupling.
            noise: Noise records by address, each ``{absolute: <sigma>}`` or
                ``{relative: <fraction>}``, standing in for the seed's.
                Empty for the seeds' noise.
            still: Addresses served without motion.
        """
        self._couple = {str(address): list(terms) for address, terms in couple.items()}
        self._noise = dict(noise)
        self._still = frozenset(str(address) for address in still)

    def motion(self, address: str, times_s: object, base: object = 0.0) -> npt.NDArray[np.float64]:
        """The motion of a channel at absolute epoch seconds.

        Drift, then the held couplings, then a ``topoff`` seed's tooth
        ``base * (s(t) / m - 1)``, then noise: the held scenario's noise
        record, else the seed's. An absolute term is its sigma times a keyed
        normal draw; a relative term scales ``base`` plus drift, couplings
        and tooth. Every term is a pure function of the address and the
        epoch time.

        Args:
            address: Any address the view declares.
            times_s: Epoch seconds; any shape.
            base: The value the motion is added to, which a relative noise
                term scales; a scalar or an array shaped like ``times_s``.

        Returns:
            A float64 array with the shape of ``times_s``; zero for a channel
            without motion, stilled, or not of type float.
        """
        times = np.asarray(times_s, dtype=np.float64)
        flat = times.reshape(-1)
        total = np.zeros(flat.shape, dtype=np.float64)
        channel = self._channels.get(address)
        if channel is None or channel.value_type != "float" or address in self._still:
            return total.reshape(times.shape)
        seed = self._seeds.get(address)
        key = series.channel_key_bytes(address)
        drift = None if seed is None else seed.drift
        if drift is not None:
            total = total + series.wander(key, flat, float(drift.amplitude), float(drift.period_s))
        for coupling in self._couple.get(address, ()):
            total = total + self._coupling(address, coupling, flat)
        record = topoff.topoff_record(seed)
        if record is not None:
            tau, period = topoff.tau_period(record)
            total = total + self._tooth(tau, period, topoff.period_mean(record), flat, base)
        term, value, subkey = self._noise_term(address)
        if value:
            normals = series.keyed_normals(key + subkey, np.rint(flat * _MS_PER_S).astype(np.int64))
            if term == _RELATIVE:
                scaled = np.asarray(base, dtype=np.float64).reshape(-1) + total
                total = total + scaled * value * normals
            else:
                total = total + value * normals
        return total.reshape(times.shape)

    def has_motion(self, address: str) -> bool:
        """Whether :meth:`motion` can be non-zero for *address*.

        True for a ``pulse`` waveform channel, whose trace changes with
        time. Otherwise true exactly for a float channel the held motion does
        not still, with seed drift, a ``topoff`` seed, a held coupling, or a
        non-zero noise term under the held motion.
        """
        if self._pulse(address) is not None:
            return True
        channel = self._channels.get(address)
        if channel is None or channel.value_type != "float" or address in self._still:
            return False
        seed = self._seeds.get(address)
        if self._couple.get(address) or topoff.topoff_record(seed) is not None:
            return True
        if address not in self._noise:
            return declares_motion(seed)
        return bool((seed is not None and seed.drift is not None) or self._noise_term(address)[1])

    def _noise_term(self, address: str) -> tuple[str, float, bytes]:
        """The noise record acting on *address*: ``(term, value, stream subkey)``.

        The held scenario's record, when there is one, stands in for the
        seed's entirely and draws from its own stream; the seed's draws from
        the channel's key. No record is an absolute term of 0.0.
        """
        replacement = self._noise.get(address)
        if replacement is None:
            seed = self._seeds.get(address)
            term, value = _noise_term(None if seed is None else seed.noise)
            return term, value, b""
        term, value = _noise_term(replacement)
        subkey = _SCENARIO_RELATIVE_SUBKEY if term == _RELATIVE else _SCENARIO_ABSOLUTE_SUBKEY
        return term, value, subkey

    @staticmethod
    def _tooth(
        tau: float, period: float, mean: float, times: npt.NDArray[np.float64], base: object
    ) -> npt.NDArray[np.float64]:
        """The top-off tooth ``base * (s(t) / m - 1)`` on the level ``base``.

        ``tau`` and ``period`` are a ``topoff`` record's, in seconds, and
        ``mean`` its sawtooth's mean ``m``.
        """
        unit = np.exp(-np.mod(times, period) / tau)
        level = np.asarray(base, dtype=np.float64).reshape(-1)
        tooth: npt.NDArray[np.float64] = level * (unit / mean - 1.0)
        return tooth

    @staticmethod
    def _coupling(
        address: str, coupling: ResolvedCoupling, times: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """``gain * (1 + gain_wander(t)) * driver(t)`` of one coupling."""
        driver = coupling.driver
        drive = coupling.drive
        signal = series.wander(
            series.driver_key_bytes(driver),
            times,
            float(drive.amplitude),
            float(drive.period_s),
        )
        gain: float | npt.NDArray[np.float64] = float(coupling.gain)
        envelope = coupling.gain_wander
        if envelope is not None:
            gain = gain * (
                1.0
                + series.wander(
                    series.channel_key_bytes(address) + _GAIN_WANDER_SUBKEY + driver.encode(),
                    times,
                    float(envelope.amplitude),
                    float(envelope.period_s),
                )
            )
        return np.asarray(gain * signal, dtype=np.float64)

    def _pulse(self, address: str) -> PulseRecord | None:
        """The ``pulse`` record of a waveform channel, ``None`` when it has none."""
        channel = self._channels.get(address)
        if channel is None or not values.is_waveform(channel.value_type):
            return None
        seed = self._seeds.get(address)
        return None if seed is None else seed.pulse

    def _pulse_trace(
        self, address: str, record: PulseRecord, t_s: float
    ) -> npt.NDArray[np.float64]:
        """The trace of a ``pulse`` channel at *t_s* from its driver and enable held values.

        A driver or enable channel the model does not hold counts as zero or
        off, so the trace reads flat zero.
        """
        driver_address = record.driver
        enable_address = record.enable
        driver = (
            values._real(self._held_value(driver_address)) if driver_address in self._held else 0.0
        )
        enable = False
        if enable_address in self._held:
            labels = self._channels[enable_address].labels
            enable = self._held_value(enable_address) == labels[-1]
        return pulse.trace(address, t_s, driver, enable)

    def clamp(self, address: str, value: float) -> float:
        """Clamp a float into the seed's ``clamp`` band; a null side is unbounded."""
        seed = self._seeds.get(address)
        band = None if seed is None else seed.clamp
        if not band:
            return value
        return series.clamp(value, band[0], band[1])

    @override
    def _get(self, names: list[str]) -> dict[str, object]:
        t_s = float(self._clock())
        outputs: dict[str, object] = {}
        for name in names:
            held = self._held_value(name)
            channel = self._channels[name]
            if channel.value_type == "float":
                level = values._real(held)
                outputs[name] = self.clamp(name, level + float(self.motion(name, t_s, base=level)))
            elif values.is_waveform(channel.value_type):
                record = self._pulse(name)
                array = held if record is None else self._pulse_trace(name, record, t_s)
                outputs[name] = np.asarray(array, dtype=np.float64).reshape(channel.shape or ())
            else:
                outputs[name] = held
        return outputs

    @override
    def _set(self, values_by_name: dict[str, object]) -> None:
        self._hold(values_by_name)

    def _hold(self, values_by_name: Mapping[str, object]) -> None:
        """Hold coerced values and their echoes; nothing is held when one is refused."""
        coerced = {
            name: self._coerce(name, value.tolist() if isinstance(value, np.ndarray) else value)
            for name, value in values_by_name.items()
        }
        echoes = {
            self._echo[name]: self._coerce(self._echo[name], value)
            for name, value in coerced.items()
            if name in self._echo
        }
        self._held.update(coerced)
        self._held.update(echoes)
