"""The texture model: the simulator's value for every channel no served model wires.

:class:`TextureModel` is a ``LUMEModel`` built from two documents of the
simulator view, ``variables.json`` and ``seeds.json``. It owns the VALUE of a
channel whose ``owner`` is ``texture`` or names a model the view marks
unserved, and the MOTION (keyed noise and drift) and clamp of every channel
the view declares, so a served model's readbacks can carry the same motion.

A texture channel reads ``clamp(held + drift(t) + couplings(t) + noise(t))``. ``held`` starts
at the channel's nominal: the seed's ``nominal``, else the unserved model's
wiring ``default``, else the zero of the channel's ``value_type``. A paired readback
starts at its setpoint's nominal. A ``linear`` channel holds the weighted sum
of its inputs' held values, a ``linear`` input counting as its own sum. Noise and drift
are pure functions of the address and the epoch millisecond, so a live read
and an archived sample at the same instant agree. Writing a setpoint holds the
value and echoes it into the readback its ``pair`` names.

The active scenarios add motion through :meth:`TextureModel.set_motion`: a
coupled channel adds ``gain * (1 + gain_wander(t)) * driver(t)`` per coupling,
where every channel coupled to one driver sees the same ``driver(t)``, and a
noise replacement stands in for the seed's noise while it is held: its
relative ``noise`` scales ``held + drift + couplings``, then its ``noise_abs``
is added.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from lume.model import LUMEModel
from lume.variables import (
    EnumVariable,
    IntVariable,
    NDVariable,
    ScalarVariable,
    StrVariable,
    Variable,
)

from osprey_connectors.simulation import series, values

__all__ = ["TEXTURE_OWNER", "TextureModel", "channel_variable"]

#: The ``owner`` the simulator view writes for a channel no model wires.
TEXTURE_OWNER = "texture"

_SETPOINT = "setpoint"
_MS_PER_S = 1000.0
_GAIN_WANDER_SUBKEY = b":gain_wander:"
_RELATIVE_NOISE_SUBKEY = b":noise"
_ABSOLUTE_NOISE_SUBKEY = b":noise_abs"
_INT_UNBOUNDED = 2**63 - 1


def _channel_value_type(channel: Mapping[str, Any]) -> str:
    return channel.get("value_type") or values.DEFAULT_VALUE_TYPE


def _is_float(channel: Mapping[str, Any]) -> bool:
    return _channel_value_type(channel) == "float"


def _value_range(channel: Mapping[str, Any]) -> tuple[Any, Any] | None:
    """The variable's range from the view's ``value_range``; a null side is unbounded."""
    bounds = channel.get("value_range")
    if bounds is None:
        return None
    low, high = bounds
    if low is None and high is None:
        return None
    if _channel_value_type(channel) == "int":
        return (
            -_INT_UNBOUNDED if low is None else int(low),
            _INT_UNBOUNDED if high is None else int(high),
        )
    return (
        -math.inf if low is None else float(low),
        math.inf if high is None else float(high),
    )


def _linear_terms(linear: Mapping[str, Any]) -> list[tuple[str, float]]:
    """``(input address, coefficient)`` pairs of a seed's ``linear`` map, sorted."""
    terms: list[tuple[str, float]] = []
    for address in sorted(linear):
        term = linear[address]
        coefficient = term["coefficient"] if isinstance(term, Mapping) else term
        terms.append((str(address), float(coefficient)))
    return terms


def channel_variable(channel: Mapping[str, Any], nominal: Any) -> Variable:
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
    address = str(channel["address"])
    read_only = not (channel.get("role") == _SETPOINT and channel.get("writable") is True)
    value_type = _channel_value_type(channel)
    unit = channel.get("unit")
    if value_type in ("bool", "enum"):
        options = list(channel.get("options") or values.DEFAULT_BOOL_OPTIONS)
        return EnumVariable(
            name=address, options=options, default_value=nominal, read_only=read_only
        )
    if value_type == "string":
        return StrVariable(name=address, default_value=nominal, read_only=read_only)
    if value_type == "waveform":
        shape = tuple(int(size) for size in channel["shape"])
        return NDVariable(
            name=address,
            shape=shape,
            default_value=np.asarray(nominal, dtype=np.float64).reshape(shape),
            unit=unit,
            read_only=read_only,
        )
    variable_class = IntVariable if value_type == "int" else ScalarVariable
    return variable_class(
        name=address,
        default_value=nominal,
        value_range=_value_range(channel),
        unit=unit,
        read_only=read_only,
    )


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
        variables: Mapping[str, Any],
        seeds: Mapping[str, Any],
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._clock = clock
        self._seeds: dict[str, Mapping[str, Any]] = {
            str(address): seed for address, seed in (seeds.get("seeds") or {}).items()
        }
        self._channels: dict[str, Mapping[str, Any]] = {
            str(channel["address"]): channel for channel in variables.get("channels", [])
        }
        unserved_defaults = self._unserved_wiring_defaults(variables.get("models", []))
        unserved = {
            str(model["name"]) for model in variables.get("models", []) if not model.get("served")
        }

        owned = sorted(
            address
            for address, channel in self._channels.items()
            if channel.get("owner", TEXTURE_OWNER) in (TEXTURE_OWNER, *unserved)
        )
        self._echo: dict[str, str] = {
            address: str(self._channels[address]["pair"])
            for address in owned
            if self._channels[address].get("role") == _SETPOINT
            and self._channels[address].get("pair") not in (None, address)
            and str(self._channels[address]["pair"]) in owned
        }

        self._nominals: dict[str, Any] = {
            address: self._nominal(address, unserved_defaults.get(address)) for address in owned
        }
        for setpoint, readback in self._echo.items():
            self._nominals[readback] = self._coerce(readback, self._nominals[setpoint])
        start = {address: self._weighted_sum(address, self._nominals) for address in owned}
        self._variables: dict[str, Variable] = {
            address: channel_variable(self._channels[address], start[address]) for address in owned
        }
        self._held: dict[str, Any] = {}
        self._couple: dict[str, list[Mapping[str, Any]]] = {}
        self._noise: dict[str, Mapping[str, Any]] = {}
        self._active: dict[str, Any] = {}
        self.reset()

    @staticmethod
    def _unserved_wiring_defaults(models: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        defaults: dict[str, Any] = {}
        for model in models:
            if model.get("served"):
                continue
            for record in model.get("wiring") or []:
                if "default" in record:
                    defaults.setdefault(str(record["address"]), record["default"])
        return defaults

    def _coerce(self, address: str, value: Any) -> Any:
        channel = self._channels[address]
        return values.coerce(
            value, _channel_value_type(channel), channel.get("options"), channel.get("shape")
        )

    def _nominal(self, address: str, wiring_default: Any) -> Any:
        channel = self._channels[address]
        seed = self._seeds.get(address, {})
        if seed.get("nominal") is not None:
            return self._coerce(address, seed["nominal"])
        if wiring_default is not None:
            return self._coerce(address, wiring_default)
        return values.zero(
            _channel_value_type(channel), channel.get("options"), channel.get("shape")
        )

    @property
    def supported_variables(self) -> dict[str, Variable]:
        """The texture-owned channels, by address."""
        return self._variables

    def reset(self) -> None:
        """Return every texture-owned channel to its nominal, then apply the active writes."""
        self._held = {address: self._nominals[address] for address in self._variables}
        self._hold(self._active)

    def set_active(self, writes: Mapping[str, Any]) -> None:
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

    def held(self, names: Sequence[str]) -> dict[str, Any]:
        """Each channel's held value, without motion or clamp.

        A ``linear`` channel holds the weighted sum of its inputs' held values;
        an input the model does not hold counts as zero.

        Args:
            names: Texture-owned addresses.

        Returns:
            The held value by address, in its stored representation.
        """
        return {name: self._held_value(name) for name in names}

    def _held_value(self, address: str) -> Any:
        return self._weighted_sum(address, self._held)

    def _weighted_sum(self, address: str, held: Mapping[str, Any]) -> Any:
        """``held[address]``, or for a ``linear`` channel the weighted sum of its inputs."""
        linear = self._seeds.get(address, {}).get("linear")
        if not linear:
            return held[address]
        return sum(
            coefficient * float(self._weighted_sum(source, held)) if source in held else 0.0
            for source, coefficient in _linear_terms(linear)
        )

    def set_motion(
        self,
        couple: Mapping[str, Sequence[Mapping[str, Any]]],
        noise: Mapping[str, Mapping[str, Any]],
    ) -> None:
        """Hold the active scenarios' motion until the next call.

        Args:
            couple: Couplings by address, each ``{driver, gain, gain_wander?,
                drive}``: ``drive`` is the driver's ``{kind, amplitude,
                period_s}`` and ``gain_wander`` an optional ``{amplitude,
                period_s}``. Empty for no coupling.
            noise: Noise replacements by address, each ``{noise, noise_abs}``;
                a key it lacks is zero. Empty for the seeds' noise.
        """
        self._couple = {str(address): list(terms) for address, terms in couple.items()}
        self._noise = {str(address): dict(entry) for address, entry in noise.items()}

    def motion(self, address: str, t_s: Any, base: Any = 0.0) -> np.ndarray:
        """The motion of a channel at absolute epoch seconds.

        Drift, then the held couplings, then noise: the seed's keyed noise, or
        the held replacement's relative term on ``base`` plus drift plus
        couplings and its absolute term. Every term is a pure function of the
        address and the epoch time.

        Args:
            address: Any address the view declares.
            t_s: Epoch seconds; any shape.
            base: The value the motion is added to, which a relative noise
                term scales; a scalar or an array shaped like ``t_s``.

        Returns:
            A float64 array with the shape of ``t_s``; zero for a channel
            without motion or not of type float.
        """
        times = np.asarray(t_s, dtype=np.float64)
        flat = times.reshape(-1)
        total = np.zeros(flat.shape, dtype=np.float64)
        channel = self._channels.get(address)
        if channel is None or not _is_float(channel):
            return total.reshape(times.shape)
        seed = self._seeds.get(address) or {}
        key = series.channel_key_bytes(address)
        drift = seed.get("drift")
        if drift:
            total = total + series.wander(
                key, flat, float(drift["amplitude"]), float(drift["period_s"])
            )
        for coupling in self._couple.get(address, ()):
            total = total + self._coupling(address, coupling, flat)
        counters_ms = np.rint(flat * _MS_PER_S).astype(np.int64)
        replacement = self._noise.get(address)
        if replacement is None:
            sigma = seed.get("noise")
            if sigma:
                total = total + float(sigma) * series.keyed_normals(key, counters_ms)
            return total.reshape(times.shape)
        relative = float(replacement.get("noise") or 0.0)
        absolute = float(replacement.get("noise_abs") or 0.0)
        if relative:
            scaled = np.asarray(base, dtype=np.float64).reshape(-1) + total
            total = total + scaled * relative * series.keyed_normals(
                key + _RELATIVE_NOISE_SUBKEY, counters_ms
            )
        if absolute:
            total = total + absolute * series.keyed_normals(
                key + _ABSOLUTE_NOISE_SUBKEY, counters_ms
            )
        return total.reshape(times.shape)

    @staticmethod
    def _coupling(address: str, coupling: Mapping[str, Any], times: np.ndarray) -> np.ndarray:
        """``gain * (1 + gain_wander(t)) * driver(t)`` of one coupling."""
        driver = str(coupling["driver"])
        drive = coupling["drive"]
        signal = series.wander(
            series.driver_key_bytes(driver),
            times,
            float(drive["amplitude"]),
            float(drive["period_s"]),
        )
        gain: Any = float(coupling["gain"])
        envelope = coupling.get("gain_wander")
        if envelope:
            gain = gain * (
                1.0
                + series.wander(
                    series.channel_key_bytes(address) + _GAIN_WANDER_SUBKEY + driver.encode(),
                    times,
                    float(envelope["amplitude"]),
                    float(envelope["period_s"]),
                )
            )
        return np.asarray(gain * signal, dtype=np.float64)

    def clamp(self, address: str, value: float) -> float:
        """Clamp a float into the seed's ``clamp`` band; a null side is unbounded."""
        band = self._seeds.get(address, {}).get("clamp")
        if not band:
            return value
        return series.clamp(value, band[0], band[1])

    def _get(self, names: list[str]) -> dict[str, Any]:
        t_s = float(self._clock())
        outputs: dict[str, Any] = {}
        for name in names:
            held = self._held_value(name)
            channel = self._channels[name]
            if _is_float(channel):
                outputs[name] = self.clamp(
                    name, float(held) + float(self.motion(name, t_s, base=float(held)))
                )
            elif _channel_value_type(channel) == "waveform":
                outputs[name] = np.asarray(held, dtype=np.float64).reshape(
                    tuple(int(size) for size in channel["shape"])
                )
            else:
                outputs[name] = held
        return outputs

    def _set(self, values_by_name: dict[str, Any]) -> None:
        self._hold(values_by_name)

    def _hold(self, values_by_name: Mapping[str, Any]) -> None:
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
