"""The texture model: the simulator's value for every channel no served model wires.

:class:`TextureModel` is a ``LUMEModel`` built from two documents of the
simulator view, ``variables.json`` and ``seeds.json``. It owns the VALUE of a
channel whose ``owner`` is ``texture`` or names a model the view marks
unserved, and the MOTION (keyed noise and drift) and clamp of every channel
the view declares, so a served model's readbacks can carry the same motion.

A texture channel reads ``clamp(held + noise(t) + drift(t))``. ``held`` starts
at the channel's nominal: the seed's ``nominal``, else the unserved model's
wiring ``default``, else the zero of the channel's ``value_type``. A paired readback
starts at its setpoint's nominal. A ``linear`` channel holds the weighted sum
of its inputs' held values, a ``linear`` input counting as its own sum. Noise and drift
are pure functions of the address and the epoch millisecond, so a live read
and an archived sample at the same instant agree. Writing a setpoint holds the
value and echoes it into the readback its ``pair`` names.
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

__all__ = ["TEXTURE_OWNER", "TextureModel"]

#: The ``owner`` the simulator view writes for a channel no model wires.
TEXTURE_OWNER = "texture"

_SETPOINT = "setpoint"
_MS_PER_S = 1000.0
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
            address: self._variable(address, self._channels[address], start[address])
            for address in owned
        }
        self._held: dict[str, Any] = {}
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

    def _variable(self, address: str, channel: Mapping[str, Any], nominal: Any) -> Variable:
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

    @property
    def supported_variables(self) -> dict[str, Variable]:
        """The texture-owned channels, by address."""
        return self._variables

    def reset(self) -> None:
        """Return every texture-owned channel to its nominal."""
        self._held = {address: self._nominals[address] for address in self._variables}

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

    def motion(self, address: str, t_s: Any) -> np.ndarray:
        """The keyed noise plus drift of a channel at absolute epoch seconds.

        Args:
            address: Any address the view declares.
            t_s: Epoch seconds; any shape.

        Returns:
            A float64 array with the shape of ``t_s``; zero for a channel
            without a seed, without noise and drift, or not of type float.
        """
        times = np.asarray(t_s, dtype=np.float64)
        flat = times.reshape(-1)
        total = np.zeros(flat.shape, dtype=np.float64)
        channel = self._channels.get(address)
        seed = self._seeds.get(address)
        if channel is None or seed is None or not _is_float(channel):
            return total.reshape(times.shape)
        key = series.channel_key_bytes(address)
        sigma = seed.get("noise")
        if sigma:
            counters_ms = np.rint(flat * _MS_PER_S).astype(np.int64)
            total = total + float(sigma) * series.keyed_normals(key, counters_ms)
        drift = seed.get("drift")
        if drift:
            total = total + series.wander(
                key, flat, float(drift["amplitude"]), float(drift["period_s"])
            )
        return total.reshape(times.shape)

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
                outputs[name] = self.clamp(name, float(held) + float(self.motion(name, t_s)))
            elif _channel_value_type(channel) == "waveform":
                outputs[name] = np.asarray(held, dtype=np.float64).reshape(
                    tuple(int(size) for size in channel["shape"])
                )
            else:
                outputs[name] = held
        return outputs

    def _set(self, values_by_name: dict[str, Any]) -> None:
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
