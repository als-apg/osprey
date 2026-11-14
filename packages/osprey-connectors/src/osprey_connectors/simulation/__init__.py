"""The simulated machine behind OSPREY's simulator connectors.

The package root holds only what every reader needs without loading a model:
the tick period and the fast period of timed passes, char-waveform decoding,
the gated and labelled value types and the model status lookups. It imports
neither numpy nor lume; the active-scenario state, value coercion, the
composite, the texture engine, the archive and the series primitives are
imported from their own modules.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from osprey_connectors.simulation.status import model_status, read_model_status
from osprey_connectors.simulation.values import GATED_TYPES, LABELLED_TYPES

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "DEFAULT_FAST_TICK_S",
    "DEFAULT_TICK_S",
    "FAST_TICK_KEY",
    "GATED_TYPES",
    "LABELLED_TYPES",
    "TICK_KEY",
    "decode_char_waveform",
    "model_status",
    "read_model_status",
    "resolve_fast_tick_s",
    "resolve_tick_s",
]

#: The config key holding the simulated machine's tick period, in seconds.
TICK_KEY = "simulation.tick_s"

#: The tick period when the config does not set one, in seconds.
DEFAULT_TICK_S = 1.0


def resolve_tick_s(config: Mapping[str, Any]) -> float:
    """The simulated machine's tick period a config asks for.

    Args:
        config: The full project config.

    Returns:
        ``simulation.tick_s`` in seconds, or :data:`DEFAULT_TICK_S` when the
        ``simulation`` section or its ``tick_s`` entry is absent.

    Raises:
        ValueError: If the value is not a number greater than zero; the
            message names the key.
    """
    return _resolve_period(config, TICK_KEY, DEFAULT_TICK_S)


#: The config key holding the fast period, in seconds: how often the served
#: machine advances its timed state between ticks.
FAST_TICK_KEY = "simulation.fast_tick_s"

#: The fast period when the config does not set one, in seconds.
DEFAULT_FAST_TICK_S = 0.1


def resolve_fast_tick_s(config: Mapping[str, Any]) -> float:
    """The fast period a config asks for.

    Args:
        config: The full project config.

    Returns:
        ``simulation.fast_tick_s`` in seconds, or :data:`DEFAULT_FAST_TICK_S`
        when the ``simulation`` section or its ``fast_tick_s`` entry is absent.

    Raises:
        ValueError: If the value is not a number greater than zero; the
            message names the key.
    """
    return _resolve_period(config, FAST_TICK_KEY, DEFAULT_FAST_TICK_S)


def _resolve_period(config: Mapping[str, Any], key: str, default: float) -> float:
    """The ``simulation.<entry>`` period ``key`` names, in seconds; ``default`` when absent."""
    section = config.get("simulation") or {}
    raw = section.get(key.removeprefix("simulation."))
    if raw is None:
        return default
    if isinstance(raw, bool) or not isinstance(raw, int | float) or raw <= 0:
        raise ValueError(f"{key} must be a number of seconds greater than 0 (got {raw!r})")
    return float(raw)


def decode_char_waveform(value: "str | Sequence[int] | np.ndarray") -> str:
    """The text a string channel holds, whichever shape it arrives in.

    A ``str`` passes unchanged. An array of character codes (a list, a tuple
    or a numpy array, signed or unsigned bytes) is read up to its first NUL
    and decoded as UTF-8, an invalid byte becoming U+FFFD.

    Args:
        value: The value read from the channel.

    Returns:
        The decoded text.
    """
    if isinstance(value, str):
        return value
    codes = value.tolist() if hasattr(value, "tolist") else list(value)
    raw = bytes(int(code) % 256 for code in codes)
    return raw.split(b"\0", 1)[0].decode("utf-8", errors="replace")
