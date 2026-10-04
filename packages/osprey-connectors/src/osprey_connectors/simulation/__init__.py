"""The simulated machine behind OSPREY's mock connectors.

The package root holds only what every reader needs without loading a model:
the tick period, value coercion, char-waveform decoding and the
active-scenario state helpers. It imports neither numpy nor lume; the engine,
machine and expression names are imported from their own modules
(:mod:`.engine`, :mod:`.machine`, :mod:`.expressions`).
"""

from collections.abc import Mapping
from typing import Any

from osprey_connectors.simulation.state import (
    ACTIVE_SCENARIOS_FILENAME,
    OVERLAP_EVENT,
    Overlap,
    format_overlap_record,
    overlap_record,
    parse_active_state,
    resolve_active_scenarios,
    validate_composition,
)
from osprey_connectors.simulation.values import coerce

__all__ = [
    "ACTIVE_SCENARIOS_FILENAME",
    "DEFAULT_TICK_S",
    "OVERLAP_EVENT",
    "Overlap",
    "TICK_KEY",
    "coerce",
    "decode_char_waveform",
    "format_overlap_record",
    "overlap_record",
    "parse_active_state",
    "resolve_active_scenarios",
    "resolve_tick_s",
    "validate_composition",
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
    section = config.get("simulation") or {}
    raw = section.get("tick_s")
    if raw is None:
        return DEFAULT_TICK_S
    if isinstance(raw, bool) or not isinstance(raw, int | float) or raw <= 0:
        raise ValueError(f"{TICK_KEY} must be a number of seconds greater than 0 (got {raw!r})")
    return float(raw)


def decode_char_waveform(value: Any) -> str:
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
