"""The rule that pairs a setpoint with its readback.

A channel's quantity is its signal less the trailing role word: a setpoint's
signal ends in a write word, a readback's in a read word, and any other signal
has no quantity. A setpoint pairs with the one readback that sits on the same
device, place or nothing, and whose signal names the same quantity. A pair a
source states wins, and the readback it names is no other setpoint's candidate.
When the location holds several setpoints or several readbacks of that quantity,
or the two differ in value type, options or shape, the setpoint stays its own pair.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

WRITE_WORDS = ("setpoint", "command")
READ_WORDS = ("readback", "status")

_SAME_SLOTS = ("value_type", "options", "shape")


def signal_quantity(signal: str | None, role: str) -> str | None:
    """The quantity a signal names on a channel of ``role``.

    Args:
        signal: The channel's vocabulary signal, if any.
        role: The channel's role (``setpoint`` or ``readback``).

    Returns:
        The signal less its trailing role word, or ``None`` when the signal is
        absent or does not end in a word of the role's side.
    """
    if not signal:
        return None
    head, _, word = str(signal).rpartition("_")
    words = WRITE_WORDS if role == "setpoint" else READ_WORDS
    return head if head and word in words else None


def location(channel: Mapping[str, Any]) -> tuple[Any, ...]:
    """Where a channel sits: its ``on`` record, its shared devices, or nothing.

    Args:
        channel: The channel's combined fields.

    Returns:
        ``("on", kind, id)``, ``("endpoint_of", ids…)`` or ``("facility",)``.
    """
    on = channel.get("on")
    if isinstance(on, Mapping):
        for kind in ("device", "place"):
            if kind in on:
                return ("on", kind, str(on[kind]))
    endpoints = channel.get("endpoint_of")
    if endpoints:
        return ("endpoint_of", tuple(sorted(str(device) for device in endpoints)))
    return ("facility",)


def derive_pairs(channels: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """The pairs the rule forms for setpoints that state none.

    Args:
        channels: Every channel's combined fields by address, after the
            ``role``, ``value_type`` and ``options`` defaults.

    Returns:
        ``{setpoint: readback}`` for each derived pair, and nothing for a
        setpoint that states ``pair`` or stays its own pair.
    """
    taken = {
        str(fields["pair"])
        for fields in channels.values()
        if fields.get("role") == "setpoint" and "pair" in fields
    }
    setpoints: dict[tuple[Any, ...], list[str]] = {}
    readbacks: dict[tuple[Any, ...], list[str]] = {}
    for address in sorted(channels):
        fields = channels[address]
        role = fields.get("role")
        if role == "setpoint" and "pair" in fields:
            continue
        if role == "readback" and address in taken:
            continue
        quantity = signal_quantity(fields.get("signal"), str(role))
        if quantity is None:
            continue
        bucket = (location(fields), quantity)
        if role == "setpoint":
            setpoints.setdefault(bucket, []).append(address)
        elif role == "readback":
            readbacks.setdefault(bucket, []).append(address)
    derived: dict[str, str] = {}
    for bucket, writers in setpoints.items():
        readers = readbacks.get(bucket, [])
        if len(writers) != 1 or len(readers) != 1:
            continue
        setpoint, readback = writers[0], readers[0]
        if all(channels[setpoint].get(s) == channels[readback].get(s) for s in _SAME_SLOTS):
            derived[setpoint] = readback
    return dict(sorted(derived.items()))
