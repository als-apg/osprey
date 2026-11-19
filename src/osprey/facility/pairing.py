"""The rule that pairs a setpoint with its readback.

A channel's quantity is its signal less the trailing role word: a setpoint's
signal ends in a write word, a readback's in a read word, and any other signal
has no quantity. A setpoint pairs with the one readback that sits on the same
device, place or nothing, and whose signal names the same quantity. A pair a
source states wins, and the readback it names is no other setpoint's candidate.
When the location holds several setpoints or several readbacks of that quantity,
the setpoint stays its own pair; so it does when the two differ in value type,
options or shape and the setpoint is no command (:func:`must_agree`). A command
and the status it reads back on are a pair whatever their types.

Several setpoints may name one readback (a set and a reset command read back on
one channel). By default the simulation follows a readback only from its one
setpoint, and only where the two hold values of one type; any other readback it
follows from no setpoint. The readback's seed states the exception both ways:
``follows: <setpoint>`` makes its pair followed, ``follows: none`` stops it
(:func:`pair_follows`, :func:`simulated_pairs`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, NamedTuple

#: The write word of a command: a setpoint whose signal ends in it is one.
COMMAND_WORD = "command"
WRITE_WORDS = ("setpoint", COMMAND_WORD)
READ_WORDS = ("readback", "status")

_SAME_SLOTS = ("value_type", "options", "shape")

#: Whether the simulation follows a pair: by default, by the readback's seed, or not.
FOLLOWED_DEFAULT = "default"
FOLLOWED_SEED = "seed"
NOT_FOLLOWED = "not followed"
FOLLOWS = (FOLLOWED_DEFAULT, FOLLOWED_SEED, NOT_FOLLOWED)
#: The seed word ``follows`` takes to say the simulation follows the readback from no setpoint.
FOLLOWS_NONE = "none"

_WHY_DEFAULT = "one setpoint names the readback and the two hold values of one type"
_WHY_SEED = "the readback's seed follows this setpoint"
_WHY_NONE = "the readback's seed follows none"
_WHY_SHARED = "several setpoints name the readback"
_WHY_TYPES = "the setpoint and the readback hold values of different types"
_WHY_OTHER = "the readback's seed follows another setpoint"

#: The rule :func:`derive_pairs` applies, named where a derived pair is shown.
PAIR_RULE = "the one readback of the same quantity where the setpoint sits"


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


def is_command_signal(signal: str | None) -> bool:
    """Whether a setpoint's vocabulary signal makes it a command: it ends in :data:`COMMAND_WORD`.

    Args:
        signal: The channel's vocabulary signal, if any.
    """
    return bool(signal) and str(signal).rpartition("_")[2] == COMMAND_WORD


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


def same_type(setpoint: Mapping[str, Any], readback: Mapping[str, Any]) -> bool:
    """Whether two channels hold values of one type: equal value type, options and shape."""
    return all(setpoint.get(slot) == readback.get(slot) for slot in _SAME_SLOTS)


def must_agree(setpoint: Mapping[str, Any], readback: Mapping[str, Any]) -> bool:
    """Whether a pair's two ends must hold values of one type: a setpoint and a
    readback whose signals name one quantity, the setpoint no command. A command
    and its status, and a pair whose quantity no signal states, may differ.

    Args:
        setpoint: The setpoint's fields.
        readback: The readback's fields.
    """
    if is_command_signal(setpoint.get("signal")):
        return False
    quantity = signal_quantity(setpoint.get("signal"), "setpoint")
    return quantity is not None and quantity == signal_quantity(readback.get("signal"), "readback")


class Follow(NamedTuple):
    """Whether the simulation follows one pair, and why."""

    setpoint: str
    readback: str
    #: One of :data:`FOLLOWS`.
    follow: str
    why: str


def seed_follows(channel: Mapping[str, Any]) -> str | None:
    """The ``follows`` a channel's seed states, if any.

    Args:
        channel: The channel's fields.
    """
    seed = channel.get("simulation")
    follows = seed.get("follows") if isinstance(seed, Mapping) else None
    return None if follows is None else str(follows)


def pair_follows(channels: Mapping[str, Mapping[str, Any]]) -> list[Follow]:
    """Whether the simulation follows each pair, and why.

    A pair is a setpoint whose ``pair`` names another channel. The readback's seed
    decides when it states ``follows``: the setpoint it names is followed, and no
    other. Otherwise the readback is followed when exactly one setpoint names it
    and the two hold values of one type.

    Args:
        channels: Every channel's fields by address, with ``role``, ``pair`` and
            the seed in ``simulation``.

    Returns:
        One :class:`Follow` per pair, sorted by readback, then setpoint.
    """
    named: dict[str, list[str]] = {}
    for address, fields in channels.items():
        if fields.get("role") != "setpoint":
            continue
        pair = str(fields.get("pair") or address)
        if pair != address:
            named.setdefault(pair, []).append(address)
    found: list[Follow] = []
    for readback, setpoints in sorted(named.items()):
        channel = channels.get(readback)
        follows = None if channel is None else seed_follows(channel)
        for setpoint in sorted(setpoints):
            if channel is None:
                found.append(Follow(setpoint, readback, NOT_FOLLOWED, "the readback is no channel"))
            elif len(setpoints) > 1:
                found.append(Follow(setpoint, readback, NOT_FOLLOWED, _WHY_SHARED))
            elif follows == FOLLOWS_NONE:
                found.append(Follow(setpoint, readback, NOT_FOLLOWED, _WHY_NONE))
            elif follows == setpoint:
                found.append(Follow(setpoint, readback, FOLLOWED_SEED, _WHY_SEED))
            elif follows is not None:
                found.append(Follow(setpoint, readback, NOT_FOLLOWED, _WHY_OTHER))
            elif same_type(channels[setpoint], channel):
                found.append(Follow(setpoint, readback, FOLLOWED_DEFAULT, _WHY_DEFAULT))
            else:
                found.append(Follow(setpoint, readback, NOT_FOLLOWED, _WHY_TYPES))
    return found


def follow_totals(follows: Iterable[Follow]) -> dict[str, int]:
    """How many pairs each word of :data:`FOLLOWS` counts.

    Args:
        follows: The pairs, as :func:`pair_follows` gives them.
    """
    totals = dict.fromkeys(FOLLOWS, 0)
    for follow in follows:
        totals[follow.follow] += 1
    return totals


def simulated_pairs(channels: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """The readbacks the simulation drives from a setpoint (:func:`pair_follows`).

    Args:
        channels: Every channel's fields by address, with ``role``, ``pair`` and
            the seed in ``simulation``.

    Returns:
        ``{readback: setpoint}`` for each followed pair.
    """
    return {
        follow.readback: follow.setpoint
        for follow in pair_follows(channels)
        if follow.follow != NOT_FOLLOWED
    }


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
        if is_command_signal(channels[setpoint].get("signal")) or same_type(
            channels[setpoint], channels[readback]
        ):
            derived[setpoint] = readback
    return dict(sorted(derived.items()))
