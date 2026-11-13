"""The model runner's configuration, decided from the simulator view alone.

The runner that serves the composite takes the configuration
``Runner.generate_config`` derives from the composite's variables and hands it
to :func:`apply_safety`, which fixes every key a write's safety depends on:

* the five write-path keys of :data:`SAFETY_KEYS` -- every write is its own
  cycle, a refused write is never echoed and raises an alarm on Channel
  Access, an out-of-band write is clamped into its band, and the runner claims
  no control channel of its own;
* each setpoint's ``value_range``, the band the view's limits records give it;
* each float channel's display ``precision``: the view's, when it states one,
  else :data:`FLOAT_PRECISION`, so no float channel is served Channel Access's
  default of none;
* each variable's PV ``mode``, stated explicitly: ``rw`` only for a setpoint
  the view marks writable -- the rule the composite itself declares a
  channel settable by -- and ``ro`` for everything else, the status addresses
  included.

It also refuses a view with a waveform channel whose array -- its declared
length times :data:`ARRAY_ELEMENT_BYTES` -- exceeds the configuration's
``max_array_bytes``: such a channel would be served truncated, so the runner
fails at start instead.

:func:`periodic_addresses` names the channels the view says refresh
``periodic`` and the IOC health addresses that move with time alone, which a
write pass leaves for the next periodic pass.

:func:`ioc_health_addresses` names every IOC health address of the view, so
the runner can serve a text one on Channel Access as the string record it
stands in for.

:func:`fast_addresses` and :func:`event_addresses` name the channels the view
says refresh ``fast`` and ``event``, which timed passes publish beside the
global tick; a view with neither runs no timed pass at all.

:func:`setting_records`, :func:`write_records` and :func:`pass_readback` name
what a write pass reads back. A physics setting reaches its own model alone, and
there moves itself, its readback and what the model computes from its settings;
a texture setpoint moves itself, its echo and the texture channels derived from
either. Every other value stands as the last pass left it -- what moves with
time alone, the next periodic pass publishes -- and reading it back again would
make each write, one per corrector for an apply, cost the whole roster.

:data:`HEALTH_KEYS` holds the defaults of the runner's health record: how many
consecutive failed publishing passes still count as ``degraded``.

Nothing here imports the serving runtime, so the configuration is decided and
tested in process.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from osprey_connectors.simulation.view import PERIODIC, REFRESH, SETPOINT

try:
    from osprey_connectors.simulation.view import PERIODIC_IOC_HEALTH
except ImportError:  # a connectors release whose view names no IOC health signals
    from osprey_connectors.simulation.view import PERIODIC as _FLOOR_STAND_IN  # noqa: F401

    PERIODIC_IOC_HEALTH = ()  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from osprey_connectors.simulation.view import SimulatorView

__all__ = [
    "ARRAY_ELEMENT_BYTES",
    "FLOAT_PRECISION",
    "HEALTH_KEYS",
    "SAFETY_KEYS",
    "apply_safety",
    "event_addresses",
    "fast_addresses",
    "ioc_health_addresses",
    "pass_readback",
    "periodic_addresses",
    "setting_records",
    "write_records",
]

#: The write-path keys every model runner configuration carries.
SAFETY_KEYS: Mapping[str, Any] = {
    "update_rate": 0.0,
    "echo_unconfirmed_writes": False,
    "alarm_on_refused_write": True,
    "clamp_writes": True,
    "control_pvs": False,
}

#: The health keys every model runner configuration carries, at their defaults:
#: more than ``failed_pass_tolerance`` consecutive failed publishing passes
#: fail the runner's health record.
HEALTH_KEYS: Mapping[str, Any] = {"failed_pass_tolerance": 3}

#: The display precision a float channel is served when the view states none:
#: four decimals read a tune to the 1e-4 a tune measurement resolves, and a
#: corrector kick in mrad below a microradian, where none would show either as 0.
FLOAT_PRECISION = 4

#: The bytes of one element of a served waveform: arrays are served as doubles.
ARRAY_ELEMENT_BYTES = 8

_FLOAT = "float"
_WAVEFORM = "waveform"
# The refresh words of the view's vocabulary, taken from it, never restated.
_, _, _FAST, _EVENT = REFRESH


def apply_safety(config: Mapping[str, Any], view: SimulatorView) -> dict[str, Any]:
    """The runner configuration ``config`` with the view's write safety applied.

    Args:
        config: A configuration shaped as ``Runner.generate_config`` returns
            it: ``{description, prefix, max_array_bytes, variables}``, each
            variable ``{name, pv, mode}`` keyed by address. Not modified.
        view: The simulator view; its channels give each address's role,
            writability and band.

    Returns:
        A new configuration: ``config`` with :data:`SAFETY_KEYS` set, every
        variable's ``mode`` stated, each setpoint's ``value_range`` set to
        the view's, and each float channel's ``precision`` set: the view's
        where it states one, else :data:`FLOAT_PRECISION`.

    Raises:
        ValueError: a waveform channel's array, its declared length times
            :data:`ARRAY_ELEMENT_BYTES`, exceeds ``max_array_bytes``.
    """
    safe: dict[str, Any] = copy.deepcopy(dict(config))
    safe.update(SAFETY_KEYS)
    for address, entry in safe["variables"].items():
        try:
            channel = view.channel(address)
        except KeyError:
            entry["mode"] = "ro"
            continue
        is_setpoint = channel.role == SETPOINT
        # A channel is served writable only as a setpoint the view marks writable.
        entry["mode"] = "rw" if is_setpoint and channel.writable else "ro"
        if is_setpoint:
            entry["value_range"] = (
                list(channel.value_range) if channel.value_range is not None else None
            )
        if channel.value_type == _FLOAT:
            entry["precision"] = (
                channel.precision if channel.precision is not None else FLOAT_PRECISION
            )
        elif channel.value_type == _WAVEFORM and channel.shape is not None:
            _check_array_size(address, channel.shape, safe["max_array_bytes"])
    return safe


def _check_array_size(address: str, shape: Iterable[int], limit: int | str) -> None:
    """Refuse the waveform at ``address`` when its array exceeds ``limit`` bytes."""
    size = math.prod(shape) * ARRAY_ELEMENT_BYTES
    if size > int(limit):
        raise ValueError(
            f"channel {address} is an array of {size} bytes, over the configuration's "
            f"array byte limit of {int(limit)} (max_array_bytes): it would be served "
            "truncated"
        )


def periodic_addresses(view: SimulatorView) -> frozenset[str]:
    """The addresses only a periodic pass publishes, across every model of the view.

    Args:
        view: The simulator view; one from before ``ioc_health_by_model`` names
            no IOC health address.

    Returns:
        The addresses bound to refresh ``periodic``, and the ``HEARTBEAT`` and
        ``UPTIME`` IOC health addresses.
    """
    return _refreshing(view, PERIODIC) | {
        address
        for signals in _ioc_health_by_model(view).values()
        for signal, address in signals.items()
        if signal in PERIODIC_IOC_HEALTH
    }


def _refreshing(view: SimulatorView, refresh: str) -> frozenset[str]:
    """The addresses whose binding refreshes ``refresh``, across every model of the view."""
    return frozenset(
        binding.address
        for binding in view.bindings(served_only=False)
        if binding.refresh == refresh
    )


def ioc_health_addresses(view: SimulatorView) -> frozenset[str]:
    """Every IOC health address of the view, across every model.

    Args:
        view: The simulator view; one from before ``ioc_health_by_model`` names
            no IOC health address.

    Returns:
        The ``ioc_health`` addresses the view lists.
    """
    return frozenset(
        address for signals in _ioc_health_by_model(view).values() for address in signals.values()
    )


def _ioc_health_by_model(view: SimulatorView) -> Mapping[str, Mapping[str, str]]:
    """The view's IOC health addresses, ``{model: {signal: address}}``; none from an older view."""
    ioc_health = getattr(view, "ioc_health_by_model", None)
    return ioc_health() if ioc_health is not None else {}


def fast_addresses(view: SimulatorView) -> frozenset[str]:
    """The addresses whose binding refreshes ``fast``, across every model of the view.

    Args:
        view: The simulator view.

    Returns:
        The bound addresses every timed pass reads and posts.
    """
    return _refreshing(view, _FAST)


def event_addresses(view: SimulatorView) -> frozenset[str]:
    """The addresses whose binding refreshes ``event``, across every model of the view.

    Args:
        view: The simulator view.

    Returns:
        The bound addresses posted when their engine emits them.
    """
    return _refreshing(view, _EVENT)


def setting_records(view: SimulatorView) -> dict[str, frozenset[str]]:
    """The records a setting's write can change, by setting address, across every model.

    A setting reaches its own model alone. There it moves itself, the readback
    it pairs with and what the model computes from all its settings -- every
    record that pairs with no setpoint, and the model's status -- but no other
    setting or the readback paired with it.

    Args:
        view: The simulator view.

    Returns:
        For each setpoint a physics model binds: itself, its readback, the
        model's records that are no setpoint and no readback paired with one,
        and the model's status address.
    """
    bindings = view.bindings(served_only=False)
    settings = [binding for binding in bindings if binding.engine_role == SETPOINT]
    pairs = {binding.address: view.channel(binding.address).pair for binding in settings}
    paired = {pair for pair in pairs.values() if pair is not None}
    computed: dict[str, set[str]] = {}
    for binding in bindings:
        if binding.engine_role != SETPOINT and binding.address not in paired:
            computed.setdefault(binding.model, set()).add(binding.address)
    for model, address in view.status_addresses().items():
        computed.setdefault(model, set()).add(address)
    records: dict[str, frozenset[str]] = {}
    for binding in settings:
        own = {binding.address, *computed.get(binding.model, ())}
        if (pair := pairs[binding.address]) is not None:
            own.add(pair)
        records[binding.address] = frozenset(own)
    return records


def write_records(
    view: SimulatorView, texture_reach: Mapping[str, frozenset[str]]
) -> dict[str, frozenset[str]]:
    """The records any setpoint's write can change, by setpoint address.

    Args:
        view: The simulator view.
        texture_reach: What a write to each settable texture channel can
            change, as the composite's ``texture_reach`` names it.

    Returns:
        ``texture_reach``, and over it the :func:`setting_records` of the
        setpoints a physics model binds.
    """
    return {**texture_reach, **setting_records(view)}


def pass_readback(
    records: Mapping[str, frozenset[str]], written: Iterable[str]
) -> frozenset[str] | None:
    """What a write pass of ``written`` reads back when ``records`` names every value of it.

    Args:
        records: The records each address's write can change, as
            :func:`write_records` names them.
        written: The addresses the pass writes.

    Returns:
        The records of every written address, or ``None`` when the pass
        writes nothing or writes any address ``records`` does not name.
    """
    addresses = set(written)
    if not addresses or not addresses <= records.keys():
        return None
    return frozenset().union(*(records[address] for address in addresses))
