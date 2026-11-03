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

:func:`periodic_addresses` names the channels the view says refresh
``periodic``, which a write pass leaves for the next periodic pass.

:func:`setting_records` and :func:`pass_readback` name what a write pass of
physics settings reads back. A physics setting reaches its own model alone, and
there moves itself, its readback and what the model computes from its settings.
Every other value stands as the last pass left it -- what moves with time alone,
the next periodic pass publishes -- and reading it back again would make each
write, one per corrector for an apply, cost the whole roster.

:data:`HEALTH_KEYS` holds the defaults of the runner's health record: how many
consecutive failed publishing passes still count as ``degraded``.

Nothing here imports the serving runtime, so the configuration is decided and
tested in process.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from osprey_connectors.simulation.view import SimulatorView

__all__ = [
    "FLOAT_PRECISION",
    "HEALTH_KEYS",
    "SAFETY_KEYS",
    "apply_safety",
    "pass_readback",
    "periodic_addresses",
    "setting_records",
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

_SETPOINT = "setpoint"
_FLOAT = "float"
_PERIODIC = "periodic"


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
    """
    safe: dict[str, Any] = copy.deepcopy(dict(config))
    safe.update(SAFETY_KEYS)
    for address, entry in safe["variables"].items():
        try:
            channel = view.channel(address)
        except KeyError:
            entry["mode"] = "ro"
            continue
        is_setpoint = channel.role == _SETPOINT
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
    return safe


def periodic_addresses(view: SimulatorView) -> frozenset[str]:
    """The addresses whose binding refreshes ``periodic``, across every model of the view.

    Args:
        view: The simulator view.

    Returns:
        The bound addresses the periodic solve alone publishes.
    """
    return frozenset(
        binding.address
        for binding in view.bindings(served_only=False)
        if binding.refresh == _PERIODIC
    )


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
    settings = [binding for binding in bindings if binding.role == _SETPOINT]
    pairs = {binding.address: view.channel(binding.address).pair for binding in settings}
    paired = {pair for pair in pairs.values() if pair is not None}
    computed: dict[str, set[str]] = {}
    for binding in bindings:
        if binding.role != _SETPOINT and binding.address not in paired:
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


def pass_readback(
    records: Mapping[str, frozenset[str]], written: Iterable[str]
) -> frozenset[str] | None:
    """What a write pass of ``written`` reads back when ``records`` names every value of it.

    Args:
        records: The records each address's write can change, as
            :func:`setting_records` names them.
        written: The addresses the pass writes.

    Returns:
        The records of every written address, or ``None`` when the pass
        writes nothing or writes any address ``records`` does not name.
    """
    addresses = set(written)
    if not addresses or not addresses <= records.keys():
        return None
    return frozenset().union(*(records[address] for address in addresses))
