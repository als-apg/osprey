"""The model runner's configuration, decided from the simulator view alone.

The runner that serves the composite takes the configuration
``Runner.generate_config`` derives from the composite's variables and hands it
to :func:`apply_safety`, which fixes every key a write's safety depends on:

* the five write-path keys of :data:`SAFETY_KEYS` -- every write is its own
  cycle, a refused write is never echoed and raises an alarm on Channel
  Access, an out-of-band write is clamped into its band, and the runner claims
  no control channel of its own;
* each setpoint's ``value_range``, the band the view's limits records give it;
* each float channel's display ``precision``, when the view states one;
* each variable's PV ``mode``, stated explicitly: ``rw`` only for a setpoint
  the view marks writable -- the rule the composite itself declares a
  channel settable by -- and ``ro`` for everything else, the status addresses
  included.

:func:`periodic_addresses` names the channels the view says refresh
``periodic``, which a write pass leaves for the next periodic pass.

:data:`HEALTH_KEYS` holds the defaults of the runner's health record: how many
consecutive failed publishing passes still count as ``degraded``.

Nothing here imports the serving runtime, so the configuration is decided and
tested in process.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from osprey_connectors.simulation.view import SimulatorView

__all__ = ["HEALTH_KEYS", "SAFETY_KEYS", "apply_safety", "periodic_addresses"]

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
        the view's, and each float channel's ``precision`` set where the view
        states one.
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
        if channel.value_type == _FLOAT and channel.precision is not None:
            entry["precision"] = channel.precision
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
