"""The model runner's configuration, decided from the simulator view alone.

The runner that serves the composite takes the configuration
``Runner.generate_config`` derives from the composite's variables and hands it
to :func:`apply_safety`, which fixes every key a write's safety depends on:

* the five write-path keys of :data:`SAFETY_KEYS` -- every write is its own
  cycle, a refused write is never echoed and raises an alarm on Channel
  Access, an out-of-band write is clamped into its band, and the runner claims
  no control channel of its own;
* each setpoint's ``value_range``, the band the view's limits records give it;
* each variable's PV ``mode``, stated explicitly: ``rw`` only for a setpoint
  the view marks writable -- the rule the composite itself declares a
  channel settable by -- and ``ro`` for everything else, the status addresses
  included.

:func:`chromaticity_addresses` names the channels wired to a physics model's
chromaticity output, which a write pass leaves for the next periodic pass.

Nothing here imports the serving runtime, so the configuration is decided and
tested in process.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

__all__ = ["SAFETY_KEYS", "apply_safety", "chromaticity_addresses"]

#: The write-path keys every model runner configuration carries.
SAFETY_KEYS: Mapping[str, Any] = {
    "update_rate": 0.0,
    "echo_unconfirmed_writes": False,
    "alarm_on_refused_write": True,
    "clamp_writes": True,
    "control_pvs": False,
}

_SETPOINT = "setpoint"
_CHROMATICITY_OUTPUT = "chromaticity"


def _settable(channel: Mapping[str, Any] | None) -> bool:
    """Whether a view channel is served writable: a setpoint the view marks writable."""
    return (
        channel is not None and channel.get("role") == _SETPOINT and channel.get("writable") is True
    )


def apply_safety(config: Mapping[str, Any], view: Mapping[str, Any]) -> dict[str, Any]:
    """The runner configuration ``config`` with the view's write safety applied.

    Args:
        config: A configuration shaped as ``Runner.generate_config`` returns
            it: ``{description, prefix, max_array_bytes, variables}``, each
            variable ``{name, pv, mode}`` keyed by address. Not modified.
        view: The simulator view's ``variables.json`` document; its
            ``channels`` give each address's role, writability and band.

    Returns:
        A new configuration: ``config`` with :data:`SAFETY_KEYS` set, every
        variable's ``mode`` stated, and each setpoint's ``value_range`` set to
        the view's.
    """
    channels = {str(channel["address"]): channel for channel in view.get("channels", [])}
    safe: dict[str, Any] = copy.deepcopy(dict(config))
    safe.update(SAFETY_KEYS)
    for address, entry in safe["variables"].items():
        channel = channels.get(address)
        entry["mode"] = "rw" if _settable(channel) else "ro"
        if channel is not None and channel.get("role") == _SETPOINT:
            entry["value_range"] = copy.deepcopy(channel.get("value_range"))
    return safe


def chromaticity_addresses(view: Mapping[str, Any]) -> frozenset[str]:
    """The addresses wired to a physics model's chromaticity output.

    A wiring entry reads that output when it names no element and no slices
    and its engine block's ``attribute`` reads the chromaticity.

    Args:
        view: The simulator view's ``variables.json`` document.

    Returns:
        The wired addresses, across every model of the view.
    """
    from osprey.simulation.engines.pyat import OPTICS_ATTRIBUTES

    attributes = {
        attribute
        for attribute, output in OPTICS_ATTRIBUTES.items()
        if output == _CHROMATICITY_OUTPUT
    }
    return frozenset(
        str(entry["address"])
        for model in view.get("models", [])
        for entry in model.get("wiring") or []
        if entry.get("element") is None
        and not entry.get("slices")
        and isinstance(entry.get("engine"), Mapping)
        and entry["engine"].get("attribute") in attributes
    )
