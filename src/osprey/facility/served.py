"""What a render serves: the facility file's models it runs and the channel it probes.

``simulation.models`` in a render's ``config.yml`` names the models. Absent or
``null`` serves every model the facility file holds; ``[]`` and ``[texture]``
serve no physics. ``texture`` is always served and always listed last, after
the physics models sorted by name, so equal configs give equal lists.

The connector blocks whose channels the build serves, the virtual
accelerator's and the stand-in's, prove their target reachable by reading
their ``probe_channel``; a stated one must be a channel of the facility file.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from osprey.facility import TEXTURE
from osprey.facility.errors import FacilityBuildError, quoted_slots
from osprey_connectors.types import LIVE_STANDIN, VIRTUAL_ACCELERATOR

__all__ = [
    "CONFIG_SOURCE",
    "SERVED_PROBE_KEYS",
    "SIMULATION_MODELS_KEY",
    "check_served_probes",
    "resolve_served",
]

#: The dotted config key that names the served models.
SIMULATION_MODELS_KEY = "simulation.models"

#: The file ``resolve_served`` reads the key from.
CONFIG_SOURCE = "config.yml"

#: The probe keys of the connector blocks whose channels the build serves, in
#: the order they are checked: the stand-in copies the virtual accelerator's.
SERVED_PROBE_KEYS: tuple[str, ...] = tuple(
    f"control_system.connector.{block}.probe_channel"
    for block in (VIRTUAL_ACCELERATOR, LIVE_STANDIN)
)

#: The role a channel record that states none has.
_DEFAULT_ROLE = "readback"


def _invalid(detail: str, remedy: str, key: str = SIMULATION_MODELS_KEY) -> FacilityBuildError:
    return FacilityBuildError(
        "profile-invalid",
        key,
        [CONFIG_SOURCE],
        remedy,
        record_kind="path",
        detail=detail,
    )


def resolve_served(rendered_config: Mapping[str, Any], facility: Mapping[str, Any]) -> list[str]:
    """Compute the models one render serves.

    Args:
        rendered_config: The render's ``config.yml``, as a nested mapping.
        facility: The build's facility file.

    Returns:
        The served model names: the physics models sorted by name, then
        ``texture``.

    Raises:
        FacilityBuildError: ``profile-invalid`` when the key is not a list of
            names, or names a model the facility file does not hold.
    """
    physics = sorted({str(m["name"]) for m in facility.get("models", [])} - {TEXTURE})
    known = [*physics, TEXTURE]
    simulation = rendered_config.get("simulation")
    if simulation is None:
        return known
    if not isinstance(simulation, Mapping):
        raise _invalid(
            f"`simulation` is {simulation!r}, not a mapping",
            f"write `{SIMULATION_MODELS_KEY}` as a list of model names",
        )
    requested = simulation.get("models")
    if requested is None:
        return known
    if not isinstance(requested, list) or not all(isinstance(n, str) for n in requested):
        raise _invalid(
            f"`{SIMULATION_MODELS_KEY}` is {requested!r}, not a list of model names",
            f"write `{SIMULATION_MODELS_KEY}` as a list of model names",
        )
    unknown = sorted(set(requested) - set(known))
    if unknown:
        raise _invalid(
            f"{quoted_slots(unknown)} is not a model in the facility file; its models are "
            f"{quoted_slots(known)}",
            f"name only models the facility file holds in `{SIMULATION_MODELS_KEY}`",
        )
    return [*sorted(set(requested) - {TEXTURE}), TEXTURE]


def check_served_probes(rendered_config: Mapping[str, Any], facility: Mapping[str, Any]) -> None:
    """Hold each served block's stated ``probe_channel`` to the facility file.

    A block that states no probe channel passes: the target switch reports a
    missing one itself.

    Args:
        rendered_config: The render's ``config.yml``, as a nested mapping.
        facility: The build's facility file.

    Raises:
        FacilityBuildError: ``profile-invalid`` naming the first key of
            ``SERVED_PROBE_KEYS`` whose address is not a channel of the
            facility file, with the first readback by address as the remedy.
    """
    channels = {str(channel["id"]): channel for channel in facility.get("channels", [])}
    for key in SERVED_PROBE_KEYS:
        address = _dotted(rendered_config, key)
        if address is None or str(address) in channels:
            continue
        readbacks = sorted(
            candidate
            for candidate, channel in channels.items()
            if channel.get("role", _DEFAULT_ROLE) == _DEFAULT_ROLE
        )
        remedy = (
            f"set `{key}` to a readback the facility file holds, such as `{readbacks[0]}`"
            if readbacks
            else f"remove `{key}` from the profile"
        )
        raise _invalid(f"`{address}` is not a channel of the facility file", remedy, key)


def _dotted(mapping: Mapping[str, Any], key: str) -> Any:
    """The value at a dotted ``key`` of a nested mapping, or ``None``."""
    node: Any = mapping
    for part in key.split("."):
        if not isinstance(node, Mapping):
            return None
        node = node.get(part)
    return node
