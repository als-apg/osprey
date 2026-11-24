"""What a render serves: the facility file's models it runs and the channels they hold.

``simulation.models`` in a render's ``config.yml`` names the models. Absent or
``null`` serves every model the facility file holds; ``[]`` and ``[texture]``
serve no physics. ``texture`` is always served and always listed last, after
the physics models sorted by name, so equal configs give equal lists.

The settings derived from what is served are listed in
:mod:`osprey.facility.settings`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from weakref import WeakKeyDictionary

from osprey_facility import TEXTURE
from osprey_facility.errors import FacilityBuildError, quoted_slots
from osprey_facility.records import ChannelSelection, FacilityRecords

__all__ = ["CONFIG_SOURCE", "SIMULATION_MODELS_KEY", "resolve_served", "served_scope"]

#: The dotted config key that names the served models.
SIMULATION_MODELS_KEY = "simulation.models"

#: The file ``resolve_served`` reads the key from.
CONFIG_SOURCE = "config.yml"


def _invalid(detail: str, remedy: str) -> FacilityBuildError:
    return FacilityBuildError(
        "profile-invalid",
        SIMULATION_MODELS_KEY,
        [CONFIG_SOURCE],
        remedy,
        record_kind="path",
        detail=detail,
    )


def resolve_served(rendered_config: Mapping[str, Any], records: FacilityRecords) -> list[str]:
    """Compute the models one render serves.

    Args:
        rendered_config: The render's ``config.yml``, as a nested mapping.
        records: The build's facility file's records.

    Returns:
        The served model names: the physics models sorted by name, then
        ``texture``.

    Raises:
        FacilityBuildError: ``profile-invalid`` when the key is not a list of
            names, or names a model the facility file does not hold.
    """
    physics = sorted(records.ids("model") - {TEXTURE})
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


def served_scope(records: FacilityRecords, served: list[str]) -> list[Mapping[str, Any]]:
    """The channels the served simulator holds.

    Each served physics model's selection, plus every channel in no model's
    selection, served or not: a channel on a device the file assigns to an
    unserved model is that model's and is not served. With no physics model
    served, every channel.
    """
    channels = records.of_kind("channel")
    physics = [name for name in served if name != TEXTURE]
    if not physics:
        return list(channels)
    scopes = _scopes.setdefault(records, {})
    key = tuple(sorted(physics))
    if key not in scopes:
        scopes[key] = _physics_scope(records, physics)
    return list(scopes[key])


#: The scope of each set of served physics models, per read model: a model's
#: selection scans every channel, and each render of one build asks for the
#: scope of the same models over the same records.
_scopes: WeakKeyDictionary[
    FacilityRecords, dict[tuple[str, ...], tuple[Mapping[str, Any], ...]]
] = WeakKeyDictionary()


def _physics_scope(records: FacilityRecords, physics: list[str]) -> tuple[Mapping[str, Any], ...]:
    """The channels in scope when the ``physics`` models are served, in document order."""
    channels = records.of_kind("channel")
    selection = {
        name: {
            str(channel["id"]) for channel in records.select_channels(ChannelSelection(model=name))
        }
        for name in records.ids("model") - {TEXTURE}
    }
    owned = set().union(*selection.values())
    in_scope = set().union(*(selection[name] for name in physics)) | {
        str(channel["id"]) for channel in channels if str(channel["id"]) not in owned
    }
    return tuple(channel for channel in channels if str(channel["id"]) in in_scope)
