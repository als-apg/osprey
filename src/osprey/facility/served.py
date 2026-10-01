"""The served-model list: which of the facility file's models a render runs.

``simulation.models`` in a render's ``config.yml`` names them. Absent or
``null`` serves every model the facility file holds; ``[]`` and ``[texture]``
serve no physics. ``texture`` is always served and always listed last, after
the physics models sorted by name, so equal configs give equal lists.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from osprey.facility import TEXTURE
from osprey.facility.errors import FacilityBuildError, quoted_slots

__all__ = ["CONFIG_SOURCE", "SIMULATION_MODELS_KEY", "resolve_served"]

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
