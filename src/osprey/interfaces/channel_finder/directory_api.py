"""The PV directory routes' one mounting point, and the ``/api/info`` descriptor.

``create_app`` imports this module and includes :data:`router` at
``/api/directory`` only when the ``pv_directory`` section is configured, so an
unconfigured panel has no directory route at all. The routes themselves live in
:mod:`.directory_search_api`, :mod:`.directory_pages_api`,
:mod:`.directory_alarms_api` and :mod:`.directory_values_api`; every one answers
through the ``DirectoryService`` the lifespan puts on ``app.state``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter

from osprey.interfaces.channel_finder import (
    directory_alarms_api,
    directory_pages_api,
    directory_search_api,
    directory_values_api,
)
from osprey.profiles.sidecar_panels import CANOPY_PANEL_ID
from osprey.profiles.web_panels import panel_spec_enabled
from osprey.services.pv_directory.resolve import ResolvedSection, predicates
from osprey.services.pv_directory.vocabulary import DIRECTORY_LABEL, source_label

__all__ = ["ariel_configured", "directory_descriptor", "router"]

#: The optional services the descriptor names, in the order it lists them.
_SERVICES = ("alarm_log", "alarm_state", "put_log")

router = APIRouter()
router.include_router(directory_search_api.router)
router.include_router(directory_pages_api.router)
router.include_router(directory_alarms_api.router)
router.include_router(directory_values_api.router)


def _panels(config: Mapping[str, Any]) -> Mapping[str, Any]:
    web = config.get("web")
    panels = web.get("panels") if isinstance(web, Mapping) else None
    return panels if isinstance(panels, Mapping) else {}


def ariel_configured(config: Mapping[str, Any]) -> bool:
    """Whether the Logbook panel is selected and an ``ariel`` section exists."""
    section = config.get("ariel")
    has_section = isinstance(section, Mapping) and bool(section)
    return has_section and panel_spec_enabled(_panels(config).get("ariel"))


async def directory_descriptor(
    config: Mapping[str, Any], resolved: ResolvedSection, service: Any
) -> dict[str, Any]:
    """The ``directory`` scope descriptor of ``/api/info``, for the current target.

    Args:
        config: The loaded config.
        resolved: The checked ``pv_directory`` section.
        service: The lifespan's ``DirectoryService``.

    Returns:
        ``{label, configured_services, canopy, ariel}``; ``label`` is the
        directory's served label, the same with or without a source.
    """
    _backend, source, _reason = await service.backend("pv_directory")
    flags = predicates(resolved)
    ariel = ariel_configured(config)
    services = [name for name in _SERVICES if flags[f"pv_directory_{name}"]]
    if ariel:
        services.append("logbook")
    return {
        "label": DIRECTORY_LABEL if source is None else source_label(source.kind, source.service),
        "configured_services": services,
        "canopy": panel_spec_enabled(_panels(config).get(CANOPY_PANEL_ID)),
        "ariel": ariel,
    }
