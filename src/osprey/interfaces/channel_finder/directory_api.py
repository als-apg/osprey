"""The channel directory routes' one mounting point, and the ``/api/info`` descriptor.

``create_app`` imports this module and includes :data:`router` at
``/api/directory`` only when the ``channel_directory`` section is configured, so an
unconfigured panel has no directory route at all. The routes themselves live in
:mod:`.directory_search_api`, :mod:`.directory_pages_api`,
:mod:`.directory_alarms_api`, :mod:`.directory_archive_api`,
:mod:`.directory_values_api` and :mod:`.directory_fleet_api`; every one but the
fleet route answers through the ``DirectoryService`` the lifespan puts on
``app.state``, and the fleet route through the fleet connection.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter

from osprey.interfaces.channel_finder import (
    directory_alarms_api,
    directory_archive_api,
    directory_fleet_api,
    directory_pages_api,
    directory_search_api,
    directory_values_api,
)
from osprey.profiles.sidecar_panels import CANOPY_PANEL_ID
from osprey.profiles.web_panels import panel_spec_enabled
from osprey.services.channel_directory.resolve import ResolvedSection, predicates
from osprey_connectors.archive_targets import archive_targets

__all__ = ["ariel_configured", "directory_descriptor", "router"]

#: The optional services the descriptor names, in the order it lists them.
_SERVICES = ("alarm_log", "alarm_state", "put_log", "fleet")

router = APIRouter()
router.include_router(directory_search_api.router)
router.include_router(directory_pages_api.router)
router.include_router(directory_alarms_api.router)
router.include_router(directory_archive_api.router)
router.include_router(directory_values_api.router)
router.include_router(directory_fleet_api.router)


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
        resolved: The checked ``channel_directory`` section.
        service: The lifespan's ``DirectoryService``.

    Returns:
        ``{label, target, words, configured_services, canopy, ariel}``;
        ``label`` is the directory's served label, the same with or without a
        source. ``words`` holds each control target's words for a channel and
        a server, by target name, and ``target`` names the current one, so the
        panel shows the words of the target an answer comes from.
        ``configured_services`` names ``fleet`` when the server fleet is on, and
        ``archive`` when the deployment's archive serves some target.
    """
    target = await service.target()
    flags = predicates(resolved)
    ariel = ariel_configured(config)
    services = [name for name in _SERVICES if flags[f"channel_directory_{name}"]]
    if archive_targets(config):
        services.append("archive")
    if ariel:
        services.append("logbook")
    return {
        "label": service.vocabulary(target).directory,
        "target": target,
        "words": {name: words.as_json() for name, words in service.words.items()},
        "configured_services": services,
        "canopy": panel_spec_enabled(_panels(config).get(CANOPY_PANEL_ID)),
        "ariel": ariel,
    }
