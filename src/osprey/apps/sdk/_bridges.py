"""The Bluesky bridges a deployment runs for a control target: :func:`bridges_for_target`."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType


def bridges_for_target(target: str) -> Mapping[str, str]:
    """Every Bluesky bridge the deployment runs for *target*, lane name to bridge URL.

    Stable: part of the application SDK's public surface. A lane is a rendered
    ``services.<lane>`` block; it serves the target its block declares, or the
    deployment baseline when it declares none (a single-lane deployment). The
    URL resolves as every framework client resolves it, the lane's
    ``<LANE>_BRIDGE_URL`` variable first.

    Args:
        target: A control target, one of
            :data:`osprey_connectors.connector_types.CONTROL_TARGETS`.

    Returns:
        The lanes serving *target*, empty when the deployment runs none.
    """
    from osprey.bluesky_bridge_connection import (
        LANE_KEYS,
        UnknownBlueskyLaneError,
        lane_declared_target,
        resolve_bridge_url,
    )
    from osprey.utils.config_paths import loaded_or_empty
    from osprey_connectors.config_sections import ControlSystemSection
    from osprey_connectors.targets import baseline_target

    config = loaded_or_empty()
    services = config.get("services")
    services = services if isinstance(services, Mapping) else {}
    baseline = baseline_target(ControlSystemSection.read(config.get("control_system") or {}))
    bridges: dict[str, str] = {}
    for lane in LANE_KEYS:
        if not isinstance(services.get(lane), Mapping):
            continue
        if (lane_declared_target(lane, config) or baseline) != target:
            continue
        try:
            bridges[lane] = resolve_bridge_url(lane)
        except UnknownBlueskyLaneError:
            continue
    return MappingProxyType(bridges)
