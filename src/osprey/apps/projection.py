"""Project a deployment's applications into rendered configuration.

The projection is pure: it reads the validated manifests and the rendered-config
mapping, writes nothing, and returns only the keys it derives. The build feeds
what it returns through the same path a profile's ``services:`` entry takes.

``config`` is the rendered-config-shaped mapping of the deployment: the profile's
``apps:`` block (``apps.<name>.{config, services.<key>.{network, env}}``) and
``deployment.port_base`` are the only keys read here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from osprey.apps.errors import AppManifestError
from osprey.apps.manifest import AppManifest, Container, launch_token_env
from osprey.port_layout import FACILITY_MAX, default_port, resolve_port_base

#: The bundled template every ``runtime: osprey`` container renders through.
APP_RUNTIME_TEMPLATE = "osprey.app_runtime"

#: The rendered ``apps.<name>.config`` file, written into each projected service's
#: directory by the build and recorded as ``services.<key>.config_file``.
APP_CONFIG_FILE = "app-config.yml"

#: The deployment directory holding application directories, relative to which an
#: ``image:`` container's fragment directory is named.
_APPS_DIR = "apps"

#: The host every projected port is published on, whatever the deployment's bind
#: address: an application panel is reached through the terminal's proxy only.
_LOOPBACK = "127.0.0.1"


def project_deployment(
    manifests: Iterable[AppManifest],
    config: Mapping[str, Any],
    *,
    claimed_ports: frozenset[int],
) -> dict[str, Any]:
    """Return the ``services.<key>`` entries of every application the profile names.

    Each entry is profile-service shaped (``{"template": ..., "config": {...}}``).
    A ``runtime: osprey`` container renders :data:`APP_RUNTIME_TEMPLATE`; an
    ``image:`` container renders its fragment directory
    ``apps/<name>/compose/<key>``, which owns its binds, volumes, env_file chain,
    depends_on and command. Applications are taken in the order the profile's
    ``apps:`` block names them, containers in manifest order.

    An unpinned container's port is the next free port of the facility band
    (``port_base + 900`` .. ``+ 999``), skipping ``claimed_ports`` and every port
    pinned by a projected container. A collision between a pinned port and the
    ledger is not checked here; the caller checks claimed and projected together.

    Args:
        manifests: The deployment's validated manifests.
        config: The rendered-config mapping (see the module docstring).
        claimed_ports: Every host port the profile already spends.

    Returns:
        ``{"services": {<key>: {"template": str, "config": dict}}}``.

    Raises:
        AppManifestError: The profile names an application with no manifest, two
            applications project the same service key, or the facility band has
            no free port left.
    """
    by_name = {manifest.name: manifest for manifest in manifests}
    apps_block = config.get("apps") or {}
    selected: list[tuple[AppManifest, Mapping[str, Any]]] = []
    for name, profile_app in apps_block.items():
        manifest = by_name.get(name)
        if manifest is None:
            raise AppManifestError(
                name,
                "",
                "missing_manifest",
                f"the profile's apps: block names {name!r} but no "
                f"{_APPS_DIR}/{name}/app.yml exists",
            )
        selected.append((manifest, profile_app or {}))

    owners: dict[str, str] = {}
    for manifest, _ in selected:
        for key in manifest.containers:
            if key in owners:
                raise AppManifestError(
                    manifest.name,
                    f"containers.{key}",
                    "service_key_taken",
                    f"service {key!r} is already projected by application {owners[key]!r}; "
                    "rename the container",
                )
            owners[key] = manifest.name

    ports = _assign_ports(selected, resolve_port_base(config), claimed_ports)

    services: dict[str, Any] = {}
    for manifest, profile_app in selected:
        axes = profile_app.get("services") or {}
        for key, container in manifest.containers.items():
            services[key] = _project_container(
                manifest, key, container, ports[key], axes.get(key) or {}
            )
    return {"services": services}


def _assign_ports(
    selected: list[tuple[AppManifest, Mapping[str, Any]]],
    base: int,
    claimed_ports: frozenset[int],
) -> dict[str, int]:
    taken = set(claimed_ports)
    for manifest, _ in selected:
        taken.update(c.port for c in manifest.containers.values() if c.port is not None)
    band = [default_port("facility", index, base) for index in range(FACILITY_MAX + 1)]
    free = (port for port in band if port not in taken)

    ports: dict[str, int] = {}
    for manifest, _ in selected:
        for key, container in manifest.containers.items():
            if container.port is not None:
                ports[key] = container.port
                continue
            port = next(free, None)
            if port is None:
                raise AppManifestError(
                    manifest.name,
                    f"containers.{key}.port",
                    "port_band_exhausted",
                    f"no free port left in the facility band {band[0]}-{band[-1]}; "
                    "pin this container's port outside the ports the deployment spends",
                )
            ports[key] = port
    return ports


def _project_container(
    manifest: AppManifest,
    key: str,
    container: Container,
    port: int,
    axes: Mapping[str, Any],
) -> dict[str, Any]:
    block: dict[str, Any] = {"app": manifest.name}
    if container.runtime is not None:
        template = APP_RUNTIME_TEMPLATE
        if container.command is not None:
            block["command"] = list(container.command)
    else:
        template = f"{_APPS_DIR}/{manifest.name}/{container.fragment or f'compose/{key}/'}"
        template = template.rstrip("/")
        block["image"] = container.image
    block["port"] = port
    block["url"] = f"http://{_LOOPBACK}:{port}"
    if container.health is not None:
        block["health"] = container.health
    block["env"] = _merge_names(container.env, axes.get("env") or [])
    if "network" in axes:
        block["network"] = axes["network"]
    if container.state is not None:
        block["state"] = container.state
    block["drives_target"] = container.drives_target
    block["control_identity"] = container.control_identity
    block["audit_identity"] = manifest.audit_identity
    block["config_file"] = f"./services/{key}/{APP_CONFIG_FILE}"

    panel = manifest.panel
    if panel is not None and panel.container == key:
        if panel.launch_token:
            block["launch_token_env"] = launch_token_env(manifest.name)
        if panel.roster_grant is not None:
            block["roster_grant"] = {
                "panel_id": panel.id,
                "single_user_alias": panel.roster_grant.single_user_alias,
            }
    return {"template": template, "config": block}


def _merge_names(*lists: Iterable[str]) -> list[str]:
    merged: list[str] = []
    for names in lists:
        for name in names:
            if name not in merged:
                merged.append(name)
    return merged
