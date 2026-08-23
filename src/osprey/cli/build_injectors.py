"""Service injectors — wire deploy-time containers into a built project.

Each injector copies a bundled compose template into ``<project>/services/``,
writes the matching ``services.<name>`` block into ``config.yml`` (and
registers it in ``deployed_services``), and prints a post-build hint. The
injectors pair 1:1 with the service dataclasses in
:mod:`osprey.cli.build_profile_schema` (``DispatchConfig``, ``BlueskyConfig``,
``BlueskyWebConfig``, ``VAConfig``, ``NextcloudBridgeProfileConfig``,
``GChatBridgeProfileConfig``) plus ``VAArchiverConfig``, whose block lives in
:mod:`osprey.cli.build_profile_archiver`.
``_copy_service_templates`` / ``_inject_profile_services`` handle the framework
and facility-declared service templates.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from osprey.errors import BuildProfileError
from osprey.utils.config_writer import anchored_append, anchored_put
from osprey.utils.logger import get_logger
from osprey_connectors import types as connector_types

if TYPE_CHECKING:
    from osprey.cli.build_profile import (
        BlueskyConfig,
        BlueskyWebConfig,
        DispatchConfig,
        GChatBridgeProfileConfig,
        NextcloudBridgeProfileConfig,
        VAConfig,
    )
    from osprey.cli.build_profile_archiver import VAArchiverConfig

logger = get_logger("build")

# Spacing between consecutive dispatch workers' host-side ports: worker ``i``
# binds ``worker_port_base + (i - 1) * _WORKER_PORT_STRIDE``. On the compose
# bridge every worker has its own network namespace and they all listen on the
# base port; on the host network they share one namespace, so the ports have to
# fan out. The stride is recorded in the worker's service config (host mode
# only) so the compose template and the host-port preflight derive the same
# ports from the same declared rule rather than each hardcoding the step.
_WORKER_PORT_STRIDE = 1


def _worker_port(worker_port_base: int, index: int) -> int:
    """Return the host-side port of dispatch worker ``index`` (1-based)."""
    return worker_port_base + (index - 1) * _WORKER_PORT_STRIDE


def _locate_pkg_services() -> Path:
    """Locate the OSPREY package's bundled ``templates/services`` directory.

    Prefers the installed ``osprey.templates`` package's location; falls back to
    a path relative to this module for source/editable checkouts where the
    package metadata is unavailable. Callers check ``.is_dir()`` on the result,
    since the directory may be absent in a stripped-down install.
    """
    try:
        import osprey.templates

        return Path(osprey.templates.__file__).parent / "services"
    except (ImportError, AttributeError):
        return Path(__file__).parent.parent / "templates" / "services"


def _user_owned_services(project_path: Path) -> set[str]:
    """Return the claimed service names from config.yml's ``scaffold.user_owned``.

    Entries use the catalog's canonical form (``services/<name>``); the
    returned set holds the bare ``<name>`` part for direct comparison with
    service keys.  Build must not refresh a claimed service template — that
    is the whole point of ``osprey scaffold claim services/<name>``.
    """
    import yaml as _yaml

    config_path = project_path / "config.yml"
    if not config_path.exists():
        return set()
    try:
        with open(config_path, encoding="utf-8") as fh:
            config = _yaml.safe_load(fh) or {}
    except Exception:
        return set()
    owned = config.get("scaffold", {}).get("user_owned", []) or []
    return {str(entry).split("/", 1)[1] for entry in owned if str(entry).startswith("services/")}


def _refresh_service_dir(src_dir: Path, dest_dir: Path, name: str, owned: set[str]) -> bool:
    """Copy a packaged service template into the project, honoring claims.

    Returns True when the template was (re)copied, False when the service is
    user-owned and the project copy was left untouched.
    """
    if name in owned:
        logger.debug(
            "  ⏭  services/%s is user-owned. The project copy was left untouched (scaffold claim).",
            name,
        )
        return False
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(src_dir, dest_dir)
    return True


#: Shared Jinja partials sitting at the packaged ``services/`` root — macro
#: files a service compose template may import. Underscore-prefixed by
#: convention, which is exactly what separates them from the root compose
#: template beside them, and what lets a future partial ship without a code
#: change here.
_SHARED_SERVICE_PARTIALS = "_*.j2"


def _copy_shared_service_partials(dest_services_root: Path) -> int:
    """Copy the shared template partials into a project's ``services/`` tree.

    A service compose template imports these by a path relative to the PROJECT
    root (``{% import "services/_network_axis.j2" %}``), because that is where
    the deploy-time renderer's loader is rooted. The partial therefore has to
    travel with every template that imports it: a project holding the template
    but not the partial fails the render outright, at ``osprey up`` time, with
    ``TemplateNotFound`` — long after the build that should have caught it.

    Called from every path that lands a service template, the whole-project
    build and each single-service injector alike, because an injector may run on
    its own and a project is only self-contained once the partial arrives with
    the first template that needs it. Always refreshed, never skipped when
    present: these are framework internals, not the user-editable service
    directories ``scaffold claim`` protects, so there is no claim to honor.

    Returns:
        Number of partials copied.
    """
    pkg_services = _locate_pkg_services()
    if not pkg_services.is_dir():
        return 0

    partials = sorted(pkg_services.glob(_SHARED_SERVICE_PARTIALS))
    for partial in partials:
        shutil.copy2(partial, dest_services_root / partial.name)
    return len(partials)


#: The one key on a `services.<name>` block that belongs to the AUTHOR rather
#: than to the injector that writes the block. Everything else an injector puts
#: there it derives from its own profile block (a port, a trigger, a path), so
#: replacing the block wholesale is right; `env:` is the exception, because it
#: is written by hand and by nothing else.
_AUTHORED_SERVICE_KEYS = ("env",)


def _carry_authored_keys(services: Any, name: str, block: dict[str, Any]) -> dict[str, Any]:
    """Carry an author-written key across an injector's whole-block replacement.

    Every dedicated injector installs its ``services.<name>`` block with
    :func:`~osprey.utils.config_writer.anchored_put`, which is a whole-VALUE
    assignment: whatever stood at that key is gone. That is deliberate for the
    keys the injector derives (the block is regenerated from the profile on
    every build, and a stale port left behind would be worse than none), but the
    env-passthrough axis is not derived from anything — ``services.<name>.env``
    is a list of host variable NAMES the author wrote, in one of two spellings,
    and both of them land in this same block *before* the injectors run:

    * nested — ``services.<name>.config.env``, written by
      :func:`_inject_profile_services`;
    * dotted — ``config: {"services.<name>.env": [...]}``, merged by
      ``build_cmd._apply_config_overrides``.

    So without this the seven services that have a dedicated injector accept the
    declaration at validation, write it to ``config.yml``, and then silently drop
    it a few steps later — the author sees no error and no passthrough, which is
    the failure the dispatch-pair rejection exists to prevent, one layer wider.
    The macro that renders the axis (``templates/services/_env_axis.j2``) reads
    exactly this key, so carrying it forward is all that is needed for the seven
    to behave like the services with no injector at all.

    Copied by reference and only when present, so a service that declares
    nothing renders byte-for-byte what it rendered before: no empty ``env: []``
    appears in any config.yml that did not already carry one.

    A key the new block already carries is left alone, which is what keeps this
    usable from :func:`_inject_profile_services` too. That injector builds its
    block from ``svc_def.config`` — the nested spelling — while the dotted one
    is already sitting in the block being replaced, so a rule of "carry
    unconditionally" would silently promote the dotted spelling over the nested
    one for every profile service. Filling only the gap leaves the existing
    precedence exactly where it was and fixes only the case where the value
    would otherwise be lost.

    Args:
        services: The ``services`` mapping being written into (a ruamel
            ``CommentedMap`` in practice), holding whatever earlier steps wrote.
        name: The service key about to be replaced.
        block: The freshly built block. Mutated in place and returned, so this
            wraps an ``anchored_put`` argument without restructuring the caller.

    Returns:
        ``block``, with any authored key the previous block carried and the new
        one does not.
    """
    previous = services.get(name) if hasattr(services, "get") else None
    if not isinstance(previous, Mapping):
        return block
    for key in _AUTHORED_SERVICE_KEYS:
        if key in previous and key not in block:
            block[key] = previous[key]
    return block


def _copy_service_templates(project_path: Path) -> int:
    """Copy service compose templates from the OSPREY package into the project.

    Copies each service's compose template directory from the package to the
    project's ``services/`` tree for the UNION of ``deployed_services`` and
    every service merely DECLARED under ``services:`` that ships a package
    template.  This makes the project self-contained so that ``osprey up``
    works directly from the deployment repo, and — crucially — bundles
    opt-in add-ons (declared but not deployed) so they can be switched on later
    via a ``deployed_services`` edit + ``osprey up`` without rebuilding.
    A bundled-but-not-deployed template is inert until deployed.

    Returns:
        Number of service template directories copied.
    """
    from ruamel.yaml import YAML

    config_path = project_path / "config.yml"
    if not config_path.exists():
        return 0

    yaml = YAML()
    with open(config_path) as fh:
        config = yaml.load(fh)

    # Locate the package's service templates directory
    pkg_services = _locate_pkg_services()

    if not pkg_services.is_dir():
        logger.warning("Service templates directory not found. Skipping the service copy.")
        return 0

    dest_services_root = project_path / "services"
    dest_services_root.mkdir(exist_ok=True)
    _copy_shared_service_partials(dest_services_root)

    # Always copy the root compose template so `osprey up` works even
    # for presets with no deployed_services (the renderer references it
    # unconditionally; without it deploy fails with TemplateNotFound).
    root_template = pkg_services / "docker-compose.yml.j2"
    if root_template.exists():
        shutil.copy2(root_template, dest_services_root / "docker-compose.yml.j2")

    services_config = config.get("services", {})

    # Bundle the UNION of deployed services and every service merely DECLARED
    # under `services:`.  deployed_services come first (preserving prior
    # behavior exactly), then any declared key not already present.  Bundling a
    # declared-but-not-deployed template keeps it inert until deployed, so
    # opt-in add-ons (e.g. the openobserve telemetry backend) can be turned on
    # later via a `deployed_services` edit + `osprey up`, no rebuild.
    deployed = [str(s) for s in config.get("deployed_services", [])]
    names = list(deployed)
    for declared in services_config:
        name = str(declared)
        if name not in names:
            names.append(name)

    if not names:
        return 0

    deployed_set = set(deployed)
    owned = _user_owned_services(project_path)

    count = 0
    for name in names:
        # Resolve package source directory
        parts = name.split(".")
        if parts[0] == "osprey" and len(parts) == 2:
            src_dir = pkg_services / parts[1]
        elif len(parts) == 1:
            src_dir = pkg_services / name
        else:
            logger.warning(
                "Skipping service %r. Its name is not a supported spelling for a template copy.",
                name,
            )
            continue

        if not src_dir.is_dir():
            # A declared-but-not-deployed service may legitimately ship no
            # package template (e.g. facility-injected elsewhere) — skip it
            # silently.  Only warn when a *deployed* service is missing its
            # template, which would break `osprey up`.
            if name in deployed_set:
                logger.warning("No package template for service %r at %s", name, src_dir)
            continue

        # Determine destination from the service config's path field
        svc_config = services_config.get(parts[-1], {})
        dest_rel = svc_config.get("path", f"./services/{parts[-1]}")
        dest_dir = project_path / dest_rel.lstrip("./")

        if _refresh_service_dir(src_dir, dest_dir, parts[-1], owned):
            count += 1

    return count


def _inject_profile_services(
    profile_dir: Path, project_path: Path, services: dict[str, Any]
) -> int:
    """Copy facility-defined service templates and register them in config.yml.

    For each service declared in the profile's ``services:`` section:
    1. Copies the template directory to ``{project}/services/{name}/``, resolving
       ``template: osprey.<name>`` to the framework's bundled template and any
       other value relative to the profile directory
    2. Writes ``services.{name}`` config entries to config.yml
    3. Appends the service to ``deployed_services``

    This lets facilities define their own containers (Typesense, Redis, etc.)
    alongside OSPREY's built-in services (PostgreSQL).

    Returns:
        Number of profile services injected.
    """
    from ruamel.yaml import YAML

    if not services:
        return 0

    config_path = project_path / "config.yml"
    if not config_path.exists():
        return 0

    yaml = YAML()
    yaml.preserve_quotes = True
    with open(config_path) as fh:
        config = yaml.load(fh)

    dest_services_root = project_path / "services"
    dest_services_root.mkdir(exist_ok=True)
    _copy_shared_service_partials(dest_services_root)
    owned = _user_owned_services(project_path)

    count = 0
    for name, svc_def in services.items():
        # Resolve the template source. ``osprey.<name>`` selects the framework's
        # own bundled template (same form _copy_service_templates accepts, and the
        # form BuildProfile.validate deliberately leaves unresolved); anything else
        # is a directory shipped inside the profile.
        parts = svc_def.template.split(".")
        if parts[0] == "osprey" and len(parts) == 2:
            src_dir = _locate_pkg_services() / parts[1]
        else:
            src_dir = profile_dir / svc_def.template

        if not src_dir.is_dir():
            # Warn and skip rather than raise, mirroring _copy_service_templates:
            # one unresolvable service must not abort the whole build. A missing
            # profile-relative dir is already a validation error, so in practice
            # this only fires for an unknown ``osprey.<name>``.
            logger.warning("No template for profile service %r at %s. Skipping it.", name, src_dir)
            continue

        # Copy template directory (skipped for claimed services)
        dest_dir = dest_services_root / name
        _refresh_service_dir(src_dir, dest_dir, name, owned)

        # Register service config in config.yml
        if "services" not in config:
            config["services"] = {}
        svc_config = {"path": f"./services/{name}"}
        svc_config.update(svc_def.config)
        anchored_put(
            config["services"], name, _carry_authored_keys(config["services"], name, svc_config)
        )

        # Add to deployed_services
        deployed = config.get("deployed_services", [])
        if name not in [str(s) for s in deployed]:
            anchored_append(deployed, name)
            config["deployed_services"] = deployed

        count += 1

    with open(config_path, "w") as fh:
        yaml.dump(config, fh)

    return count


def _inject_dispatch(dispatch: DispatchConfig, profile_dir: Path, project_path: Path) -> None:
    """Wire the event-dispatch feature into a built project.

    1. Resolve and copy the triggers file to ``<project>/triggers.yml``.
    2. Copy the bundled event_dispatcher + dispatch_worker compose templates
       into ``<project>/services/``.
    3. Write ``services.{event_dispatcher,dispatch_worker}`` config + register
       both in ``deployed_services``.
    4. Print a post-build hint (dashboard URL + sample curl + image prerequisite).

    The pair shares one network: ``dispatch.network`` is written into BOTH
    service configs, since a dispatcher on the compose bridge and workers on the
    host network could not reach each other. On the host network the addresses
    the build emits change with it — the dispatcher reaches a worker at
    ``localhost``, not at a compose DNS name — so step 1a also rewrites the
    copied triggers file's ``dispatch_target``, and the worker's per-index port
    rule is recorded alongside it. Both are written only in host mode: an
    unset (bridge) ``dispatch.network`` leaves every artifact exactly as it was
    before the axis existed.

    Args:
        dispatch: Validated dispatch configuration from the build profile.
        profile_dir: Directory containing the build profile (triggers source).
        project_path: Root of the built project.

    Raises:
        BuildProfileError: If the configured triggers file cannot be resolved.
    """
    from ruamel.yaml import YAML

    from osprey.cli.build_profile import _triggers_dir
    from osprey.cli.build_profile_schema import DEFAULT_NETWORK_MODE

    # The default is spelled twice on purpose (here and on ``DispatchConfig``):
    # every read site of a build knob says what an omitted knob means, so this
    # function is readable on its own and stays correct for a hand-built
    # DispatchConfig that never went through the profile loader.
    network = dispatch.network or DEFAULT_NETWORK_MODE
    on_host_network = network == "host"

    # 1. Resolve + copy triggers file (profile-relative path or bundled triggers name).
    if (profile_dir / dispatch.triggers).is_file():
        triggers_src = profile_dir / dispatch.triggers
    elif (_triggers_dir() / dispatch.triggers).is_file():
        triggers_src = _triggers_dir() / dispatch.triggers
    else:
        raise BuildProfileError(f"dispatch.triggers not found: {dispatch.triggers!r}")
    triggers_dest = project_path / "triggers.yml"
    shutil.copy2(triggers_src, triggers_dest)

    # 1a. Make the preset the single source of truth for pool limits. The bundled
    # triggers file hardcodes its own dispatcher.max_concurrent_runs/max_queue_depth
    # (the dispatcher reads them from triggers.yml at runtime), so a profile that
    # overrides dispatch.max_concurrent_runs/max_queue_depth would otherwise be
    # silently ignored. Patch the copied file's dispatcher block to match the
    # validated DispatchConfig.
    _trigger_yaml = YAML()
    _trigger_yaml.preserve_quotes = True
    with open(triggers_dest) as fh:
        triggers_doc = _trigger_yaml.load(fh)
    #
    # The same patch carries the routing address. ``dispatch_target`` in a
    # triggers file names the first worker by its compose service DNS name
    # (``http://dispatch-worker-1:9190``), which resolves only on the compose
    # bridge. On the host network the dispatcher and the worker share the host's
    # namespace, so the worker is reachable at ``localhost`` on the port it
    # binds — worker 1's, i.e. the base port. Left alone in bridge mode, so a
    # facility-authored target keeps whatever it points at.
    if triggers_doc is not None:
        dispatcher_block = triggers_doc.setdefault("dispatcher", {})
        dispatcher_block["max_concurrent_runs"] = dispatch.max_concurrent_runs
        dispatcher_block["max_queue_depth"] = dispatch.max_queue_depth
        if on_host_network:
            worker_one_port = _worker_port(dispatch.worker_port_base, 1)
            dispatcher_block["dispatch_target"] = f"http://localhost:{worker_one_port}"
        with open(triggers_dest, "w") as fh:
            _trigger_yaml.dump(triggers_doc, fh)

    # 2. Copy bundled compose templates (located the same way as service templates).
    pkg_services = _locate_pkg_services()

    dest_services_root = project_path / "services"
    dest_services_root.mkdir(exist_ok=True)
    _copy_shared_service_partials(dest_services_root)
    owned = _user_owned_services(project_path)

    for name in ("event_dispatcher", "dispatch_worker"):
        src_dir = pkg_services / name
        if not src_dir.is_dir():
            logger.warning("No package template for dispatch service %r at %s", name, src_dir)
            continue
        dest_dir = dest_services_root / name
        _refresh_service_dir(src_dir, dest_dir, name, owned)

    # 3. Write config.yml entries + register in deployed_services.
    config_path = project_path / "config.yml"
    if not config_path.exists():
        logger.warning("config.yml not found. Skipping the dispatch config registration.")
        return

    yaml = YAML()
    yaml.preserve_quotes = True
    with open(config_path) as fh:
        config = yaml.load(fh)

    # No ``image`` key on either service, so each falls to its compose default:
    # the event-dispatcher builds the project's <project>-dispatch:local image (its
    # own compose ``build:`` block), and the dispatch worker runs <project>:local —
    # the project image ``osprey up`` builds from the project Dockerfile
    # (the worker has no build block of its own, to avoid racing the dispatcher).
    # Override with OSPREY_DISPATCH_IMAGE/OSPREY_WORKER_IMAGE, or set
    # ``services.<name>.image`` here, to use a prebuilt/published image.
    config.setdefault("services", {})
    dispatcher_config: dict[str, Any] = {
        "path": "./services/event_dispatcher",
        "port": dispatch.dispatcher_port,
        "facility_name": dispatch.facility_name,
        "pv_strip_prefix": dispatch.pv_strip_prefix,
        # Copy the project's triggers.yml into the service build context so the
        # compose ``./triggers.yml`` bind-mount resolves to a file (otherwise the
        # container runtime auto-creates an empty directory at the mount source).
        "additional_dirs": [{"src": "triggers.yml", "dst": "triggers.yml"}],
    }
    worker_config: dict[str, Any] = {
        "path": "./services/dispatch_worker",
        "worker_count": dispatch.worker_count,
        "worker_port_base": dispatch.worker_port_base,
        "workspace_mode": dispatch.workspace_mode,
        "timeout_sec": dispatch.timeout_sec,
        "inactivity_sec": dispatch.inactivity_sec,
    }
    if on_host_network:
        # One profile knob, both halves: the compose templates read the mode off
        # their own service block, so the single ``dispatch.network`` value is
        # written into each. A directly-authored ``network:`` on either half is
        # rejected by profile validation, which runs before this — nothing
        # authored can be overwritten here.
        #
        # Written only off the default: with the axis unset, config.yml is
        # byte-for-byte what it was before the knob existed, and the templates
        # fall back to the same default the schema declares.
        dispatcher_config["network"] = network
        worker_config["network"] = network
        # Per-worker port derivation, recorded rather than hardcoded in the
        # template: worker ``i`` binds ``worker_port_base + (i - 1) * stride``.
        # The compose render (per-worker DISPATCH_WORKER_PORT + healthcheck) and
        # the host-port preflight both derive from these three keys.
        worker_config["worker_port_stride"] = _WORKER_PORT_STRIDE
    anchored_put(config["services"], "event_dispatcher", dispatcher_config)
    anchored_put(config["services"], "dispatch_worker", worker_config)
    deployed = config.get("deployed_services", []) or []
    for name in ("event_dispatcher", "dispatch_worker"):
        if name not in [str(s) for s in deployed]:
            anchored_append(deployed, name)
    config["deployed_services"] = deployed

    # Derive web.panels.events.url from dispatcher_port so the port is a single
    # source of truth.  Write only if the profile has not already set an explicit
    # ``web.panels.events.url`` via a config override (merged earlier in the
    # build); explicit overrides take precedence.
    #
    # Emit a bare-host ``url`` plus a ``/dashboard`` ``path`` (rather than baking
    # ``/dashboard`` into ``url``) to match the custom-panel proxy convention:
    # the web terminal composes ``url.rstrip('/') + '/' + path``, so a path baked
    # into ``url`` double-prefixes sub-routes. ``setdefault`` on ``path`` honors a
    # facility that pinned its own ``web.panels.events.path``.
    existing_events_url = config.get("web", {}).get("panels", {}).get("events", {}).get("url", "")
    if not existing_events_url:
        panels = config.setdefault("web", {}).setdefault("panels", {})
        events_panel = panels.get("events")
        if events_panel is None:
            # Put the complete panel in one shot: anchored_put re-anchors a
            # section comment beneath the new entry, which needs the entry's
            # full subtree to exist at insertion time.
            anchored_put(
                panels,
                "events",
                {"url": f"http://localhost:{dispatch.dispatcher_port}", "path": "/dashboard"},
            )
        else:
            anchored_put(events_panel, "url", f"http://localhost:{dispatch.dispatcher_port}")
            if "path" not in events_panel:
                anchored_put(events_panel, "path", "/dashboard")

    with open(config_path, "w") as fh:
        yaml.dump(config, fh)

    # 4. Post-build hint.
    logger.debug(
        "  ✓ Injected event dispatch (%d worker(s), port %d)",
        dispatch.worker_count,
        dispatch.dispatcher_port,
    )
    logger.debug("    Dashboard:  http://localhost:%d/dashboard", dispatch.dispatcher_port)
    logger.debug(
        "    Token:      `osprey up` writes EVENT_DISPATCHER_TOKEN to .env; "
        "load it with: export $(grep -E '^EVENT_DISPATCHER_TOKEN=' .env | xargs)"
    )
    logger.debug(
        "    Try it:     curl -X POST http://localhost:%d/webhook/hello-dispatch "
        '-H "Authorization: Bearer $EVENT_DISPATCHER_TOKEN" '
        "-H 'Content-Type: application/json' -d '{}'",
        dispatch.dispatcher_port,
    )
    logger.debug(
        "    Images:     `osprey up` builds the dispatch image and the "
        "worker's project image locally (first run is slow). Use `--dev` to bake "
        "in your local osprey checkout; set OSPREY_DISPATCH_IMAGE/OSPREY_WORKER_IMAGE "
        "to use a published image."
    )


#: The two control-system targets a bluesky plan lane can serve, keyed by the
#: ``control_system.type`` each is spelled with in a rendered config.yml — taken
#: from the connector package's constants rather than respelled here, so a
#: renamed type cannot leave this mapping silently matching nothing.
#: ``MOCK`` and ``DOOCS`` are deliberately absent: they are not switch targets,
#: so a deployment on one of them has no second lane to render.
_LANE_TARGET_BY_CONTROL_SYSTEM_TYPE = {
    connector_types.EPICS: "live",
    connector_types.VIRTUAL_ACCELERATOR: "va",
}

#: Lane 1 always keeps the historical service key. Lane 2 is named for the
#: TARGET it serves, never for its index — a lane's identity is fixed at render
#: time, and ``bluesky_2`` would leave every reader (compose, host-port map,
#: approval prompt) to look up which machine it talks to.
_SECOND_LANE_SERVICE_KEY = {"live": "bluesky_live", "va": "bluesky_va"}

#: Address of the live lane's control-system gateway, as a REQUIRED compose
#: variable rather than a bare ``${EPICS_CA_NAME_SERVERS}`` passthrough.
#:
#: The distinction is the whole point: compose interpolates an unset bare
#: reference to the empty string, so the lane would come up looking healthy and
#: quietly search for PVs at nowhere — a live lane that silently talks to
#: nothing is the failure this string exists to prevent. ``:?`` makes compose
#: refuse to start the deployment and say which variable is missing. Same
#: judgement, and the same spelling, as the archiver recorder's external-IOC
#: branch (``templates/services/archiver_recorder/docker-compose.yml.j2``).
#:
#: The VA lane has no counterpart: its gateway is the co-deployed
#: ``virtual-accelerator`` service at a port the build already knows, so there
#: is nothing for an operator to supply and nothing to refuse over.
_LIVE_LANE_CA_NAME_SERVERS = (
    "${EPICS_CA_NAME_SERVERS:?set EPICS_CA_NAME_SERVERS to <host>:<port> of the live "
    "control-system gateway the bluesky live lane queues plans against}"
)


def _baseline_lane_target(config: Any, virtual_accelerator: VAConfig | None) -> str:
    """Resolve which target the deployment baseline lane serves.

    Read from the rendered ``control_system.type`` rather than from the profile,
    because that is the value every other holder resolves the deployment
    baseline from — injectors run after ``_apply_config_overrides``, so the key
    is already final by the time this is called.

    The VA service is checked here too, because the two questions have one
    answer: on a LIVE baseline the second lane is the VA lane, and a VA lane
    addresses a soft-IOC that has to actually be part of the deployment. There
    is no mirror check on a VA baseline — the second lane is then the LIVE lane,
    whose gateway is a facility address no build can verify, which is exactly
    why its addressing is the ``${EPICS_CA_NAME_SERVERS:?}`` requirement that
    fails loudly at ``osprey up`` instead (see
    :data:`_LIVE_LANE_CA_NAME_SERVERS`).

    Args:
        config: The loaded config.yml mapping.
        virtual_accelerator: The profile's ``virtual_accelerator:`` block, or
            ``None`` when the profile deploys no VA service.

    Returns:
        ``"live"`` or ``"va"``.

    Raises:
        BuildProfileError: If the baseline is a control-system type that is not
            a switch target, or if a live baseline's VA lane would have no VA
            service to address.
    """
    control_system = config.get("control_system") or {}
    cs_type = (
        str(control_system.get("type") or "mock") if hasattr(control_system, "get") else "mock"
    )
    target = _LANE_TARGET_BY_CONTROL_SYSTEM_TYPE.get(cs_type)
    if target is None:
        raise BuildProfileError(
            f"bluesky.second_lane needs a switchable deployment baseline: the lane "
            f"pair is {'/'.join(sorted(_SECOND_LANE_SERVICE_KEY))}, and "
            f"control_system.type is {cs_type!r}. Set it to one of "
            f"{', '.join(repr(t) for t in sorted(_LANE_TARGET_BY_CONTROL_SYSTEM_TYPE))}, "
            f"or drop bluesky.second_lane and deploy the single lane."
        )
    if target == "live" and virtual_accelerator is None:
        raise BuildProfileError(
            f"bluesky.second_lane on a {cs_type!r} baseline renders a "
            f"{_SECOND_LANE_SERVICE_KEY['va']} lane that queues plans against the "
            f"co-deployed virtual accelerator, and this profile deploys none. Add a "
            f"'virtual_accelerator:' block to the profile, or drop "
            f"bluesky.second_lane and deploy the single live lane."
        )
    return target


def _inject_bluesky(
    bluesky: BlueskyConfig,
    project_path: Path,
    virtual_accelerator: VAConfig | None = None,
) -> None:
    """Wire the Bluesky bridge feature into a built project.

    1. Copy the bundled ``templates/services/bluesky/`` compose template into
       ``<project>/services/bluesky/``.
    2. Write one ``services.<lane>`` config block per PLAN LANE + register each
       in ``deployed_services`` (so ``find_service_config`` resolves them,
       mirroring ``_inject_dispatch``).
    3. Print a post-build hint (launch-token env var + image prerequisite).

    A project deploys ONE lane — one bluesky-bridge process — unless the profile
    sets ``bluesky.second_lane``, in which case it deploys two: one per
    control-system target, so a session switched away from the deployment
    baseline still has a lane to queue plans on. Lane 1 keeps the ``bluesky``
    service key and serves the baseline target; lane 2 is a sibling block named
    for the target it serves (``bluesky_live`` / ``bluesky_va``), on its own
    derived port. Tiled is the one shared component and stays on lane 1.

    Still simpler than ``_inject_dispatch``: no triggers file to resolve, and
    the lane count is a fixed one-or-two rather than a worker loop. The
    ``bluesky`` MCP server itself is a separate, always-available framework
    server (see ``osprey.mcp_server.bluesky``); this step only wires the
    *deploy-time* containers that server talks to over HTTP.

    Args:
        bluesky: Validated bluesky configuration from the build profile.
        project_path: Root of the built project.
        virtual_accelerator: The profile's ``virtual_accelerator:`` block, read
            ONLY to check that a live baseline's VA lane has a soft-IOC to
            address. Passed rather than read back from ``config.yml`` because
            ``_inject_va`` writes that block AFTER this injector runs, so the
            rendered config cannot answer the question yet. Defaults to
            ``None`` — the right answer for every single-lane caller, which
            never reaches the check.

    Raises:
        BuildProfileError: If ``second_lane`` is set on a deployment whose
            baseline control-system type is not a switch target, or on a live
            baseline that deploys no virtual accelerator.
    """
    from ruamel.yaml import YAML

    # 1. Copy the bundled compose template (located the same way as service templates).
    pkg_services = _locate_pkg_services()

    src_dir = pkg_services / "bluesky"
    if not src_dir.is_dir():
        logger.warning("No package template for bluesky service at %s", src_dir)
        return

    dest_services_root = project_path / "services"
    dest_services_root.mkdir(exist_ok=True)
    _copy_shared_service_partials(dest_services_root)
    # ONE template directory serves every lane: the lanes differ in their config
    # block (port, target, addressing), not in their compose source, and a second
    # copied directory would be a second thing to keep in step with the first.
    dest_dir = dest_services_root / "bluesky"
    _refresh_service_dir(src_dir, dest_dir, "bluesky", _user_owned_services(project_path))

    # 2. Write config.yml entries + register in deployed_services.
    config_path = project_path / "config.yml"
    if not config_path.exists():
        logger.warning("config.yml not found. Skipping the bluesky config registration.")
        return

    yaml = YAML()
    yaml.preserve_quotes = True
    with open(config_path) as fh:
        config = yaml.load(fh)

    # No ``image`` key: the service builds the local bluesky-bridge image on
    # first ``osprey up``. Override with OSPREY_BLUESKY_BRIDGE_IMAGE, or
    # set ``services.bluesky.image`` here, to use a prebuilt/published image.
    config.setdefault("services", {})
    svc_config: dict[str, Any] = {
        "path": "./services/bluesky",
        "port": bluesky.port,
        "tiled_enabled": bluesky.tiled_enabled,
        "tiled_port": bluesky.tiled_port,
    }
    if bluesky.plan_dir:
        # Only written when configured — its absence is what keeps a
        # bridge-only deploy (no facility plan directory) free of the mount:
        # the compose template's {% if %} guard reads this same key, so an
        # unset plan_dir means no mount and no BLUESKY_PLAN_DIRS env var at all.
        svc_config["plan_dir"] = bluesky.plan_dir
    if bluesky.excluded_plans:
        # Only written when non-empty — its absence keeps a deploy with no
        # exclusions free of the variable: the compose template's
        # {% if %} guard reads this same key, so an empty list means no
        # BLUESKY_EXCLUDED_PLANS env var at all. The os.pathsep join is done
        # Python-side because the Jinja render context has no `os` module.
        svc_config["excluded_plans"] = os.pathsep.join(bluesky.excluded_plans)

    lanes: list[tuple[str, dict[str, Any]]] = [("bluesky", svc_config)]
    if bluesky.second_lane:
        # Two lanes: lane 1 serves the deployment baseline, lane 2 the other
        # target. Both carry `target` — a lane's identity is fixed here, at
        # render time, and the bridge reads it rather than inferring a session
        # target it is never told about. The keys below are written ONLY on a
        # two-lane deploy, which is what keeps a single-lane block byte-for-byte
        # what it has always been.
        baseline = _baseline_lane_target(config, virtual_accelerator)
        second = "va" if baseline == "live" else "live"
        second_config: dict[str, Any] = {
            "path": "./services/bluesky",
            "port": bluesky.second_lane_port(),
        }
        # Plans are a property of the facility, not of a target: both lanes
        # load the same plan directory and hide the same exclusions.
        if bluesky.plan_dir:
            second_config["plan_dir"] = bluesky.plan_dir
        if bluesky.excluded_plans:
            second_config["excluded_plans"] = os.pathsep.join(bluesky.excluded_plans)
        # No tiled keys: tiled is shared, and lane 1 is where it lives.
        for lane_config, lane_target in ((svc_config, baseline), (second_config, second)):
            lane_config["target"] = lane_target
            if lane_target == "live":
                lane_config["ca_name_servers"] = _LIVE_LANE_CA_NAME_SERVERS
        lanes.append((_SECOND_LANE_SERVICE_KEY[second], second_config))

    deployed = config.get("deployed_services", []) or []
    for lane_key, lane_config in lanes:
        anchored_put(
            config["services"],
            lane_key,
            _carry_authored_keys(config["services"], lane_key, lane_config),
        )
        if lane_key not in [str(s) for s in deployed]:
            anchored_append(deployed, lane_key)
    config["deployed_services"] = deployed

    with open(config_path, "w") as fh:
        yaml.dump(config, fh)

    # 3. Post-build hint.
    logger.debug("  ✓ Injected Bluesky bridge (port %d)", bluesky.port)
    if len(lanes) > 1:
        second_key, second_block = lanes[1]
        logger.debug(
            "    Lanes:      bluesky serves the %s target, %s serves %s (port %d)",
            svc_config["target"],
            second_key,
            second_block["target"],
            second_block["port"],
        )
        if any("ca_name_servers" in block for _, block in lanes):
            logger.debug(
                "    Live lane:  set EPICS_CA_NAME_SERVERS to <host>:<port> of the "
                "live control-system gateway; `osprey up` refuses without it."
            )
    logger.debug(
        "    Token:      `osprey up` writes BLUESKY_LAUNCH_TOKEN to .env; "
        "a host-run agent's queue tools read it automatically. Deployed web "
        "terminals receive it only for personas entitled to it; the rest are "
        "refused with launch_token_required and hand arming to the operator."
    )
    logger.debug(
        "    Images:     `osprey up` builds the bluesky-bridge image locally "
        "(first run is slow). Use `--dev` to bake in your local osprey checkout; "
        "set OSPREY_BLUESKY_BRIDGE_IMAGE to use a published image."
    )
    if bluesky.tiled_enabled:
        logger.debug("    Tiled:      enabled on port %d", bluesky.tiled_port)
    if bluesky.plan_dir:
        logger.debug(
            "    Plan dir:   %s mounted read-only into the bridge; its plans "
            "load as the 'facility' trust tier (BLUESKY_PLAN_DIRS)",
            bluesky.plan_dir,
        )


#: One gateway row of the VA connector table the injector installs: localhost
#: over CA name-server (TCP) mode — the one host<->container CA configuration
#: that works across container runtimes. Deliberately NO ``port``: with it unset
#: the connector follows ``services.virtual_accelerator.port``, so the deployed
#: soft-IOC's port is stated once and the two cannot drift (Ledger 56).
_VA_GATEWAY_ROW: dict[str, Any] = {"address": "localhost", "use_name_server": True}

#: Path of the block the VA gateways live under, in the nested spelling a
#: rendered ``config.yml`` is read in. Its last element is also the connector
#: type, which is what makes it the block the factory reads.
_VA_CONNECTOR_PATH = ("control_system", "connector", connector_types.VIRTUAL_ACCELERATOR)


def _ensure_va_connector_gateways(config: Any) -> bool:
    """Give a config that has no VA gateway table the canonical one.

    A session can only be switched to a target the rendered config already
    describes — the switch never edits ``config.yml`` — so a project that
    deploys the VA service but carries no ``control_system.connector.
    virtual_accelerator.gateways`` has a service running and nothing able to
    point at it. Projects built from the current generic template render that
    block themselves; this covers the configs that predate it, and attached
    projects whose ``config.yml`` was written by hand.

    Never clobbers. A ``gateways`` key that is already there belongs to whoever
    wrote it and is left exactly as written — including a table missing one of
    the two roles, because the presence of the key means the author owns the
    table and "filling in" the role they left out is an edit they did not ask
    for. Only the levels that do not exist are created, in one
    :func:`~osprey.utils.config_writer.anchored_put` so a section comment
    trailing the last entry stays anchored where it was.

    ``probe_channel`` is deliberately NOT written. The channel a VA serves comes
    from that project's own machine model, so no value could be derived here,
    and a placeholder would make the target look eligible while naming a channel
    nothing serves. Eligibility reports the missing probe_channel as the reason
    the target is not switchable yet, which is the honest thing for it to say.

    Args:
        config: The round-trip-loaded ``config.yml`` document, mutated in place.

    Returns:
        Whether anything was written.
    """
    if not isinstance(config, Mapping):
        return False

    node: Any = config
    for depth, key in enumerate(_VA_CONNECTOR_PATH):
        if key not in node:
            # Missing from here down: the remaining path and the table are one
            # value, so one write installs them and one comment re-anchors.
            value: dict[str, Any] = {"gateways": _va_gateways()}
            for parent in reversed(_VA_CONNECTOR_PATH[depth + 1 :]):
                value = {parent: value}
            anchored_put(node, key, value)
            return True
        child = node[key]
        if not isinstance(child, Mapping):
            # Something is there that is not a section. Writing through it would
            # replace whatever the author meant by it, which is the one thing
            # this step must never do.
            logger.warning(
                "'%s' in config.yml is not a mapping; leaving the "
                "virtual_accelerator gateways to whoever wrote it.",
                ".".join(_VA_CONNECTOR_PATH[: depth + 1]),
            )
            return False
        node = child

    if "gateways" in node:
        return False
    anchored_put(node, "gateways", _va_gateways())
    return True


def _va_gateways() -> dict[str, Any]:
    """A fresh copy of the two-role gateway table, safe to hand to ruamel."""
    return {"read_only": dict(_VA_GATEWAY_ROW), "write_access": dict(_VA_GATEWAY_ROW)}


def _inject_va(va: VAConfig, project_path: Path) -> None:
    """Wire the Virtual Accelerator soft-IOC into a built project.

    1. Copy the bundled ``templates/services/virtual_accelerator/`` compose
       template into ``<project>/services/virtual_accelerator/``.
    2. Write ``services.virtual_accelerator`` config + register it in
       ``deployed_services`` (so ``find_service_config`` resolves it,
       mirroring ``_inject_bluesky``).
    3. Make sure the deployed soft-IOC has a target block pointing at it —
       ``control_system.connector.virtual_accelerator.gateways`` — when the
       config carries none (see :func:`_ensure_va_connector_gateways`, which
       never touches an existing one).
    4. Print a post-build hint (data/simulation prerequisite + image note).

    Thin mirror of :func:`_inject_bluesky`: one soft-IOC container, one
    config block — no source-tree staging, no registry logic.

    Args:
        va: Validated Virtual Accelerator configuration from the build profile.
        project_path: Root of the built project.
    """
    from ruamel.yaml import YAML

    # 1. Copy the bundled compose template (located the same way as service templates).
    pkg_services = _locate_pkg_services()

    src_dir = pkg_services / "virtual_accelerator"
    if not src_dir.is_dir():
        logger.warning("No package template for virtual_accelerator service at %s", src_dir)
        return

    dest_services_root = project_path / "services"
    dest_services_root.mkdir(exist_ok=True)
    _copy_shared_service_partials(dest_services_root)
    dest_dir = dest_services_root / "virtual_accelerator"
    _refresh_service_dir(
        src_dir, dest_dir, "virtual_accelerator", _user_owned_services(project_path)
    )

    # 2. Write config.yml entries + register in deployed_services.
    config_path = project_path / "config.yml"
    if not config_path.exists():
        logger.warning(
            "config.yml not found. Skipping the virtual_accelerator config registration."
        )
        return

    yaml = YAML()
    yaml.preserve_quotes = True
    with open(config_path) as fh:
        config = yaml.load(fh)

    # No scenario-state directory is created here. Pre-creating the compose bind
    # source — so the container runtime cannot materialize it root-owned — only
    # helps where it survives, and anything created under the RENDER root is
    # wiped by the next build. The path compose actually binds is anchored on
    # the repo's own `var/agent_data`, and
    # `compose_generator._ensure_agent_data_structure` pre-creates it there,
    # which is the copy that survives long enough to be a mount source.

    # No ``image`` key: the service builds the local VA image on first
    # ``osprey up``. Override with OSPREY_VA_IMAGE, or set
    # ``services.virtual_accelerator.image`` here, to use a prebuilt/published image.
    config.setdefault("services", {})
    anchored_put(
        config["services"],
        "virtual_accelerator",
        _carry_authored_keys(
            config["services"],
            "virtual_accelerator",
            {
                "path": "./services/virtual_accelerator",
                "port": va.port,
            },
        ),
    )
    deployed = config.get("deployed_services", []) or []
    if "virtual_accelerator" not in [str(s) for s in deployed]:
        anchored_append(deployed, "virtual_accelerator")
    config["deployed_services"] = deployed

    # 3. Give the deployed soft-IOC a target block to be reached through, unless
    # the config already has one — deploying the service and leaving nothing
    # able to point at it is what would otherwise need a hand edit.
    wrote_gateways = _ensure_va_connector_gateways(config)

    with open(config_path, "w") as fh:
        yaml.dump(config, fh)

    # 4. Post-build hint.
    logger.debug("  ✓ Injected Virtual Accelerator soft-IOC (CA port %d)", va.port)
    if wrote_gateways:
        logger.debug(
            "    Target:     wrote control_system.connector.virtual_accelerator."
            "gateways (localhost, name-server mode; port follows "
            "services.virtual_accelerator.port). Set that block's `probe_channel` "
            "to a channel your machine model serves to make the target switchable."
        )
    logger.debug(
        "    Data:       requires <project>/data/simulation/machine.json "
        "(the simulation preset provisions this; without it the IOC SystemExits)."
    )
    logger.debug(
        "    Images:     `osprey up` builds the virtual-accelerator image "
        "locally for your native architecture. The first run is slow: the native "
        "deps are compiled from source, so no prebuilt aarch64 wheels are "
        "needed. Use `--dev` to bake in your local osprey checkout; "
        "set OSPREY_VA_IMAGE to use a published image."
    )


def _fill_panel_defaults(panel_cfg: Any, url: str, path: str, label: str) -> None:
    """Fill a partially-specified ``web.panels.<id>`` entry in place.

    Explicit values win: a facility override merged earlier in the build keeps
    whatever it set, and only the keys it left out are derived. ``url`` is
    additionally treated as unset when empty, so a blank override cannot leave
    the panel pointing nowhere.
    """
    if not panel_cfg.get("url"):
        anchored_put(panel_cfg, "url", url)
    if "path" not in panel_cfg:
        anchored_put(panel_cfg, "path", path)
    if "label" not in panel_cfg:
        anchored_put(panel_cfg, "label", label)


def _inject_bluesky_web(bluesky_web: BlueskyWebConfig, project_path: Path) -> None:
    """Wire the bluesky-web sidecar + its web panels into a built project.

    1. Copy the bundled ``templates/services/bluesky_web/`` compose template
       into ``<project>/services/bluesky_web/``.
    2. Write ``services.bluesky_web`` config + register it in
       ``deployed_services`` (so ``find_service_config`` resolves it,
       mirroring ``_inject_bluesky``).
    3. Register the ``web.panels.bluesky`` entry pointing at the sidecar's root
       URL, mirroring ``_inject_dispatch``'s ``events`` panel registration: the
       panel points the proxy at the sidecar ROOT and uses ``path`` to select
       the panel's static mount, so the panel HTML loads there while its
       prefix-relative API fetches reach the sidecar root.
    4. Print a post-build hint (image prerequisite).

    Thin mirror of :func:`_inject_va`/:func:`_inject_bluesky` for the compose
    + config wiring, plus :func:`_inject_dispatch`'s ``web.panels`` setdefault
    idiom for the panel registration.

    Args:
        bluesky_web: Validated bluesky-web configuration from the build profile.
        project_path: Root of the built project.
    """
    from ruamel.yaml import YAML

    # 1. Copy the bundled compose template (located the same way as service templates).
    pkg_services = _locate_pkg_services()

    src_dir = pkg_services / "bluesky_web"
    if not src_dir.is_dir():
        logger.warning("No package template for bluesky_web service at %s", src_dir)
        return

    dest_services_root = project_path / "services"
    dest_services_root.mkdir(exist_ok=True)
    _copy_shared_service_partials(dest_services_root)
    dest_dir = dest_services_root / "bluesky_web"
    _refresh_service_dir(src_dir, dest_dir, "bluesky_web", _user_owned_services(project_path))

    # 2. Write config.yml entries + register in deployed_services.
    config_path = project_path / "config.yml"
    if not config_path.exists():
        logger.warning("config.yml not found. Skipping the bluesky_web config registration.")
        return

    yaml = YAML()
    yaml.preserve_quotes = True
    with open(config_path) as fh:
        config = yaml.load(fh)

    # No ``image`` key: the service builds the local bluesky-web image on
    # first ``osprey up``. Override with OSPREY_BLUESKY_WEB_IMAGE, or
    # set ``services.bluesky_web.image`` here, to use a prebuilt/published image.
    config.setdefault("services", {})
    anchored_put(
        config["services"],
        "bluesky_web",
        _carry_authored_keys(
            config["services"],
            "bluesky_web",
            {
                "path": "./services/bluesky_web",
                "port": bluesky_web.port,
            },
        ),
    )
    deployed = config.get("deployed_services", []) or []
    if "bluesky_web" not in [str(s) for s in deployed]:
        anchored_append(deployed, "bluesky_web")
    config["deployed_services"] = deployed

    # 3. Register the web.panels.bluesky entry. Derive its url from
    # bluesky_web.port so the port is a single source of truth (mirroring the
    # events-panel comment in _inject_dispatch), but write only when the
    # profile has not already set an explicit `web.panels.<id>.url` via a
    # config override (merged earlier in the build); explicit overrides take
    # precedence. Emit a bare sidecar-root `url` plus a per-panel `path`
    # (rather than baking the panel path into `url`) to match the
    # custom-panel proxy convention: the web terminal composes
    # `url.rstrip('/') + '/' + path`, so a path baked into `url` would
    # double-prefix sub-routes. `setdefault` on `path`/`label` honors a
    # facility override.
    default_url = f"${{BLUESKY_WEB_URL:-http://localhost:{bluesky_web.port}}}"
    panel_specs = (("bluesky", "/bluesky/", "BLUESKY"),)
    panels = config.setdefault("web", {}).setdefault("panels", {})
    for panel_id, panel_path, label in panel_specs:
        panel_cfg = panels.get(panel_id)
        if panel_cfg is None:
            # Put the complete panel in one shot: anchored_put re-anchors a
            # section comment beneath the new entry, which needs the entry's
            # full subtree to exist at insertion time.
            anchored_put(panels, panel_id, {"url": default_url, "path": panel_path, "label": label})
            continue
        _fill_panel_defaults(panel_cfg, default_url, panel_path, label)

    with open(config_path, "w") as fh:
        yaml.dump(config, fh)

    # 4. Post-build hint.
    logger.debug("  ✓ Injected bluesky-web sidecar (port %d)", bluesky_web.port)
    logger.debug(
        "    Panels:     BLUESKY (Plans | Queue | Results). Reach it through the "
        "web-terminal proxy at /panel/bluesky."
    )
    logger.debug(
        "    Images:     `osprey up` builds the bluesky-web image locally "
        "(first run is slow). Use `--dev` to bake in your local osprey checkout; "
        "set OSPREY_BLUESKY_WEB_IMAGE to use a published image."
    )


def _inject_nextcloud_bridge(
    nextcloud_bridge: NextcloudBridgeProfileConfig, project_path: Path
) -> None:
    """Wire the Nextcloud Talk bridge service into a built project.

    1. Copy the bundled ``templates/services/nextcloud_bridge/`` compose
       template into ``<project>/services/nextcloud_bridge/``.
    2. Write ``services.nextcloud_bridge`` config + register it in
       ``deployed_services`` (so ``find_service_config`` resolves it,
       mirroring ``_inject_bluesky``).
    3. Print a post-build hint naming the bot credentials the operator must
       supply — unlike every other injected service, this one cannot come up on
       deploy-minted secrets alone.

    Structurally a thin mirror of :func:`_inject_bluesky`: one poller container,
    one config block, no source-tree staging. It must run *after*
    :func:`_inject_dispatch`, which is what puts ``event_dispatcher`` /
    ``dispatch_worker`` into ``deployed_services`` — the compose template gates
    both its ``depends_on`` and its in-network ``DISPATCHER_URL``/``WORKER_URL``
    on their presence there.

    Args:
        nextcloud_bridge: Validated bridge configuration from the build profile.
            ``BuildProfile.validate`` has already established that a
            ``dispatch:`` block exists and that ``trigger`` names a trigger
            declared in the resolved triggers file.
        project_path: Root of the built project.
    """
    from ruamel.yaml import YAML

    # 1. Copy the bundled compose template (located the same way as service templates).
    pkg_services = _locate_pkg_services()

    src_dir = pkg_services / "nextcloud_bridge"
    if not src_dir.is_dir():
        logger.warning("No package template for nextcloud_bridge service at %s", src_dir)
        return

    dest_services_root = project_path / "services"
    dest_services_root.mkdir(exist_ok=True)
    _copy_shared_service_partials(dest_services_root)
    dest_dir = dest_services_root / "nextcloud_bridge"
    _refresh_service_dir(src_dir, dest_dir, "nextcloud_bridge", _user_owned_services(project_path))

    # 2. Write config.yml entries + register in deployed_services.
    config_path = project_path / "config.yml"
    if not config_path.exists():
        logger.warning("config.yml not found. Skipping the nextcloud_bridge config registration.")
        return

    yaml = YAML()
    yaml.preserve_quotes = True
    with open(config_path) as fh:
        config = yaml.load(fh)

    # ``trigger`` is the one key the compose template reads with NO ``| default``
    # — it renders DISPATCH_TRIGGER straight from here, so a missing key must
    # break the render loudly rather than silently firing another facility's
    # trigger. No ``image`` key: the service builds the local
    # <project>-nextcloud-bridge image on first ``osprey up`` (the
    # template's own ``| default`` supplies that tag). Override with
    # OSPREY_NEXTCLOUD_BRIDGE_IMAGE, or set ``services.nextcloud_bridge.image``
    # here, to use a prebuilt/published image.
    config.setdefault("services", {})
    anchored_put(
        config["services"],
        "nextcloud_bridge",
        _carry_authored_keys(
            config["services"],
            "nextcloud_bridge",
            {
                "path": "./services/nextcloud_bridge",
                "trigger": nextcloud_bridge.trigger,
            },
        ),
    )
    deployed = config.get("deployed_services", []) or []
    if "nextcloud_bridge" not in [str(s) for s in deployed]:
        anchored_append(deployed, "nextcloud_bridge")
    config["deployed_services"] = deployed

    with open(config_path, "w") as fh:
        yaml.dump(config, fh)

    # 3. Post-build hint.
    logger.debug("  ✓ Injected Nextcloud Talk bridge (trigger %r)", nextcloud_bridge.trigger)
    logger.debug(
        "    Credentials: set NEXTCLOUD_BASE_URL, NEXTCLOUD_BOT_ACCOUNT, "
        "NEXTCLOUD_APP_PASSWORD and NEXTCLOUD_ROOMS in the project .env before "
        "`osprey up`. These are user-supplied. Unlike the dispatch tokens, deploy "
        "does not mint them, and the bridge aborts at boot naming whichever is "
        "missing."
    )
    logger.debug(
        "    Rooms:       NEXTCLOUD_ROOMS is a comma-separated list of Talk room "
        "tokens. Room membership is the access gate: whoever can post in a "
        "listed room can reach the agent."
    )
    logger.debug(
        "    Images:     `osprey up` builds the nextcloud-bridge image locally "
        "(first run is slow). Use `--dev` to bake in your local osprey checkout; "
        "set OSPREY_NEXTCLOUD_BRIDGE_IMAGE to use a published image."
    )


def _inject_gchat_bridge(gchat_bridge: GChatBridgeProfileConfig, project_path: Path) -> None:
    """Wire the Google Chat bridge service into a built project.

    1. Copy the bundled ``templates/services/gchat_bridge/`` compose template
       into ``<project>/services/gchat_bridge/``.
    2. Write ``services.gchat_bridge`` config + register it in
       ``deployed_services`` (so ``find_service_config`` resolves it,
       mirroring ``_inject_nextcloud_bridge``).
    3. Print a post-build hint naming the Google credentials the operator must
       supply, and the one deployment rule the compose file cannot enforce
       (one bridge per Pub/Sub subscription).

    Structurally a thin mirror of :func:`_inject_nextcloud_bridge` — the other
    chat channel — down to the single ``trigger`` key the template reads with no
    ``| default``. Like it, this must run *after* :func:`_inject_dispatch`,
    which is what puts ``event_dispatcher`` / ``dispatch_worker`` into
    ``deployed_services``; the compose template gates both its ``depends_on``
    and its in-network ``DISPATCHER_URL``/``WORKER_URL`` on their presence there.

    Args:
        gchat_bridge: Validated bridge configuration from the build profile.
            ``BuildProfile.validate`` has already established that a
            ``dispatch:`` block exists and that ``trigger`` names a trigger
            declared in the resolved triggers file.
        project_path: Root of the built project.
    """
    from ruamel.yaml import YAML

    # 1. Copy the bundled compose template (located the same way as service templates).
    pkg_services = _locate_pkg_services()

    src_dir = pkg_services / "gchat_bridge"
    if not src_dir.is_dir():
        logger.warning("No package template for gchat_bridge service at %s", src_dir)
        return

    dest_services_root = project_path / "services"
    dest_services_root.mkdir(exist_ok=True)
    _copy_shared_service_partials(dest_services_root)
    dest_dir = dest_services_root / "gchat_bridge"
    _refresh_service_dir(src_dir, dest_dir, "gchat_bridge", _user_owned_services(project_path))

    # 2. Write config.yml entries + register in deployed_services.
    config_path = project_path / "config.yml"
    if not config_path.exists():
        logger.warning("config.yml not found. Skipping the gchat_bridge config registration.")
        return

    yaml = YAML()
    yaml.preserve_quotes = True
    with open(config_path) as fh:
        config = yaml.load(fh)

    # ``trigger`` is the one key the compose template reads with NO ``| default``
    # — it renders DISPATCH_TRIGGER straight from here, so a missing key must
    # break the render loudly rather than silently firing another facility's
    # trigger. No ``image`` key: the service builds the local
    # <project>-gchat-bridge image on first ``osprey up`` (the template's
    # own ``| default`` supplies that tag). Override with
    # OSPREY_GCHAT_BRIDGE_IMAGE, or set ``services.gchat_bridge.image`` here, to
    # use a prebuilt/published image.
    config.setdefault("services", {})
    config["services"]["gchat_bridge"] = _carry_authored_keys(
        config["services"],
        "gchat_bridge",
        {
            "path": "./services/gchat_bridge",
            "trigger": gchat_bridge.trigger,
        },
    )
    deployed = config.get("deployed_services", []) or []
    if "gchat_bridge" not in [str(s) for s in deployed]:
        deployed.append("gchat_bridge")
    config["deployed_services"] = deployed

    with open(config_path, "w") as fh:
        yaml.dump(config, fh)

    # 3. Post-build hint.
    logger.debug("  ✓ Injected Google Chat bridge (trigger %r)", gchat_bridge.trigger)
    logger.debug(
        "    Credentials: set GCHAT_SA_KEY (host path to the service-account JSON "
        "key, mounted read-only at the same path in the container), "
        "GCHAT_SUBSCRIPTION and GCHAT_APP_ID in the project .env before "
        "`osprey up`. These are user-supplied. Unlike the dispatch tokens, deploy "
        "does not mint them, and the bridge aborts at boot naming whichever is "
        "missing."
    )
    logger.debug(
        "    Subscription: deploy exactly ONE bridge per Pub/Sub subscription. "
        "Pub/Sub load-balances a subscription across its consumers, so a second "
        "deployment on the same name does not duplicate events. It silently splits "
        "them, and each half answers only what it received. Give every "
        "deployment its own subscription on the topic."
    )
    logger.debug(
        "    Images:     `osprey up` builds the gchat-bridge image locally "
        "(first run is slow). Use `--dev` to bake in your local osprey checkout; "
        "set OSPREY_GCHAT_BRIDGE_IMAGE to use a published image."
    )


def _inject_va_archiver(va_archiver: VAArchiverConfig, project_path: Path) -> None:
    """Wire the stored archiver — store plus recorder — into a built project.

    1. Copy the bundled ``mongodb`` and ``archiver_recorder`` compose templates
       into ``<project>/services/``.
    2. Write ``services.{mongodb,archiver_recorder}`` config + register both in
       ``deployed_services``.
    3. Print a post-build hint (deploy-time seed, minted password, what the
       recorder waits for).

    A two-service injector like :func:`_inject_dispatch`, and paired for the
    same reason: a store nothing writes to and a recorder with nowhere to write
    are each half a feature. They are deployed together and the recorder's
    compose template health-gates on the store.

    Must run *after* :func:`_inject_va`, which is what puts
    ``virtual_accelerator`` into ``deployed_services`` — the recorder template
    reads that membership to decide whether it can reuse the VA's image and
    address the IOC in-network, or must be told both from the environment.

    The connection block the agent reads (``archiver.mongodb_archiver.*``) and
    the archive's shape knobs (``va_archiver.*``) are deliberately NOT written
    here: they come from :func:`~osprey.cli.build_profile_archiver.va_archiver_config_overrides`
    on the ordinary config-override path, which an attached project reaches and
    this injector does not.

    Args:
        va_archiver: Validated archiver configuration from the build profile.
        project_path: Root of the built project.
    """
    from ruamel.yaml import YAML

    # 1. Copy the bundled compose templates (located the same way as service
    #    templates). The recorder ships no Dockerfile — it runs the VA's image
    #    with a different command — so there is nothing to build here either.
    pkg_services = _locate_pkg_services()

    dest_services_root = project_path / "services"
    dest_services_root.mkdir(exist_ok=True)
    _copy_shared_service_partials(dest_services_root)
    owned = _user_owned_services(project_path)

    for name in ("mongodb", "archiver_recorder"):
        src_dir = pkg_services / name
        if not src_dir.is_dir():
            logger.warning("No package template for archiver service %r at %s", name, src_dir)
            continue
        _refresh_service_dir(src_dir, dest_services_root / name, name, owned)

    # 1a. The recorder bind-mounts the simulation data dir read-only to read the
    # channel manifest. An app bundle that ships no such tree would leave the
    # mount source missing, and the container runtime materializes a missing
    # source itself, root-owned — which then locks the host out of a directory
    # inside its own project. Same guard, same reason, as the VA injector's.
    (project_path / "data" / "simulation").mkdir(parents=True, exist_ok=True)

    # 2. Write config.yml entries + register in deployed_services.
    config_path = project_path / "config.yml"
    if not config_path.exists():
        logger.warning("config.yml not found. Skipping the archiver config registration.")
        return

    yaml = YAML()
    yaml.preserve_quotes = True
    with open(config_path) as fh:
        config = yaml.load(fh)

    # Only the keys the compose templates read: how the store publishes itself,
    # who it creates, and how it compresses. They restate profile knobs the
    # agent-side connection block also carries — one profile value rendered into
    # the two places that read it, which is derivation rather than a second home
    # for the fact. No ``image`` key on either service: the store falls to the
    # template's pinned upstream tag, and the recorder to the VA's image.
    config.setdefault("services", {})
    anchored_put(
        config["services"],
        "mongodb",
        _carry_authored_keys(
            config["services"],
            "mongodb",
            {
                "path": "./services/mongodb",
                "port_host": va_archiver.port_host,
                "username": va_archiver.username,
                "compression": va_archiver.compression,
            },
        ),
    )
    anchored_put(
        config["services"],
        "archiver_recorder",
        _carry_authored_keys(
            config["services"],
            "archiver_recorder",
            {"path": "./services/archiver_recorder"},
        ),
    )
    deployed = config.get("deployed_services", []) or []
    for name in ("mongodb", "archiver_recorder"):
        if name not in [str(s) for s in deployed]:
            anchored_append(deployed, name)
    config["deployed_services"] = deployed

    with open(config_path, "w") as fh:
        yaml.dump(config, fh)

    # 3. Post-build hint.
    logger.debug(
        "  ✓ Injected archiver store + recorder (port %d, %d-day retention)",
        va_archiver.port_host,
        va_archiver.retention_days,
    )
    logger.debug(
        "    History:    `osprey up` seeds the base series before the "
        "stack starts (minutes on a first deploy, skipped when the knobs have "
        "not changed). Until then the archive is empty and archiver reads "
        "honestly return nothing."
    )
    logger.debug(
        "    Password:   `osprey up` mints %s into .env; the store, the "
        "recorder and the agent all authenticate with that one value.",
        va_archiver.password_env,
    )
    logger.debug(
        "    Recording:  the recorder writes only while control_system.type is "
        "'virtual_accelerator'. On any other control system it idles. It "
        "re-reads that setting on an interval, so the flip needs no restart."
    )
