"""Service injectors — wire deploy-time containers into a built project.

Each injector copies a bundled compose template into ``<project>/services/``,
writes the matching ``services.<name>`` block into ``config.yml`` (and
registers it in ``deployed_services``), and prints a post-build hint. The
injectors pair 1:1 with the service dataclasses in
:mod:`osprey.cli.build_profile_schema` (``DispatchConfig``, ``BlueskyConfig``,
``BlueskyPanelsConfig``, ``VAConfig``, ``NextcloudBridgeProfileConfig``).
``_copy_service_templates`` / ``_inject_profile_services`` handle the framework
and facility-declared service templates.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from osprey.errors import BuildProfileError
from osprey.utils.config_writer import anchored_append, anchored_put
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from osprey.cli.build_profile import (
        BlueskyConfig,
        BlueskyPanelsConfig,
        DispatchConfig,
        NextcloudBridgeProfileConfig,
        VAConfig,
    )

logger = get_logger("build")


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
        logger.info("  ⏭  services/%s is user-owned — left untouched (scaffold claim)", name)
        return False
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(src_dir, dest_dir)
    return True


def _copy_service_templates(project_path: Path) -> int:
    """Copy service compose templates from the OSPREY package into the project.

    Copies each service's compose template directory from the package to the
    project's ``services/`` tree for the UNION of ``deployed_services`` and
    every service merely DECLARED under ``services:`` that ships a package
    template.  This makes the project self-contained so that ``osprey deploy
    up`` works directly from the project directory, and — crucially — bundles
    opt-in add-ons (declared but not deployed) so they can be switched on later
    via a ``deployed_services`` edit + ``osprey deploy up`` without rebuilding.
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
        logger.warning("Service templates directory not found — skipping")
        return 0

    dest_services_root = project_path / "services"
    dest_services_root.mkdir(exist_ok=True)

    # Always copy the root compose template so `osprey deploy up` works even
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
    # later via a `deployed_services` edit + `osprey deploy up`, no rebuild.
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
            logger.warning("Skipping service %r — unsupported naming for template copy", name)
            continue

        if not src_dir.is_dir():
            # A declared-but-not-deployed service may legitimately ship no
            # package template (e.g. facility-injected elsewhere) — skip it
            # silently.  Only warn when a *deployed* service is missing its
            # template, which would break `osprey deploy up`.
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
            logger.warning("No template for profile service %r at %s — skipping", name, src_dir)
            continue

        # Copy template directory (skipped for claimed services)
        dest_dir = dest_services_root / name
        _refresh_service_dir(src_dir, dest_dir, name, owned)

        # Register service config in config.yml
        if "services" not in config:
            config["services"] = {}
        svc_config = {"path": f"./services/{name}"}
        svc_config.update(svc_def.config)
        anchored_put(config["services"], name, svc_config)

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

    Args:
        dispatch: Validated dispatch configuration from the build profile.
        profile_dir: Directory containing the build profile (triggers source).
        project_path: Root of the built project.

    Raises:
        BuildProfileError: If the configured triggers file cannot be resolved.
    """
    from ruamel.yaml import YAML

    from osprey.cli.build_profile import _triggers_dir

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
    if triggers_doc is not None:
        dispatcher_block = triggers_doc.setdefault("dispatcher", {})
        dispatcher_block["max_concurrent_runs"] = dispatch.max_concurrent_runs
        dispatcher_block["max_queue_depth"] = dispatch.max_queue_depth
        with open(triggers_dest, "w") as fh:
            _trigger_yaml.dump(triggers_doc, fh)

    # 2. Copy bundled compose templates (located the same way as service templates).
    pkg_services = _locate_pkg_services()

    dest_services_root = project_path / "services"
    dest_services_root.mkdir(exist_ok=True)
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
        logger.warning("config.yml not found — skipping dispatch config registration")
        return

    yaml = YAML()
    yaml.preserve_quotes = True
    with open(config_path) as fh:
        config = yaml.load(fh)

    # No ``image`` key on either service, so each falls to its compose default:
    # the event-dispatcher builds the project's <project>-dispatch:local image (its
    # own compose ``build:`` block), and the dispatch worker runs <project>:local —
    # the project image ``osprey deploy up`` builds from the project Dockerfile
    # (the worker has no build block of its own, to avoid racing the dispatcher).
    # Override with OSPREY_DISPATCH_IMAGE/OSPREY_WORKER_IMAGE, or set
    # ``services.<name>.image`` here, to use a prebuilt/published image.
    config.setdefault("services", {})
    anchored_put(
        config["services"],
        "event_dispatcher",
        {
            "path": "./services/event_dispatcher",
            "port": dispatch.dispatcher_port,
            "facility_name": dispatch.facility_name,
            "pv_strip_prefix": dispatch.pv_strip_prefix,
            # Copy the project's triggers.yml into the service build context so the
            # compose ``./triggers.yml`` bind-mount resolves to a file (otherwise the
            # container runtime auto-creates an empty directory at the mount source).
            "additional_dirs": [{"src": "triggers.yml", "dst": "triggers.yml"}],
        },
    )
    anchored_put(
        config["services"],
        "dispatch_worker",
        {
            "path": "./services/dispatch_worker",
            "worker_count": dispatch.worker_count,
            "worker_port_base": dispatch.worker_port_base,
            "workspace_mode": dispatch.workspace_mode,
            "timeout_sec": dispatch.timeout_sec,
            "inactivity_sec": dispatch.inactivity_sec,
        },
    )
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
    logger.info(
        "  ✓ Injected event dispatch (%d worker(s), port %d)",
        dispatch.worker_count,
        dispatch.dispatcher_port,
    )
    logger.info("    Dashboard:  http://localhost:%d/dashboard", dispatch.dispatcher_port)
    logger.info(
        "    Token:      `osprey deploy up` writes EVENT_DISPATCHER_TOKEN to .env; "
        "load it with: export $(grep -E '^EVENT_DISPATCHER_TOKEN=' .env | xargs)"
    )
    logger.info(
        "    Try it:     curl -X POST http://localhost:%d/webhook/hello-dispatch "
        '-H "Authorization: Bearer $EVENT_DISPATCHER_TOKEN" '
        "-H 'Content-Type: application/json' -d '{}'",
        dispatch.dispatcher_port,
    )
    logger.info(
        "    Images:     `osprey deploy up` builds the dispatch image and the "
        "worker's project image locally (first run is slow). Use `--dev` to bake "
        "in your local osprey checkout; set OSPREY_DISPATCH_IMAGE/OSPREY_WORKER_IMAGE "
        "to use a published image."
    )


def _inject_bluesky(bluesky: BlueskyConfig, project_path: Path) -> None:
    """Wire the Bluesky scan-bridge feature into a built project.

    1. Copy the bundled ``templates/services/bluesky/`` compose template into
       ``<project>/services/bluesky/``.
    2. Write ``services.bluesky`` config + register it in ``deployed_services``
       (so ``find_service_config`` resolves it, mirroring ``_inject_dispatch``).
    3. Print a post-build hint (launch-token env var + image prerequisite).

    Simpler than ``_inject_dispatch``: no triggers file to resolve and no
    multi-instance worker loop — a project deploys exactly one bluesky-bridge
    process. The ``scan`` MCP server itself is a separate, always-available
    framework server (see ``osprey.mcp_server.bluesky``); this step only wires
    the *deploy-time* container that server talks to over HTTP.

    Args:
        bluesky: Validated bluesky configuration from the build profile.
        project_path: Root of the built project.
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
    dest_dir = dest_services_root / "bluesky"
    _refresh_service_dir(src_dir, dest_dir, "bluesky", _user_owned_services(project_path))

    # 2. Write config.yml entries + register in deployed_services.
    config_path = project_path / "config.yml"
    if not config_path.exists():
        logger.warning("config.yml not found — skipping bluesky config registration")
        return

    yaml = YAML()
    yaml.preserve_quotes = True
    with open(config_path) as fh:
        config = yaml.load(fh)

    # No ``image`` key: the service builds the local bluesky-bridge image on
    # first ``osprey deploy up``. Override with OSPREY_BLUESKY_BRIDGE_IMAGE, or
    # set ``services.bluesky.image`` here, to use a prebuilt/published image.
    config.setdefault("services", {})
    svc_config: dict[str, Any] = {
        "path": "./services/bluesky",
        "port": bluesky.port,
        "tiled_enabled": bluesky.tiled_enabled,
        "tiled_port": bluesky.tiled_port,
        "demo_runner": bluesky.demo_runner,
    }
    if bluesky.plan_dir:
        # Only written when configured — its absence is what keeps a
        # bridge-only deploy (no facility plan directory) rendering exactly
        # as before: the compose template's {% if %} guard reads this same
        # key, so an unset plan_dir means no mount and no BLUESKY_PLAN_DIRS
        # env var at all (Task 1.4).
        svc_config["plan_dir"] = bluesky.plan_dir
    if bluesky.excluded_plans:
        # Only written when non-empty — its absence keeps a deploy with no
        # exclusions rendering exactly as before: the compose template's
        # {% if %} guard reads this same key, so an empty list means no
        # BLUESKY_EXCLUDED_PLANS env var at all. The os.pathsep join is done
        # Python-side because the Jinja render context has no `os` module.
        svc_config["excluded_plans"] = os.pathsep.join(bluesky.excluded_plans)
    anchored_put(config["services"], "bluesky", svc_config)
    deployed = config.get("deployed_services", []) or []
    if "bluesky" not in [str(s) for s in deployed]:
        anchored_append(deployed, "bluesky")
    config["deployed_services"] = deployed

    with open(config_path, "w") as fh:
        yaml.dump(config, fh)

    # 3. Post-build hint.
    logger.info("  ✓ Injected Bluesky scan bridge (port %d)", bluesky.port)
    logger.info(
        "    Token:      `osprey deploy up` writes BLUESKY_LAUNCH_TOKEN to .env; "
        "the `scan` MCP server's launch_run tool reads it automatically."
    )
    logger.info(
        "    Images:     `osprey deploy up` builds the bluesky-bridge image locally "
        "(first run is slow). Use `--dev` to bake in your local osprey checkout; "
        "set OSPREY_BLUESKY_BRIDGE_IMAGE to use a published image."
    )
    if bluesky.tiled_enabled:
        logger.info("    Tiled:      enabled on port %d", bluesky.tiled_port)
    if bluesky.plan_dir:
        logger.info(
            "    Plan dir:   %s mounted read-only into the bridge; its plans "
            "load as the 'facility' trust tier (BLUESKY_PLAN_DIRS)",
            bluesky.plan_dir,
        )
    if bluesky.demo_runner:
        logger.warning(
            "    Demo mode:  BLUESKY_DEMO_RUNNER is set — the bridge runs a real "
            "bluesky RunEngine against MOCK devices only. Never enable this for a "
            "facility wiring real EPICS hardware."
        )


def _inject_va(va: VAConfig, project_path: Path) -> None:
    """Wire the Virtual Accelerator soft-IOC into a built project.

    1. Copy the bundled ``templates/services/virtual_accelerator/`` compose
       template into ``<project>/services/virtual_accelerator/``.
    2. Write ``services.virtual_accelerator`` config + register it in
       ``deployed_services`` (so ``find_service_config`` resolves it,
       mirroring ``_inject_bluesky``).
    3. Print a post-build hint (data/simulation prerequisite + image note).

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
    dest_dir = dest_services_root / "virtual_accelerator"
    _refresh_service_dir(
        src_dir, dest_dir, "virtual_accelerator", _user_owned_services(project_path)
    )

    # 2. Write config.yml entries + register in deployed_services.
    config_path = project_path / "config.yml"
    if not config_path.exists():
        logger.warning("config.yml not found — skipping virtual_accelerator config registration")
        return

    yaml = YAML()
    yaml.preserve_quotes = True
    with open(config_path) as fh:
        config = yaml.load(fh)

    # No ``image`` key: the service builds the local VA image on first
    # ``osprey deploy up``. Override with OSPREY_VA_IMAGE, or set
    # ``services.virtual_accelerator.image`` here, to use a prebuilt/published image.
    config.setdefault("services", {})
    anchored_put(
        config["services"],
        "virtual_accelerator",
        {
            "path": "./services/virtual_accelerator",
            "port": va.port,
        },
    )
    deployed = config.get("deployed_services", []) or []
    if "virtual_accelerator" not in [str(s) for s in deployed]:
        anchored_append(deployed, "virtual_accelerator")
    config["deployed_services"] = deployed

    with open(config_path, "w") as fh:
        yaml.dump(config, fh)

    # 3. Post-build hint.
    logger.info("  ✓ Injected Virtual Accelerator soft-IOC (CA port %d)", va.port)
    logger.info(
        "    Data:       requires <project>/data/simulation/machine.json "
        "(the simulation preset provisions this; without it the IOC SystemExits)."
    )
    logger.info(
        "    Images:     `osprey deploy up` builds the virtual-accelerator image "
        "locally for your native architecture (first run is slow — the native deps "
        "PyAT/softioc are compiled from source, so no prebuilt aarch64 wheels are "
        "needed). Use `--dev` to bake in your local osprey checkout; "
        "set OSPREY_VA_IMAGE to use a published image."
    )


def _inject_bluesky_panels(bluesky_panels: BlueskyPanelsConfig, project_path: Path) -> None:
    """Wire the bluesky-panels sidecar + its three web panels into a built project.

    1. Copy the bundled ``templates/services/bluesky_panels/`` compose template
       into ``<project>/services/bluesky_panels/``.
    2. Write ``services.bluesky_panels`` config + register it in
       ``deployed_services`` (so ``find_service_config`` resolves it,
       mirroring ``_inject_bluesky``).
    3. Register the three ``web.panels.<id>`` entries (``plan``,
       ``results``, ``health``) pointing at the sidecar's root URL,
       mirroring ``_inject_dispatch``'s ``events`` panel registration: each
       panel points the proxy at the sidecar ROOT and uses ``path`` to select
       the panel's static mount, so the panel HTML loads there while its
       prefix-relative API fetches reach the sidecar root.
    4. Print a post-build hint (image prerequisite).

    Thin mirror of :func:`_inject_va`/:func:`_inject_bluesky` for the compose
    + config wiring, plus :func:`_inject_dispatch`'s ``web.panels`` setdefault
    idiom for the panel registration.

    Args:
        bluesky_panels: Validated bluesky-panels configuration from the build profile.
        project_path: Root of the built project.
    """
    from ruamel.yaml import YAML

    # 1. Copy the bundled compose template (located the same way as service templates).
    pkg_services = _locate_pkg_services()

    src_dir = pkg_services / "bluesky_panels"
    if not src_dir.is_dir():
        logger.warning("No package template for bluesky_panels service at %s", src_dir)
        return

    dest_services_root = project_path / "services"
    dest_services_root.mkdir(exist_ok=True)
    dest_dir = dest_services_root / "bluesky_panels"
    _refresh_service_dir(src_dir, dest_dir, "bluesky_panels", _user_owned_services(project_path))

    # 2. Write config.yml entries + register in deployed_services.
    config_path = project_path / "config.yml"
    if not config_path.exists():
        logger.warning("config.yml not found — skipping bluesky_panels config registration")
        return

    yaml = YAML()
    yaml.preserve_quotes = True
    with open(config_path) as fh:
        config = yaml.load(fh)

    # No ``image`` key: the service builds the local bluesky-panels image on
    # first ``osprey deploy up``. Override with OSPREY_BLUESKY_PANELS_IMAGE, or
    # set ``services.bluesky_panels.image`` here, to use a prebuilt/published image.
    config.setdefault("services", {})
    anchored_put(
        config["services"],
        "bluesky_panels",
        {
            "path": "./services/bluesky_panels",
            "port": bluesky_panels.port,
        },
    )
    deployed = config.get("deployed_services", []) or []
    if "bluesky_panels" not in [str(s) for s in deployed]:
        anchored_append(deployed, "bluesky_panels")
    config["deployed_services"] = deployed

    # 3. Register the two web.panels.<id> entries. Derive each url from
    # bluesky_panels.port so the port is a single source of truth (mirroring the
    # events-panel comment in _inject_dispatch), but write only when the
    # profile has not already set an explicit `web.panels.<id>.url` via a
    # config override (merged earlier in the build); explicit overrides take
    # precedence. Emit a bare sidecar-root `url` plus a per-panel `path`
    # (rather than baking the panel path into `url`) to match the
    # custom-panel proxy convention: the web terminal composes
    # `url.rstrip('/') + '/' + path`, so a path baked into `url` would
    # double-prefix sub-routes. `setdefault` on `path`/`label` honors a
    # facility override.
    default_url = f"${{BLUESKY_PANELS_URL:-http://localhost:{bluesky_panels.port}}}"
    panel_specs = (
        ("plan", "/plan/", "PLAN"),
        ("results", "/results/", "RESULTS"),
    )
    for panel_id, panel_path, label in panel_specs:
        panels = config.setdefault("web", {}).setdefault("panels", {})
        panel_cfg = panels.get(panel_id)
        if panel_cfg is None:
            # Put the complete panel in one shot: anchored_put re-anchors a
            # section comment beneath the new entry, which needs the entry's
            # full subtree to exist at insertion time.
            anchored_put(panels, panel_id, {"url": default_url, "path": panel_path, "label": label})
            continue
        if not panel_cfg.get("url"):
            anchored_put(panel_cfg, "url", default_url)
        if "path" not in panel_cfg:
            anchored_put(panel_cfg, "path", panel_path)
        if "label" not in panel_cfg:
            anchored_put(panel_cfg, "label", label)

    with open(config_path, "w") as fh:
        yaml.dump(config, fh)

    # 4. Post-build hint.
    logger.info("  ✓ Injected bluesky-panels sidecar (port %d)", bluesky_panels.port)
    logger.info(
        "    Panels:     PLAN, RESULTS — reached through the "
        "web-terminal proxy at /panel/{plan,results}."
    )
    logger.info(
        "    Images:     `osprey deploy up` builds the bluesky-panels image locally "
        "(first run is slow). Use `--dev` to bake in your local osprey checkout; "
        "set OSPREY_BLUESKY_PANELS_IMAGE to use a published image."
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
    dest_dir = dest_services_root / "nextcloud_bridge"
    _refresh_service_dir(src_dir, dest_dir, "nextcloud_bridge", _user_owned_services(project_path))

    # 2. Write config.yml entries + register in deployed_services.
    config_path = project_path / "config.yml"
    if not config_path.exists():
        logger.warning("config.yml not found — skipping nextcloud_bridge config registration")
        return

    yaml = YAML()
    yaml.preserve_quotes = True
    with open(config_path) as fh:
        config = yaml.load(fh)

    # ``trigger`` is the one key the compose template reads with NO ``| default``
    # — it renders DISPATCH_TRIGGER straight from here, so a missing key must
    # break the render loudly rather than silently firing another facility's
    # trigger. No ``image`` key: the service builds the local
    # <project>-nextcloud-bridge image on first ``osprey deploy up`` (the
    # template's own ``| default`` supplies that tag). Override with
    # OSPREY_NEXTCLOUD_BRIDGE_IMAGE, or set ``services.nextcloud_bridge.image``
    # here, to use a prebuilt/published image.
    config.setdefault("services", {})
    anchored_put(
        config["services"],
        "nextcloud_bridge",
        {
            "path": "./services/nextcloud_bridge",
            "trigger": nextcloud_bridge.trigger,
        },
    )
    deployed = config.get("deployed_services", []) or []
    if "nextcloud_bridge" not in [str(s) for s in deployed]:
        anchored_append(deployed, "nextcloud_bridge")
    config["deployed_services"] = deployed

    with open(config_path, "w") as fh:
        yaml.dump(config, fh)

    # 3. Post-build hint.
    logger.info("  ✓ Injected Nextcloud Talk bridge (trigger %r)", nextcloud_bridge.trigger)
    logger.info(
        "    Credentials: set NEXTCLOUD_BASE_URL, NEXTCLOUD_BOT_ACCOUNT, "
        "NEXTCLOUD_APP_PASSWORD and NEXTCLOUD_ROOMS in the project .env before "
        "`osprey deploy up`. These are user-supplied — unlike the dispatch "
        "tokens, deploy does not mint them, and the bridge aborts at boot "
        "naming whichever is missing."
    )
    logger.info(
        "    Rooms:       NEXTCLOUD_ROOMS is a comma-separated list of Talk room "
        "tokens. Room membership is the access gate: whoever can post in a "
        "listed room can reach the agent."
    )
    logger.info(
        "    Images:     `osprey deploy up` builds the nextcloud-bridge image locally "
        "(first run is slow). Use `--dev` to bake in your local osprey checkout; "
        "set OSPREY_NEXTCLOUD_BRIDGE_IMAGE to use a published image."
    )
