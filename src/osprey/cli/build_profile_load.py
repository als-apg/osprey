"""YAML-to-``BuildProfile`` parsing and the single-file loader.

Turns a raw profile mapping into the typed dataclasses — validating the
per-server and per-section shapes that only the parser can see (MCP
command/url/port exclusivity, mapping-vs-scalar sections) and warning on
top-level keys that are almost certainly typos. :func:`load_profile` is the
plain single-file entry point; the preset/override/``--set`` layering path in
:mod:`osprey.cli.build_profile_resolve` reuses :func:`_parse_profile` after it has
assembled its own raw dict.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from osprey.errors import BuildProfileError

from .build_profile_merge import _resolve_extends
from .build_profile_model import BuildProfile
from .build_profile_schema import (
    BlueskyConfig,
    BlueskyPanelsConfig,
    DispatchConfig,
    EnvConfig,
    LifecycleConfig,
    LifecycleStep,
    McpServerDef,
    NextcloudBridgeProfileConfig,
    ServiceDef,
    VAConfig,
)

_LOGGER = logging.getLogger(__name__)


def load_profile(path: Path) -> BuildProfile:
    """Load a build profile from YAML.

    Args:
        path: Path to the profile YAML file.

    Returns:
        Parsed and validated BuildProfile.

    Raises:
        BuildProfileError: If the file is invalid or validation fails.
    """
    if not path.exists():
        raise BuildProfileError(f"Profile not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise BuildProfileError(f"Invalid YAML in {path}: {e}") from e

    if not isinstance(raw, dict):
        raise BuildProfileError(f"Profile must be a YAML mapping, got {type(raw).__name__}")

    raw = _resolve_extends(raw, path.resolve())

    profile = _parse_profile(raw)
    profile_dir = path.parent

    profile.validate(profile_dir)
    return profile


# Top-level keys recognized by BuildProfile. Anything else is almost certainly
# a typo of one of these (e.g. mcp_server vs mcp_servers).
_KNOWN_PROFILE_KEYS = frozenset(
    {
        "name",
        "extends",
        "exclude",
        "data_bundle",
        "deploy_services",
        "provider",
        "model",
        "channel_finder_mode",
        "tier",
        "config",
        "overlay",
        "mcp_servers",
        "services",
        "lifecycle",
        "env",
        "dependencies",
        "requires_osprey_version",
        "osprey_install",
        "python_env",
        "hooks",
        "rules",
        "skills",
        "agents",
        "output_styles",
        "web_panels",
        "default_panel",
        "panel_presets",
        "claude_md_template",
        "categories",
        "dispatch",
        "bluesky",
        "virtual_accelerator",
        "bluesky_panels",
        "nextcloud_bridge",
    }
)


def _warn_unknown_keys(raw: dict[str, Any]) -> None:
    """Warn (don't abort) on unknown top-level profile keys.

    Lenient first; promote to a hard error after one release cycle if it
    stays clean. See cleanup item C11.
    """
    unknown = sorted(set(raw.keys()) - _KNOWN_PROFILE_KEYS)
    for key in unknown:
        _LOGGER.warning(
            "Unknown profile key %r — ignored. Did you mean one of: %s?",
            key,
            ", ".join(sorted(_KNOWN_PROFILE_KEYS)),
        )


def _parse_profile(raw: dict[str, Any]) -> BuildProfile:
    """Parse raw YAML dict into a BuildProfile."""
    _warn_unknown_keys(raw)
    mcp_servers: dict[str, McpServerDef] = {}
    for name, sdef in raw.get("mcp_servers", {}).items():
        if not isinstance(sdef, dict):
            raise BuildProfileError(f"MCP server '{name}' must be a mapping")
        perms = sdef.get("permissions", {})
        url = sdef.get("url")
        command = sdef.get("command", "")
        port = sdef.get("port")
        if port is not None and (
            not isinstance(port, int) or isinstance(port, bool) or not (1 <= port <= 65535)
        ):
            raise BuildProfileError(
                f"MCP server '{name}' port must be an integer in 1..65535 (got {port!r})"
            )
        if url and command:
            raise BuildProfileError(
                f"MCP server '{name}' has both 'command' and 'url' — use one or the other"
            )
        if port is not None and command:
            raise BuildProfileError(
                f"MCP server '{name}' has both 'command' and 'port' — stdio servers cannot declare a port"
            )
        # Derive url from port when only port is set (HTTP host-published service).
        # Web terminals run host-networked, so localhost is the right host for .mcp.json.
        if port is not None and not url:
            url = f"http://localhost:{port}/mcp"
        if not url and not command:
            raise BuildProfileError(f"MCP server '{name}' must have either 'command' or 'url'")
        mcp_servers[name] = McpServerDef(
            command=command,
            args=sdef.get("args", []),
            env=sdef.get("env", {}),
            permissions={
                "allow": perms.get("allow", []),
                "ask": perms.get("ask", []),
            },
            url=url,
            port=port,
        )

    services: dict[str, ServiceDef] = {}
    for name, sdef in raw.get("services", {}).items():
        if not isinstance(sdef, dict):
            raise BuildProfileError(f"Service '{name}' must be a mapping")
        services[name] = ServiceDef(
            template=sdef.get("template", ""),
            config=sdef.get("config", {}),
        )

    lifecycle_raw = raw.get("lifecycle", {})
    lifecycle = LifecycleConfig(
        pre_build=[LifecycleStep(**s) for s in lifecycle_raw.get("pre_build", [])],
        post_build=[LifecycleStep(**s) for s in lifecycle_raw.get("post_build", [])],
        validate=[LifecycleStep(**s) for s in lifecycle_raw.get("validate", [])],
    )

    env_raw = raw.get("env", {})
    env = EnvConfig(
        required=env_raw.get("required", []),
        defaults=env_raw.get("defaults", {}),
        file=env_raw.get("file"),
    )

    dependencies = raw.get("dependencies", [])

    dispatch_raw = raw.get("dispatch")
    dispatch = None
    if dispatch_raw is not None:
        if not isinstance(dispatch_raw, dict):
            raise BuildProfileError("Profile 'dispatch' must be a mapping")
        dispatch = DispatchConfig(
            triggers=dispatch_raw.get("triggers", ""),
            worker_count=dispatch_raw.get("worker_count", 1),
            workspace_mode=dispatch_raw.get("workspace_mode", "isolated"),
            max_concurrent_runs=dispatch_raw.get("max_concurrent_runs", 2),
            max_queue_depth=dispatch_raw.get("max_queue_depth", 50),
            dispatcher_port=dispatch_raw.get("dispatcher_port", 8020),
            worker_port_base=dispatch_raw.get("worker_port_base", 9190),
            timeout_sec=dispatch_raw.get("timeout_sec", 300),
            inactivity_sec=dispatch_raw.get("inactivity_sec", 120),
            facility_name=dispatch_raw.get("facility_name", ""),
            pv_strip_prefix=dispatch_raw.get("pv_strip_prefix", ""),
        )

    bluesky_raw = raw.get("bluesky")
    bluesky = None
    if bluesky_raw is not None:
        if not isinstance(bluesky_raw, dict):
            raise BuildProfileError("Profile 'bluesky' must be a mapping")
        excluded_plans = bluesky_raw.get("excluded_plans", [])
        if not isinstance(excluded_plans, list) or not all(
            isinstance(p, str) for p in excluded_plans
        ):
            raise BuildProfileError(
                "bluesky.excluded_plans must be a list of plan-name strings "
                f"(got {excluded_plans!r})"
            )
        bluesky = BlueskyConfig(
            port=bluesky_raw.get("port", 8090),
            tiled_enabled=bluesky_raw.get("tiled_enabled", False),
            tiled_port=bluesky_raw.get("tiled_port", 8091),
            demo_runner=bluesky_raw.get("demo_runner", False),
            plan_dir=bluesky_raw.get("plan_dir"),
            excluded_plans=excluded_plans,
        )

    va_raw = raw.get("virtual_accelerator")
    virtual_accelerator = None
    if va_raw is not None:
        if not isinstance(va_raw, dict):
            raise BuildProfileError("Profile 'virtual_accelerator' must be a mapping")
        virtual_accelerator = VAConfig(
            port=va_raw.get("port", 5064),
        )

    bluesky_panels_raw = raw.get("bluesky_panels")
    bluesky_panels = None
    if bluesky_panels_raw is not None:
        if not isinstance(bluesky_panels_raw, dict):
            raise BuildProfileError("Profile 'bluesky_panels' must be a mapping")
        bluesky_panels = BlueskyPanelsConfig(
            port=bluesky_panels_raw.get("port", 8095),
        )

    nextcloud_bridge_raw = raw.get("nextcloud_bridge")
    nextcloud_bridge = None
    if nextcloud_bridge_raw is not None:
        if not isinstance(nextcloud_bridge_raw, dict):
            raise BuildProfileError("Profile 'nextcloud_bridge' must be a mapping")
        nextcloud_bridge = NextcloudBridgeProfileConfig(
            trigger=nextcloud_bridge_raw.get("trigger", "nextcloud-question"),
        )

    return BuildProfile(
        name=raw.get("name", ""),
        data_bundle=raw.get("data_bundle", "control_assistant"),
        deploy_services=raw.get("deploy_services", True),
        provider=raw.get("provider"),
        model=raw.get("model"),
        channel_finder_mode=raw.get("channel_finder_mode"),
        tier=(int(raw["tier"]) if raw.get("tier") is not None else None),
        config=raw.get("config", {}),
        overlay=raw.get("overlay", {}),
        mcp_servers=mcp_servers,
        services=services,
        lifecycle=lifecycle,
        env=env,
        dependencies=dependencies,
        requires_osprey_version=raw.get("requires_osprey_version"),
        osprey_install=raw.get("osprey_install", "local"),
        python_env=raw.get("python_env", "project"),
        hooks=raw.get("hooks", []),
        rules=raw.get("rules", []),
        skills=raw.get("skills", []),
        agents=raw.get("agents", []),
        output_styles=raw.get("output_styles", []),
        web_panels=raw.get("web_panels", []),
        default_panel=raw.get("default_panel"),
        panel_presets=raw.get("panel_presets", {}),
        claude_md_template=raw.get("claude_md_template"),
        categories=raw.get("categories", {}),
        dispatch=dispatch,
        bluesky=bluesky,
        virtual_accelerator=virtual_accelerator,
        bluesky_panels=bluesky_panels,
        nextcloud_bridge=nextcloud_bridge,
    )
