"""Build profile data model and loader.

A build profile is a YAML file that describes how to assemble a
facility-specific assistant project from an OSPREY template plus
overlay files, config overrides, and MCP server definitions.
"""

from __future__ import annotations

import importlib.resources
import logging
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from osprey.build.build_tiers import (
    VALID_CHANNEL_FINDER_MODES,
    default_tier_for_mode,
    tier_mode_conflict,
)
from osprey.errors import BuildProfileError
from osprey.profiles.web_panels import BUILTIN_PANELS

_LOGGER = logging.getLogger("osprey.cli.build_profile")

# VALID_CHANNEL_FINDER_MODES / default_tier_for_mode / tier_mode_conflict are
# re-imported from the build-time kernel (osprey.build.build_tiers) so this
# module's validators can use them while the definitions live below the cli
# layer.


@dataclass
class McpServerDef:
    """Definition of an MCP server to inject into a built project."""

    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    permissions: dict[str, list[str]] = field(default_factory=dict)
    # permissions: {"allow": ["tool1"], "ask": ["tool2"]}
    url: str | None = None  # HTTP/SSE transport URL (mutually exclusive with command)
    # Single port the HTTP MCP service binds AND publishes. Compose maps
    # host:port → container:port 1:1, so consumers can derive every URL
    # variant from this single value. Mutually exclusive with command;
    # compatible with url (a port hint for non-Claude consumers).
    port: int | None = None


@dataclass
class LifecycleStep:
    """A single command to run during a lifecycle phase."""

    name: str
    run: str
    cwd: str | None = None
    timeout: int = 120  # seconds; override per-step in YAML
    stream: bool = False  # stream stdout in real-time for this step


@dataclass
class LifecycleConfig:
    """Lifecycle commands run before/after build and for validation."""

    pre_build: list[LifecycleStep] = field(default_factory=list)
    post_build: list[LifecycleStep] = field(default_factory=list)
    validate: list[LifecycleStep] = field(default_factory=list)


@dataclass
class EnvConfig:
    """Environment variable template configuration."""

    required: list[str] = field(default_factory=list)
    defaults: dict[str, str] = field(default_factory=dict)
    file: str | None = None  # Profile-relative path to copy as .env


@dataclass
class ServiceDef:
    """Definition of a container service for ``osprey deploy``."""

    template: str  # Path to template dir (relative to profile dir)
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class DispatchConfig:
    """Event-dispatch configuration for a build profile (opt-in via the ``dispatch:`` key).

    Consumed by the build pipeline's dispatch-injection step to deploy the
    event_dispatcher + dispatch_worker services. All ports/counts are validated
    by :meth:`BuildProfile.validate`.
    """

    # Bundled trigger-file name (e.g. "tutorial_triggers.yml") or profile-relative path.
    triggers: str
    worker_count: int = 1
    workspace_mode: Literal["isolated", "shared"] = "isolated"
    max_concurrent_runs: int = 2
    max_queue_depth: int = 50
    dispatcher_port: int = 8020
    worker_port_base: int = 9190
    timeout_sec: int = 300
    inactivity_sec: int = 120
    facility_name: str = ""
    pv_strip_prefix: str = ""


@dataclass
class BlueskyConfig:
    """Bluesky scan-bridge configuration for a build profile (opt-in via the ``bluesky:`` key).

    Consumed by the build pipeline's bluesky-injection step to deploy the
    single ``bluesky_bridge`` service (see NAMING-ADDENDUM.md: deploy key
    ``bluesky``, env var ``BLUESKY_LAUNCH_TOKEN``, MCP server name ``scan``).
    Ports are validated by :meth:`BuildProfile.validate`.
    """

    port: int = 8090
    tiled_enabled: bool = False
    tiled_port: int = 8091
    demo_runner: bool = False
    """Opt-in only for the deploy-smoke-demo / tutorial case: wires the
    container's bridge process to a real bluesky RunEngine against mock
    ophyd-async devices (``devices/mock.py``) via app.py's guarded startup
    hook (task 2.14a), instead of the Phase 1 no-op ``FakePlanRunner`` default.
    MUST stay False for any facility wiring real EPICS hardware — turning
    this on would silently override real device/plan wiring with an
    in-memory mock runner.
    """
    plan_dir: str | None = None
    """Optional host directory of facility plan files (Task 1.4),
    bind-mounted read-only into the bridge container and surfaced to the
    plan loader as a ``BLUESKY_PLAN_DIRS`` (facility-tier) layer — see
    ``plan_loader.py``. ``None`` (default) deploys the bridge with no
    facility plan directory, matching every prior bluesky-only build.
    """
    excluded_plans: list[str] = field(default_factory=list)
    """Named plans to hide from the agent while the bluesky server stays
    enabled (dev/local convenience). Production uses the
    ``BLUESKY_EXCLUDED_PLANS`` env var instead.
    """


@dataclass
class VAConfig:
    """Virtual Accelerator soft-IOC configuration for a build profile (opt-in
    via the ``virtual_accelerator:`` key).

    Consumed by the build pipeline's VA-injection step to deploy the single
    ``virtual_accelerator`` service (compose service ``virtual-accelerator``,
    container ``<project>-virtual-accelerator``). Port is validated by
    :meth:`BuildProfile.validate`.
    """

    port: int = 5064
    """Channel Access TCP port the soft-IOC serves PVs on (see
    src/osprey/services/virtual_accelerator/entrypoint.py's run contract)."""


@dataclass
class BlueskyPanelsConfig:
    """Scan-panels sidecar configuration for a build profile (opt-in via the
    ``bluesky_panels:`` key).

    Consumed by the build pipeline's bluesky-panels-injection step
    (``_inject_bluesky_panels`` in ``build_cmd.py``) to deploy the single
    ``bluesky_panels`` FastAPI sidecar (compose service ``bluesky-panels``) that
    serves the three operator web panels (``plan``, ``results``,
    ``health``) and read-proxies the bluesky bridge. Port is validated
    by :meth:`BuildProfile.validate`.
    """

    port: int = 8095
    """Host/container port the sidecar's uvicorn process binds and publishes
    (see ``templates/services/bluesky_panels/docker-compose.yml.j2``)."""


_ENV_VAR_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")


# ---------------------------------------------------------------------------
# Profile inheritance helpers
# ---------------------------------------------------------------------------


def _merge_lists(base: list, child: list) -> list:
    """Merge two YAML lists.

    String lists: union with dedup, base order preserved.
    Other lists (e.g. lifecycle step dicts): concatenate.
    """
    if not base and not child:
        return []
    all_items = base + child
    if all(isinstance(x, str) for x in all_items):
        seen: set[str] = set()
        merged: list[str] = []
        for item in all_items:
            if item not in seen:
                seen.add(item)
                merged.append(item)
        return merged
    return list(base) + list(child)


def _deep_merge(base: dict, child: dict) -> dict:
    """Deep-merge two raw YAML profile dicts (child wins on conflict)."""
    merged = dict(base)
    for key, child_val in child.items():
        if key not in base:
            merged[key] = child_val
        else:
            base_val = base[key]
            if isinstance(base_val, dict) and isinstance(child_val, dict):
                merged[key] = _deep_merge(base_val, child_val)
            elif isinstance(base_val, list) and isinstance(child_val, list):
                merged[key] = _merge_lists(base_val, child_val)
            else:
                merged[key] = child_val
    return merged


# String-list profile fields that ``exclude:`` may subtract from. Deliberately
# excludes dict-shaped fields (config, overlay, mcp_servers, services, ...) —
# list subtraction only makes sense for the plain string collections a child
# inherits via ``extends``.
_EXCLUDABLE_FIELDS: frozenset[str] = frozenset(
    {
        "skills",
        "rules",
        "hooks",
        "agents",
        "output_styles",
        "web_panels",
        "dependencies",
    }
)


def _apply_exclude(merged: dict[str, Any], exclude: Any) -> None:
    """Subtract ``exclude`` entries from the string-list fields of ``merged`` in place.

    ``exclude`` is a mapping of field name (one of :data:`_EXCLUDABLE_FIELDS`) to a
    list of entries to remove. Excluding an entry that is not present is a silent
    no-op. Because this runs after each ``_deep_merge`` in :func:`_resolve_extends`,
    a deeper ``extends`` layer that re-adds an entry merges in afterwards and wins;
    an entry re-added by an override file or ``--set`` merges *before* extends
    resolution and is stripped again here, so it cannot win.

    Args:
        merged: The merged raw profile dict (mutated in place).
        exclude: The raw ``exclude`` value from a profile layer.

    Raises:
        BuildProfileError: If ``exclude`` is not a mapping, names an unknown or
            non-list-shaped field, or maps a field to a non-list value.
    """
    if not isinstance(exclude, dict):
        raise BuildProfileError(
            f"Profile 'exclude' must be a mapping of field name to list "
            f"(got {type(exclude).__name__})"
        )
    for field_name, entries in exclude.items():
        if field_name not in _EXCLUDABLE_FIELDS:
            raise BuildProfileError(
                f"exclude: unknown or non-list field {field_name!r} "
                f"(must be one of {sorted(_EXCLUDABLE_FIELDS)})"
            )
        if not isinstance(entries, list):
            raise BuildProfileError(
                f"exclude.{field_name} must be a list of entries to remove "
                f"(got {type(entries).__name__})"
            )
        current = merged.get(field_name)
        if not isinstance(current, list):
            continue
        removal = set(entries)
        merged[field_name] = [item for item in current if item not in removal]


def _resolve_extends(
    raw: dict[str, Any], profile_path: Path, chain: list[Path] | None = None
) -> dict[str, Any]:
    """Resolve ``extends`` chain, returning a fully merged raw YAML dict.

    Args:
        raw: The raw YAML dict from the current file.
        profile_path: Resolved path to the current YAML file.
        chain: Paths already visited (for circular-reference detection).

    Returns:
        Merged raw dict with ``extends`` consumed.

    Raises:
        BuildProfileError: On missing base, circular reference, or bad YAML.
    """
    if chain is None:
        chain = []

    resolved = profile_path.resolve()
    if resolved in chain:
        cycle = " -> ".join(str(p) for p in chain) + f" -> {resolved}"
        raise BuildProfileError(f"Circular extends detected: {cycle}")
    chain.append(resolved)

    extends_value = raw.pop("extends", None)
    if extends_value is None:
        # No base to subtract from — ``exclude`` here can only touch this file's
        # own declarations, which is an author mistake. Apply-to-self (a no-op in
        # practice) and log so it's discoverable, matching the recursive path's
        # "pop exclude before returning" contract.
        exclude_value = raw.pop("exclude", None)
        if exclude_value is not None:
            _LOGGER.debug(
                "Profile %s declares 'exclude' without 'extends'; it can only "
                "affect its own declarations (no inherited entries to remove).",
                resolved,
            )
            _apply_exclude(raw, exclude_value)
        return raw

    # Try a bundled preset by name first; fall through to filesystem-path
    # resolution. Path-shaped values like ``als-base.yml`` correctly miss the
    # preset probe (it looks up ``als-base.yml.yml``) and resolve as paths,
    # preserving the sibling-file semantics ALS-style profiles depend on.
    preset_path = _preset_exists(extends_value)
    if preset_path is not None:
        base_path = preset_path
    else:
        base_path = (profile_path.parent / extends_value).resolve()
        if not base_path.exists():
            available = ", ".join(list_presets()) or "(none)"
            raise BuildProfileError(
                f"Cannot resolve extends: {extends_value!r}. "
                f"No bundled preset by that name (available: {available}), "
                f"and no file at {base_path}."
            )

    try:
        base_raw = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise BuildProfileError(f"Invalid YAML in {base_path}: {e}") from e

    if not isinstance(base_raw, dict):
        raise BuildProfileError(f"Extended profile must be a YAML mapping: {base_path}")

    # Recurse: the base may itself extend another profile
    base_raw = _resolve_extends(base_raw, base_path, chain)

    merged = _deep_merge(base_raw, raw)
    # Apply this layer's ``exclude`` to the merged result and consume it. The
    # recursively-resolved ``base_raw`` has already had its own ``exclude``
    # popped, so the only ``exclude`` present here is this layer's own.
    exclude_value = merged.pop("exclude", None)
    if exclude_value is not None:
        _apply_exclude(merged, exclude_value)
    return merged


@dataclass
class BuildProfile:
    """Complete build profile parsed from YAML."""

    name: str
    data_bundle: str = "control_assistant"
    deploy_services: bool = True
    """Whether this project scaffolds its own container-services stack.

    ``True`` (default) builds a self-contained, deployable project: service
    templates are copied and ``services.*``/``deployed_services`` config is
    written for every declared/injected service.

    ``False`` marks an *attached* project — one that connects to a services
    stack deployed by another OSPREY project on the same host. Service sections
    in the profile (own or inherited) are parsed and validated but scaffold
    nothing: no ``services/`` directory, no ``services.*`` blocks, and an empty
    ``deployed_services`` list. Its terminal images reach the shared stack via
    client config (e.g. ``bluesky.bridge_url``) over host networking.
    """
    provider: str | None = None
    model: str | None = None
    channel_finder_mode: str | None = None
    tier: int | None = None
    """Channel-database tier (1|3) selecting which preset `tiers/tier{N}` DB
    is materialized at build time to the flat `data/channel_databases/<name>.json`
    location. Tier 1 is in_context-only; tier 3 carries all three paradigms.
    When ``None``, the build resolves a paradigm-aware default via
    :meth:`resolved_tier` (in_context → 1, hierarchical/middle_layer → 3).
    This is build-time only and is NOT rendered into `config.yml`; the runtime
    config carries no tier knob. Facility profiles can ignore it because the
    DB they overlay overwrites whatever the preset put there.
    """
    config: dict[str, Any] = field(default_factory=dict)
    overlay: dict[str, str] = field(default_factory=dict)
    mcp_servers: dict[str, McpServerDef] = field(default_factory=dict)
    services: dict[str, ServiceDef] = field(default_factory=dict)
    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    dependencies: list[str] = field(default_factory=list)
    requires_osprey_version: str | None = None  # PEP 440 specifier, e.g. ">=0.12.0"
    osprey_install: str = (
        # "local" (auto-detect from importlib.metadata: editable → source tree,
        # otherwise pin to running version) | "pip" | PEP 508 spec
        # (e.g. "osprey-framework==2026.5.0")
        "local"
    )
    python_env: str = "project"  # "project" | "build" | absolute path to Python executable
    hooks: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)
    output_styles: list[str] = field(default_factory=list)
    web_panels: list[str] = field(default_factory=list)
    default_panel: str | None = None
    panel_presets: dict[str, list[str]] = field(default_factory=dict)
    """Named panel layouts ("presets") rendered into ``web.presets``. Each key is
    the display label, each value a list of member panel ids (built-ins or
    custom ``web.panels.<id>.url``-backed ids). A human applies one from the
    Web Terminal "+" popover's "Layouts" section. Empty (the default) renders no
    ``web.presets`` block. Members are typo-validated at build time, mirroring
    :attr:`default_panel`.
    """
    claude_md_template: str | None = None
    """Bundled `templates/claude_code/<filename>` to render as CLAUDE.md
    (default: "CLAUDE.md.j2"). Lets a preset pick an alternate persona
    (e.g. "CLAUDE.ariel.md.j2" for the logbook-research bundle). Internal
    preset-author primitive — facility profiles override CLAUDE.md via
    overlay, not via this key.
    """
    categories: dict[str, dict[str, str]] = field(default_factory=dict)
    dispatch: DispatchConfig | None = None
    bluesky: BlueskyConfig | None = None
    virtual_accelerator: VAConfig | None = None
    bluesky_panels: BlueskyPanelsConfig | None = None

    def resolved_tier(self) -> int:
        """Resolve the build-time tier, applying a paradigm-aware default.

        Returns ``self.tier`` if set; otherwise picks tier 1 for ``in_context``
        and tier 3 for ``hierarchical``/``middle_layer``.  Callers that need a
        concrete integer (the build pipeline, the materializer) MUST go through
        this method rather than reading ``self.tier`` directly.
        """
        if self.tier is not None:
            return self.tier
        return default_tier_for_mode(self.channel_finder_mode)

    def _is_known_panel_id(self, pid: str) -> bool:
        """Return True if ``pid`` names a panel this profile could render.

        A panel id is known when it is a framework built-in, a declared
        ``web_panels`` entry, or a custom panel backed by a
        ``web.panels.<id>.url`` config override. Shared by the ``default_panel``
        and ``panel_presets`` member validation so both reject the same typos
        with the same predicate (a single source of truth, not two drifting
        membership checks).
        """
        if pid in BUILTIN_PANELS:
            return True
        if pid in self.web_panels:
            return True
        return f"web.panels.{pid}.url" in self.config

    def validate(self, profile_dir: Path) -> None:
        """Validate profile consistency. Raises BuildProfileError with all issues."""
        errors: list[str] = []

        if not self.name:
            errors.append("Profile 'name' is required")

        if not isinstance(self.deploy_services, bool):
            errors.append(
                f"deploy_services must be a boolean (got {type(self.deploy_services).__name__})"
            )

        if self.tier is not None and self.tier not in (1, 3):
            errors.append(f"tier must be 1 or 3 (got {self.tier!r})")

        # Tier 1 ships only the in_context paradigm DB; reject a tier/paradigm
        # mismatch here with a rule-naming message (see tier_mode_conflict) so
        # the failure is legible on every configuration path rather than an
        # opaque FileNotFoundError deep in materialize_tier_artifacts.
        conflict = tier_mode_conflict(self.tier, self.channel_finder_mode)
        if conflict:
            errors.append(conflict)

        if (
            self.channel_finder_mode is not None
            and self.channel_finder_mode not in VALID_CHANNEL_FINDER_MODES
        ):
            errors.append(
                f"channel_finder_mode must be one of {VALID_CHANNEL_FINDER_MODES} "
                f"(got {self.channel_finder_mode!r})"
            )

        # Validate overlay source paths exist
        for src, _dst in self.overlay.items():
            src_path = profile_dir / src
            if not src_path.exists():
                errors.append(f"Overlay source not found: {src} (resolved: {src_path})")

        # Path traversal guard on overlay destinations
        for _src, dst in self.overlay.items():
            normalized = Path(dst)
            if normalized.is_absolute() or ".." in normalized.parts:
                errors.append(f"Overlay destination must be relative without '..': {dst}")

        # Validate MCP server definitions
        for name, server in self.mcp_servers.items():
            if not server.command and not server.url:
                errors.append(f"MCP server '{name}' missing 'command' or 'url'")

        # Validate service definitions
        for name, svc in self.services.items():
            if not svc.template:
                errors.append(f"Service '{name}' missing 'template'")
            elif svc.template.startswith("osprey."):
                # Bundled template (e.g. "osprey.event_dispatcher") — resolved at copy
                # time by _copy_service_templates; no profile-dir file to validate.
                continue
            else:
                tmpl_path = profile_dir / svc.template
                if not tmpl_path.is_dir():
                    errors.append(f"Service '{name}' template dir not found: {tmpl_path}")
                elif not (tmpl_path / "docker-compose.yml.j2").exists():
                    errors.append(f"Service '{name}' template dir missing docker-compose.yml.j2")

        # Validate lifecycle steps
        for phase_name in ("pre_build", "post_build", "validate"):
            for step in getattr(self.lifecycle, phase_name):
                if not step.name:
                    errors.append(f"Lifecycle {phase_name} step missing 'name'")
                if not step.run:
                    errors.append(f"Lifecycle {phase_name} step missing 'run'")
                if step.cwd:
                    cwd_path = Path(step.cwd)
                    if cwd_path.is_absolute() or ".." in cwd_path.parts:
                        errors.append(
                            f"Lifecycle {phase_name} step '{step.name}' cwd must be"
                            f" relative without '..': {step.cwd}"
                        )
                if step.timeout <= 0:
                    errors.append(
                        f"Lifecycle {phase_name} step '{step.name}' timeout must be"
                        f" positive: {step.timeout}"
                    )

        # Validate env var names
        for var in self.env.required:
            if not _ENV_VAR_RE.match(var):
                errors.append(f"Invalid env var name: {var}")

        # Validate env file path
        if self.env.file:
            env_file_path = profile_dir / self.env.file
            if not env_file_path.is_file():
                errors.append(f"env.file not found: {self.env.file} (resolved: {env_file_path})")

        # Validate dependencies
        for dep in self.dependencies:
            if not isinstance(dep, str) or not dep.strip():
                errors.append(f"Dependency must be a non-empty string: {dep!r}")

        # Validate requires_osprey_version specifier
        if self.requires_osprey_version:
            try:
                from packaging.specifiers import SpecifierSet

                SpecifierSet(self.requires_osprey_version)
            except Exception:
                errors.append(
                    f"Invalid requires_osprey_version specifier: "
                    f"'{self.requires_osprey_version}' (must be PEP 440, e.g. '>=0.12.0')"
                )

        # Validate web_panels: each entry must either be a built-in (rendered
        # by the framework) or a custom panel backed by a ``web.panels.<id>.url``
        # config override (rendered as an iframe by the web terminal). Catches
        # typos in shipped presets and missing URL backing for facility panels.
        for panel in self.web_panels:
            if panel in BUILTIN_PANELS:
                continue
            url_key = f"web.panels.{panel}.url"
            if url_key in self.config:
                continue
            # The ``events`` panel URL is derived post-build from the dispatch
            # block (``_inject_dispatch`` in build_cmd.py), which runs after this
            # validator. So a dispatch-backed events panel is legitimately
            # url-less here — accept it rather than aborting the build.
            if panel == "events" and self.dispatch is not None:
                continue
            # The three panel ids' URLs are likewise derived post-build
            # (``_inject_bluesky_panels`` in build_cmd.py, which runs after this
            # validator) from the bluesky_panels sidecar's port — so they are
            # legitimately url-less here when a bluesky_panels block is present.
            if panel in ("plan", "results", "health") and self.bluesky_panels is not None:
                continue
            errors.append(
                f"Unknown web_panel {panel!r}: not in BUILTIN_PANELS "
                f"({sorted(BUILTIN_PANELS)}) and no '{url_key}' config override"
            )

        # Validate default_panel: must be a built-in, a declared web_panels
        # entry, or a custom panel backed by a `web.panels.<id>.url` override.
        # Catches typos like `default_panel: areil` that would otherwise
        # silently fall back to the frontend DEFAULT_PANEL_FALLBACK at runtime.
        if self.default_panel is not None and not self._is_known_panel_id(self.default_panel):
            errors.append(
                f"Unknown default_panel {self.default_panel!r}: not in BUILTIN_PANELS "
                f"({sorted(BUILTIN_PANELS)}), not in web_panels, and no "
                f"'web.panels.{self.default_panel}.url' config override"
            )

        # Validate panel_presets: each member id must resolve the same way a
        # default_panel does (built-in, declared web_panels, or url-backed
        # custom). Catches typos in a preset's member list at build time so a
        # facility author gets the same fail-fast feedback as default_panel.
        for preset_name, members in self.panel_presets.items():
            if not isinstance(members, list):
                errors.append(
                    f"panel_presets[{preset_name!r}] must be a list of panel ids "
                    f"(got {type(members).__name__})"
                )
                continue
            for member in members:
                if not self._is_known_panel_id(member):
                    errors.append(
                        f"Unknown panel_presets[{preset_name!r}] member {member!r}: not in "
                        f"BUILTIN_PANELS ({sorted(BUILTIN_PANELS)}), not in web_panels, and no "
                        f"'web.panels.{member}.url' config override"
                    )

        # Validate custom category definitions
        import re

        _hex_re = re.compile(r"^#[0-9a-fA-F]{6}$")
        for cat_key, cat_spec in self.categories.items():
            if not isinstance(cat_spec, dict):
                errors.append(f"Category '{cat_key}' must be a mapping with label and color")
                continue
            if "label" not in cat_spec or not isinstance(cat_spec.get("label"), str):
                errors.append(f"Category '{cat_key}' missing or invalid 'label'")
            if "color" not in cat_spec or not _hex_re.match(str(cat_spec.get("color", ""))):
                errors.append(f"Category '{cat_key}' missing or invalid 'color' (must be #RRGGBB)")

        # Validate dispatch configuration
        if self.dispatch is not None:
            d = self.dispatch
            if d.worker_count < 1:
                errors.append(f"dispatch.worker_count must be >= 1 (got {d.worker_count})")
            if not (1 <= d.dispatcher_port <= 65535):
                errors.append(
                    f"dispatch.dispatcher_port must be in 1..65535 (got {d.dispatcher_port})"
                )
            if not (1 <= d.worker_port_base <= 65535):
                errors.append(
                    f"dispatch.worker_port_base must be in 1..65535 (got {d.worker_port_base})"
                )
            elif d.worker_count >= 1 and (d.worker_port_base + d.worker_count - 1) > 65535:
                errors.append(
                    f"dispatch.worker_port_base + worker_count - 1 exceeds 65535 "
                    f"({d.worker_port_base} + {d.worker_count} - 1)"
                )
            if d.workspace_mode not in ("isolated", "shared"):
                errors.append(
                    f"dispatch.workspace_mode must be 'isolated' or 'shared' "
                    f"(got {d.workspace_mode!r})"
                )
            if d.max_concurrent_runs < 1:
                errors.append(
                    f"dispatch.max_concurrent_runs must be >= 1 (got {d.max_concurrent_runs})"
                )
            if d.max_queue_depth < 1:
                errors.append(f"dispatch.max_queue_depth must be >= 1 (got {d.max_queue_depth})")
            if d.timeout_sec <= 0:
                errors.append(f"dispatch.timeout_sec must be > 0 (got {d.timeout_sec})")
            if d.inactivity_sec <= 0:
                errors.append(f"dispatch.inactivity_sec must be > 0 (got {d.inactivity_sec})")
            # triggers must be a non-empty, resolvable file
            # (profile-relative OR bundled preset name)
            if not d.triggers:
                errors.append(
                    "dispatch.triggers is required (bundled name or profile-relative path)"
                )
            elif (
                not (profile_dir / d.triggers).is_file()
                and not (_triggers_dir() / d.triggers).is_file()
            ):
                errors.append(
                    f"dispatch.triggers file not found: {d.triggers!r} "
                    f"(looked in profile dir {profile_dir} and bundled triggers)"
                )
            # Advisory: multiple workers sharing one workspace can corrupt each other.
            if d.worker_count > 1 and d.workspace_mode == "shared":
                warnings.warn(
                    "dispatch.workspace_mode='shared' with worker_count>1: workers share one "
                    "workspace volume and may clobber each other's files; consider 'isolated'.",
                    UserWarning,
                    stacklevel=2,
                )

        # Validate bluesky configuration
        if self.bluesky is not None:
            b = self.bluesky
            if not (1 <= b.port <= 65535):
                errors.append(f"bluesky.port must be in 1..65535 (got {b.port})")
            if b.tiled_enabled:
                if not (1 <= b.tiled_port <= 65535):
                    errors.append(f"bluesky.tiled_port must be in 1..65535 (got {b.tiled_port})")
                elif b.tiled_port == b.port:
                    errors.append(
                        f"bluesky.tiled_port must differ from bluesky.port (both {b.port})"
                    )

        # Validate virtual_accelerator configuration
        if self.virtual_accelerator is not None:
            va = self.virtual_accelerator
            if not (1 <= va.port <= 65535):
                errors.append(f"virtual_accelerator.port must be in 1..65535 (got {va.port})")

        # Validate bluesky_panels configuration
        if self.bluesky_panels is not None:
            sp = self.bluesky_panels
            if not (1 <= sp.port <= 65535):
                errors.append(f"bluesky_panels.port must be in 1..65535 (got {sp.port})")

        if errors:
            raise BuildProfileError(
                "Build profile validation failed:\n  - " + "\n  - ".join(errors)
            )


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
    }
)


def _warn_unknown_keys(raw: dict[str, Any]) -> None:
    """Warn (don't abort) on unknown top-level profile keys.

    Lenient first; promote to a hard error after one release cycle if it
    stays clean. See cleanup item C11.
    """
    import logging

    logger = logging.getLogger("osprey.cli.build_profile")
    unknown = sorted(set(raw.keys()) - _KNOWN_PROFILE_KEYS)
    for key in unknown:
        logger.warning(
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
    )


# ---------------------------------------------------------------------------
# Bundled-preset helpers
# ---------------------------------------------------------------------------


_PRESETS_PACKAGE = "osprey.profiles.presets"


def _normalize_preset_name(name: str) -> str:
    """Normalize CLI preset spelling to the on-disk filename form.

    CLI accepts both ``control-assistant`` and ``control_assistant``;
    bundled YAML files are hyphenated.
    """
    return name.replace("_", "-")


def _presets_dir() -> Path:
    """Return the directory containing bundled preset YAMLs."""
    return Path(str(importlib.resources.files(_PRESETS_PACKAGE)))


_TRIGGERS_PACKAGE = "osprey.profiles.triggers"


def _triggers_dir() -> Path:
    """Return the directory containing bundled trigger-config YAMLs.

    Distinct from :func:`_presets_dir` — trigger configs are not build presets
    and must not appear in the preset namespace (``--list-presets``).
    """
    return Path(str(importlib.resources.files(_TRIGGERS_PACKAGE)))


def _preset_exists(name: str) -> Path | None:
    """Return the resolved preset path if ``name`` matches a bundled preset, else None.

    Non-raising probe; mirrors :func:`_load_preset_raw`'s lookup so callers
    that need to *try* preset resolution before falling back can do so
    without absorbing an exception. Note that :func:`_normalize_preset_name`
    only translates ``_`` → ``-``; values containing ``.yml`` (e.g. path-style
    ``extends: als-base.yml``) probe as ``als-base.yml.yml`` and correctly miss.
    """
    normalized = _normalize_preset_name(name)
    candidate = _presets_dir() / f"{normalized}.yml"
    return candidate if candidate.is_file() else None


def list_presets() -> list[str]:
    """Return the sorted list of bundled preset names (hyphenated)."""
    return sorted(
        p.name.removesuffix(".yml")
        for p in _presets_dir().iterdir()
        if p.name.endswith(".yml") and not p.name.startswith("_")
    )


def _load_preset_raw(name: str) -> tuple[dict[str, Any], Path]:
    """Read a bundled preset YAML; return (raw_dict, preset_file_path).

    Raises ``BuildProfileError`` if the preset is unknown or invalid YAML.
    """
    normalized = _normalize_preset_name(name)
    target = _presets_dir() / f"{normalized}.yml"
    if not target.exists():
        available = ", ".join(list_presets()) or "(none)"
        raise BuildProfileError(f"Unknown preset {name!r}. Available: {available}")
    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise BuildProfileError(f"Invalid YAML in preset {name!r}: {e}") from e
    if not isinstance(raw, dict):
        raise BuildProfileError(f"Preset {name!r} must be a YAML mapping")
    return raw, target


def _hash_resolved_profile(raw: dict[str, Any], profile_path: Path) -> str:
    """Canonical content hash of a profile dict after ``extends`` resolution.

    Hashes the *resolved* content (canonical JSON, sorted keys) rather than
    file bytes, so comment/ordering churn is invisible while a change in any
    ``extends`` parent is not.
    """
    import hashlib
    import json

    resolved = _resolve_extends(dict(raw), profile_path)
    canonical = json.dumps(resolved, sort_keys=True, default=str)
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def compute_preset_hash(preset_name: str) -> str | None:
    """Content hash of a bundled preset as resolved (post-``extends``).

    Stamped into ``.osprey-manifest.json`` at build time and compared by the
    deploy-side staleness advisory
    (:mod:`osprey.deployment.staleness`). Returns ``None`` when the preset is
    unknown or unreadable — callers treat that as "cannot compare", never as
    drift.
    """
    try:
        raw, path = _load_preset_raw(preset_name)
        return _hash_resolved_profile(raw, path)
    except Exception:
        return None


def compute_profile_hash(profile_path: Path) -> str | None:
    """Content hash of a positional profile YAML as resolved (post-``extends``).

    Counterpart of :func:`compute_preset_hash` for ``osprey build NAME
    PROFILE.yml`` invocations. Returns ``None`` when the file is missing or
    unreadable.
    """
    try:
        path = Path(profile_path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return None
        return _hash_resolved_profile(raw, path)
    except Exception:
        return None


def _parse_set_pairs(pairs: tuple[str, ...]) -> dict[str, Any]:
    """Parse ``--set KEY.PATH=VALUE`` pairs into a nested dict.

    The right-hand side is parsed with ``yaml.safe_load`` so callers get
    type coercion for free: ``true``/``false`` -> bool, ``[a,b]`` -> list,
    bare ints/floats -> numeric, anything else -> string.
    """
    result: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise BuildProfileError(f"--set expects KEY=VALUE (with '='), got: {pair!r}")
        key, _, raw_value = pair.partition("=")
        key = key.strip()
        if not key:
            raise BuildProfileError(f"--set key must be non-empty: {pair!r}")
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as e:
            raise BuildProfileError(f"--set value for {key!r} is not valid YAML: {e}") from e
        target: dict[str, Any] = result
        parts = key.split(".")
        for part in parts[:-1]:
            existing = target.get(part)
            if existing is None:
                existing = {}
                target[part] = existing
            elif not isinstance(existing, dict):
                raise BuildProfileError(
                    f"--set key {key!r} conflicts with earlier scalar at {part!r}"
                )
            target = existing
        target[parts[-1]] = value
    return result


# The model-selection shorthand keys a user can override via `--set` whose
# explicit use is recorded in the project manifest (extract_build_args) and
# re-applied by persona auto-render, so one parent-build override retints
# every derived persona project.
MODEL_SELECTION_OVERRIDE_KEYS = ("provider", "model", "channel_finder_mode")


def explicit_model_override_keys(set_pairs: tuple[str, ...]) -> list[str]:
    """Model-selection keys the user explicitly overrode via bare ``--set``.

    Only top-level shorthand keys count (``--set provider=x``); a dotted path
    into ``config:`` addresses the rendered config directly and carries no
    whole-stack intent, so it is never forwarded to persona renders.

    Returns the matching keys in :data:`MODEL_SELECTION_OVERRIDE_KEYS` order.
    """
    parsed = _parse_set_pairs(set_pairs)
    return [key for key in MODEL_SELECTION_OVERRIDE_KEYS if key in parsed]


def resolve_build_profile(
    profile_path: Path | None,
    preset: str | None,
    overrides: tuple[Path, ...] = (),
    set_pairs: tuple[str, ...] = (),
) -> tuple[BuildProfile, Path]:
    """Resolve a build profile from any combination of preset / file / overlays.

    Mode is determined by which of ``profile_path`` and ``preset`` is given;
    they are mutually exclusive and exactly one is required.

    Layers are applied in order: base -> override file(s) -> --set values.
    All layers are merged via :func:`_deep_merge` (string lists union-dedup,
    other lists concatenate) before ``extends:`` is resolved.

    Returns:
        ``(profile, profile_dir)``. ``profile_dir`` anchors overlay/services
        path lookups in :meth:`BuildProfile.validate`. For preset mode it is
        the bundled ``profiles/presets/`` package directory.

    Raises:
        BuildProfileError: For mutual-exclusion violations, missing files,
        invalid YAML, or validation failures.
    """
    if profile_path is not None and preset is not None:
        raise BuildProfileError("Pass either a profile path or --preset, not both.")
    if profile_path is None and preset is None:
        raise BuildProfileError("Either a profile path or --preset is required.")

    if preset is not None:
        raw, base_anchor = _load_preset_raw(preset)
        profile_dir = base_anchor.parent
    else:
        assert profile_path is not None  # narrows for type-checkers
        if not profile_path.exists():
            raise BuildProfileError(f"Profile not found: {profile_path}")
        try:
            raw = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            raise BuildProfileError(f"Invalid YAML in {profile_path}: {e}") from e
        if not isinstance(raw, dict):
            raise BuildProfileError(f"Profile must be a YAML mapping, got {type(raw).__name__}")
        base_anchor = profile_path.resolve()
        profile_dir = profile_path.parent

    for override_path in overrides:
        if not override_path.exists():
            raise BuildProfileError(f"Override not found: {override_path}")
        try:
            override_raw = yaml.safe_load(override_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            raise BuildProfileError(f"Invalid YAML in {override_path}: {e}") from e
        if override_raw is None:
            continue
        if not isinstance(override_raw, dict):
            raise BuildProfileError(f"Override must be a YAML mapping: {override_path}")
        raw = _deep_merge(raw, override_raw)

    if set_pairs:
        raw = _deep_merge(raw, _parse_set_pairs(set_pairs))

    raw = _resolve_extends(raw, base_anchor)

    profile = _parse_profile(raw)
    profile.validate(profile_dir)
    return profile, profile_dir
