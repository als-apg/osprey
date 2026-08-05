"""The ``BuildProfile`` dataclass — the parsed shape of a profile and its validator.

Holds the 34 profile fields, the paradigm-aware tier default, and the
consistency checks a profile must pass before a build touches disk. Kept
separate from the YAML loader so the shape and its rules can be imported (and
constructed in tests) without pulling in preset resolution or ``extends``
merging.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from osprey.build.build_tiers import (
    VALID_CHANNEL_FINDER_MODES,
    default_tier_for_mode,
    tier_mode_conflict,
)
from osprey.errors import BuildProfileError
from osprey.profiles.web_panels import BUILTIN_PANELS

from .build_profile_presets import _triggers_dir
from .build_profile_schema import (
    _ENV_VAR_RE,
    BlueskyConfig,
    BlueskyPanelsConfig,
    DispatchConfig,
    EnvConfig,
    EnvironmentConfig,
    LifecycleConfig,
    McpServerDef,
    NextcloudBridgeProfileConfig,
    ServiceDef,
    VAConfig,
)

# VALID_CHANNEL_FINDER_MODES / default_tier_for_mode / tier_mode_conflict are
# imported from the build-time kernel (osprey.build.build_tiers) so the
# validators below can use them while the definitions live below the cli layer.


@dataclass
class BuildProfile:
    """Complete build profile parsed from YAML."""

    name: str
    data_bundle: str = "control_assistant"
    data: str | None = None
    """Facility data tree this profile carries, as a path relative to the
    profile directory (``data`` for a materialized profile, ``../data`` for a
    sibling persona profile that shares its parent's tree; may therefore
    resolve above the profile dir). When set, the build copies this tree
    instead of the bundled ``apps/<data_bundle>/data/`` — a full replacement,
    not a layered fallback. Meaningless for ``--preset`` builds, which have no
    profile directory to anchor it against; the resolution point for both the
    validator and the build is :meth:`resolved_data_root`.
    """
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
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    """Python-environment declaration (``environment:``). Always present; an
    absent block parses to an all-default :class:`EnvironmentConfig` whose
    ``python`` is ``None`` and whose lists are empty, so consumers never need a
    ``None`` check. Coexists with :attr:`dependencies` — both contribute
    packages to the built environment."""
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
    artifact_server: dict[str, Any] = field(default_factory=dict)
    """Overrides merged into config.yml's ``artifact_server`` block (the
    artifacts-gallery server). Supported subkeys: ``host``, ``port``,
    ``auto_launch``, and ``categories`` — custom gallery categories as
    ``{key: {label, color}}`` that facility MCP tools save artifacts under via
    ``category="<key>"``.
    """
    dispatch: DispatchConfig | None = None
    bluesky: BlueskyConfig | None = None
    virtual_accelerator: VAConfig | None = None
    bluesky_panels: BlueskyPanelsConfig | None = None
    nextcloud_bridge: NextcloudBridgeProfileConfig | None = None

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

    def resolved_data_root(self, profile_dir: Path) -> Path | None:
        """Resolve the profile's ``data:`` tree against its profile directory.

        The single anchoring point shared by :meth:`validate` and the build, so
        the tree the validator checks is the tree the build copies.

        Args:
            profile_dir: Directory holding the profile file.

        Returns:
            The resolved data root, or ``None`` when the profile declares no
            ``data:`` (or declares it with a non-string value, which
            :meth:`validate` reports).
        """
        if not isinstance(self.data, str) or not self.data.strip():
            return None
        return (profile_dir / self.data).resolve()

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

    def _validate_environment(self) -> list[str]:
        """Return validation errors for the ``environment:`` block (empty when clean).

        Takes no ``profile_dir``: ``environment.python`` names a host
        interpreter, not a profile-relative asset, so it must be absolute. That
        keeps :meth:`EnvironmentConfig.resolved_python` and
        :meth:`EnvironmentConfig.venv_base` usable by build-time consumers that
        hold only the profile.
        """
        errors: list[str] = []
        cfg = self.environment

        python_usable = False
        if cfg.python is not None:
            if not isinstance(cfg.python, str) or not cfg.python.strip():
                errors.append(
                    f"environment.python must be a non-empty string path (got {cfg.python!r})"
                )
            else:
                resolved = cfg.resolved_python()
                assert resolved is not None  # non-empty string → never None
                if not resolved.is_absolute():
                    errors.append(
                        f"environment.python must be an absolute path to a Python "
                        f"interpreter (got {cfg.python!r})"
                    )
                elif not resolved.exists():
                    errors.append(f"environment.python not found: {resolved}")
                elif not (resolved.is_file() and os.access(resolved, os.X_OK)):
                    errors.append(f"environment.python is not an executable file: {resolved}")
                else:
                    python_usable = True

        for label, entries in (
            ("packages", cfg.packages),
            ("inherit_exclude", cfg.inherit_exclude),
        ):
            if not isinstance(entries, list):
                errors.append(
                    f"environment.{label} must be a list of strings (got {type(entries).__name__})"
                )
                continue
            for entry in entries:
                if not isinstance(entry, str) or not entry.strip():
                    errors.append(
                        f"environment.{label} entries must be non-empty strings (got {entry!r})"
                    )

        # inherit_exclude only means something when there is a venv base to
        # exclude distributions *from*. A bare interpreter carries no installed
        # set to freeze, so the key would be a silent no-op — reject it rather
        # than let a facility believe an exclusion took effect. Stays quiet when
        # the interpreter itself is already reported bad: one root cause, one error.
        if isinstance(cfg.inherit_exclude, list) and cfg.inherit_exclude:
            if cfg.python is None:
                errors.append(
                    "environment.inherit_exclude requires environment.python to point at an "
                    "existing venv's interpreter (there is nothing to exclude from)"
                )
            elif python_usable and cfg.venv_base() is None:
                errors.append(
                    f"environment.inherit_exclude requires a venv base, but "
                    f"environment.python is a bare interpreter (no pyvenv.cfg beside "
                    f"{cfg.resolved_python()})"
                )

        return errors

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

        # The data tree may legitimately sit above the profile dir (persona
        # profiles share their parent's tree via ``data: ../data``), so only
        # existence and shape are checked, never containment.
        if self.data is not None:
            if not isinstance(self.data, str) or not self.data.strip():
                errors.append(f"data must be a non-empty directory path (got {self.data!r})")
            else:
                data_root = self.resolved_data_root(profile_dir)
                assert data_root is not None  # narrows for type-checkers
                if not data_root.exists():
                    errors.append(f"data directory not found: {self.data} (resolved: {data_root})")
                elif not data_root.is_dir():
                    errors.append(f"data must be a directory: {self.data} (resolved: {data_root})")

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

        # Validate the environment block (interpreter base + extra packages)
        errors.extend(self._validate_environment())

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

        # Validate the artifact_server override block (gallery server settings
        # + custom category definitions)
        import re

        _hex_re = re.compile(r"^#[0-9a-fA-F]{6}$")
        if not isinstance(self.artifact_server, dict):
            errors.append(
                f"artifact_server must be a mapping (got {type(self.artifact_server).__name__})"
            )
        else:
            _allowed = {"host", "port", "auto_launch", "categories"}
            for unknown in sorted(set(self.artifact_server) - _allowed):
                errors.append(
                    f"artifact_server.{unknown} is not a supported key "
                    f"(must be one of {sorted(_allowed)})"
                )
            raw_categories = self.artifact_server.get("categories", {})
            if not isinstance(raw_categories, dict):
                errors.append("artifact_server.categories must be a mapping of category ids")
                raw_categories = {}
            for cat_key, cat_spec in raw_categories.items():
                if not isinstance(cat_spec, dict):
                    errors.append(f"Category '{cat_key}' must be a mapping with label and color")
                    continue
                if "label" not in cat_spec or not isinstance(cat_spec.get("label"), str):
                    errors.append(f"Category '{cat_key}' missing or invalid 'label'")
                if "color" not in cat_spec or not _hex_re.match(str(cat_spec.get("color", ""))):
                    errors.append(
                        f"Category '{cat_key}' missing or invalid 'color' (must be #RRGGBB)"
                    )

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

        # Validate nextcloud_bridge configuration
        if self.nextcloud_bridge is not None:
            nb = self.nextcloud_bridge
            if not nb.trigger:
                errors.append(
                    "nextcloud_bridge.trigger is required: name the dispatch trigger the "
                    "bridge fires, declared in the dispatch.triggers file"
                )
            # The bridge does nothing but POST questions to the dispatcher's
            # webhook, so a bridge without the dispatch pair would deploy and
            # then fail every message. Reject it at build time instead.
            if self.dispatch is None:
                errors.append(
                    "nextcloud_bridge requires a 'dispatch:' block: the bridge dispatches "
                    "every Talk mention to the event dispatcher's webhook. Add a dispatch "
                    f"block whose triggers file declares {nb.trigger!r}, or remove the "
                    "nextcloud_bridge block."
                )
            elif nb.trigger and self.dispatch.triggers:
                # Check the trigger against the SOURCE triggers file, resolved the
                # same way the dispatch block above resolves it (profile-relative
                # first, then bundled). A bridge pointed at an undeclared trigger
                # builds and deploys cleanly and then 404s on every message.
                triggers_file = next(
                    (
                        candidate
                        for candidate in (
                            profile_dir / self.dispatch.triggers,
                            _triggers_dir() / self.dispatch.triggers,
                        )
                        if candidate.is_file()
                    ),
                    None,
                )
                # An unresolvable path is already reported by the dispatch block.
                if triggers_file is not None:
                    # Deferred import: keeps osprey.dispatch out of this module's
                    # import graph for every profile that declares no bridge.
                    from osprey.dispatch.trigger_config import load_triggers

                    try:
                        _, declared_triggers = load_triggers(str(triggers_file))
                    except (OSError, ValueError, yaml.YAMLError) as e:
                        errors.append(
                            f"nextcloud_bridge.trigger cannot be checked: dispatch.triggers "
                            f"file {triggers_file} failed to parse ({e}); fix that file first"
                        )
                    else:
                        names = sorted(t.name for t in declared_triggers)
                        if nb.trigger not in names:
                            errors.append(
                                f"nextcloud_bridge.trigger {nb.trigger!r} is not declared in "
                                f"dispatch.triggers file {self.dispatch.triggers!r} "
                                f"(declares: {names}). Add a trigger named {nb.trigger!r} to "
                                f"that file, or set nextcloud_bridge.trigger to one of them."
                            )

        if errors:
            raise BuildProfileError(
                "Build profile validation failed:\n  - " + "\n  - ".join(errors)
            )
