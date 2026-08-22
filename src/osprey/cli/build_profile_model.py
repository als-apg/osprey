"""The ``BuildProfile`` dataclass — the parsed shape of a profile and its validator.

Holds the 37 profile fields, the paradigm-aware tier default, and the
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
from osprey.deployment.graphdb_service import (
    GRAPHDB_SERVICE_NAME,
    resolve_graphdb_service_config,
)
from osprey.errors import BuildProfileError
from osprey.profiles.web_panels import BUILTIN_PANELS

from .build_profile_archiver import (
    VAArchiverConfig,
    va_archiver_errors,
    va_mock_archiver_errors,
)
from .build_profile_deploy import DeployConfig
from .build_profile_presets import _triggers_dir
from .build_profile_schema import (
    _ENV_VAR_RE,
    BlueskyConfig,
    BlueskyWebConfig,
    DispatchConfig,
    EnvConfig,
    EnvironmentConfig,
    GChatBridgeProfileConfig,
    LifecycleConfig,
    McpServerDef,
    NextcloudBridgeProfileConfig,
    ProfileProvenance,
    ServiceDef,
    VAConfig,
    network_mode_errors,
)
from .profile_conventions import validate_convention_sources

DISPATCH_PAIR_SERVICES = ("event_dispatcher", "dispatch_worker")
"""The two services the ``dispatch:`` block deploys as one unit. They share a
network by construction, so ``dispatch.network:`` is their only network knob
and a per-half ``network:`` is rejected rather than honored."""

_GRAPH_MODE = "graph"
"""The one channel-finder paradigm that answers from a graph store rather than a
channel-database file, and so the one with a service prerequisite. A member name
of :data:`VALID_CHANNEL_FINDER_MODES`, not a second enumeration — the rule below
is about this paradigm alone, exactly as ``tier_mode_conflict`` is."""

_APP_TEMPLATE_ROOT = Path(__file__).resolve().parent.parent / "templates" / "apps"
"""Where the bundled app templates live; ``app_template:`` names one by directory."""

_GRAPHDB_CONFIG_PREFIX = f"services.{GRAPHDB_SERVICE_NAME}"
"""Dotted-key spelling of the graph-store block on a profile's ``config:`` surface."""

_CHANNEL_FINDER_SERVER_ENABLED_KEY = "claude_code.servers.channel-finder.enabled"
"""The switch a persona flips to take the channel finder out of its render."""

_MISSING = object()
"""Sentinel for "this profile says nothing about that key"."""


def _config_lookup(config: Any, dotted_key: str) -> Any:
    """Read *dotted_key* off a profile's ``config:`` overlay, or ``_MISSING``.

    The overlay is authored in dotted keys and that is the only spelling the
    renderer applies, but a hand-written profile can nest the same path — and
    can nest part of it (``claude_code.servers:`` holding a ``channel-finder:``
    mapping). Longest match at each step, so a key that IS spelled dotted is
    found before the walk tries to descend into it.

    Args:
        config: A profile's ``config:`` block, whatever shape it parsed as.
        dotted_key: The full path, in the dotted spelling.

    Returns:
        The value, or :data:`_MISSING` when no spelling of that path is set.
    """
    node: Any = config
    parts = dotted_key.split(".")
    index = 0
    while index < len(parts):
        if not isinstance(node, dict):
            return _MISSING
        for end in range(len(parts), index, -1):
            candidate = ".".join(parts[index:end])
            if candidate in node:
                node = node[candidate]
                index = end
                break
        else:
            return _MISSING
    return node


def _app_template_declares_graphdb(data_bundle: str) -> bool:
    """Whether app template *data_bundle* renders a ``services.graphdb`` block.

    Read off the template source rather than off a render: this validator runs
    before anything reaches disk, and the block is a fixed part of the app
    template — what varies per profile is the ``config:`` overlay applied after
    the render, which :meth:`BuildProfile._renders_graphdb_block` reads itself.

    Args:
        data_bundle: App template directory name (the ``app_template:`` key).

    Returns:
        Whether that template's ``services:`` section carries a ``graphdb``
        key. False for a bundle that ships no ``config.yml.j2``, which is also
        how an unknown bundle name reads here — the build names that fault
        itself, and reporting it twice would not help.
    """
    template = _APP_TEMPLATE_ROOT / data_bundle / "config.yml.j2"
    try:
        text = template.read_text(encoding="utf-8")
    except OSError:
        return False
    # Which top-level section each indented key belongs to is all this needs,
    # so the file is walked by indentation rather than parsed: it is a Jinja
    # template, not YAML, until it is rendered.
    section: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "{%", "{{")):
            continue
        if not line[0].isspace():
            section = stripped.split(":", 1)[0]
            continue
        if section == "services" and stripped.split(":", 1)[0] == GRAPHDB_SERVICE_NAME:
            return True
    return False


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
    deploy: DeployConfig | None = None
    """Where this project is built, pushed, and run (``deploy:``).

    ``None`` for a profile that declares no deployment coordinates — the
    default, and correct for anything only ever built locally. The CI
    scaffolding verbs read it; nothing in the ordinary build path does. Not to
    be confused with :attr:`deploy_services`, which is about the project's own
    container stack rather than where that stack lands.
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
    (e.g. "CLAUDE.ariel.md.j2" for the logbook-research bundle). This is the
    one channel for the rendered CLAUDE.md: the project copy is build-owned, so
    the ``project/`` mirror rejects a hand-written one.
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
    bluesky_web: BlueskyWebConfig | None = None
    nextcloud_bridge: NextcloudBridgeProfileConfig | None = None
    gchat_bridge: GChatBridgeProfileConfig | None = None
    va_archiver: VAArchiverConfig | None = None
    """The stored archive a simulated deployment keeps its history in
    (``va_archiver:``).

    ``None`` for a profile that declares none — history then comes from
    whatever archiver ``config:`` selects, including the synthesizing mock. When
    present, the block carries every knob the store's shape depends on and the
    build derives the connector's connection keys from it (see
    :mod:`osprey.cli.build_profile_archiver`).
    """
    provenance: ProfileProvenance | None = None
    """What ``osprey init`` materialized this profile from (``provenance:``).

    ``None`` for a bundled preset and for a hand-written profile, neither of
    which was materialized from anything. Carried through resolution unchanged:
    the build reads it to compare against the installed preset, and never
    rewrites it — the profile records its own origin, not the last build's.
    """

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

    def _validate_chat_bridge(
        self, key: str, trigger: str, ingested: str, profile_dir: Path
    ) -> list[str]:
        """Return validation errors for one chat-bridge block (empty when clean).

        Every chat bridge is the same shape — an outbound-only ingester that
        does nothing but POST questions to the dispatcher's webhook — so they
        all answer the same three questions: is a trigger named, is there a
        ``dispatch:`` block to post to, and is that trigger actually declared in
        the triggers file. Shared here so a new channel inherits the checks
        rather than re-deriving them.

        Args:
            key: The profile key being validated, used verbatim in every message
                (e.g. ``"nextcloud_bridge"``).
            trigger: The bridge's configured trigger name.
            ingested: What the bridge forwards, for the missing-dispatch message
                (e.g. ``"every Talk mention"``).
            profile_dir: Directory the profile was loaded from, used to resolve a
                profile-relative triggers file.

        Returns:
            The errors found, for the caller to fold into its aggregate.
        """
        errors: list[str] = []

        if not trigger:
            errors.append(
                f"{key}.trigger is required: name the dispatch trigger the "
                "bridge fires, declared in the dispatch.triggers file"
            )
        # The bridge does nothing but POST questions to the dispatcher's
        # webhook, so a bridge without the dispatch pair would deploy and
        # then fail every message. Reject it at build time instead.
        if self.dispatch is None:
            errors.append(
                f"{key} requires a 'dispatch:' block: the bridge dispatches "
                f"{ingested} to the event dispatcher's webhook. Add a dispatch "
                f"block whose triggers file declares {trigger!r}, or remove the "
                f"{key} block."
            )
        elif trigger and self.dispatch.triggers:
            # Check the trigger against the SOURCE triggers file, resolved the
            # same way the dispatch block resolves it (profile-relative first,
            # then bundled). A bridge pointed at an undeclared trigger builds
            # and deploys cleanly and then 404s on every message.
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
                        f"{key}.trigger cannot be checked: dispatch.triggers "
                        f"file {triggers_file} failed to parse ({e}); fix that file first"
                    )
                else:
                    names = sorted(t.name for t in declared_triggers)
                    if trigger not in names:
                        errors.append(
                            f"{key}.trigger {trigger!r} is not declared in "
                            f"dispatch.triggers file {self.dispatch.triggers!r} "
                            f"(declares: {names}). Add a trigger named {trigger!r} to "
                            f"that file, or set {key}.trigger to one of them."
                        )

        return errors

    def _validate_network_axis(self) -> list[str]:
        """Return validation errors for every ``network:`` declaration.

        Runs on the profile as authored, BEFORE the build injects anything, so
        a ``network:`` on a dispatch-pair half is caught as something a person
        wrote rather than as the value the build is about to write there
        itself. Both authoring surfaces are checked — the ``services:`` block
        and the dotted ``config:`` overrides — because either spelling reaches
        the same key in the rendered ``config.yml``.

        Returns:
            Human-readable error messages; empty when every declaration names a
            valid mode and the dispatch pair keeps its single knob.
        """
        errors: list[str] = []
        pair_reported: set[str] = set()

        def check(name: str, value: Any, key: str) -> None:
            """Record the problem with one service's declared mode, if any."""
            if name in DISPATCH_PAIR_SERVICES:
                if name in pair_reported:
                    return
                pair_reported.add(name)
                errors.append(
                    f"{key} is not a per-service knob: the event dispatcher and its "
                    f"workers share one network, set by dispatch.network: "
                    f"(got {value!r}). Remove {key} and set dispatch.network instead."
                )
            else:
                errors.extend(network_mode_errors(value, key))

        if self.dispatch is not None:
            errors.extend(network_mode_errors(self.dispatch.network, "dispatch.network"))

        for name, svc in self.services.items():
            # A bare `on:`/`no:` service name is read on the YAML 1.1 resolver
            # as a boolean, which would reach the dotted-key formatting below as
            # a non-string. Report the resolver rather than crash on it.
            if not isinstance(name, str):
                errors.append(
                    f"services keys must be strings (got {name!r}); a bare "
                    "yes/no/on/off in YAML parses as a boolean — quote the service name"
                )
                continue
            if isinstance(svc.config, dict) and "network" in svc.config:
                check(name, svc.config["network"], f"services.{name}.network")

        if isinstance(self.config, dict):
            for key, value in self.config.items():
                if not isinstance(key, str):
                    errors.append(
                        f"config keys must be strings (got {key!r}); a bare "
                        "yes/no/on/off in YAML parses as a boolean — quote the key"
                    )
                    continue
                parts = key.split(".")
                if len(parts) == 3 and parts[0] == "services" and parts[2] == "network":
                    check(parts[1], value, key)

        return errors

    def _runs_the_channel_finder(self) -> bool:
        """Whether this profile keeps the channel-finder MCP server switched on.

        A persona narrows the deployment it is rendered from by switching
        servers off in its ``config:`` overlay, and one of them takes the whole
        channel finder — server, tools and subagent — out of that render. Such a
        persona still inherits ``channel_finder_mode`` from the profile it
        extends, because the mode belongs to the deployment rather than to the
        login; it simply has no channel finder for the mode to configure.

        Only an explicit ``false`` counts. The key is absent from every profile
        that keeps the server, so absence reads as on.

        Returns:
            Whether the rendered project has a channel finder at all.
        """
        return _config_lookup(self.config, _CHANNEL_FINDER_SERVER_ENABLED_KEY) is not False

    def _renders_graphdb_block(self) -> bool:
        """Whether this profile's build renders a ``services.graphdb`` block.

        Assembled from the two surfaces that decide it. The app template carries
        the block for a deployment that runs its own store; the profile's
        ``config:`` overlay, applied after the render, is how a store the
        facility already runs is named instead (``services.graphdb.uri``) and
        how a template's block is dropped. Whether what those produce counts as
        a store is
        :func:`~osprey.deployment.graphdb_service.resolve_graphdb_service_config`'s
        own decision, borrowed rather than re-implemented so this validator and
        the render-time ``graphdb_configured`` gate that ships the graph MCP
        server cannot disagree about whether one exists.

        Returns:
            Whether the rendered ``config.yml`` will carry a graph store to dial.
        """
        overrides = self.config if isinstance(self.config, dict) else {}
        sub_keys = [
            key
            for key in overrides
            if isinstance(key, str) and key.startswith(f"{_GRAPHDB_CONFIG_PREFIX}.")
        ]
        if sub_keys:
            # `services.graphdb.uri:` and its siblings write into the block,
            # creating it when the app template carries none.
            block: Any = {key.split(".", 2)[2]: overrides[key] for key in sub_keys}
        elif _GRAPHDB_CONFIG_PREFIX in overrides:
            # A whole-block override replaces what the template rendered —
            # including the bare `services.graphdb:` that removes it.
            block = overrides[_GRAPHDB_CONFIG_PREFIX]
        elif not self.deploy_services:
            # An attached project renders `services: {}` whatever its app
            # template says: it scaffolds nothing and reaches a shared stack
            # through client config, so its store has to be named above.
            return False
        else:
            return GRAPHDB_SERVICE_NAME in self.services or _app_template_declares_graphdb(
                self.data_bundle
            )

        try:
            resolved = resolve_graphdb_service_config({"services": {GRAPHDB_SERVICE_NAME: block}})
        except ValueError:
            # A malformed key (a port out of range, a heap size the JVM would
            # not parse) still names a store this profile means to dial. The
            # resolver raises about it where it can be acted on — the deploy
            # preflight — and answering "no block" here would report the wrong
            # fault.
            return True
        return resolved is not None

    def validate(self, profile_dir: Path) -> None:
        """Validate profile consistency. Raises BuildProfileError with all issues."""
        errors: list[str] = []

        if not self.name:
            errors.append("Profile 'name' is required")

        if not isinstance(self.deploy_services, bool):
            errors.append(
                f"deploy_services must be a boolean (got {type(self.deploy_services).__name__})"
            )

        # Every reader of `config:` treats it as a mapping of dotted keys — the
        # renderer, the web-stack lint, the deploy block's duplicate-key probe.
        # Rejected by name here rather than left to whichever of them a given
        # command reaches first: a list arrives as an unhandled TypeError deep in
        # one of those, and a block this malformed has no partial meaning worth
        # salvaging.
        if not isinstance(self.config, dict):
            errors.append(
                f"config must be a mapping of dotted keys to values "
                f"(got {type(self.config).__name__})"
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

        # `graph` is the one paradigm whose store is a service rather than a
        # database file, so it is the one paradigm with a prerequisite outside
        # the profile's own artifacts. Without a `services.graphdb` block the
        # build would render a channel finder with nothing behind it, and the
        # same `graphdb_configured` gate the registry applies would leave the
        # graph MCP server out without a word — an agent holding the mode and
        # none of its tools. Named here instead, on every configuration path.
        #
        # Skipped for a profile that switches the channel-finder server off — a
        # persona rendered as a logbook terminal, say. It inherits the mode from
        # the deployment it narrows, because the mode is the deployment's, but
        # it ships no channel finder for the mode to leave toolless. Requiring a
        # store there would make every graph deployment configure one for a
        # login that cannot query it.
        if (
            self.channel_finder_mode == _GRAPH_MODE
            and self._runs_the_channel_finder()
            and not self._renders_graphdb_block()
        ):
            attached = "" if self.deploy_services else ", deploy_services: false"
            errors.append(
                f"channel_finder_mode: {_GRAPH_MODE} needs a "
                f"services.{GRAPHDB_SERVICE_NAME} block and this build renders none "
                f"(app_template: {self.data_bundle}{attached}). Build on an app "
                f"template that deploys a graph store, or name one the facility "
                f"already runs with `config: services.{GRAPHDB_SERVICE_NAME}.uri: "
                f"bolt://host:7687` plus GRAPHDB_PASSWORD in the project .env."
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

        # Validate the profile's convention directories (shape of each source,
        # plus the reserved paths the project/ mirror may not write). Reported
        # as one entry so its own multi-problem message stays intact.
        try:
            validate_convention_sources(profile_dir)
        except BuildProfileError as e:
            errors.append(str(e))

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

        # Validate the network axis across both authoring surfaces
        errors.extend(self._validate_network_axis())

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
            # The bluesky panel's URL is likewise derived post-build
            # (``_inject_bluesky_web`` in build_cmd.py, which runs after this
            # validator) from the bluesky_web sidecar's port — so it is
            # legitimately url-less here when a bluesky_web block is present.
            if panel == "bluesky" and self.bluesky_web is not None:
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

        # Validate bluesky_web configuration
        if self.bluesky_web is not None:
            sp = self.bluesky_web
            if not (1 <= sp.port <= 65535):
                errors.append(f"bluesky_web.port must be in 1..65535 (got {sp.port})")

        # Validate the archiver store's knobs, and its agreement with the rest
        # of the profile — see va_archiver_errors for why the rules live beside
        # the block but are reported from here.
        errors.extend(va_archiver_errors(self.va_archiver, self.config, bool(self.deploy_services)))

        # The honesty rule's build-time site: a simulated machine may not be
        # paired with an archiver that invents its history. Checked against the
        # resolved `config:` rather than the block, because a profile can reach
        # the pairing by declaring no block at all — which is the common way in.
        errors.extend(va_mock_archiver_errors(self.config))

        # Validate the chat bridges — same checks per channel, see _validate_chat_bridge.
        if self.nextcloud_bridge is not None:
            errors.extend(
                self._validate_chat_bridge(
                    "nextcloud_bridge",
                    self.nextcloud_bridge.trigger,
                    "every Talk mention",
                    profile_dir,
                )
            )
        if self.gchat_bridge is not None:
            errors.extend(
                self._validate_chat_bridge(
                    "gchat_bridge",
                    self.gchat_bridge.trigger,
                    "every Chat message",
                    profile_dir,
                )
            )

        if errors:
            raise BuildProfileError(
                "Build profile validation failed:\n  - " + "\n  - ".join(errors)
            )
