"""Build profile data model and loader.

A build profile is a YAML file that describes how to assemble a
facility-specific assistant project from an OSPREY template plus
overlay files, config overrides, and MCP server definitions.
"""

from __future__ import annotations

import importlib.resources
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from osprey.errors import BuildProfileError


@dataclass
class McpServerDef:
    """Definition of an MCP server to inject into a built project."""

    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    permissions: dict[str, list[str]] = field(default_factory=dict)
    # permissions: {"allow": ["tool1"], "ask": ["tool2"]}
    url: str | None = None  # HTTP/SSE transport URL (mutually exclusive with command)


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
        return raw

    base_path = (profile_path.parent / extends_value).resolve()
    if not base_path.exists():
        raise BuildProfileError(
            f"Extended profile not found: {extends_value} (resolved to {base_path})"
        )

    try:
        base_raw = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise BuildProfileError(f"Invalid YAML in {base_path}: {e}") from e

    if not isinstance(base_raw, dict):
        raise BuildProfileError(f"Extended profile must be a YAML mapping: {base_path}")

    # Recurse: the base may itself extend another profile
    base_raw = _resolve_extends(base_raw, base_path, chain)

    return _deep_merge(base_raw, raw)


@dataclass
class BuildProfile:
    """Complete build profile parsed from YAML."""

    name: str
    data_bundle: str = "control_assistant"
    provider: str | None = None
    model: str | None = None
    channel_finder_mode: str | None = None
    tier: int = 1
    """Channel-database tier (1|2|3) controlling which `tiers/tier{N}` DB the
    rendered config points at. Tier 1 is the smallest slice and the default
    for benchmarks; tier 3 is the full preset. Same value gates query and
    DB scope so per-paradigm benchmarks stay coherent.
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
        "local"  # "local" | "pip" | PEP 508 spec (e.g. "osprey-framework==2026.5.0")
    )
    python_env: str = "project"  # "project" | "build" | absolute path to Python executable
    hooks: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)
    output_styles: list[str] = field(default_factory=list)
    web_panels: list[str] = field(default_factory=list)
    categories: dict[str, dict[str, str]] = field(default_factory=dict)

    def validate(self, profile_dir: Path) -> None:
        """Validate profile consistency. Raises BuildProfileError with all issues."""
        errors: list[str] = []

        if not self.name:
            errors.append("Profile 'name' is required")

        if self.tier not in (1, 2, 3):
            errors.append(f"tier must be 1, 2, or 3 (got {self.tier!r})")

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
        from osprey.profiles.web_panels import BUILTIN_PANELS

        for panel in self.web_panels:
            if panel in BUILTIN_PANELS:
                continue
            url_key = f"web.panels.{panel}.url"
            if url_key not in self.config:
                errors.append(
                    f"Unknown web_panel {panel!r}: not in BUILTIN_PANELS "
                    f"({sorted(BUILTIN_PANELS)}) and no '{url_key}' config override"
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
        "data_bundle",
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
        "categories",
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
        if url and command:
            raise BuildProfileError(
                f"MCP server '{name}' has both 'command' and 'url' — use one or the other"
            )
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

    return BuildProfile(
        name=raw.get("name", ""),
        data_bundle=raw.get("data_bundle", "control_assistant"),
        provider=raw.get("provider"),
        model=raw.get("model"),
        channel_finder_mode=raw.get("channel_finder_mode"),
        tier=int(raw.get("tier", 1)),
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
        categories=raw.get("categories", {}),
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


def list_presets() -> list[str]:
    """Return the sorted list of bundled preset names (hyphenated)."""
    presets_root = importlib.resources.files(_PRESETS_PACKAGE)
    return sorted(
        p.name.removesuffix(".yml")
        for p in presets_root.iterdir()
        if p.name.endswith(".yml") and not p.name.startswith("_")
    )


def _load_preset_raw(name: str) -> tuple[dict[str, Any], Path]:
    """Read a bundled preset YAML; return (raw_dict, preset_file_path).

    Raises ``BuildProfileError`` if the preset is unknown or invalid YAML.
    """
    normalized = _normalize_preset_name(name)
    presets_root = importlib.resources.files(_PRESETS_PACKAGE)
    presets_dir = Path(str(presets_root))
    target = presets_dir / f"{normalized}.yml"
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
