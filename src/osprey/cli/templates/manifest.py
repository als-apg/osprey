"""Manifest generation, checksums, constants, and version utilities."""

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from osprey.profiles.web_panels import BUILTIN_PANELS
from osprey.services.prompts.catalog import PromptCatalog

logger = logging.getLogger("osprey.cli.templates")


class ManifestError(ValueError):
    """Raised when a template manifest references unknown artifacts or panels."""

# Manifest schema version for future compatibility
MANIFEST_SCHEMA_VERSION = "1.2.0"

# File used to store project manifest
MANIFEST_FILENAME = ".osprey-manifest.json"

# Maps manifest YAML category keys to PromptCatalog canonical-name prefixes.
# Needed because YAML convention uses underscores while registry uses hyphens.
_MANIFEST_CATEGORY_PREFIX = {
    "hooks": "hooks/",
    "rules": "rules/",
    "skills": "skills/",
    "agents": "agents/",
    "output_styles": "output-styles/",
}

# Known framework-managed files for checksum collection during regen.
REGEN_TRACKED_FILES = [
    "CLAUDE.md",
    ".mcp.json",
    ".claude/settings.json",
    ".claude/statusline.py",
    ".claude/rules/safety.md",
    ".claude/rules/error-handling.md",
    ".claude/rules/artifacts.md",
    ".claude/rules/workflows.md",
    ".claude/rules/facility.md",
    ".claude/hooks/osprey_writes_check.py",
    ".claude/hooks/osprey_limits.py",
    ".claude/hooks/osprey_approval.py",
    ".claude/hooks/osprey_error_guidance.py",
    ".claude/hooks/osprey_notebook_update.py",
    ".claude/hooks/osprey_cf_feedback_capture.py",
    ".claude/hooks/osprey_hook_log.py",
    ".claude/hooks/hook_config.json",
    ".claude/hooks/osprey_memory_guard.py",
    ".claude/rules/python-execution.md",
    ".claude/rules/data-visualization.md",
    ".claude/rules/control-system-safety.md",
    ".claude/skills/diagnose/SKILL.md",
    ".claude/skills/session-report/SKILL.md",
    ".claude/skills/session-report/reference.md",
    ".claude/skills/setup-mode/SKILL.md",
    ".claude/skills/demo-gallery/SKILL.md",
    ".claude/rules/timezone.md",
    ".claude/output-styles/control-operator.md",
]


def load_template_manifest(
    template_root: Path,
    template_name: str,
    project_dir: Path | None = None,
) -> dict | None:
    """Load manifest.yml for a template, if it exists.

    When the template-level ``manifest.yml`` does not exist, falls back to
    reading the ``"artifacts"`` section from the project-local
    ``.osprey-manifest.json`` (when ``project_dir`` is provided).

    Args:
        template_root: Path to osprey's bundled templates directory
        template_name: Name of the application template (e.g. "control_assistant")
        project_dir: Optional project directory. When given and the template
            manifest does not exist, artifacts are read from the project's
            ``.osprey-manifest.json`` instead.

    Returns:
        Parsed YAML-like dict, or None if no manifest source is available.
    """
    manifest_path = template_root / "apps" / template_name / "manifest.yml"
    if not manifest_path.exists():
        # Fall back to project-local manifest artifacts
        if project_dir is not None:
            osprey_manifest_path = project_dir / MANIFEST_FILENAME
            if osprey_manifest_path.exists():
                try:
                    osprey_data = json.loads(osprey_manifest_path.read_text(encoding="utf-8"))
                    stored_artifacts = osprey_data.get("artifacts")
                    if stored_artifacts:
                        return {"artifacts": stored_artifacts}
                except (json.JSONDecodeError, OSError):
                    pass
        # Fall back to the bundled preset profile (manifest.yml was removed; preset
        # profiles are now the canonical source of artifact declarations per data bundle)
        _preset_name = template_name.replace("_", "-") + ".yml"
        try:
            import importlib.resources

            profile_text = (
                importlib.resources.files("osprey.profiles.presets")
                .joinpath(_preset_name)
                .read_text(encoding="utf-8")
            )
            profile_data = yaml.safe_load(profile_text) or {}
            artifact_keys = ("hooks", "rules", "skills", "agents", "output_styles", "web_panels")
            artifacts = {k: profile_data.get(k, []) for k in artifact_keys if k in profile_data}
            if artifacts:
                return {"artifacts": artifacts}
        except (FileNotFoundError, ModuleNotFoundError, OSError):
            pass
        return None

    with open(manifest_path, encoding="utf-8") as f:
        manifest = yaml.safe_load(f) or {}

    # Validate entries against the prompt registry
    registry = PromptCatalog.default()
    artifacts = manifest.get("artifacts", {})
    for category, entries in artifacts.items():
        prefix = _MANIFEST_CATEGORY_PREFIX.get(category)
        if prefix is None:
            logger.warning(
                "Unknown manifest category '%s' in template '%s'", category, template_name
            )
            continue
        for entry_name in entries:
            canonical_prefix = prefix + entry_name
            # Check if any registry artifact starts with this prefix
            matches = [
                a
                for a in registry.all_artifacts()
                if a.canonical_name == canonical_prefix
                or a.canonical_name.startswith(canonical_prefix + "/")
            ]
            if not matches:
                logger.warning(
                    "Manifest entry '%s/%s' in template '%s' not found in prompt registry",
                    category,
                    entry_name,
                    template_name,
                )

    # Validate web_panels entries against the shared registry. Unreachable for
    # presets today (they short-circuit at the artifacts wrapper above) but the
    # load-bearing gate for any future template manifest.
    for panel_id in manifest.get("web_panels", []):
        if panel_id not in BUILTIN_PANELS:
            raise ManifestError(
                f"Unknown web_panel {panel_id!r} in template {template_name!r} "
                f"(valid: {sorted(BUILTIN_PANELS)})"
            )

    return manifest


def resolve_manifest_outputs(manifest: dict) -> set[str]:
    """Resolve a template manifest to the set of output paths that should be generated.

    Config artifacts (CLAUDE.md, .mcp.json, .claude/settings.json, .claude/statusline.py) are always included.
    For each manifest entry, prefix-matching is used against the prompt registry to
    handle multi-file artifacts (e.g. session-report -> SKILL.md + reference.md).

    Args:
        manifest: Parsed manifest dict (from load_template_manifest)

    Returns:
        Set of output paths (relative to project root) that the manifest allows.
    """
    result = {"CLAUDE.md", ".mcp.json", ".claude/settings.json", ".claude/statusline.py"}

    registry = PromptCatalog.default()
    all_artifacts = registry.all_artifacts()

    for category, entries in manifest.get("artifacts", {}).items():
        prefix = _MANIFEST_CATEGORY_PREFIX.get(category)
        if prefix is None:
            continue
        for entry_name in entries:
            canonical_prefix = prefix + entry_name
            # Prefix-match: include artifacts whose canonical name matches exactly
            # OR starts with prefix + "/". This handles multi-file artifacts like
            # session-report (skills/session-report + skills/session-report/reference).
            for artifact in all_artifacts:
                if (
                    artifact.canonical_name == canonical_prefix
                    or artifact.canonical_name.startswith(canonical_prefix + "/")
                ):
                    result.add(artifact.output_path)

    return result


def get_tracked_files(
    template_root: Path,
    template_name: str,
    project_dir: Path | None = None,
) -> list[str]:
    """Get the list of tracked files for regen, based on manifest if available.

    Resolution order:
    1. If ``project_dir`` is given and its ``.osprey-manifest.json`` contains
       an ``"artifacts"`` key, use those artifact selections.
    2. Otherwise fall back to the template-level ``manifest.yml``.
    3. If neither exists, return the static ``REGEN_TRACKED_FILES`` list.

    Args:
        template_root: Path to osprey's bundled templates directory
        template_name: Name of the application template
        project_dir: Optional project directory; when provided, the project-local
            ``.osprey-manifest.json`` is checked for stored artifact selections.

    Returns:
        Sorted list of output paths that should be tracked during regen.
    """
    # 1. Try project-local manifest first
    if project_dir is not None:
        osprey_manifest_path = project_dir / MANIFEST_FILENAME
        if osprey_manifest_path.exists():
            try:
                osprey_data = json.loads(osprey_manifest_path.read_text(encoding="utf-8"))
                stored_artifacts = osprey_data.get("artifacts")
                if stored_artifacts:
                    return sorted(resolve_manifest_outputs({"artifacts": stored_artifacts}))
            except (json.JSONDecodeError, OSError):
                pass

    # 2. Fall back to template manifest.yml
    tmpl_manifest = load_template_manifest(template_root, template_name)
    if tmpl_manifest is not None:
        return sorted(resolve_manifest_outputs(tmpl_manifest))
    return list(REGEN_TRACKED_FILES)


def get_framework_version() -> str:
    """Get current osprey version.

    Returns:
        Version string (e.g., "0.7.0") or ``"unknown"`` if the version
        symbol cannot be imported (broken environment / partial install).
        Manifest readers can branch on the sentinel.
    """
    try:
        from osprey import __version__

        return __version__
    except (ImportError, AttributeError):
        return "unknown"


def sha256_file(file_path: Path) -> str:
    """Calculate SHA256 hash of a file.

    Args:
        file_path: Path to the file

    Returns:
        Hex-encoded SHA256 hash
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def extract_build_args(
    project_name: str,
    preset_name: str | None,
    profile_path: str | None,
    data_bundle: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Extract build invocation arguments for manifest storage.

    Captures both the user-facing options (provider/model/etc.) and the
    invocation source (preset vs. positional profile path) so that
    ``build_reproducible_command`` can render the matching CLI form.

    Exactly one of ``preset_name`` and ``profile_path`` must be set.

    Args:
        project_name: Name of the project.
        preset_name: Hyphenated preset name (e.g. "hello-world") if the build
            was invoked via ``--preset``; otherwise ``None``.
        profile_path: Path string to the positional profile YAML if the build
            was invoked positionally; otherwise ``None``.
        data_bundle: The underlying app bundle resolved from the profile
            (e.g. "hello_world", "control_assistant").
        context: Full template context.

    Returns:
        Dictionary of build arguments suitable for round-trip reproduction.
    """
    build_args: dict[str, Any] = {
        "project_name": project_name,
        "source": "preset" if preset_name else "profile",
        "data_bundle": data_bundle,
    }
    if preset_name:
        build_args["preset"] = preset_name
    if profile_path:
        build_args["profile_path"] = profile_path

    optional_keys = [
        ("default_provider", "provider"),
        ("default_model", "model"),
        ("channel_finder_mode", "channel_finder_mode"),
    ]
    for context_key, arg_key in optional_keys:
        if context_key in context and context[context_key] is not None:
            value = context[context_key]
            if isinstance(value, bool) or value:
                build_args[arg_key] = value

    return build_args


def build_reproducible_command(build_args: dict[str, Any]) -> str:
    """Render a reproducible ``osprey build`` command from build args.

    Branches on ``build_args["source"]``:
      * ``"preset"``  -> ``osprey build NAME --preset PRESET [--set ...]``
      * ``"profile"`` -> ``osprey build NAME PROFILE_PATH [--set ...]``

    Args:
        build_args: Output of :func:`extract_build_args`.

    Returns:
        CLI command string that recreates the project.
    """
    parts = ["osprey", "build", build_args["project_name"]]

    source = build_args.get("source", "preset")
    if source == "preset" and build_args.get("preset"):
        parts.extend(["--preset", build_args["preset"]])
    elif source == "profile" and build_args.get("profile_path"):
        parts.append(build_args["profile_path"])
    elif build_args.get("data_bundle"):
        # Defensive fallback: legacy manifests without source/preset/profile_path.
        parts.extend(["--preset", build_args["data_bundle"].replace("_", "-")])

    for key in ("provider", "model", "channel_finder_mode"):
        if build_args.get(key):
            parts.extend(["--set", f"{key}={build_args[key]}"])

    return " ".join(parts)


def calculate_file_checksums(project_dir: Path) -> dict[str, str]:
    """Calculate SHA256 checksums for trackable project files.

    Trackable files are those that come from templates and may change
    between OSPREY versions. This excludes:
    - .env files (contain secrets)
    - _agent_data/ (runtime data)
    - data/ directories (user data)
    - __pycache__/ and .pyc files
    - .git/ directory

    Args:
        project_dir: Root directory of the project

    Returns:
        Dictionary mapping relative file paths to their SHA256 checksums
    """
    checksums = {}

    # Patterns to exclude
    exclude_patterns = {
        ".env",
        ".git",
        "__pycache__",
        ".pyc",
        "_agent_data",
        "data",
        ".osprey-manifest.json",  # Don't checksum ourselves
    }

    # Walk the project directory
    for file_path in project_dir.rglob("*"):
        if not file_path.is_file():
            continue

        # Get relative path
        rel_path = file_path.relative_to(project_dir)
        rel_path_str = str(rel_path)

        # Skip excluded patterns
        skip = False
        for pattern in exclude_patterns:
            if pattern in rel_path.parts or rel_path_str.startswith(pattern):
                skip = True
                break
        if skip:
            continue

        # Skip binary and large files
        if file_path.suffix in [".pyc", ".pyo", ".so", ".dll", ".dylib"]:
            continue

        # Calculate checksum
        try:
            checksum = sha256_file(file_path)
            checksums[rel_path_str] = f"sha256:{checksum}"
        except OSError:
            # Skip files that can't be read
            continue

    return checksums


def build_user_owned_manifest(
    template_root: Path,
    jinja_env,
    project_dir: Path,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Build user_owned section for the manifest.

    For each user-owned artifact, records the SHA-256 of the framework
    template as rendered at claim time. During regen, if the framework
    hash changes, a drift warning is shown.

    Args:
        template_root: Path to osprey's bundled templates directory
        jinja_env: Jinja2 environment for template rendering
        project_dir: Root directory of the project
        context: Template context with ``user_owned`` key

    Returns:
        Dict mapping canonical names to user_owned metadata, or empty dict.
    """
    user_owned = context.get("user_owned", [])
    if not user_owned:
        return {}

    import tempfile

    registry = PromptCatalog.default()
    result: dict[str, Any] = {}
    claude_code_dir = template_root / "claude_code"

    for canonical_name in user_owned:
        artifact = registry.get(canonical_name)
        if artifact is None:
            continue

        framework_hash = None
        template_file = claude_code_dir / artifact.template_path
        if template_file.exists():
            try:
                if template_file.suffix == ".j2":
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=template_file.stem, delete=False, encoding="utf-8"
                    ) as tmp:
                        template_rel = f"claude_code/{artifact.template_path}"
                        template = jinja_env.get_template(template_rel)
                        rendered = template.render(**context)
                        tmp.write(rendered)
                        tmp_path = Path(tmp.name)
                    framework_hash = f"sha256:{sha256_file(tmp_path)}"
                    tmp_path.unlink(missing_ok=True)
                else:
                    framework_hash = f"sha256:{sha256_file(template_file)}"
            except Exception:
                pass  # Best-effort

        entry: dict[str, Any] = {
            "claimed_at": datetime.now(UTC).isoformat(),
        }
        if framework_hash:
            entry["framework_hash"] = framework_hash

        result[canonical_name] = entry

    return result


def generate_manifest(
    template_root: Path,
    jinja_env,
    project_dir: Path,
    project_name: str,
    template_name: str,
    context: dict[str, Any],
    *,
    artifacts: dict[str, list[str]] | None = None,
    preset_name: str | None = None,
    profile_path: str | None = None,
) -> dict[str, Any]:
    """Generate ``.osprey-manifest.json`` for a freshly built project.

    Args:
        template_root: Root of the bundled templates package.
        jinja_env: The Jinja2 environment used during rendering.
        project_dir: Project root directory (where the manifest is written).
        project_name: Name of the project.
        template_name: Underlying app bundle (data_bundle) used to render.
        context: Full template-render context.
        artifacts: Profile-driven artifact selection.
        preset_name: Hyphenated preset name if invoked via ``--preset``.
        profile_path: Path string to the positional profile, if any.

    Returns:
        The manifest dict written to disk.
    """
    build_args = extract_build_args(
        project_name=project_name,
        preset_name=preset_name,
        profile_path=profile_path,
        data_bundle=template_name,
        context=context,
    )
    reproducible_command = build_reproducible_command(build_args)
    file_checksums = calculate_file_checksums(project_dir)
    framework_version = get_framework_version()
    user_owned_manifest = build_user_owned_manifest(template_root, jinja_env, project_dir, context)

    # The 'template' field carries the original preset name (hyphenated) when
    # the user invoked --preset; otherwise it carries the bundle name as a
    # best-effort fallback. 'data_bundle' is always the underlying app bundle.
    template_field = preset_name if preset_name else template_name

    creation_block: dict[str, Any] = {
        "osprey_version": framework_version,
        "timestamp": datetime.now(UTC).isoformat(),
        "template": template_field,
        "data_bundle": template_name,
        "claude_code_only": True,
    }

    manifest_data: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "creation": creation_block,
        "build_args": build_args,
        "reproducible_command": reproducible_command,
        "file_checksums": file_checksums,
    }

    if artifacts:
        manifest_data["artifacts"] = artifacts
    if user_owned_manifest:
        manifest_data["user_owned"] = user_owned_manifest

    manifest_path = project_dir / MANIFEST_FILENAME
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, sort_keys=False)

    return manifest_data
