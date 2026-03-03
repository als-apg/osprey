"""Manifest generation, checksums, constants, and version utilities."""

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from osprey.cli.prompt_registry import PromptRegistry

logger = logging.getLogger("osprey.cli.templates")

# Manifest schema version for future compatibility
MANIFEST_SCHEMA_VERSION = "1.1.0"

# File used to store project manifest
MANIFEST_FILENAME = ".osprey-manifest.json"

# Maps manifest YAML category keys to PromptRegistry canonical-name prefixes.
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
    ".claude/hooks/osprey_memory_guard.py",
    ".claude/rules/python-execution.md",
    ".claude/rules/data-visualization.md",
    ".claude/rules/control-system-safety.md",
    ".claude/skills/diagnose/SKILL.md",
    ".claude/skills/session-report/SKILL.md",
    ".claude/skills/session-report/reference.md",
    ".claude/skills/setup-mode/SKILL.md",
    ".claude/rules/timezone.md",
    ".claude/output-styles/control-operator.md",
    # lattice_design skills and rules (render empty for other templates)
    ".claude/skills/load-lattice/SKILL.md",
    ".claude/skills/compute-optics/SKILL.md",
    ".claude/skills/fit-tune/SKILL.md",
    ".claude/skills/fit-chromaticity/SKILL.md",
    ".claude/skills/compare-optics/SKILL.md",
    ".claude/rules/lattice-physics.md",
]


def load_template_manifest(template_root: Path, template_name: str) -> dict | None:
    """Load manifest.yml for a template, if it exists.

    Args:
        template_root: Path to osprey's bundled templates directory
        template_name: Name of the application template (e.g. "control_assistant")

    Returns:
        Parsed YAML dict, or None if no manifest.yml exists for this template.
    """
    manifest_path = template_root / "apps" / template_name / "manifest.yml"
    if not manifest_path.exists():
        return None

    with open(manifest_path, encoding="utf-8") as f:
        manifest = yaml.safe_load(f) or {}

    # Validate entries against the prompt registry
    registry = PromptRegistry.default()
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

    # Validate web_panels entries
    valid_panel_ids = {"ariel", "channel-finder", "tuning"}
    for panel_id in manifest.get("web_panels", []):
        if panel_id not in valid_panel_ids:
            logger.warning("Unknown web_panel '%s' in template '%s'", panel_id, template_name)

    return manifest


def resolve_manifest_outputs(manifest: dict) -> set[str]:
    """Resolve a template manifest to the set of output paths that should be generated.

    Config artifacts (CLAUDE.md, .mcp.json, .claude/settings.json) are always included.
    For each manifest entry, prefix-matching is used against the prompt registry to
    handle multi-file artifacts (e.g. session-report -> SKILL.md + reference.md).

    Args:
        manifest: Parsed manifest dict (from load_template_manifest)

    Returns:
        Set of output paths (relative to project root) that the manifest allows.
    """
    result = {"CLAUDE.md", ".mcp.json", ".claude/settings.json"}

    registry = PromptRegistry.default()
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


def get_tracked_files(template_root: Path, template_name: str) -> list[str]:
    """Get the list of tracked files for regen, based on manifest if available.

    Args:
        template_root: Path to osprey's bundled templates directory
        template_name: Name of the application template

    Returns:
        Sorted list of output paths that should be tracked during regen.
    """
    manifest = load_template_manifest(template_root, template_name)
    if manifest is not None:
        return sorted(resolve_manifest_outputs(manifest))
    return list(REGEN_TRACKED_FILES)


def get_framework_version() -> str:
    """Get current osprey version.

    Returns:
        Version string (e.g., "0.7.0")
    """
    try:
        from osprey import __version__

        return __version__
    except (ImportError, AttributeError):
        return "0.7.0"


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


def extract_init_args(
    project_name: str,
    template_name: str,
    registry_style: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Extract init arguments from context for manifest storage.

    This extracts the user-facing init options from the full template context,
    filtering out derived values and internal state.

    Args:
        project_name: Name of the project
        template_name: Template used
        registry_style: Registry style
        context: Full template context

    Returns:
        Dictionary of init arguments that can be used to recreate the project
    """
    # Base arguments that are always present
    init_args = {
        "project_name": project_name,
        "template": template_name,
        "registry_style": registry_style,
    }

    # Optional arguments that may be in context
    optional_keys = [
        ("default_provider", "provider"),
        ("default_model", "model"),
        ("channel_finder_mode", "channel_finder_mode"),
        ("code_generator", "code_generator"),
    ]

    for context_key, arg_key in optional_keys:
        if context_key in context and context[context_key] is not None:
            # For boolean keys, include even when False; for strings, skip empty
            value = context[context_key]
            if isinstance(value, bool) or value:
                init_args[arg_key] = value

    return init_args


def build_reproducible_command(init_args: dict[str, Any]) -> str:
    """Build a reproducible CLI command from init arguments.

    Args:
        init_args: Dictionary of init arguments

    Returns:
        CLI command string that can recreate the project
    """
    parts = ["osprey", "init", init_args["project_name"]]

    # Always include template for reproducibility (default may change)
    if init_args.get("template"):
        parts.extend(["--template", init_args["template"]])

    # Add provider if specified
    if init_args.get("provider"):
        parts.extend(["--provider", init_args["provider"]])

    # Add model if specified
    if init_args.get("model"):
        parts.extend(["--model", init_args["model"]])

    # Add channel_finder_mode if specified
    if init_args.get("channel_finder_mode"):
        parts.extend(["--channel-finder-mode", init_args["channel_finder_mode"]])

    # Add code_generator if specified
    if init_args.get("code_generator"):
        parts.extend(["--code-generator", init_args["code_generator"]])

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

    registry = PromptRegistry.default()
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
    registry_style: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Generate a project manifest for migration support.

    The manifest captures all information needed to recreate the project
    with the same OSPREY version and settings. This enables future migrations
    by providing a baseline for three-way diffs.

    Args:
        template_root: Path to osprey's bundled templates directory
        jinja_env: Jinja2 environment for template rendering
        project_dir: Root directory of the created project
        project_name: Name of the project
        template_name: Template used to create the project
        registry_style: Registry style ("extend" or "standalone")
        context: Full context dict used during template rendering

    Returns:
        Dictionary containing the manifest data that was written to file
    """
    # Build init_args from context - extract the user-facing options
    init_args = extract_init_args(project_name, template_name, registry_style, context)

    # Build reproducible command string
    reproducible_command = build_reproducible_command(init_args)

    # Calculate file checksums for trackable files
    file_checksums = calculate_file_checksums(project_dir)

    # Get framework version
    framework_version = get_framework_version()

    # Build user_owned section from context
    user_owned_manifest = build_user_owned_manifest(template_root, jinja_env, project_dir, context)

    # Build manifest
    manifest_data = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "creation": {
            "osprey_version": framework_version,
            "timestamp": datetime.now(UTC).isoformat(),
            "template": template_name,
            "registry_style": registry_style,
            "claude_code_only": True,
        },
        "init_args": init_args,
        "reproducible_command": reproducible_command,
        "file_checksums": file_checksums,
    }

    if user_owned_manifest:
        manifest_data["user_owned"] = user_owned_manifest

    # Write manifest to file
    manifest_path = project_dir / MANIFEST_FILENAME
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, sort_keys=False)

    return manifest_data
