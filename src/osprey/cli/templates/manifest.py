"""Manifest generation, checksums, constants, and version utilities.

The leaf primitives ``MANIFEST_FILENAME`` and ``sha256_file`` live in the
build-time kernel (:mod:`osprey.build.manifest`) so lower layers can consume
them without importing ``cli``; they are re-imported here for the
catalog-aware generation/validation logic that stays in this module.
"""

import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from osprey.build.manifest import MANIFEST_FILENAME, sha256_file
from osprey.profiles.web_panels import BUILTIN_PANELS
from osprey.services.build_artifacts.catalog import BuildArtifactCatalog

logger = logging.getLogger("osprey.cli.templates")


class ManifestError(ValueError):
    """Raised when a template manifest references unknown artifacts or panels."""


# Manifest schema version for future compatibility
MANIFEST_SCHEMA_VERSION = "1.2.0"

# Maps manifest YAML category keys to BuildArtifactCatalog canonical-name prefixes.
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
    ".claude/hooks/osprey_focus_validate.py",
    ".claude/hooks/osprey_config_drift.py",
    ".claude/hooks/osprey_panels_context.py",
    ".claude/rules/python-execution.md",
    ".claude/rules/control-system-safety.md",
    ".claude/skills/diagnose/SKILL.md",
    ".claude/skills/session-report/SKILL.md",
    ".claude/skills/session-report/reference.md",
    ".claude/skills/setup-mode/SKILL.md",
    ".claude/skills/demo-gallery/SKILL.md",
    ".claude/skills/demo-ui/SKILL.md",
    ".claude/skills/writing-bluesky-plans/SKILL.md",
    ".claude/skills/operating-bluesky-scans/SKILL.md",
    ".claude/rules/timezone.md",
    ".claude/output-styles/control-operator.md",
]


def framework_template_hash(
    claude_code_dir: Path,
    template_path: str,
    jinja_env: Any,
    context: dict[str, Any],
) -> str | None:
    """``sha256:`` digest of the framework's own version of one artifact.

    Recorded when an artifact is claimed and recomputed on every regen, so the
    two must be computed identically or every regen would report drift that is
    not there. That is the whole reason this lives in one function: the two
    callers are in different modules and would otherwise be free to differ on
    the render context, the encoding, or the ``sha256:`` prefix.

    A ``.j2`` template is rendered first — the digest is of what the framework
    would *write*, not of the template that writes it, so a context change is
    drift and a comment change in the template is not.

    Args:
        claude_code_dir: The ``claude_code`` template directory.
        template_path: The artifact's template path below it.
        jinja_env: Jinja environment the render goes through.
        context: Template context for the render.

    Returns:
        ``sha256:<hex>``, or ``None`` when the template is missing or will not
        render. Callers treat ``None`` as "no comparison possible" rather than
        as drift: a template that cannot render is a framework problem, and
        reporting it as the operator's artifact having drifted would misdirect.
    """
    template_file = claude_code_dir / template_path
    if not template_file.exists():
        return None
    try:
        if template_file.suffix != ".j2":
            return f"sha256:{sha256_file(template_file)}"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=template_file.stem, delete=False, encoding="utf-8"
        ) as tmp:
            template = jinja_env.get_template(f"claude_code/{template_path}")
            tmp.write(template.render(**context))
            tmp_path = Path(tmp.name)
        digest = f"sha256:{sha256_file(tmp_path)}"
        tmp_path.unlink(missing_ok=True)
        return digest
    except Exception:
        return None


def _stored_artifacts(project_dir: Path | None) -> dict | None:
    """The artifact selections a built project recorded, or ``None``.

    Read from the project's own ``.osprey-manifest.json`` — what the build that
    made this project actually selected, and therefore the most specific answer
    available. ``None`` covers every way there is no answer: no project to ask,
    no manifest, a manifest too damaged to parse, or one recording no
    selections. A damaged manifest is not an error here: the callers all have a
    further fallback, and refusing to regenerate a project because its manifest
    got truncated would be worse than regenerating from the template's list.
    """
    if project_dir is None:
        return None
    manifest_path = project_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data.get("artifacts") or None


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
        stored_artifacts = _stored_artifacts(project_dir)
        if stored_artifacts:
            return {"artifacts": stored_artifacts}
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
    registry = BuildArtifactCatalog.default()
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

    registry = BuildArtifactCatalog.default()
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
    stored_artifacts = _stored_artifacts(project_dir)
    if stored_artifacts:
        return sorted(resolve_manifest_outputs({"artifacts": stored_artifacts}))

    # 2. Fall back to template manifest.yml
    tmpl_manifest = load_template_manifest(template_root, template_name)
    if tmpl_manifest is not None:
        return sorted(resolve_manifest_outputs(tmpl_manifest))
    return list(REGEN_TRACKED_FILES)


def get_framework_version() -> str:
    """Get the running osprey version, for display in generated projects.

    This is the version a human reads — it carries distance past the last release
    (``2026.6.2.post783+g83fda5e60``) so a project rendered from a development
    checkout says so. Anything comparing versions wants
    :func:`osprey.version.get_release_version` instead; see
    :func:`get_framework_release_version`.

    Returns:
        Version string, or ``"unknown"`` if the version module cannot be imported
        (broken environment / partial install). Manifest readers branch on the
        sentinel.
    """
    try:
        from osprey.version import get_running_version

        return get_running_version()
    except (ImportError, AttributeError):
        return "unknown"


def get_framework_release_version() -> str:
    """Get the release this osprey descends from, for version *comparisons*.

    Stamped into the manifest's ``creation.osprey_version`` and read back by
    :mod:`osprey.deployment.staleness`, which compares it by string equality. Using
    the running version on either side would make every commit in a development
    checkout register as drift — inverting the advisory's rule that "can't compare"
    must never read as drift into "always reads as drift".

    Returns:
        Release version string (e.g. ``"2026.6.2"``), or ``"unknown"`` if the
        version module cannot be imported.
    """
    try:
        from osprey.version import get_release_version

        return get_release_version()
    except (ImportError, AttributeError):
        return "unknown"


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

    ``profile_path`` stays the string the user typed — it is what
    ``reproducible_command`` shows them. ``profile_path_abs`` is recorded
    separately, from ``context``, and on *every* build: it names the profile
    directory the project was actually built from (for a ``--preset`` build,
    the one that invocation materialized), which is what the deploy side
    follows to write minted secrets back to their source. A relative CLI string
    re-resolves against whatever directory the deploy runs from, so it is the
    absolute form readers follow.

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
    profile_path_abs = context.get("profile_path_abs")
    if profile_path_abs:
        build_args["profile_path_abs"] = str(profile_path_abs)

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
      * ``"preset"``  -> ``osprey build NAME --preset PRESET``
      * ``"profile"`` -> ``osprey build NAME PROFILE_PATH``

    No ``--set`` is replayed. Provider, model and channel-finder paradigm are
    recorded above as the values this build *resolved*, and the profile the
    command names already carries them — a build overriding one writes it into
    that profile rather than layering it at build time. Replaying them would
    put stale copies on a command line that re-reads the source of truth
    anyway.

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

    return " ".join(parts)


def calculate_file_checksums(project_dir: Path) -> dict[str, str]:
    """Calculate SHA256 checksums for trackable project files.

    Trackable files are those the build renders into the project — everything
    a rebuild reproduces, so a checksum mismatch means the file drifted from
    its source. ``data/`` is included: it is build-owned (machine models,
    channel databases, benchmark query sets all come from the profile or the
    preset), so a runtime writer landing there would show up here as
    unexplained drift. Runtime state belongs under ``_agent_data/``.

    Excluded:
    - ``.env`` (secrets, never rendered deterministically)
    - ``_agent_data/`` (runtime state — the one directory runtime writers own)
    - ``__pycache__/`` and ``.pyc`` files
    - ``.git/``
    - the manifest itself

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

    registry = BuildArtifactCatalog.default()
    result: dict[str, Any] = {}
    claude_code_dir = template_root / "claude_code"

    for canonical_name in user_owned:
        artifact = registry.get(canonical_name)
        if artifact is None:
            continue

        framework_hash = framework_template_hash(
            claude_code_dir, artifact.template_path, jinja_env, context
        )

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
    # The release lineage, not the running version — staleness compares this field
    # by string equality, so a development build must not read as drift.
    framework_version = get_framework_release_version()
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
    # Stamp a content hash of what this project was rendered from, so `osprey
    # deploy` can detect a project whose render predates its source — the
    # osprey_version alone can't, since a --dev checkout keeps one version
    # string across commits.
    #
    # That source is the PROFILE, on every build. A `--preset` build renders
    # from the profile it materialized, not from the bundled preset, so hashing
    # the preset here would leave a hand edit to that profile invisible to the
    # staleness advisory — the one edit the advisory most needs to see, since
    # editing the profile is how a facility is meant to change its project. The
    # preset a profile came from stays recorded in the profile's own
    # `provenance:` block, which the build compares separately for its
    # materialization-time drift advisory. The preset hash remains the fallback
    # for a build that somehow resolved no profile path.
    #
    # Best-effort: an unhashable source omits the key (staleness checks then
    # skip the content comparison) rather than failing the build.
    try:
        from osprey.cli import build_profile as _build_profile

        # The path the build resolved, never the string the user typed: the two
        # only agree while the process stays in the invocation directory, and a
        # profile's data/overlay trees are anchored on it.
        hashed_profile = build_args.get("profile_path_abs") or profile_path
        if hashed_profile:
            preset_hash = _build_profile.compute_profile_hash(Path(hashed_profile))
        elif preset_name:
            preset_hash = _build_profile.compute_preset_hash(preset_name)
        else:
            preset_hash = None
    except Exception:
        preset_hash = None
    if preset_hash:
        creation_block["preset_hash"] = preset_hash

    # Preserve the CLAUDE.md template choice so `osprey claude regen` can
    # re-render against the same persona (e.g. CLAUDE.ariel.md.j2 for the
    # ARIEL standalone preset). Default is the control-system persona.
    claude_md_template = context.get("claude_md_template")
    if claude_md_template and claude_md_template != "CLAUDE.md.j2":
        creation_block["claude_md_template"] = claude_md_template

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


#: Manifest key recording whether the last deploy persisted the service secrets
#: it minted into the profile that owns them. Absent means "never asked": a
#: project built before the key existed, or one no deploy has touched yet.
SECRETS_SYNCED_KEY = "secrets_synced_to_profile"


def load_project_manifest(project_dir: Path) -> dict[str, Any] | None:
    """Read a built project's ``.osprey-manifest.json``.

    Returns ``None`` when the file is missing, unparseable, or not an object —
    every reader of build provenance is advisory and must fail open rather than
    break the operation it is annotating.

    :param project_dir: Project root (the directory holding the manifest).
    """
    try:
        raw = (Path(project_dir) / MANIFEST_FILENAME).read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def manifest_profile_path(project_dir: Path) -> Path | None:
    """The absolute profile file a project was built from, per its manifest.

    Reads ``build_args.profile_path_abs`` — the path ``osprey build`` actually
    resolved. The ``profile_path`` spelling beside it is the string the user
    typed and is deliberately *not* used as a fallback here: a relative one
    re-resolves against whatever directory the deploy runs from, so following it
    would write a facility's secrets into an arbitrary directory that happens to
    match. ``None`` for a legacy manifest or no manifest at all; every build
    records it, a ``--preset`` build naming the profile it materialized.

    :param project_dir: Project root (the directory holding the manifest).
    """
    manifest = load_project_manifest(project_dir)
    build_args = (manifest or {}).get("build_args")
    if not isinstance(build_args, dict):
        return None
    profile_path = build_args.get("profile_path_abs")
    if not isinstance(profile_path, str) or not profile_path:
        return None
    return Path(profile_path)


def read_secrets_synced_to_profile(project_dir: Path) -> bool | None:
    """Whether the last deploy synced this project's secrets to its profile.

    ``None`` when the manifest does not say — a project built before the flag
    existed, or one that has never been deployed. Callers must treat that as
    "unknown", never as "not synced": an unanswerable question must not read as
    a problem.

    :param project_dir: Project root (the directory holding the manifest).
    """
    manifest = load_project_manifest(project_dir)
    value = (manifest or {}).get(SECRETS_SYNCED_KEY)
    return value if isinstance(value, bool) else None


def write_secrets_synced_to_profile(project_dir: Path, synced: bool) -> bool:
    """Record whether a deploy persisted its minted secrets to the profile.

    Rewrites the manifest in place (temp file + ``os.replace``, so a concurrent
    reader never sees a half-written manifest). The flag is stamped at deploy
    time, long after the build wrote the file, and the value can flip back to
    ``True`` when a later deploy reaches a profile an earlier one could not.

    :param project_dir: Project root (the directory holding the manifest).
    :param synced: ``True`` when the profile now carries the secrets.
    :returns: ``False`` when there is no readable manifest to stamp — deploying
        a project without one is legal, and losing the flag is not worth
        failing a deploy over.
    """
    project_dir = Path(project_dir)
    manifest = load_project_manifest(project_dir)
    if manifest is None:
        return False
    if manifest.get(SECRETS_SYNCED_KEY) is synced:
        return True
    manifest[SECRETS_SYNCED_KEY] = synced

    manifest_path = project_dir / MANIFEST_FILENAME
    fd, tmp_name = tempfile.mkstemp(dir=project_dir, prefix=MANIFEST_FILENAME, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=False)
        os.replace(tmp_name, manifest_path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return True
