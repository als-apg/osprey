"""Claude Code artifact rendering, regeneration, and user-ownership."""

import json
import logging
import shutil
import warnings
from datetime import UTC, datetime
from pathlib import Path

import yaml

from osprey.cli.styles import console
from osprey.cli.templates import manifest as manifest_mod
from osprey.cli.templates._rendering import render_template
from osprey.services.prompts.catalog import PromptCatalog
from osprey.utils.config import resolve_env_vars

logger = logging.getLogger("osprey.cli.templates")


def build_claude_code_context(
    template_root: Path,
    jinja_env,
    project_dir: Path,
    config: dict,
    project_root_override: Path | str | None = None,
) -> dict:
    """Build template context for Claude Code artifact rendering.

    Reconstructs the template context needed by Claude Code templates
    (.mcp.json, CLAUDE.md, settings.json, agents) from the project's
    config.yml and manifest.

    Args:
        template_root: Path to osprey's bundled templates directory
        jinja_env: Jinja2 environment for template rendering
        project_dir: Root directory of the project
        config: Parsed config.yml dictionary

    Returns:
        Template context dict suitable for Claude Code templates
    """
    import sys

    project_name = config.get("project_name", project_dir.name)
    package_name = project_name.replace("-", "_").lower()

    # Read template_name and artifact selections from manifest if available
    manifest_path = project_dir / manifest_mod.MANIFEST_FILENAME
    template_name = "control_assistant"
    data_bundle = "control_assistant"
    artifacts: dict[str, list[str]] = {}
    if manifest_path.exists():
        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            creation = manifest_data.get("creation", {})
            template_name = creation.get("template", "control_assistant")
            data_bundle = creation.get("data_bundle", template_name)
            artifacts = manifest_data.get("artifacts", {})
        except (json.JSONDecodeError, OSError):
            pass

    # Fall back to template manifest.yml artifact list when manifest has no artifacts
    # (projects created before artifact persistence was introduced)
    if not artifacts:
        tmpl_manifest = manifest_mod.load_template_manifest(template_root, template_name)
        if tmpl_manifest:
            artifacts = tmpl_manifest.get("artifacts", {})

    # Derive feature flags from artifact selections
    has_lattice_physics = "lattice-physics" in artifacts.get("rules", [])
    selected_hooks = artifacts.get("hooks", [])

    ctx = {
        "project_name": project_name,
        "package_name": package_name,
        "project_root": str(project_root_override) if project_root_override else str(project_dir.absolute()),
        "current_python_env": (
            config.get("execution", {}).get("python_env_path")
            or sys.executable
        ),
        "template_name": template_name,
        "data_bundle": data_bundle,
        "facility_name": config.get("facility_name", project_name),
        "system_timezone": config.get("system", {}).get("timezone", "UTC"),
        "has_lattice_physics": has_lattice_physics,
        "selected_hooks": selected_hooks,
    }

    # Derive channel finder configuration
    channel_finder = config.get("channel_finder")
    if channel_finder and "channel-finder" in artifacts.get("agents", []):
        pipeline_mode = channel_finder.get("pipeline_mode", "hierarchical")
        ctx["channel_finder_pipeline"] = pipeline_mode
        ctx["channel_finder_mode"] = pipeline_mode
        ctx["default_pipeline"] = pipeline_mode

        # Embed hierarchy info at render time so the agent doesn't need
        # a separate hierarchy_info() tool call.
        if pipeline_mode == "hierarchical":
            try:
                db_path = (
                    channel_finder.get("pipelines", {})
                    .get("hierarchical", {})
                    .get("database", {})
                    .get("path", "")
                )
                if db_path:
                    from osprey.services.channel_finder.databases.hierarchical import (
                        HierarchicalChannelDatabase,
                    )

                    resolved = (project_dir / db_path).resolve()
                    db = HierarchicalChannelDatabase(str(resolved))
                    ctx["channel_finder_hierarchy"] = {
                        "hierarchy_levels": db.hierarchy_levels,
                        "hierarchy_config": db.hierarchy_config,
                        "naming_pattern": db.naming_pattern,
                    }
            except Exception:
                logger.warning(
                    "Could not load hierarchy info for template rendering",
                    exc_info=True,
                )

    ctx.setdefault("channel_finder_hierarchy", None)

    # Claude Code server + agent resolution (data-driven registry)
    claude_code_config = config.get("claude_code", {})
    ctx["facility_permissions"] = claude_code_config.get("permissions", {})

    from osprey.registry.mcp import resolve_agents, resolve_servers

    ctx["servers"] = resolve_servers(claude_code_config, ctx)
    ctx["agents"] = resolve_agents(claude_code_config, ctx, project_dir, ctx["servers"])

    ctx["enabled_servers"] = {s["name"] for s in ctx["servers"] if s["enabled"]}
    ctx["enabled_agents"] = {a["name"] for a in ctx["agents"] if a["enabled"]}
    # User-owned files: regen skips these, users edit in-place
    ctx["user_owned"] = config.get("prompts", {}).get("user_owned", [])

    # Textbooks root -- resolve relative to project directory (repo root)
    _textbooks_dir = project_dir.parent / "data" / "textbooks"
    ctx["textbooks_root"] = str(_textbooks_dir) if _textbooks_dir.is_dir() else None
    # Tilde variant for permission matching (models abbreviate /Users/x to ~)
    import os as _os

    _home = _os.path.expanduser("~")
    if ctx["textbooks_root"] and ctx["textbooks_root"].startswith(_home):
        ctx["textbooks_root_tilde"] = "~" + ctx["textbooks_root"][len(_home) :]
    else:
        ctx["textbooks_root_tilde"] = None

    # Model provider resolution for Claude Code
    from osprey.cli.claude_code_resolver import ClaudeCodeModelResolver

    api_providers = config.get("api", {}).get("providers", {})
    try:
        model_spec = ClaudeCodeModelResolver.resolve(claude_code_config, api_providers)
    except ValueError as exc:
        warnings.warn(str(exc), stacklevel=2)
        model_spec = None
    ctx["claude_code_model_spec"] = model_spec

    # Write tools blocked by the writes kill switch (for hook_config.json)
    ctx["control_system_write_tools"] = config.get("control_system", {}).get("write_tools", [])

    # Control system type for protocol-aware safety rules
    ctx["control_system_type"] = config.get("control_system", {}).get("type", "mock")

    return ctx


def compute_regen_summary(ctx: dict) -> dict:
    """Compute active/disabled server and agent lists from template context.

    Args:
        ctx: Template context dict with ``servers`` and ``agents`` lists
             (populated by ``resolve_servers`` / ``resolve_agents``).

    Returns:
        Dict with active_servers, disabled_servers, extra_servers,
        active_agents, disabled_agents keys.
    """
    servers = ctx.get("servers", [])
    agents = ctx.get("agents", [])

    return {
        "active_servers": [s["name"] for s in servers if s["enabled"]],
        "disabled_servers": [s["name"] for s in servers if not s["enabled"]],
        "extra_servers": [s["name"] for s in servers if s.get("is_custom")],
        "active_agents": [a["name"] for a in agents if a["enabled"]],
        "disabled_agents": [a["name"] for a in agents if not a["enabled"]],
    }


def is_user_owned(rel_path: str, ctx: dict) -> bool:
    """Check if a file is user-owned (regen should skip it).

    User-owned files are listed in ``prompts.user_owned`` in config.yml.
    During init (empty list), nothing is user-owned so all files are written.
    Agent and skill files are never user-owned (always auto-managed).

    Args:
        rel_path: Relative path from project root (e.g. ".claude/rules/safety.md")
        ctx: Template context (must contain "user_owned" key)
    """
    if rel_path.startswith(".claude/agents/"):
        return False  # agents always auto-managed
    if rel_path.startswith(".claude/skills/"):
        return False  # skills always auto-managed
    user_owned = ctx.get("user_owned", [])
    if not user_owned:
        return False
    registry = PromptCatalog.default()
    art = registry.get_by_output(rel_path)
    return art is not None and art.canonical_name in user_owned


def auto_register_user_owned(project_dir: Path, canonical_name: str):
    """Add a canonical name to ``prompts.user_owned`` in config.yml.

    Used during init to mark facility.md as user-owned so regen
    never overwrites user customizations.  Uses ruamel.yaml round-trip
    mode to preserve comments and formatting.
    """
    from osprey.utils.config_writer import config_add_to_list

    config_path = project_dir / "config.yml"
    if not config_path.exists():
        return
    config_add_to_list(config_path, ["prompts", "user_owned"], canonical_name)


def output_path_to_canonical(output_path: str, registry: PromptCatalog) -> str | None:
    """Reverse-lookup: map an output file path to its canonical artifact name."""
    art = registry.get_by_output(output_path)
    return art.canonical_name if art else None


def _build_framework_hook_rules(
    selected_hooks: list[str],
) -> tuple[list[dict], list[dict]]:
    """Build HookRule dicts for standalone framework hooks.

    Resolves each selected hook name to its file path, parses frontmatter,
    and builds wiring entries for hooks that declare ``wiring: standalone``.

    Returns:
        ``(pre_rules, post_rules)`` — same dict shape as server hook rules.
    """
    from osprey.cli.templates.artifact_library import parse_hook_frontmatter, resolve_artifact

    pre_rules: list[tuple[int, dict]] = []  # (safety_layer, rule)
    post_rules: list[tuple[int, dict]] = []

    for hook_name in selected_hooks:
        try:
            hook_path = resolve_artifact("hooks", hook_name)
        except ValueError:
            continue

        meta = parse_hook_frontmatter(hook_path)
        if meta is None:
            continue

        rule = {
            "matcher": meta["tools"],
            "hooks": [
                {
                    "type": "command",
                    "command": f'python "$CLAUDE_PROJECT_DIR/.claude/hooks/{hook_path.name}"',
                    "timeout": meta["timeout"],
                }
            ],
        }

        if meta["event"] == "PreToolUse":
            pre_rules.append((meta["safety_layer"], rule))
        elif meta["event"] == "PostToolUse":
            post_rules.append((meta["safety_layer"], rule))

    # Sort by safety_layer ascending (lower = outermost gate)
    pre_rules.sort(key=lambda x: x[0])
    post_rules.sort(key=lambda x: x[0])

    return [r for _, r in pre_rules], [r for _, r in post_rules]


def create_claude_code_integration(
    template_root: Path,
    jinja_env,
    project_dir: Path,
    ctx: dict,
    allowed_outputs: set[str] | None = None,
):
    """Create Claude Code integration files for the project.

    Copies template files from templates/claude_code/ into the project,
    applying dotless-to-dotted naming convention (claude/ -> .claude/,
    mcp.json.j2 -> .mcp.json).

    User-owned files (listed in ``ctx["user_owned"]``) are skipped during
    regeneration, preserving user customizations.

    When ``allowed_outputs`` is provided (from a template manifest), only
    files whose output path is in the set are generated. Config artifacts
    (CLAUDE.md, .mcp.json, .claude/settings.json) should already be in the
    set. If ``allowed_outputs`` is None, all files are generated (backward compat).

    Args:
        template_root: Path to osprey's bundled templates directory
        jinja_env: Jinja2 environment for template rendering
        project_dir: Root directory of the project
        ctx: Template context variables
        allowed_outputs: If set, only generate files whose output path is in this set.
            When None, all files are generated (no manifest filtering).
    """
    claude_code_dir = template_root / "claude_code"

    if not claude_code_dir.exists():
        console.print(
            "  [warning]⚠[/warning] Claude Code templates not found — skipping",
            style="yellow",
        )
        return

    # Build framework hook rules from selected hooks' frontmatter
    fw_pre, fw_post = _build_framework_hook_rules(ctx.get("selected_hooks", []))
    ctx["framework_pre_hooks"] = fw_pre
    ctx["framework_post_hooks"] = fw_post

    files_created = 0

    # 1. Render mcp.json.j2 -> .mcp.json
    mcp_template = claude_code_dir / "mcp.json.j2"
    if mcp_template.exists() and not is_user_owned(".mcp.json", ctx):
        render_template(jinja_env, "claude_code/mcp.json.j2", ctx, project_dir / ".mcp.json")
        files_created += 1

    # 2. Render CLAUDE.md.j2 -> CLAUDE.md
    claude_md_j2 = claude_code_dir / "CLAUDE.md.j2"
    claude_md_static = claude_code_dir / "CLAUDE.md"
    if not is_user_owned("CLAUDE.md", ctx):
        if claude_md_j2.exists():
            render_template(jinja_env, "claude_code/CLAUDE.md.j2", ctx, project_dir / "CLAUDE.md")
        elif claude_md_static.exists():
            shutil.copy2(claude_md_static, project_dir / "CLAUDE.md")
        files_created += 1

    # 2b. Create facility.md -- user-owned artifact
    # During init, render the template in-place and auto-register as
    # user-owned so regen never overwrites user customizations.
    facility_md = project_dir / ".claude" / "rules" / "facility.md"
    facility_j2 = claude_code_dir / "claude" / "rules" / "facility.md.j2"
    if allowed_outputs is not None and ".claude/rules/facility.md" not in allowed_outputs:
        pass  # Skip -- not in manifest
    elif is_user_owned(".claude/rules/facility.md", ctx):
        pass  # Skip -- user owns it
    elif not facility_md.exists() and facility_j2.exists():
        facility_md.parent.mkdir(parents=True, exist_ok=True)
        render_template(jinja_env, "claude_code/claude/rules/facility.md.j2", ctx, facility_md)
        # Auto-register as user-owned so regen preserves user edits
        auto_register_user_owned(project_dir, "rules/facility")
        files_created += 1

    # 3. Recursively copy/render claude/ -> .claude/ (dotless to dotted)
    #    Files with .j2 extension are rendered as Jinja2 templates.
    #    facility.md.j2 is handled above (create-only), so skip it here.
    claude_src = claude_code_dir / "claude"
    if claude_src.exists():
        for src_file in claude_src.rglob("*"):
            if not src_file.is_file():
                continue
            rel_path = src_file.relative_to(claude_src)

            # Skip files in _-prefixed directories (include-only fragments)
            if any(part.startswith("_") for part in rel_path.parts[:-1]):
                continue

            if src_file.suffix == ".j2":
                output_rel = rel_path.with_suffix("")
                dst_rel = f".claude/{output_rel}"

                # Skip facility.md -- handled above (create-only semantics)
                if str(output_rel) == "rules/facility.md":
                    continue

                # Skip files not in manifest (when manifest is active)
                if allowed_outputs is not None and dst_rel not in allowed_outputs:
                    continue

                # Skip user-owned files
                if is_user_owned(dst_rel, ctx):
                    continue

                # Render Jinja2 template, strip .j2 extension
                dst_file = project_dir / ".claude" / output_rel
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                template_path = f"claude_code/claude/{rel_path}"
                render_template(jinja_env, template_path, ctx, dst_file)

                # Clean up empty rendered files (template-conditional content)
                if dst_file.exists() and not dst_file.read_text(encoding="utf-8").strip():
                    dst_file.unlink()
                    # Remove empty parent dir (e.g., .claude/skills/load-lattice/)
                    if dst_file.parent != project_dir and not any(dst_file.parent.iterdir()):
                        dst_file.parent.rmdir()
                    continue
            else:
                dst_rel = f".claude/{rel_path}"

                # Skip files not in manifest (when manifest is active)
                if allowed_outputs is not None and dst_rel not in allowed_outputs:
                    continue

                # Skip user-owned files
                if is_user_owned(dst_rel, ctx):
                    continue

                dst_file = project_dir / ".claude" / rel_path
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
            files_created += 1

    # 4. Set hook scripts executable
    hooks_dir = project_dir / ".claude" / "hooks"
    if hooks_dir.exists():
        for hook in hooks_dir.iterdir():
            if hook.is_file() and hook.suffix == ".py":
                hook.chmod(hook.stat().st_mode | 0o755)

    console.print(f"  [success]✓[/success] Created {files_created} Claude Code integration file(s)")


def check_user_owned_drift(
    template_root: Path,
    jinja_env,
    project_dir: Path,
    ctx: dict,
) -> list[str]:
    """Check if framework templates changed since user claimed ownership.

    Compares the current rendered framework hash against the hash stored
    in the manifest at claim time.

    Args:
        template_root: Path to osprey's bundled templates directory
        jinja_env: Jinja2 environment for template rendering
        project_dir: Root directory of the project
        ctx: Template context dict

    Returns:
        List of canonical names whose framework template has drifted.
    """
    manifest_path = project_dir / manifest_mod.MANIFEST_FILENAME
    if not manifest_path.exists():
        return []

    try:
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    user_owned_meta = manifest_data.get("user_owned", {})
    if not user_owned_meta:
        return []

    import tempfile

    registry = PromptCatalog.default()
    claude_code_dir = template_root / "claude_code"
    drift: list[str] = []

    for canonical_name, meta in user_owned_meta.items():
        stored_hash = meta.get("framework_hash")
        if not stored_hash:
            continue

        artifact = registry.get(canonical_name)
        if artifact is None:
            continue

        template_file = claude_code_dir / artifact.template_path
        if not template_file.exists():
            continue

        current_hash = None
        try:
            if template_file.suffix == ".j2":
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=template_file.stem, delete=False, encoding="utf-8"
                ) as tmp:
                    template_rel = f"claude_code/{artifact.template_path}"
                    template = jinja_env.get_template(template_rel)
                    rendered = template.render(**ctx)
                    tmp.write(rendered)
                    tmp_path = Path(tmp.name)
                current_hash = f"sha256:{manifest_mod.sha256_file(tmp_path)}"
                tmp_path.unlink(missing_ok=True)
            else:
                current_hash = f"sha256:{manifest_mod.sha256_file(template_file)}"
        except Exception:
            continue

        if current_hash and current_hash != stored_hash:
            drift.append(canonical_name)
            console.print(
                f"  [warning]⚠[/warning] Framework updated {canonical_name} since you claimed it.\n"
                f"    Run `osprey prompts diff {canonical_name}` to review changes.",
                style="yellow",
            )

    return drift


def regenerate_claude_code(
    template_root: Path,
    jinja_env,
    project_dir: Path,
    dry_run: bool = False,
    project_root_override: Path | str | None = None,
) -> dict:
    """Regenerate Claude Code artifacts from current config.yml.

    Reads config.yml, reconstructs the template context, and re-renders
    all Claude Code .j2 templates. Backs up existing files before overwriting.

    Args:
        template_root: Path to osprey's bundled templates directory
        jinja_env: Jinja2 environment for template rendering
        project_dir: Root directory of the project
        dry_run: If True, report what would change without writing files
        project_root_override: If set, use this path as ``project_root`` in
            the rendered context instead of ``project_dir``.  ``project_dir``
            is still used for all file I/O (reading config, writing output).

    Returns:
        Dict with 'changed', 'unchanged', and 'backup_dir' keys

    Raises:
        FileNotFoundError: If config.yml doesn't exist in project_dir
    """
    config_file = project_dir / "config.yml"
    if not config_file.exists():
        raise FileNotFoundError(
            f"No config.yml found in {project_dir}. Are you in an OSPREY project directory?"
        )

    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    config = resolve_env_vars(config)

    ctx = build_claude_code_context(
        template_root, jinja_env, project_dir, config,
        project_root_override=project_root_override,
    )

    # Resolve allowed_outputs from .osprey-manifest.json artifact list.
    # Fall back to loading the template's manifest.yml for legacy projects.
    template_name = ctx.get("template_name", "control_assistant")
    osprey_manifest_path = project_dir / manifest_mod.MANIFEST_FILENAME
    regen_manifest: dict | None = None
    stored_artifacts: dict | None = None
    if osprey_manifest_path.exists():
        try:
            osprey_manifest_data = json.loads(osprey_manifest_path.read_text(encoding="utf-8"))
            stored_artifacts = osprey_manifest_data.get("artifacts") or None
            if stored_artifacts:
                # Build an in-memory manifest dict in the same format as manifest.yml
                regen_manifest = {"artifacts": stored_artifacts}
        except (json.JSONDecodeError, OSError):
            pass
    if regen_manifest is None:
        regen_manifest = manifest_mod.load_template_manifest(template_root, template_name)

    allowed_outputs = (
        manifest_mod.resolve_manifest_outputs(regen_manifest) if regen_manifest else None
    )

    # Filter agents to allowed outputs
    if allowed_outputs is not None:
        ctx["agents"] = [
            a for a in ctx["agents"] if f".claude/agents/{a['name']}.md" in allowed_outputs
        ]

    # Collect checksums of existing Claude Code files before regeneration.
    # When stored_artifacts are present, derive tracked files from the manifest;
    # otherwise fall back to the template's static tracked-file list.
    if stored_artifacts and allowed_outputs is not None:
        claude_code_files = sorted(allowed_outputs)
    else:
        claude_code_files = manifest_mod.get_tracked_files(
            template_root, template_name, project_dir
        )
    agents_dir = project_dir / ".claude" / "agents"
    if agents_dir.exists():
        for agent_file in agents_dir.iterdir():
            if agent_file.is_file() and agent_file.suffix == ".md":
                rel = f".claude/agents/{agent_file.name}"
                if rel not in claude_code_files:
                    claude_code_files.append(rel)

    old_checksums = {}
    for rel_path in claude_code_files:
        file_path = project_dir / rel_path
        if file_path.exists():
            old_checksums[rel_path] = manifest_mod.sha256_file(file_path)

    if dry_run:
        # Render to temp dir and compare
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            # Create necessary subdirectories
            (tmp_dir / ".claude").mkdir(parents=True, exist_ok=True)
            create_claude_code_integration(template_root, jinja_env, tmp_dir, ctx, allowed_outputs)

            changed = []
            unchanged = []
            for rel_path in claude_code_files:
                tmp_file = tmp_dir / rel_path
                orig_file = project_dir / rel_path
                if tmp_file.exists():
                    new_checksum = manifest_mod.sha256_file(tmp_file)
                    old_checksum = old_checksums.get(rel_path)
                    if old_checksum != new_checksum:
                        changed.append(rel_path)
                    else:
                        unchanged.append(rel_path)
                elif orig_file.exists():
                    changed.append(rel_path)  # File would be removed

            # Check for new files in tmp that aren't in old list
            for tmp_file in Path(tmp).rglob("*"):
                if not tmp_file.is_file():
                    continue
                rel = str(tmp_file.relative_to(tmp))
                if rel not in claude_code_files and rel not in changed:
                    changed.append(rel)

            summary = compute_regen_summary(ctx)
            return {"changed": changed, "unchanged": unchanged, "backup_dir": None, **summary}

    # Create backup
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    backup_dir = project_dir / "_agent_data" / "backup" / f"claude-code-{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    for rel_path in claude_code_files:
        src = project_dir / rel_path
        if src.exists():
            dst = backup_dir / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    # Regenerate
    create_claude_code_integration(template_root, jinja_env, project_dir, ctx, allowed_outputs)

    # Compare checksums
    changed = []
    unchanged = []
    for rel_path in claude_code_files:
        file_path = project_dir / rel_path
        if file_path.exists():
            new_checksum = manifest_mod.sha256_file(file_path)
            old_checksum = old_checksums.get(rel_path)
            if old_checksum != new_checksum:
                changed.append(rel_path)
            else:
                unchanged.append(rel_path)

    # Check for newly created files (e.g., new agents)
    new_agents_dir = project_dir / ".claude" / "agents"
    if new_agents_dir.exists():
        for agent_file in new_agents_dir.iterdir():
            if agent_file.is_file() and agent_file.suffix == ".md":
                rel = f".claude/agents/{agent_file.name}"
                if rel not in claude_code_files and rel not in changed:
                    changed.append(rel)

    # Check for user-owned drift (framework template changed since claiming)
    drift_warnings = check_user_owned_drift(template_root, jinja_env, project_dir, ctx)

    # Compute active/disabled summary
    summary = compute_regen_summary(ctx)

    return {
        "changed": changed,
        "unchanged": unchanged,
        "backup_dir": str(backup_dir),
        "drift_warnings": drift_warnings,
        **summary,
    }
