"""Build command — assemble a facility-specific assistant from a build profile.

Reads a YAML build profile that specifies a base template, config overrides,
file overlays, and MCP server definitions. Produces a standalone, self-contained
project directory (wipe-and-rebuild safe).

Usage:
    osprey build my-assistant profile.yml
    osprey build my-assistant profile.yml --variant prod --force
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import click

from .styles import Messages, Styles, console
from .templates.manager import TemplateManager

logger = logging.getLogger(__name__)


@click.command()
@click.argument("project_name")
@click.argument("profile", type=click.Path(exists=True, dir_okay=False))
@click.option("--variant", "-v", default=None, help="Profile variant to apply (e.g., prod, test)")
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default=".",
    help="Output directory for project (default: current directory)",
)
@click.option("--force", "-f", is_flag=True, help="Force overwrite if project directory exists")
def build(
    project_name: str,
    profile: str,
    variant: str | None,
    output_dir: str,
    force: bool,
) -> None:
    """Build a facility-specific assistant from a profile.

    Assembles a standalone project by rendering a base template, applying
    config overrides, copying overlay files, and injecting MCP servers.

    PROJECT_NAME: Name of the project directory to create

    PROFILE: Path to a YAML build profile

    Examples:

    \b
      # Build from profile
      $ osprey build als-test ~/profiles/als-dev.yml

      # Build with variant
      $ osprey build als-prod ~/profiles/als-dev.yml --variant prod

      # Force overwrite
      $ osprey build als-test ~/profiles/als-dev.yml --force
    """
    from .build_profile import load_profile
    from .init_cmd import _clear_claude_code_project_state

    console.print(f"🔨 Building project: [header]{project_name}[/header]")

    try:
        # 1. Load and validate profile
        profile_path = Path(profile).resolve()
        profile_dir = profile_path.parent
        build_profile = load_profile(profile_path, variant=variant)

        console.print(f"  📋 Profile: [accent]{build_profile.name}[/accent]")
        console.print(f"  📦 Template: [accent]{build_profile.base_template}[/accent]")
        if variant:
            console.print(f"  🔀 Variant: [accent]{variant}[/accent]")

        # 2. Resolve output path
        output_path = Path(output_dir).resolve()
        project_path = output_path / project_name

        # 3. Handle --force / directory existence
        if project_path.exists():
            if force:
                msg = Messages.warning(f"Removing existing directory: {project_path}")
                console.print(f"  ⚠️  {msg}")
                shutil.rmtree(project_path)
                console.print(f"  {Messages.success('Removed existing directory')}")
            else:
                console.print(
                    f"❌ Directory '{project_path}' already exists.\n"
                    f"   Use --force to overwrite, or choose a different name.",
                    style=Styles.ERROR,
                )
                raise click.Abort()

        # 4. Clear Claude Code project state
        _clear_claude_code_project_state(project_path)

        # 5. Build context from profile fields
        context: dict[str, Any] = {}
        if build_profile.provider:
            context["default_provider"] = build_profile.provider
        if build_profile.model:
            context["default_model"] = build_profile.model
        if build_profile.channel_finder_mode:
            context["channel_finder_mode"] = build_profile.channel_finder_mode

        # 6. Create project from template
        manager = TemplateManager()
        project_path = manager.create_project(
            project_name=project_name,
            output_dir=output_path,
            template_name=build_profile.base_template,
            registry_style="extend",
            context=context,
        )
        console.print("  ✓ Base template rendered", style=Styles.SUCCESS)

        # 7. Apply config overrides
        if build_profile.config:
            _apply_config_overrides(project_path, build_profile.config)
            console.print(
                f"  ✓ Applied {len(build_profile.config)} config override(s)",
                style=Styles.SUCCESS,
            )

        # 8. Copy overlay files
        if build_profile.overlay:
            _copy_overlay_files(profile_dir, project_path, build_profile.overlay)
            console.print(
                f"  ✓ Copied {len(build_profile.overlay)} overlay(s)", style=Styles.SUCCESS
            )

        # 9. Inject MCP servers
        if build_profile.mcp_servers:
            _inject_mcp_servers(project_path, build_profile.mcp_servers)
            console.print(
                f"  ✓ Injected {len(build_profile.mcp_servers)} MCP server(s)",
                style=Styles.SUCCESS,
            )

        # 10. Generate manifest
        manifest_context = {
            "default_provider": build_profile.provider or "anthropic",
            "default_model": build_profile.model or "haiku",
        }
        if build_profile.channel_finder_mode:
            manifest_context["channel_finder_mode"] = build_profile.channel_finder_mode
        manager.generate_manifest(
            project_dir=project_path,
            project_name=project_name,
            template_name=build_profile.base_template,
            registry_style="extend",
            context=manifest_context,
        )

        # 11. Git init + commit
        _git_init_and_commit(project_path)

        console.print(f"\n✅ Project built successfully at: [bold]{project_path}[/bold]")

    except click.Abort:
        raise
    except ValueError as e:
        console.print(f"❌ Error: {e}", style=Styles.ERROR)
        raise click.Abort() from e
    except Exception as e:
        console.print(f"❌ Unexpected error: {e}", style=Styles.ERROR)
        import traceback

        console.print(traceback.format_exc(), style=Styles.DIM)
        raise click.Abort() from e


def _apply_config_overrides(project_path: Path, config_dict: dict[str, Any]) -> None:
    """Apply dot-notation config overrides to the project's config.yml."""
    from osprey.utils.config_writer import config_update_fields

    config_path = project_path / "config.yml"
    if not config_path.exists():
        logger.warning("config.yml not found at %s — skipping config overrides", config_path)
        return
    config_update_fields(config_path, config_dict)


def _copy_overlay_files(
    profile_dir: Path, project_path: Path, overlay_dict: dict[str, str]
) -> None:
    """Copy overlay files/directories from profile dir into the project.

    Args:
        profile_dir: Directory containing the profile and overlay sources.
        project_path: Root of the built project.
        overlay_dict: Mapping of source (relative to profile_dir) → destination
            (relative to project_path).
    """
    for src_rel, dst_rel in overlay_dict.items():
        src = (profile_dir / src_rel).resolve()
        dst = (project_path / dst_rel).resolve()

        # Path traversal guard
        if not dst.is_relative_to(project_path.resolve()):
            raise ValueError(f"Overlay destination escapes project root: {dst_rel}")

        dst.parent.mkdir(parents=True, exist_ok=True)

        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

        logger.info("Overlay: %s → %s", src_rel, dst_rel)


def _inject_mcp_servers(project_path: Path, mcp_servers: dict[str, Any]) -> None:
    """Inject MCP server definitions into .mcp.json and .claude/settings.json.

    For each server:
      - Adds command/args/env to .mcp.json mcpServers
      - Adds tool permissions to .claude/settings.json permissions.allow/ask
      - Resolves {project_root} placeholders in args and env values
    """
    from .build_profile import McpServerDef

    # --- .mcp.json ---
    mcp_json_path = project_path / ".mcp.json"
    if mcp_json_path.exists():
        mcp_data = json.loads(mcp_json_path.read_text(encoding="utf-8"))
    else:
        mcp_data = {}

    mcp_servers_section = mcp_data.setdefault("mcpServers", {})

    for name, server in mcp_servers.items():
        if not isinstance(server, McpServerDef):
            continue
        if name in mcp_servers_section:
            console.print(
                f"  ⚠️  MCP server '{name}' already exists in .mcp.json — skipping",
                style=Styles.WARNING,
            )
            continue

        entry: dict[str, Any] = {
            "command": server.command,
            "args": [_resolve_placeholders(a, project_path) for a in server.args],
        }
        if server.env:
            entry["env"] = {
                k: _resolve_placeholders(v, project_path) for k, v in server.env.items()
            }
        mcp_servers_section[name] = entry

    mcp_json_path.write_text(json.dumps(mcp_data, indent=2) + "\n", encoding="utf-8")

    # --- .claude/settings.json ---
    settings_path = project_path / ".claude" / "settings.json"
    if settings_path.exists():
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    else:
        settings = {}

    permissions = settings.setdefault("permissions", {})
    allow_list: list[str] = permissions.setdefault("allow", [])
    ask_list: list[str] = permissions.setdefault("ask", [])

    for name, server in mcp_servers.items():
        if not isinstance(server, McpServerDef):
            continue
        for tool in server.permissions.get("allow", []):
            entry = f"mcp__{name}__{tool}"
            if entry not in allow_list:
                allow_list.append(entry)
        for tool in server.permissions.get("ask", []):
            entry = f"mcp__{name}__{tool}"
            if entry not in ask_list:
                ask_list.append(entry)

    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


def _resolve_placeholders(value: str, project_path: Path) -> str:
    """Replace {project_root} with the actual project path."""
    return value.replace("{project_root}", str(project_path))


def _git_init_and_commit(project_path: Path) -> None:
    """Initialize a git repo and create an initial commit."""
    import os
    import subprocess

    # Check if project is inside an existing git repo
    inside_existing_repo = False
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=project_path,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            parent_root = Path(result.stdout.strip()).resolve()
            if parent_root != project_path.resolve():
                inside_existing_repo = True
    except FileNotFoundError:
        pass

    try:
        subprocess.run(["git", "init"], cwd=project_path, check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=project_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial project from osprey build"],
            cwd=project_path,
            check=True,
            capture_output=True,
            env={
                **os.environ,
                "GIT_AUTHOR_NAME": "osprey",
                "GIT_AUTHOR_EMAIL": "osprey@build",
                "GIT_COMMITTER_NAME": "osprey",
                "GIT_COMMITTER_EMAIL": "osprey@build",
            },
        )
        console.print("  ✓ Initialized git repository", style=Styles.SUCCESS)
        if inside_existing_repo:
            console.print(
                f"  ⚠️  Note: created a nested git repo inside {parent_root}.\n"
                "     This is required for Claude Code project isolation (it uses\n"
                "     the git root to discover .claude/ settings). The parent repo\n"
                "     will treat this directory as opaque.",
                style=Styles.WARNING,
            )
    except FileNotFoundError:
        console.print(
            "  ⚠️  git not found — project created but not initialized as a git repo.\n"
            "     Claude Code requires git. Run 'git init && git add . && git commit'"
            " manually.",
            style=Styles.WARNING,
        )
    except subprocess.CalledProcessError:
        console.print(
            "  ⚠️  git init succeeded but initial commit failed.\n"
            "     Run 'git add . && git commit' manually.",
            style=Styles.WARNING,
        )
