"""CLI subcommands for managing prompt artifact ownership.

Provides ``osprey prompts list|claim|diff|unclaim`` commands
for inspecting and customizing the Claude Code prompt artifacts
that OSPREY generates during ``osprey init`` / ``osprey claude regen``.
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

import click
import yaml

from osprey.cli.styles import console
from osprey.cli.templates.manager import TemplateManager
from osprey.services.prompts.catalog import PromptCatalog
from osprey.services.prompts.ownership import (
    get_user_owned,
    update_config_add_user_owned,
    update_config_remove_user_owned,
    update_manifest_add_user_owned,
    update_manifest_remove_user_owned,
)
from osprey.utils.config import resolve_env_vars


def _load_config(project_dir: Path) -> dict[str, Any]:
    """Load and return config.yml from project_dir."""
    config_file = project_dir / "config.yml"
    if not config_file.exists():
        raise click.ClickException(
            f"No config.yml found in {project_dir}. Are you in an OSPREY project directory?"
        )
    with open(config_file, encoding="utf-8") as f:
        return resolve_env_vars(yaml.safe_load(f) or {})


@click.group(name="prompts", invoke_without_command=True)
@click.pass_context
def prompts(ctx):
    """Manage prompt artifact ownership.

    Framework-managed prompt artifacts can be claimed per-facility
    for in-place editing. Use the subcommands to inspect, claim, diff,
    and unclaim artifacts.

    Examples:

    \b
      osprey prompts list                       # Show all artifacts
      osprey prompts claim agents/channel-finder # Claim for editing
      osprey prompts diff agents/channel-finder  # Compare yours vs framework
      osprey prompts unclaim agents/channel-finder # Restore framework management
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@prompts.command(name="list")
@click.option(
    "--project",
    "-p",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    default=None,
    help="Project directory (default: current directory)",
)
def list_artifacts(project):
    """List all prompt artifacts and their ownership status."""
    project_dir = Path(project) if project else Path.cwd()

    try:
        config = _load_config(project_dir)
    except click.ClickException:
        config = {}

    user_owned = get_user_owned(config)
    registry = PromptCatalog.default()

    framework_managed = []
    owned = []

    for art in registry.all_artifacts():
        if art.canonical_name in user_owned:
            owned.append(art)
        else:
            framework_managed.append(art)

    console.print("\n[bold]Prompt Artifacts[/bold]\n")

    if framework_managed:
        console.print("  [dim]Framework-managed:[/dim]")
        for art in framework_managed:
            console.print(
                f"    [success]\u2713[/success] {art.canonical_name:<35s} {art.description}"
            )

    if owned:
        console.print("\n  [dim]User-owned:[/dim]")
        for art in owned:
            console.print(f"    [bold]\u2605[/bold] {art.canonical_name:<35s} {art.output_path}")

    console.print()


@prompts.command(name="claim")
@click.argument("name")
@click.option(
    "--project",
    "-p",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    default=None,
    help="Project directory (default: current directory)",
)
def claim(name, project):
    """Claim ownership of a framework artifact for in-place editing.

    If the file doesn't exist yet, renders the framework template in-place
    at the canonical output path. If it already exists, just marks it as
    user-owned. Regen will skip user-owned files.

    Examples:

    \b
      osprey prompts claim agents/channel-finder
      osprey prompts claim rules/safety
    """
    project_dir = Path(project) if project else Path.cwd()
    registry = PromptCatalog.default()
    artifact = registry.get(name)

    if artifact is None:
        known = ", ".join(registry.all_names())
        raise click.ClickException(f"Unknown artifact '{name}'. Known artifacts:\n  {known}")

    config = _load_config(project_dir)
    user_owned = get_user_owned(config)

    if name in user_owned:
        raise click.ClickException(
            f"'{name}' is already user-owned. Edit it directly at {artifact.output_path}."
        )

    # Build template context
    manager = TemplateManager()
    from osprey.cli.templates.claude_code import build_claude_code_context

    ctx = build_claude_code_context(manager.template_root, manager.jinja_env, project_dir, config)

    # If file doesn't exist, render the framework template in-place
    output_file = project_dir / artifact.output_path
    if not output_file.exists():
        claude_code_dir = manager.template_root / "claude_code"
        template_file = claude_code_dir / artifact.template_path

        if not template_file.exists():
            raise click.ClickException(f"Template file not found: {artifact.template_path}")

        if template_file.suffix == ".j2":
            template_rel = f"claude_code/{artifact.template_path}"
            template = manager.jinja_env.get_template(template_rel)
            content = template.render(**ctx)
        else:
            content = template_file.read_text(encoding="utf-8")

        if not content.strip():
            console.print(
                "[warning]\u26a0[/warning] Template renders to empty content "
                "(likely a Jinja2 condition is not met for your config).",
                style="yellow",
            )
            if not click.confirm("Create an empty file anyway?"):
                return

        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(content, encoding="utf-8")
        if output_file.suffix == ".py":
            output_file.chmod(output_file.stat().st_mode | 0o755)

        console.print(f"  [success]\u2713[/success] Rendered {name} \u2192 {artifact.output_path}")
    else:
        console.print(f"  [success]\u2713[/success] File already exists at {artifact.output_path}")

    # Update config.yml
    added = update_config_add_user_owned(project_dir, name)
    if added:
        console.print(
            f"  [success]\u2713[/success] Updated config.yml \u2014 prompts.user_owned += {name}"
        )

    # Update manifest
    update_manifest_add_user_owned(project_dir, manager, ctx, name)

    console.print(f"\n  Edit [path]{artifact.output_path}[/path] \u2014 regen will skip it.\n")


@prompts.command(name="diff")
@click.argument("name")
@click.option(
    "--project",
    "-p",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    default=None,
    help="Project directory (default: current directory)",
)
def diff(name, project):
    """Show diff between a framework template and your file.

    Renders the current framework template and compares it against
    your file at the canonical output path using a unified diff.

    Examples:

    \b
      osprey prompts diff agents/channel-finder
      osprey prompts diff rules/facility
    """
    project_dir = Path(project) if project else Path.cwd()
    config = _load_config(project_dir)
    user_owned = get_user_owned(config)

    if name not in user_owned:
        raise click.ClickException(
            f"'{name}' is not user-owned in config.yml. Run `osprey prompts claim {name}` first."
        )

    registry = PromptCatalog.default()
    artifact = registry.get(name)
    if artifact is None:
        raise click.ClickException(f"Unknown artifact '{name}'.")

    # Read user's file from canonical output path
    user_file = project_dir / artifact.output_path
    if not user_file.exists():
        raise click.ClickException(f"File not found: {artifact.output_path}")
    user_lines = user_file.read_text(encoding="utf-8").splitlines(keepends=True)

    # Render framework template
    manager = TemplateManager()
    from osprey.cli.templates.claude_code import build_claude_code_context

    ctx = build_claude_code_context(manager.template_root, manager.jinja_env, project_dir, config)
    claude_code_dir = manager.template_root / "claude_code"
    template_file = claude_code_dir / artifact.template_path

    if template_file.suffix == ".j2":
        template_rel = f"claude_code/{artifact.template_path}"
        template = manager.jinja_env.get_template(template_rel)
        framework_content = template.render(**ctx)
    else:
        framework_content = template_file.read_text(encoding="utf-8")

    framework_lines = framework_content.splitlines(keepends=True)

    # Generate unified diff
    diff_lines = difflib.unified_diff(
        framework_lines,
        user_lines,
        fromfile=f"framework:{artifact.template_path}",
        tofile=f"yours:{artifact.output_path}",
    )

    output = "".join(diff_lines)
    if output:
        click.echo(output)
    else:
        console.print(
            "[success]\u2713[/success] Your file matches the current framework template "
            "\u2014 no differences."
        )


@prompts.command(name="unclaim")
@click.argument("name")
@click.option(
    "--project",
    "-p",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    default=None,
    help="Project directory (default: current directory)",
)
def unclaim(name, project):
    """Release ownership and restore framework management.

    Removes the artifact from the user_owned list in config.yml and
    .osprey-manifest.json. The next ``osprey claude regen`` will
    overwrite the file with the framework template.

    Examples:

    \b
      osprey prompts unclaim agents/channel-finder
      osprey prompts unclaim rules/safety
    """
    project_dir = Path(project) if project else Path.cwd()
    config = _load_config(project_dir)
    user_owned = get_user_owned(config)

    if name not in user_owned:
        raise click.ClickException(f"'{name}' is not user-owned in config.yml.")

    # Remove from config.yml
    update_config_remove_user_owned(project_dir, name)

    # Remove from manifest
    update_manifest_remove_user_owned(project_dir, name)

    console.print(f"  [success]\u2713[/success] Released ownership of {name}")
    console.print("\n  Next `osprey claude regen` will overwrite with the framework template.\n")
