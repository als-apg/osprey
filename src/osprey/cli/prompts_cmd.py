"""CLI subcommands for managing prompt artifact overrides.

Provides ``osprey prompts list|scaffold|diff|unoverride`` commands
for inspecting and customizing the Claude Code prompt artifacts
that OSPREY generates during ``osprey init`` / ``osprey claude regen``.
"""

from __future__ import annotations

import difflib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
import yaml

from osprey.cli.prompt_registry import PromptRegistry
from osprey.cli.styles import console
from osprey.cli.templates import MANIFEST_FILENAME, TemplateManager


def _load_config(project_dir: Path) -> dict[str, Any]:
    """Load and return config.yml from project_dir."""
    config_file = project_dir / "config.yml"
    if not config_file.exists():
        raise click.ClickException(
            f"No config.yml found in {project_dir}. "
            "Are you in an OSPREY project directory?"
        )
    with open(config_file, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get_overrides(config: dict) -> dict[str, str]:
    """Extract prompts.overrides dict from config."""
    return config.get("prompts", {}).get("overrides", {})


@click.group(name="prompts", invoke_without_command=True)
@click.pass_context
def prompts(ctx):
    """Manage prompt artifact overrides.

    Framework-managed prompt artifacts can be customized per-facility
    using overrides. Use the subcommands to inspect, scaffold, diff,
    and remove overrides.

    Examples:

    \b
      osprey prompts list                       # Show all artifacts
      osprey prompts scaffold agents/channel-finder  # Create editable copy
      osprey prompts diff agents/channel-finder      # Compare override vs framework
      osprey prompts unoverride agents/channel-finder # Remove override
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
    """List all prompt artifacts and their override status."""
    project_dir = Path(project) if project else Path.cwd()

    try:
        config = _load_config(project_dir)
    except click.ClickException:
        config = {}

    overrides = _get_overrides(config)
    registry = PromptRegistry.default()

    framework_managed = []
    overridden = []

    for art in registry.all_artifacts():
        if art.canonical_name in overrides:
            overridden.append((art, overrides[art.canonical_name]))
        else:
            framework_managed.append(art)

    console.print("\n[bold]Prompt Artifacts[/bold]\n")

    if framework_managed:
        console.print("  [dim]Framework-managed:[/dim]")
        for art in framework_managed:
            console.print(
                f"    [success]\u2713[/success] {art.canonical_name:<35s} {art.description}"
            )

    if overridden:
        console.print("\n  [dim]Overridden:[/dim]")
        for art, path in overridden:
            console.print(
                f"    [bold]\u2605[/bold] {art.canonical_name:<35s} {path}"
            )

    console.print()


@prompts.command(name="scaffold")
@click.argument("name")
@click.option(
    "--project",
    "-p",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    default=None,
    help="Project directory (default: current directory)",
)
def scaffold(name, project):
    """Create an editable override from a framework artifact.

    Renders the framework template with the current config context and
    writes it to overrides/<output_path>. Also adds the override entry
    to config.yml and updates .osprey-manifest.json.

    Examples:

    \b
      osprey prompts scaffold agents/channel-finder
      osprey prompts scaffold rules/safety
    """
    project_dir = Path(project) if project else Path.cwd()
    registry = PromptRegistry.default()
    artifact = registry.get(name)

    if artifact is None:
        known = ", ".join(registry.all_names())
        raise click.ClickException(
            f"Unknown artifact '{name}'. Known artifacts:\n  {known}"
        )

    config = _load_config(project_dir)
    overrides = _get_overrides(config)

    if name in overrides:
        existing = project_dir / overrides[name]
        if existing.exists():
            raise click.ClickException(
                f"Override for '{name}' already exists at {overrides[name]}. "
                f"Edit it directly or run `osprey prompts unoverride {name}` first."
            )

    # Build template context
    manager = TemplateManager()
    ctx = manager._build_claude_code_context(project_dir, config)

    # Render the framework template
    claude_code_dir = manager.template_root / "claude_code"
    template_file = claude_code_dir / artifact.template_path

    if not template_file.exists():
        raise click.ClickException(
            f"Template file not found: {artifact.template_path}"
        )

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
        if not click.confirm("Create an empty override file anyway?"):
            return

    # Determine override path
    override_rel = f"overrides/{artifact.output_path}"
    override_path = project_dir / override_rel
    override_path.parent.mkdir(parents=True, exist_ok=True)
    override_path.write_text(content, encoding="utf-8")
    if override_path.suffix == ".py":
        override_path.chmod(override_path.stat().st_mode | 0o755)

    console.print(
        f"  [success]\u2713[/success] Scaffolded {name} \u2192 {override_rel}"
    )

    # Update config.yml using ruamel.yaml for comment-preserving round-trip
    _update_config_add_override(project_dir, name, override_rel)

    # Update manifest
    _update_manifest_add_override(project_dir, manager, ctx, name, override_rel)

    console.print(
        f"\n  Edit [path]{override_rel}[/path] to customize, then run "
        f"[command]osprey claude regen[/command].\n"
    )


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
    """Show diff between a framework template and your override.

    Renders the current framework template and compares it against
    your override file using a unified diff.

    Examples:

    \b
      osprey prompts diff agents/channel-finder
    """
    project_dir = Path(project) if project else Path.cwd()
    config = _load_config(project_dir)
    overrides = _get_overrides(config)

    if name not in overrides:
        raise click.ClickException(
            f"'{name}' is not overridden in config.yml. "
            f"Run `osprey prompts scaffold {name}` first."
        )

    registry = PromptRegistry.default()
    artifact = registry.get(name)
    if artifact is None:
        raise click.ClickException(f"Unknown artifact '{name}'.")

    # Read override file
    override_path = project_dir / overrides[name]
    if not override_path.exists():
        raise click.ClickException(f"Override file not found: {overrides[name]}")
    override_lines = override_path.read_text(encoding="utf-8").splitlines(keepends=True)

    # Render framework template
    manager = TemplateManager()
    ctx = manager._build_claude_code_context(project_dir, config)
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
        override_lines,
        fromfile=f"framework:{artifact.template_path}",
        tofile=f"override:{overrides[name]}",
    )

    output = "".join(diff_lines)
    if output:
        click.echo(output)
    else:
        console.print(
            "[success]\u2713[/success] Override matches the current framework template — no differences."
        )


@prompts.command(name="unoverride")
@click.argument("name")
@click.option(
    "--project",
    "-p",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    default=None,
    help="Project directory (default: current directory)",
)
@click.option(
    "--delete-file/--keep-file",
    default=False,
    help="Delete the override file (default: keep it)",
)
def unoverride(name, project, delete_file):
    """Remove an override and restore framework management.

    Removes the override entry from config.yml and .osprey-manifest.json.
    By default keeps the override file on disk.

    Examples:

    \b
      osprey prompts unoverride agents/channel-finder
      osprey prompts unoverride agents/channel-finder --delete-file
    """
    project_dir = Path(project) if project else Path.cwd()
    config = _load_config(project_dir)
    overrides = _get_overrides(config)

    if name not in overrides:
        raise click.ClickException(
            f"'{name}' is not overridden in config.yml."
        )

    override_rel = overrides[name]

    # Remove from config.yml
    _update_config_remove_override(project_dir, name)

    # Remove from manifest
    _update_manifest_remove_override(project_dir, name)

    # Optionally delete the file
    if delete_file:
        override_path = project_dir / override_rel
        if override_path.exists():
            override_path.unlink()
            console.print(f"  [success]\u2713[/success] Deleted {override_rel}")
        # Clean up empty parent directories
        _cleanup_empty_dirs(override_path.parent, project_dir / "overrides")
    else:
        console.print(f"  [dim]Override file kept at {override_rel}[/dim]")

    console.print(
        f"  [success]\u2713[/success] Removed override for {name}"
    )
    console.print(
        "\n  Next `osprey claude regen` will use the framework template.\n"
    )


# ── Config.yml helpers ───────────────────────────────────────────────


def _update_config_add_override(project_dir: Path, name: str, override_rel: str):
    """Add a prompts.overrides entry to config.yml, preserving comments."""
    config_path = project_dir / "config.yml"

    try:
        from ruamel.yaml import YAML

        ry = YAML(typ="rt")
        ry.preserve_quotes = True
        with open(config_path, encoding="utf-8") as f:
            data = ry.load(f)

        if data is None:
            data = {}

        if "prompts" not in data:
            data["prompts"] = {}
        if "overrides" not in data["prompts"]:
            data["prompts"]["overrides"] = {}

        data["prompts"]["overrides"][name] = override_rel

        with open(config_path, "w", encoding="utf-8") as f:
            ry.dump(data, f)

        console.print(
            f"  [success]\u2713[/success] Updated config.yml — "
            f"prompts.overrides.{name}"
        )
    except ImportError:
        # Fallback: append to file as a comment-less YAML block
        _append_override_to_config(config_path, name, override_rel)


def _append_override_to_config(config_path: Path, name: str, override_rel: str):
    """Fallback: append override entry when ruamel.yaml is not available."""
    content = config_path.read_text(encoding="utf-8")

    if "prompts:" in content and "overrides:" in content:
        # Append under existing overrides section
        content += f"    {name}: {override_rel}\n"
    elif "prompts:" in content:
        content += f"  overrides:\n    {name}: {override_rel}\n"
    else:
        content += f"\nprompts:\n  overrides:\n    {name}: {override_rel}\n"

    config_path.write_text(content, encoding="utf-8")
    console.print(
        f"  [success]\u2713[/success] Updated config.yml — "
        f"prompts.overrides.{name}"
    )


def _update_config_remove_override(project_dir: Path, name: str):
    """Remove a prompts.overrides entry from config.yml."""
    config_path = project_dir / "config.yml"

    try:
        from ruamel.yaml import YAML

        ry = YAML(typ="rt")
        ry.preserve_quotes = True
        with open(config_path, encoding="utf-8") as f:
            data = ry.load(f)

        if data and "prompts" in data and "overrides" in data["prompts"]:
            overrides = data["prompts"]["overrides"]
            if name in overrides:
                del overrides[name]
                # Clean up empty sections
                if not overrides:
                    del data["prompts"]["overrides"]
                if not data["prompts"]:
                    del data["prompts"]

        with open(config_path, "w", encoding="utf-8") as f:
            ry.dump(data, f)
    except ImportError:
        # Fallback: use pyyaml (loses comments, but functional)
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if "prompts" in data and "overrides" in data["prompts"]:
            data["prompts"]["overrides"].pop(name, None)
            if not data["prompts"]["overrides"]:
                del data["prompts"]["overrides"]
            if not data["prompts"]:
                del data["prompts"]
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# ── Manifest helpers ─────────────────────────────────────────────────


def _update_manifest_add_override(
    project_dir: Path,
    manager: TemplateManager,
    ctx: dict,
    name: str,
    override_rel: str,
):
    """Add an override entry to .osprey-manifest.json."""
    manifest_path = project_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return

    if "overrides" not in manifest:
        manifest["overrides"] = {}

    # Compute framework hash
    registry = PromptRegistry.default()
    artifact = registry.get(name)
    framework_hash = None
    if artifact:
        claude_code_dir = manager.template_root / "claude_code"
        template_file = claude_code_dir / artifact.template_path
        if template_file.exists():
            try:
                if template_file.suffix == ".j2":
                    import tempfile

                    template_rel = f"claude_code/{artifact.template_path}"
                    template = manager.jinja_env.get_template(template_rel)
                    rendered = template.render(**ctx)
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".tmp", delete=False, encoding="utf-8"
                    ) as tmp:
                        tmp.write(rendered)
                        tmp_path = Path(tmp.name)
                    framework_hash = f"sha256:{manager._sha256_file(tmp_path)}"
                    tmp_path.unlink(missing_ok=True)
                else:
                    framework_hash = f"sha256:{manager._sha256_file(template_file)}"
            except Exception:
                pass

    entry: dict[str, Any] = {
        "override_path": override_rel,
        "scaffolded_at": datetime.now(UTC).isoformat(),
    }
    if framework_hash:
        entry["framework_hash"] = framework_hash

    manifest["overrides"][name] = entry

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)


def _update_manifest_remove_override(project_dir: Path, name: str):
    """Remove an override entry from .osprey-manifest.json."""
    manifest_path = project_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return

    overrides = manifest.get("overrides", {})
    if name in overrides:
        del overrides[name]
        if not overrides:
            manifest.pop("overrides", None)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)


def _cleanup_empty_dirs(directory: Path, stop_at: Path):
    """Remove empty directories up to stop_at."""
    current = directory
    while current != stop_at and current.exists():
        try:
            if not any(current.iterdir()):
                current.rmdir()
            else:
                break
        except OSError:
            break
        current = current.parent
